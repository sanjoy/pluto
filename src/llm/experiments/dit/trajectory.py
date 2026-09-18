"""Recorded plain-SGD trajectories for Dynamic Influence Tracker.

Step t always means the parameters *before* update t, i.e. theta^[t].  The
manifest records the actual example IDs and learning rate of every update, not
just a seed from which a future implementation would have to guess them.
Checkpoint replay is the storage-saving variant of Algorithms 1/3 in
arXiv:2502.10793v1.  There is no momentum, weight decay, or Adam state here.

The loss callback returns one scalar per example. For language modeling that
scalar is the mean token cross entropy of a complete context. Training averages
those scalars over the original batch, including during deletion replay.
Callbacks must be deterministic and sample-separable: the loss of an example
must not depend on its batch companions. They must not change model buffers or
sample random values. GPT-2 with dropout disabled meets these requirements;
batch-statistic normalization does not.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import torch

LossFunction = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
State = dict[str, torch.Tensor]
_FORMAT = "pluto-dit-sgd"
_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(model: torch.nn.Module) -> State:
    """Copy all parameters/buffers to independent CPU tensors, without grads."""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _parameter_layout(model: torch.nn.Module) -> list[dict[str, Any]]:
    # Keep aliases in the schema so loading an untied model into a tied-model
    # trajectory fails even when names/shapes happen to match. PyTorch's normal
    # parameters() iterator deduplicates ties when we actually apply SGD.
    first_names: dict[int, str] = {}
    layout = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        first_name = first_names.setdefault(id(parameter), name)
        layout.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
                "alias_of": first_name,
            }
        )
    return layout


def _state_layout(state: State) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in state.items()
    ]


def _integer(value: Any, description: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}")
    return value


def _rate(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("learning rate must be finite and positive")
    return float(value)


def _validate_shape(shape: Any) -> None:
    if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError("tensor shape must be a list of nonnegative integers")


def _validate_dtype(dtype: Any) -> None:
    if (
        not isinstance(dtype, str)
        or not dtype.startswith("torch.")
        or not isinstance(getattr(torch, dtype[6:], None), torch.dtype)
    ):
        raise ValueError("invalid tensor dtype")


def _validate_descriptor(descriptor: Any) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError("trajectory file descriptor must be an object")
    filename = descriptor.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in (".", "..")
    ):
        raise ValueError("trajectory file must be a simple relative filename")
    digest = descriptor.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid trajectory SHA-256 checksum")


def _validate_layout(layout: Any, *, parameters: bool = False) -> None:
    if not isinstance(layout, list) or not layout:
        raise ValueError("missing model layout")
    entries = {}
    for entry in layout:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError("invalid model layout entry")
        name = entry["name"]
        if not name or name in entries:
            raise ValueError("duplicate or empty model tensor name")
        _validate_shape(entry.get("shape"))
        _validate_dtype(entry.get("dtype"))
        if parameters:
            if type(entry.get("requires_grad")) is not bool:
                raise ValueError("invalid parameter requires_grad flag")
            alias = entry.get("alias_of")
            if not isinstance(alias, str) or (alias != name and alias not in entries):
                raise ValueError("invalid tied-parameter alias")
            if alias != name:
                previous = entries[alias]
                if any(
                    entry[key] != previous[key]
                    for key in ("shape", "dtype", "requires_grad", "alias_of")
                ):
                    raise ValueError("inconsistent tied-parameter layout")
        entries[name] = entry


def _device(model: torch.nn.Module) -> torch.device:
    parameters = list(model.parameters())
    if not parameters or not any(p.requires_grad for p in parameters):
        raise ValueError("model needs at least one trainable parameter")
    devices = {p.device for p in parameters}
    if len(devices) != 1:
        raise ValueError("all model parameters must be on the same device")
    return parameters[0].device


def _all_finite(values: Sequence[torch.Tensor]) -> bool:
    # All model tensors use one device. Combining scalar checks before reading
    # their result avoids one CPU/GPU synchronization per parameter tensor.
    return not values or bool(
        torch.stack([torch.isfinite(v).all() for v in values]).all()
    )


def _validate_replay_model(model: torch.nn.Module) -> None:
    _device(model)
    # This experiment deliberately has no stochastic forward state to replay.
    # GPT-2 is configured with dropout=0. Custom callbacks must likewise be
    # deterministic and must not mutate buffers or sample random numbers.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p:
            raise ValueError("trajectory replay requires dropout=0")
        # Replay explicitly switches to train mode, so even a currently eval()
        # BatchNorm would mix examples and mutate its running-statistic buffers.
        # track_running_stats=False still mixes examples through batch moments.
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            raise ValueError(
                "trajectory replay requires sample-separable loss; BatchNorm is unsupported"
            )


def sgd_step(
    model: torch.nn.Module,
    examples: torch.Tensor,
    batch_ids: Sequence[int],
    learning_rate: float,
    loss_fn: LossFunction,
    *,
    omitted_sample: int | None = None,
) -> float:
    """Apply one recorded SGD update, optionally deleting a training example.

    The denominator is ALWAYS len(batch_ids), not the number of surviving
    examples. This is the counterfactual in Eq. (4): deleting a sample zeros its
    contribution without increasing the learning rates of all other samples.
    An empty surviving batch is a no-op, not a division by zero.
    loss_fn must be deterministic, sample-separable, and leave buffers intact.
    """
    if not isinstance(examples, torch.Tensor) or examples.ndim < 1:
        raise ValueError("examples must be a tensor indexed by sample")
    if not isinstance(batch_ids, (list, tuple)) or not batch_ids:
        raise ValueError("SGD batch must not be empty")
    Trajectory._validate_batch(batch_ids, len(batch_ids), len(examples))
    if omitted_sample is not None:
        _integer(omitted_sample, "omitted sample")
        if omitted_sample >= len(examples):
            raise ValueError("omitted sample outside corpus")
    rate = _rate(learning_rate)
    kept = [i for i in batch_ids if i != omitted_sample]
    model.zero_grad(set_to_none=True)
    if not kept:
        return 0.0
    batch = examples[kept].to(_device(model))
    losses = loss_fn(model, batch)
    if (
        not isinstance(losses, torch.Tensor)
        or losses.ndim != 1
        or losses.shape[0] != len(kept)
    ):
        raise ValueError("loss_fn must return one scalar per example")
    if not torch.isfinite(losses).all():
        raise ValueError("non-finite training loss")
    loss = losses.sum() / len(batch_ids)
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    if not _all_finite([g for g in gradients if g is not None]):
        raise ValueError("non-finite SGD gradient")
    with torch.no_grad():
        # Validate every candidate before changing any parameter. A divergent
        # update must not leave half of the model modified on an error path.
        updates = [
            (p, p.add(g, alpha=-rate))
            for p, g in zip(parameters, gradients)
            if g is not None
        ]
        if not _all_finite([value for _, value in updates]):
            raise ValueError("non-finite SGD parameter after update")
        for parameter, value in updates:
            parameter.copy_(value)
    return float(loss.detach())


class Trajectory:
    """A completed, checksummed trajectory; architecture/loss remain external.

    Construction verifies metadata and the examples file. Checkpoint checksums
    are verified when loaded, avoiding a full scan of a potentially large run.
    Files use torch.load(weights_only=True); arbitrary Python pickle objects
    are never deserialized. This is integrity checking, not authenticity against
    an attacker who can also replace the manifest.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("not a completed DIT trajectory: missing manifest.json")
        with manifest_path.open(encoding="utf-8") as source:
            self.manifest = json.load(source)
        self._validate_manifest()
        path = self._checked_file(self.manifest["examples"])
        self.examples = torch.load(path, map_location="cpu", weights_only=True)
        descriptor = self.manifest["examples"]
        if (
            not isinstance(self.examples, torch.Tensor)
            or list(self.examples.shape) != descriptor["shape"]
            or str(self.examples.dtype) != descriptor["dtype"]
        ):
            raise ValueError("examples do not match their recorded shape/dtype")
        self.metadata = self.manifest["metadata"]

    @classmethod
    def open(cls, directory: str | Path) -> Trajectory:
        return cls(directory)

    @property
    def steps(self) -> int:
        return self.manifest["steps"]

    @property
    def sample_count(self) -> int:
        return self.manifest["sample_count"]

    def _checked_file(self, descriptor: dict[str, Any]) -> Path:
        _validate_descriptor(descriptor)
        filename = descriptor["file"]
        path = self.directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or symlinked trajectory file: {filename}")
        if _sha256(path) != descriptor.get("sha256"):
            raise ValueError(f"checksum mismatch: {filename}")
        return path

    def _validate_manifest(self) -> None:
        m = self.manifest
        if not isinstance(m, dict):
            raise ValueError("trajectory manifest must be a JSON object")
        required = {
            "format",
            "version",
            "complete",
            "steps",
            "sample_count",
            "batch_size",
            "checkpoint_interval",
            "seed",
            "metadata",
            "schedule",
            "checkpoints",
            "examples",
            "parameter_layout",
            "state_layout",
            "device_type",
        }
        if not required.issubset(m):
            raise ValueError("incomplete trajectory manifest")
        if (
            m["format"] != _FORMAT
            or type(m["version"]) is not int
            or m["version"] != _VERSION
            or m["complete"] is not True
        ):
            raise ValueError("unsupported or incomplete DIT trajectory")
        steps = _integer(m["steps"], "steps")
        count = _integer(m["sample_count"], "sample_count", 1)
        size = _integer(m["batch_size"], "batch_size", 1)
        interval = _integer(m["checkpoint_interval"], "checkpoint_interval", 1)
        _integer(m["seed"], "seed")
        if size > count or not isinstance(m["metadata"], dict):
            raise ValueError("invalid batch size or metadata")
        json.dumps(m["metadata"], allow_nan=False)
        if m["device_type"] not in ("cpu", "cuda"):
            raise ValueError("unsupported trajectory device type")
        if not isinstance(m["schedule"], list) or len(m["schedule"]) != steps:
            raise ValueError("schedule length does not match steps")
        for record in m["schedule"]:
            if (
                not isinstance(record, dict)
                or "batch" not in record
                or "learning_rate" not in record
            ):
                raise ValueError("invalid recorded update")
            self._validate_batch(record["batch"], size, count)
            _rate(record["learning_rate"])
            loss = record.get("mean_loss")
            if type(loss) not in (float, int) or not math.isfinite(loss):
                raise ValueError("invalid recorded mean loss")
        if not isinstance(m["checkpoints"], list):
            raise ValueError("invalid checkpoints")
        expected_steps = sorted(set(range(0, steps + 1, interval)) | {steps})
        for item in m["checkpoints"]:
            _validate_descriptor(item)
            _integer(item.get("step"), "checkpoint step")
        recorded_steps = [item["step"] for item in m["checkpoints"]]
        if recorded_steps != expected_steps:
            raise ValueError("missing, duplicate, or incorrectly ordered checkpoints")
        _validate_layout(m["parameter_layout"], parameters=True)
        _validate_layout(m["state_layout"])
        state_entries = {entry["name"]: entry for entry in m["state_layout"]}
        for entry in m["parameter_layout"]:
            state_entry = state_entries.get(entry["name"], {})
            if any(entry[key] != state_entry.get(key) for key in ("shape", "dtype")):
                raise ValueError("parameter and state layouts disagree")
        descriptor = m["examples"]
        if (
            not isinstance(descriptor, dict)
            or not isinstance(descriptor.get("shape"), list)
            or not descriptor["shape"]
            or descriptor["shape"][0] != count
        ):
            raise ValueError("examples shape does not match sample count")
        _validate_descriptor(descriptor)
        _validate_shape(descriptor["shape"])
        _validate_dtype(descriptor.get("dtype"))
        filenames = [c["file"] for c in m["checkpoints"]] + [descriptor["file"]]
        if len(set(filenames)) != len(filenames):
            raise ValueError("trajectory files must have distinct filenames")

    @staticmethod
    def _validate_batch(batch: Any, size: int, count: int) -> None:
        if not isinstance(batch, (list, tuple)) or len(batch) != size:
            raise ValueError("recorded batch has incorrect size")
        if any(type(i) is not int or not 0 <= i < count for i in batch):
            raise ValueError("recorded batch contains an invalid sample ID")
        if len(set(batch)) != len(batch):
            raise ValueError("a batch must sample without replacement")

    def batch(self, step: int) -> tuple[int, ...]:
        self._check_step(step, allow_final=False)
        return tuple(self.manifest["schedule"][step]["batch"])

    def learning_rate(self, step: int) -> float:
        self._check_step(step, allow_final=False)
        return self.manifest["schedule"][step]["learning_rate"]

    def _check_step(self, step: int, *, allow_final: bool = True) -> None:
        _integer(step, "step")
        if step > self.steps or (not allow_final and step == self.steps):
            raise ValueError("step outside recorded trajectory")

    def _validate_model(self, model: torch.nn.Module) -> None:
        _validate_replay_model(model)
        if _parameter_layout(model) != self.manifest["parameter_layout"]:
            raise ValueError(
                "model parameter layout/dtype/tying differs from trajectory"
            )
        if _state_layout(model.state_dict()) != self.manifest["state_layout"]:
            raise ValueError("model state layout differs from trajectory")

    def _load_checkpoint(
        self, model: torch.nn.Module, descriptor: dict[str, Any]
    ) -> None:
        path = self._checked_file(descriptor)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(v, torch.Tensor) for v in state.values()
        ):
            raise ValueError("checkpoint must contain a state dictionary of tensors")
        if _state_layout(state) != self.manifest["state_layout"]:
            raise ValueError("checkpoint state layout differs from manifest")
        if any(
            v.is_floating_point() and not torch.isfinite(v).all()
            for v in state.values()
        ):
            raise ValueError("checkpoint contains non-finite parameters")
        for entry in self.manifest["parameter_layout"]:
            if entry["alias_of"] != entry["name"]:
                if not torch.equal(state[entry["name"]], state[entry["alias_of"]]):
                    raise ValueError("checkpoint contains inconsistent tied weights")
        model.load_state_dict(state, strict=True)

    def load_state(
        self, model: torch.nn.Module, step: int, loss_fn: LossFunction | None = None
    ) -> None:
        """Install theta_step, replaying from the nearest preceding checkpoint.

        A saved checkpoint can be loaded without a loss callback, including for
        ordinary inference. Unsaved intermediate states require deterministic
        replay on the original device type and an equivalent loss callback.
        This mutates model weights, clears parameter gradients, and sets train
        mode. GPT-2 has no dropout, so inference values are mode-independent.
        """
        self._check_step(step)
        self._validate_model(model)
        descriptor = max(
            (c for c in self.manifest["checkpoints"] if c["step"] <= step),
            key=lambda c: c["step"],
        )
        if descriptor["step"] != step:
            if loss_fn is None:
                raise ValueError("intermediate state replay requires loss_fn")
            if _device(model).type != self.manifest["device_type"]:
                raise ValueError("exact replay requires the original device type")
        self._load_checkpoint(model, descriptor)
        model.train()
        model.zero_grad(set_to_none=True)
        for t in range(descriptor["step"], step):
            sgd_step(
                model, self.examples, self.batch(t), self.learning_rate(t), loss_fn
            )

    def iter_reverse_segments(
        self, model: torch.nn.Module, start: int, end: int, loss_fn: LossFunction
    ) -> Iterator[tuple[int, torch.nn.Module]]:
        """Yield (t, model at theta_t) for t=end-1,...,start using bounded RAM.

        CPU snapshots cover at most one checkpoint interval, never all T states.
        A yielded model is borrowed: consume its gradients/HVPs before advancing
        the iterator. Do not retain the model as a snapshot. For DIT on [t1,t2],
        callers must request start=0, NOT t1: earlier removals still affect the
        parameters and therefore influence accrued in a later window.
        """
        self._check_step(start)
        self._check_step(end)
        if start > end:
            raise ValueError("reverse interval requires start <= end")
        self._validate_model(model)
        if _device(model).type != self.manifest["device_type"]:
            raise ValueError("exact replay requires the original device type")
        stop = end
        while stop > start:
            descriptor = max(
                (c for c in self.manifest["checkpoints"] if c["step"] < stop),
                key=lambda c: c["step"],
            )
            first = descriptor["step"]
            self.load_state(model, first)
            states = []
            for t in range(first, stop):
                if t >= start:
                    states.append((t, snapshot(model)))
                # The last update is unnecessary: we need theta_(stop-1), not
                # theta_stop. Avoid one extra expensive model backward pass.
                if t + 1 < stop:
                    sgd_step(
                        model,
                        self.examples,
                        self.batch(t),
                        self.learning_rate(t),
                        loss_fn,
                    )
            for t, state in reversed(states):
                model.load_state_dict(state, strict=True)
                model.zero_grad(set_to_none=True)
                yield t, model
            del states, state
            stop = first

    def leave_one_out_state(
        self,
        model: torch.nn.Module,
        omitted_sample: int,
        *,
        end: int | None = None,
        loss_fn: LossFunction,
    ) -> None:
        """Replay true deletion training from theta_0 through theta_end.

        Never start a late-window counterfactual from the original theta_t1:
        that would erase deletion effects accumulated before the window.
        """
        _integer(omitted_sample, "omitted sample")
        if omitted_sample >= self.sample_count:
            raise ValueError("omitted sample outside corpus")
        end = self.steps if end is None else end
        self._check_step(end)
        if _device(model).type != self.manifest["device_type"]:
            raise ValueError("exact replay requires the original device type")
        self.load_state(model, 0)
        for t in range(end):
            sgd_step(
                model,
                self.examples,
                self.batch(t),
                self.learning_rate(t),
                loss_fn,
                omitted_sample=omitted_sample,
            )


def train_trajectory(
    model: torch.nn.Module,
    examples: torch.Tensor,
    directory: str | Path,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float | Sequence[float],
    seed: int,
    checkpoint_interval: int,
    loss_fn: LossFunction,
    metadata: dict[str, Any] | None = None,
    schedule: Sequence[Sequence[int]] | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> Trajectory:
    """Train plain SGD and atomically mark the complete trajectory readable.

    The supplied model is theta_0; initialization is the caller's responsibility.
    Each random batch samples distinct example IDs, independently between steps.
    A fixed schedule and a per-step learning-rate sequence support controlled
    research comparisons. Existing output paths are never overwritten. Failure
    leaves an incomplete directory with no manifest, not a loadable partial run.
    Metadata must be JSON data and should include architecture/corpus provenance.
    progress(completed_steps, batch_mean_loss) runs after an applied update.
    """
    steps = _integer(steps, "steps")
    batch_size = _integer(batch_size, "batch_size", 1)
    checkpoint_interval = _integer(checkpoint_interval, "checkpoint_interval", 1)
    _integer(seed, "seed")
    _validate_replay_model(model)
    if (
        not isinstance(examples, torch.Tensor)
        or examples.ndim < 1
        or len(examples) == 0
    ):
        raise ValueError("examples must be a nonempty tensor indexed by sample")
    if batch_size > len(examples):
        raise ValueError("batch_size exceeds sample count")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    # Validate before making the output directory. allow_nan=False also rejects
    # NaN/Inf in metadata, rather than silently emitting nonstandard JSON.
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    if isinstance(learning_rate, Sequence):
        rates = [_rate(value) for value in learning_rate]
        if len(rates) != steps:
            raise ValueError("learning-rate sequence length must equal steps")
    else:
        rates = [_rate(learning_rate)] * steps
    if schedule is None:
        generator = random.Random(seed)
        batches = [
            generator.sample(range(len(examples)), batch_size) for _ in range(steps)
        ]
    else:
        if len(schedule) != steps:
            raise ValueError("schedule length must equal steps")
        batches = [list(batch) for batch in schedule]
    for batch in batches:
        Trajectory._validate_batch(batch, batch_size, len(examples))
    initial_state = snapshot(model)
    if any(
        value.is_floating_point() and not torch.isfinite(value).all()
        for value in initial_state.values()
    ):
        raise ValueError("initial model contains non-finite parameters")
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)
    examples = examples.detach().cpu().clone()
    examples_path = directory / "examples.pt"
    torch.save(examples, examples_path)
    manifest = {
        "format": _FORMAT,
        "version": _VERSION,
        "complete": True,
        "steps": steps,
        "sample_count": len(examples),
        "batch_size": batch_size,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "metadata": metadata,
        "torch_version": torch.__version__,
        "device_type": _device(model).type,
        "parameter_layout": _parameter_layout(model),
        "state_layout": _state_layout(initial_state),
        "examples": {
            "file": examples_path.name,
            "sha256": _sha256(examples_path),
            "shape": list(examples.shape),
            "dtype": str(examples.dtype),
        },
        "schedule": [],
        "checkpoints": [],
    }

    def save_checkpoint(step: int, state: State) -> None:
        path = directory / f"step_{step:08d}.pt"
        torch.save(state, path)
        manifest["checkpoints"].append(
            {"step": step, "file": path.name, "sha256": _sha256(path)}
        )

    save_checkpoint(0, initial_state)
    del initial_state
    model.train()
    for t, (batch, rate) in enumerate(zip(batches, rates)):
        loss = sgd_step(model, examples, batch, rate, loss_fn)
        manifest["schedule"].append(
            {"batch": batch, "learning_rate": rate, "mean_loss": loss}
        )
        if (t + 1) % checkpoint_interval == 0 or t + 1 == steps:
            save_checkpoint(t + 1, snapshot(model))
        if progress is not None:
            progress(t + 1, loss)
    temporary_path = directory / "manifest.json.tmp"
    with temporary_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, directory / "manifest.json")
    return Trajectory(directory)
