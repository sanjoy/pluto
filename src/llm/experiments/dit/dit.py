"""Train GPT-2 on Shakespeare, infer DIT influence, and generate ordinary text.

Run from the repository root with python -m src.llm.experiments.dit.dit.
The PyTorch backend is intentional: the native cuTile Layer API has only a
first backward, while DIT differentiates the entire loss backward once more.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform

# cuBLAS must see this before the first CUDA context/handle is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from src.llm.experiments.dit.data import load_text_examples, load_tokenizer
from src.llm.experiments.dit.influence import compute_influence, validate_leave_one_out
from src.llm.experiments.dit.model import GPT2Config, Gpt2
from src.llm.experiments.dit.trajectory import Trajectory, train_trajectory


def training_loss(model, batch):
    return model.per_example_loss(batch)


def _source_fingerprints():
    directory = Path(__file__).parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("dit.py", "model.py", "data.py", "trajectory.py", "influence.py")
    }


def configure_runtime(device, cpu_threads=1):
    """Fix precision and deterministic choices for training and replay alike."""
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "device": device,
        "cpu_threads": cpu_threads,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": True,
        "tf32": False,
    }


def _train(args):
    config = GPT2Config()
    if not 1 <= args.context_tokens <= config.context_length:
        raise ValueError("context_tokens must fit the GPT-2 context length")
    runtime = configure_runtime(args.device, args.cpu_threads)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("this recipe requires the GPT-2 50,257-token vocabulary")
    data = load_text_examples(
        args.corpus,
        args.tokenizer_dir,
        args.context_tokens,
        max_examples=args.max_examples,
        offset=args.token_offset,
    )
    model = Gpt2(
        config, seed=args.seed, device=args.device, dtype=getattr(torch, args.dtype)
    )
    initial_checkpoint = None
    if args.initial_checkpoint is not None:
        model.load_pluto_checkpoint(args.initial_checkpoint)
        initial_checkpoint = str(Path(args.initial_checkpoint).resolve())
    metadata = {
        "architecture": "pluto-gpt2",
        "config": asdict(config),
        "dtype": args.dtype,
        "context_tokens": args.context_tokens,
        "corpus": str(Path(args.corpus).resolve()),
        "tokenizer_dir": str(Path(args.tokenizer_dir).resolve()),
        "corpus_sha256": data.corpus_sha256,
        "tokenizer_sha256": data.tokenizer_sha256,
        "token_offsets": list(data.offsets),
        "initial_checkpoint": initial_checkpoint,
        "runtime": runtime,
        "source_sha256": _source_fingerprints(),
        "loss": "mean-per-example-next-token-cross-entropy-v1",
    }
    print(
        f"Training {sum(p.numel() for p in model.parameters()):,} parameters; "
        f"{len(data.offsets)} examples, {args.context_tokens} targets/example",
        flush=True,
    )
    run = train_trajectory(
        model,
        data.tokens,
        args.run_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        loss_fn=training_loss,
        metadata=metadata,
        progress=lambda step, loss: print(
            f"step {step}: batch mean loss {loss:.9f}", flush=True
        ),
    )
    if args.export_checkpoint:
        model.write_pluto_checkpoint(args.export_checkpoint)
    print(f"Completed DIT trajectory: {run.directory.resolve()}", flush=True)


def _load_run(args, *, require_replay=True):
    run = Trajectory.open(args.run_dir)
    metadata = run.metadata
    if (
        metadata.get("architecture") != "pluto-gpt2"
        or metadata.get("loss") != "mean-per-example-next-token-cross-entropy-v1"
    ):
        raise ValueError("not a GPT-2 next-token DIT trajectory")
    config = GPT2Config(**metadata["config"])
    device = args.device or run.manifest["device_type"]
    recorded_runtime = metadata["runtime"]
    runtime = configure_runtime(device, recorded_runtime["cpu_threads"])
    # Deterministic replay is only promised for the recorded backend/version
    # and precision settings, not across devices or changed implementations.
    if require_replay and (
        runtime != recorded_runtime
        or _source_fingerprints() != metadata["source_sha256"]
    ):
        raise ValueError(
            "replay runtime/source differs from recorded training; use the original environment/code"
        )
    if metadata["dtype"] not in ("float32", "float64"):
        raise ValueError("invalid recorded model dtype")
    model = Gpt2(
        config,
        seed=run.manifest["seed"],
        device=device,
        dtype=getattr(torch, metadata["dtype"]),
    )
    tokenizer_dir = args.tokenizer_dir or metadata["tokenizer_dir"]
    tokenizer_file = Path(tokenizer_dir) / "tokenizer.json"
    if (
        hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
        != metadata["tokenizer_sha256"]
    ):
        raise ValueError("tokenizer differs from the one used to train this trajectory")
    tokenizer = load_tokenizer(tokenizer_dir)
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("tokenizer vocabulary differs from model")
    return run, model, tokenizer


def _prompt_tokens(model, tokenizer, prompt):
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not 1 <= len(ids) <= model.config.context_length:
        raise ValueError("prompt must encode to 1..context_length tokens")
    return torch.tensor(
        [ids], dtype=torch.int64, device=next(model.parameters()).device
    )


def _query(args, run, model, tokenizer, t2):
    if args.query == "parameter":
        if args.prompt is not None or args.target_id is not None:
            raise ValueError("parameter queries do not take prompt or target_id")
        parameters = dict(model.named_parameters())
        if args.parameter not in parameters or args.coordinate is None:
            raise ValueError(
                "parameter query requires a valid --parameter and --coordinate"
            )
        coordinate = tuple(int(part) for part in args.coordinate.split(","))
        shape = parameters[args.parameter].shape
        if len(coordinate) != len(shape) or any(
            not 0 <= i < n for i, n in zip(coordinate, shape)
        ):
            raise ValueError("parameter coordinate is outside its shape")
        return (
            lambda m: dict(m.named_parameters())[args.parameter][coordinate],
            {
                "kind": "parameter",
                "parameter": args.parameter,
                "coordinate": coordinate,
            },
        )
    if args.parameter is not None or args.coordinate is not None:
        raise ValueError("parameter and coordinate are only for parameter queries")
    if args.prompt is None:
        raise ValueError("prediction/loss queries require --prompt")
    tokens = _prompt_tokens(model, tokenizer, args.prompt)
    target = args.target_id
    if target is None:
        if args.query == "loss":
            raise ValueError(
                "loss queries require --target_id (the expected next token)"
            )
        # Choose once at t2, then freeze across both endpoints and ALL removal
        # replays. Changing argmax per model would change the question asked.
        run.load_state(model, t2, training_loss)
        with torch.no_grad():
            target = int(model(tokens)[0, -1].argmax())
    if not 0 <= target < model.config.vocab_size:
        raise ValueError("target_id is outside the logical vocabulary")

    def query(model):
        logits = model(tokens)[0, -1]
        if args.query == "logit":
            return logits[target]
        if args.query == "probability":
            return logits.softmax(dim=-1)[target]
        return -logits.log_softmax(dim=-1)[target]

    return query, {
        "kind": args.query,
        "prompt": args.prompt,
        "prompt_ids": tokens[0].tolist(),
        "target_id": target,
        "target_text": tokenizer.decode([target], skip_special_tokens=False),
    }


def _infer(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite influence report: {output}")
    run, model, tokenizer = _load_run(args)
    t2 = run.steps if args.t2 is None else args.t2
    if not 0 <= args.t1 < t2 <= run.steps:
        raise ValueError("window must satisfy 0 <= t1 < t2 <= training steps")
    selected = (
        None
        if args.sample_ids is None
        else [int(part) for part in args.sample_ids.split(",")]
    )
    query, description = _query(args, run, model, tokenizer, t2)
    result = compute_influence(
        model,
        run,
        training_loss,
        query,
        t1=args.t1,
        t2=t2,
        sample_ids=selected,
        progress=lambda step, count: print(
            f"reverse step {step}: {count} selected examples", flush=True
        ),
    )
    validation = None
    if args.validate_loo:
        print("Validating with finite leave-one-out retraining...", flush=True)
        validation = validate_leave_one_out(
            model, run, training_loss, query, t1=args.t1, t2=t2, sample_ids=selected
        )
    rows = []
    for sample, score in sorted(result.scores.items(), key=lambda item: -abs(item[1])):
        ids = run.examples[sample].tolist()
        row = {
            "example_id": sample,
            "token_offset": run.metadata["token_offsets"][sample],
            "token_ids": ids,
            "text": tokenizer.decode(ids, skip_special_tokens=False),
            "dit_score": score,
        }
        if validation is not None:
            row.update(
                finite_loo=validation[sample],
                absolute_error=abs(score - validation[sample]),
            )
        rows.append(row)
    report = {
        "format": "pluto-dit-influence",
        "version": 1,
        "run_dir": str(run.directory.resolve()),
        "window": [args.t1, t2],
        "query": description,
        "query_start": result.query_start,
        "query_end": result.query_end,
        "semantics": "linearized removal effect at end minus at start; positive increases query",
        "finite_loo_is_approximation_target_not_identity": True,
        "model_config": asdict(model.config),
        "source_sha256": _source_fingerprints(),
        "examples": rows,
        "contributions": result.contributions,
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    print(f"Query: {json.dumps(description, ensure_ascii=False)}")
    print(f"Original query: {result.query_start:.9g} -> {result.query_end:.9g}")
    for row in rows[: args.top]:
        suffix = (
            f"; finite LOO {row['finite_loo']:+.9g}" if validation is not None else ""
        )
        print(
            f"example {row['example_id']}: {row['dit_score']:+.9g}{suffix}  {row['text']!r}"
        )
    print(f"Influence report: {output.resolve()}")


def _generate(args):
    if args.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    run, model, tokenizer = _load_run(args, require_replay=False)
    step = run.steps if args.step is None else args.step
    # Only saved checkpoints may be used across runtimes. Intermediate steps
    # require the same strict deterministic replay contract as DIT inference.
    if step not in [checkpoint["step"] for checkpoint in run.manifest["checkpoints"]]:
        run, model, tokenizer = _load_run(args)
    run.load_state(model, step, training_loss)
    tokens = _prompt_tokens(model, tokenizer, args.prompt)[0].tolist()
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = tokens[-model.config.context_length :]
            batch = torch.tensor([context], dtype=torch.int64, device=device)
            token = int(model(batch)[0, -1].argmax())
            tokens.append(token)
            if token == tokenizer.token_to_id("<|endoftext|>"):
                break
    print(tokenizer.decode(tokens, skip_special_tokens=False))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    train = commands.add_parser(
        "train", help="record a plain-SGD GPT-2 training trajectory"
    )
    train.add_argument(
        "--run_dir", required=True, help="new output directory; never overwritten"
    )
    train.add_argument("--corpus", default="testdata/shakespeare.txt")
    train.add_argument("--tokenizer_dir", required=True)
    train.add_argument("--steps", type=int, default=100)
    train.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="number of independently removable contexts",
    )
    train.add_argument("--context_tokens", type=int, default=1024)
    train.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="omit to use all complete corpus windows",
    )
    train.add_argument("--token_offset", type=int, default=0)
    train.add_argument("--learning_rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--checkpoint_interval", type=int, default=1)
    train.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    train.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    train.add_argument("--cpu_threads", type=int, default=1)
    train.add_argument(
        "--initial_checkpoint",
        help="optional native weight_N.bin directory, defining theta_0",
    )
    train.add_argument(
        "--export_checkpoint",
        help="optional new directory for native-compatible final weights",
    )
    train.set_defaults(action=_train)
    for name, action, help_text in (
        ("infer", _infer, "compute the paper's time-window influence"),
        (
            "generate",
            _generate,
            "ordinary greedy text inference from a recorded checkpoint",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--run_dir", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default=None)
        command.add_argument(
            "--tokenizer_dir", help="override saved tokenizer path; contents must match"
        )
        command.set_defaults(action=action)
        if name == "infer":
            command.add_argument("--output", required=True, help="new JSON report path")
            command.add_argument("--t1", type=int, default=0)
            command.add_argument(
                "--t2", type=int, help="completed updates; defaults to final step"
            )
            command.add_argument(
                "--query",
                choices=("logit", "loss", "probability", "parameter"),
                default="logit",
            )
            command.add_argument("--prompt")
            command.add_argument(
                "--target_id", type=int, help="fixed next-token ID; required for loss"
            )
            command.add_argument("--parameter", help="named_parameters() tensor name")
            command.add_argument("--coordinate", help="comma-separated tensor indices")
            command.add_argument(
                "--sample_ids", help="comma-separated example IDs; default all"
            )
            command.add_argument(
                "--validate_loo",
                action="store_true",
                help="also measure expensive finite deletion retraining",
            )
            command.add_argument("--top", type=int, default=10)
        else:
            command.add_argument(
                "--step", type=int, help="completed updates; defaults to final"
            )
            command.add_argument("--prompt", required=True)
            command.add_argument("--max_new_tokens", type=int, default=40)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    args.action(args)


if __name__ == "__main__":
    main()
