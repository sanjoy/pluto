"""An explicit, read-only map of the *Pluto recipe* GPT-2 checkpoint format.

These files are NOT Hugging Face GPT-2 checkpoints. C++ WriteToDirectory writes
each unique GPU allocation verbatim, in recursive Layer.weights() order. There
is no header, shape, dtype, endian marker, model identifier, optimizer state, or
provenance in the files. A caller must supply the architecture; the defaults
below match src/llm/recipes/gpt2.h. Size validation can reject an SAE checkpoint,
but cannot prove that a same-shaped checkpoint came from the intended model.

All weights in this recipe are FP32 master parameters, even when inference
uses BF16 activations. The GH200 checkpoint producer is little-endian, hence
the explicit '<f4' interpretation; we do not guess from the reader's machine.
Every 2-D array is C-order (last index varies fastest). FullyConnected stores
W[input, output], not the common PyTorch W[output, input]: y = x @ W + b.

Token embeddings are [padded_vocabulary, model_width]. The LM head reads the
SAME allocation transposed, logits = x @ embedding.T; it has no separate file.
Padded vocabulary rows are not tokens and must be excluded from token analyses.
The attention projection concatenates Q, K, V along its output axis, then
contiguous head channels within each component; it does not interleave Q/K/V.

Example (this only reads weights; it never executes a model):
    checkpoint = GPT2Checkpoint('/path/to/step_13030', check_finite=True)
    keys = checkpoint.mlp_keys(0)      # [d_ff, d_model], a strided view
    values = checkpoint.mlp_values(0)  # [d_ff, d_model]
    w_query_head_2 = checkpoint.qkv(0, 'q', head=2)

The manifest and provenance() give exact filenames, local byte ranges, shapes,
sharing, and hashes of the *currently inspected* source. Source hashes are not
claimed to identify the historical checkpoint producer. Optional checkpoint
hashes identify the actual analyzed bytes, at the cost of a full sequential read.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class GPT2Config:
    """Explicit physical layout; small dimensions are allowed for unit fixtures.

    Toy fixtures need not meet cuTile's execution alignment constraints. This
    loader validates byte layout, not whether a CUDA kernel could run a shape.
    """

    vocab_size: int = 50257
    padded_vocab_size: int = 50272
    context_length: int = 1024
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.vocab_size > self.padded_vocab_size:
            raise ValueError('vocab_size exceeds physical padded_vocab_size')
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class TensorSpec:
    """One unique on-disk allocation, not one occurrence in Layer.weights().

    byte_range is a half-open interval within filename, never an offset into
    an imaginary concatenated checkpoint. element_byte_range locates a single
    scalar for reproducible weight-to-feature attribution.
    """

    index: int
    name: str
    role: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    shared_with: tuple[str, ...] = ()
    dtype: str = '<f4'

    @property
    def filename(self) -> str:
        return f'weight_{self.index}.bin'

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * 4

    @property
    def byte_range(self) -> tuple[int, int]:
        return (0, self.nbytes)

    def element_byte_range(self, *indices: int) -> tuple[int, int]:
        if len(indices) != len(self.shape):
            raise IndexError('one index is required for each tensor dimension')
        offset = 0
        for index, dimension in zip(indices, self.shape):
            if not isinstance(index, (int, np.integer)) or not 0 <= index < dimension:
                raise IndexError(f'index {index} outside dimension {dimension}')
            offset = offset * dimension + int(index)
        return (4 * offset, 4 * (offset + 1))

    def to_dict(self) -> dict:
        return dict(asdict(self), filename=self.filename,
                    byte_range=self.byte_range, nbytes=self.nbytes,
                    storage_order='C')


def tensor_manifest(config: GPT2Config = GPT2Config()) -> tuple[TensorSpec, ...]:
    """Reproduce recipe order and checkpoint pointer-identity deduplication.

    Residual/Composed layers concatenate their children's parameters, while
    attention and GELU own none. Each block therefore contributes 12 files.
    The final LM head's embedding occurrence is deduplicated by the writer.
    """
    result = []

    def add(name, role, shape, axes, shared_with=()):
        result.append(TensorSpec(len(result), name, role, shape, axes, shared_with))

    d = config.d_model
    f = config.d_ff
    add('token_embedding.weight', 'Token embedding and tied LM output dictionary',
        (config.padded_vocab_size, d), ('token_id_including_padding', 'residual'),
        ('lm_head.weight',))
    add('position_embedding.weight', 'Learned absolute position embedding',
        (config.context_length, d), ('position', 'residual'))
    for block in range(config.n_layers):
        prefix = f'blocks.{block}'
        add(f'{prefix}.ln1.scale', 'Pre-attention LayerNorm gamma', (d,), ('residual',))
        add(f'{prefix}.ln1.bias', 'Pre-attention LayerNorm beta', (d,), ('residual',))
        add(f'{prefix}.attn.qkv.weight', 'Packed Q/K/V input projection',
            (d, 3 * d), ('residual_in', 'component_then_head_then_channel'))
        add(f'{prefix}.attn.qkv.bias', 'Packed Q/K/V projection bias',
            (3 * d,), ('component_then_head_then_channel',))
        add(f'{prefix}.attn.output.weight', 'Attention output projection',
            (d, d), ('head_then_channel', 'residual_out'))
        add(f'{prefix}.attn.output.bias', 'Attention output projection bias',
            (d,), ('residual_out',))
        add(f'{prefix}.ln2.scale', 'Pre-MLP LayerNorm gamma', (d,), ('residual',))
        add(f'{prefix}.ln2.bias', 'Pre-MLP LayerNorm beta', (d,), ('residual',))
        add(f'{prefix}.mlp.input.weight', 'MLP feature input directions (columns)',
            (d, f), ('residual_in', 'mlp_neuron'))
        add(f'{prefix}.mlp.input.bias', 'MLP pre-GELU bias', (f,), ('mlp_neuron',))
        add(f'{prefix}.mlp.output.weight', 'MLP feature output directions (rows)',
            (f, d), ('mlp_neuron', 'residual_out'))
        add(f'{prefix}.mlp.output.bias', 'MLP output bias', (d,), ('residual_out',))
    add('final_norm.scale', 'Final LayerNorm gamma', (d,), ('residual',))
    add('final_norm.bias', 'Final LayerNorm beta', (d,), ('residual',))
    return tuple(result)


SOURCE_FILES = (
    'src/llm/recipes/gpt2.h', 'src/llm/recipes/gpt2.cc',
    'src/llm/checkpoint.cc', 'src/llm/layers/combinators.cc',
    'src/llm/layers/embedding.h', 'src/llm/layers/embedding.cc',
    'src/llm/layers/fully_connected.h', 'src/llm/layers/fully_connected.cc',
    'src/llm/layers/norm.h', 'src/llm/layers/norm.cc',
    'src/llm/layers/attention.cc',
)


def sha256_file(path: str | Path) -> str:
    """Stream a file hash without allocating a checkpoint-sized byte string."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint_directory(parent_directory: str | Path) -> Path:
    """Find greatest numeric step among directories; never unpack archives.

    This does not silently fall back if the latest checkpoint is malformed.
    Construct GPT2Checkpoint on the result to perform strict layout checks.
    """
    candidates = []
    for path in Path(parent_directory).iterdir():
        match = re.fullmatch(r'step_([0-9]+)', path.name)
        if match and path.is_dir():
            candidates.append((int(match[1]), path.name, path))
    if not candidates:
        raise FileNotFoundError(f'no uncompressed step_N directories in {parent_directory}')
    return max(candidates)[2]


class GPT2Checkpoint:
    """Named, read-only NumPy memmaps with mandatory count and size validation.

    Only canonical weight_N.bin names are permitted for weight files. Other
    files (e.g. a training log) are ignored. Missing, extra, malformed, and
    non-regular weight files are rejected before any mapping is created.
    This deliberately does not accept prefix or four-file SAE checkpoints.
    Arrays borrow OS mappings, and views keep those mappings alive; no explicit
    close() invalidates outstanding analysis views. Files must remain immutable
    during an analysis (use a completed checkpoint, not one being written).
    """

    def __init__(self, checkpoint_dir: str | Path,
                 config: GPT2Config = GPT2Config(), *, check_finite: bool = False):
        self.directory = Path(checkpoint_dir).resolve()
        self.config = config
        self.manifest = tensor_manifest(config)
        self.specs = MappingProxyType({spec.name: spec for spec in self.manifest})
        if not self.directory.is_dir():
            raise ValueError(f'checkpoint is not a directory: {self.directory}')
        expected = {spec.filename for spec in self.manifest}
        actual = {path.name for path in self.directory.iterdir()
                  if path.name.startswith('weight_') and path.name.endswith('.bin')}
        if actual != expected:
            raise ValueError('GPT-2 checkpoint file set mismatch; '
                             f'missing={sorted(expected - actual)}, '
                             f'extra={sorted(actual - expected)}')
        for spec in self.manifest:
            path = self.directory / spec.filename
            if not path.is_file():
                raise ValueError(f'weight is not a regular file: {path}')
            if path.stat().st_size != spec.nbytes:
                raise ValueError(f'{spec.filename} ({spec.name}): expected '
                                 f'{spec.nbytes} bytes, got {path.stat().st_size}')
        self.tensors = MappingProxyType({
            spec.name: np.memmap(self.directory / spec.filename, dtype=spec.dtype,
                                 mode='r', shape=spec.shape, order='C')
            for spec in self.manifest
        })
        if check_finite:
            self.validate_finite()

    def __getitem__(self, name: str) -> np.memmap:
        # Both names refer to the same storage and orientation; callers multiply
        # by its transpose when projecting a residual direction into vocabulary.
        return self.tensors['token_embedding.weight' if name == 'lm_head.weight' else name]

    def __iter__(self) -> Iterator[TensorSpec]:
        return iter(self.manifest)

    @property
    def token_embedding(self) -> np.ndarray:
        """Logical vocabulary only: padding is never a valid token ID."""
        return self['token_embedding.weight'][:self.config.vocab_size]

    def _block_prefix(self, block: int) -> str:
        if type(block) is not int or not 0 <= block < self.config.n_layers:
            raise IndexError(f'invalid block index: {block}')
        return f'blocks.{block}'

    def _head_slice(self, head: int | None) -> slice:
        if head is None:
            return slice(0, self.config.d_model)
        if type(head) is not int or not 0 <= head < self.config.n_heads:
            raise IndexError(f'invalid head index: {head}')
        start = head * self.config.head_dim
        return slice(start, start + self.config.head_dim)

    def _qkv_slice(self, component: str, head: int | None) -> slice:
        if component not in ('q', 'k', 'v'):
            raise ValueError("component must be 'q', 'k', or 'v'")
        offset = ('q', 'k', 'v').index(component) * self.config.d_model
        channels = self._head_slice(head)
        return slice(offset + channels.start, offset + channels.stop)

    def qkv(self, block: int, component: str, head: int | None = None) -> np.ndarray:
        """Projection view [residual_in, head_channel] (or all head channels)."""
        return self[f'{self._block_prefix(block)}.attn.qkv.weight'][
            :, self._qkv_slice(component, head)]

    def qkv_bias(self, block: int, component: str,
                 head: int | None = None) -> np.ndarray:
        return self[f'{self._block_prefix(block)}.attn.qkv.bias'][
            self._qkv_slice(component, head)]

    def attention_output_head(self, block: int, head: int) -> np.ndarray:
        """Output rows [head_channel, residual_out]; unlike QKV, slice rows."""
        return self[f'{self._block_prefix(block)}.attn.output.weight'][self._head_slice(head), :]

    def mlp_keys(self, block: int) -> np.ndarray:
        """Input columns as [neuron, residual]; not standalone token patterns.

        Actual triggering also depends on preceding layers, LayerNorm, bias,
        and GELU. Calling these directions 'keys' is analysis terminology.
        """
        return self[f'{self._block_prefix(block)}.mlp.input.weight'].T

    def mlp_values(self, block: int) -> np.ndarray:
        """Output rows [neuron, residual]; these are residual updates, not logits."""
        return self[f'{self._block_prefix(block)}.mlp.output.weight']

    def validate_finite(self, chunk_elements: int = 1024 * 1024) -> None:
        """Scan parameters with bounded temporary memory, including pad rows."""
        if type(chunk_elements) is not int or chunk_elements <= 0:
            raise ValueError('chunk_elements must be a positive integer')
        for name, tensor in self.tensors.items():
            flat = tensor.reshape(-1)
            for start in range(0, flat.size, chunk_elements):
                if not np.isfinite(flat[start:start + chunk_elements]).all():
                    raise ValueError(f'non-finite parameter in {name}')

    def provenance(self, *, hash_weights: bool = False,
                   repository_root: str | Path | None = None) -> dict:
        """JSON-serializable description with an explicit provenance limitation."""
        root = (Path(repository_root) if repository_root is not None
                else Path(__file__).resolve().parents[2])
        sources = {name: sha256_file(root / name) for name in SOURCE_FILES}
        result = {
            'format': 'pluto-gpt2-raw-unique-fp32-weights',
            'checkpoint_directory': str(self.directory),
            'config': asdict(self.config),
            'head_dim': self.config.head_dim,
            'storage_dtype': '<f4',
            'parameter_count': sum(spec.nbytes // 4 for spec in self.manifest),
            'checkpoint_bytes': sum(spec.nbytes for spec in self.manifest),
            'unique_weight_files': len(self.manifest),
            'source_sha256': sources,
            'source_hash_scope': 'Current analysis source, not authenticated checkpoint producer',
            'checkpoint_metadata': 'None: dtype, byte order, and architecture are caller assumptions',
            'tensors': [spec.to_dict() for spec in self.manifest],
        }
        if hash_weights:
            result['weight_sha256'] = {
                spec.filename: sha256_file(self.directory / spec.filename)
                for spec in self.manifest
            }
        return result
