"""Local GPT-2 tokenization and immutable, identifiable Shakespeare examples."""

from dataclasses import dataclass
import hashlib
import mmap
from pathlib import Path

import torch
from tokenizers import Tokenizer


@dataclass(frozen=True)
class TextExamples:
    """One row is a context plus its final target; ids are stable row indices.

    offsets are absolute offsets in the *tokenized* corpus, not byte offsets.
    Keeping every training row with its trajectory avoids silently attributing
    a saved update to a different version of the text/tokenizer later on.
    """

    tokens: torch.Tensor
    offsets: tuple[int, ...]
    corpus_sha256: str
    tokenizer_sha256: str

    def __post_init__(self):
        if (
            self.tokens.dtype != torch.int64
            or self.tokens.ndim != 2
            or self.tokens.shape[0] == 0
            or self.tokens.shape[1] < 2
        ):
            raise ValueError("tokens must be nonempty int64 [examples, context+1]")
        if len(self.offsets) != self.tokens.shape[0]:
            raise ValueError("one corpus token offset is required per example")
        if any(type(offset) is not int or offset < 0 for offset in self.offsets):
            raise ValueError("corpus token offsets must be nonnegative integers")

    def example_metadata(self, indices=None):
        if indices is None:
            indices = range(len(self.offsets))
        return [
            {
                "example_id": int(index),
                "token_offset": self.offsets[index],
                "token_ids": self.tokens[index].tolist(),
            }
            for index in indices
        ]


def load_tokenizer(directory):
    """Read tokenizer.json locally; never download or add special tokens."""
    return _tokenizer_from_bytes((Path(directory) / "tokenizer.json").read_bytes())


def _tokenizer_from_bytes(contents):
    # Parse the same bytes we fingerprint, so replacing the file during a run
    # cannot silently associate the examples with the wrong tokenizer hash.
    tokenizer = Tokenizer.from_str(contents.decode("utf-8"))
    # Sampling boundaries belong to this experiment, not tokenizer settings.
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def make_text_examples(
    token_ids,
    context_length,
    *,
    max_examples=None,
    offset=0,
    stride=None,
    corpus_sha256="",
    tokenizer_sha256=""
):
    """Select reproducible, target-inclusive nonoverlapping windows.

    Every row consumes context_length+1 corpus tokens. The first L tokens are
    model input and the final L (shifted by one) are targets. Skipping surplus
    tail tokens is intentional: padding would change a training example's loss.
    """
    if type(context_length) is not int or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if max_examples is not None and (
        type(max_examples) is not int or max_examples <= 0
    ):
        raise ValueError("max_examples must be a positive integer")
    if stride is None:
        stride = context_length + 1
    if type(stride) is not int or stride < context_length + 1:
        raise ValueError("stride must keep target-inclusive examples disjoint")
    tokens = torch.as_tensor(token_ids, device="cpu")
    if tokens.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("token_ids must contain integers, not floating-point values")
    if tokens.ndim != 1 or bool((tokens < 0).any()):
        raise ValueError("token_ids must be a nonnegative one-dimensional sequence")
    tokens = tokens.to(dtype=torch.int64)
    offsets = tuple(range(offset, len(tokens) - context_length, stride))
    if max_examples is not None:
        offsets = offsets[:max_examples]
    if not offsets:
        raise ValueError("corpus contains no complete selected examples")
    windows = torch.stack(
        [tokens[start : start + context_length + 1] for start in offsets]
    )
    return TextExamples(windows, offsets, corpus_sha256, tokenizer_sha256)


def load_text_examples(
    corpus, tokenizer_dir, context_length, *, max_examples=None, offset=0, stride=None
):
    """Memory-map UTF-8 text, encode once, and retain content fingerprints."""
    corpus = Path(corpus)
    if corpus.stat().st_size == 0:
        raise ValueError("corpus is empty")
    with corpus.open("rb") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            content = mapped[:]
    tokenizer_bytes = (Path(tokenizer_dir) / "tokenizer.json").read_bytes()
    tokenizer = _tokenizer_from_bytes(tokenizer_bytes)
    token_ids = tokenizer.encode(content.decode("utf-8"), add_special_tokens=False).ids
    return make_text_examples(
        token_ids,
        context_length,
        max_examples=max_examples,
        offset=offset,
        stride=stride,
        corpus_sha256=hashlib.sha256(content).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tokenizer_bytes).hexdigest(),
    )
