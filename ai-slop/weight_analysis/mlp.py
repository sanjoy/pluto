#!/usr/bin/env python3
"""Corpus-blind token associations from paired MLP key/value weight vectors.

This is NOT a transformer forward pass. It projects static weight directions
onto the tied embedding vocabulary, then optionally enumerates paths in the
resulting directed graph. GELU gates, attention, positions, residual inputs,
and LayerNorm are deliberately absent. Consequently these are hypotheses about
stored associations, not a claim that the original text has been decompressed.
No argument or API here accepts training text or corpus token frequencies.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .checkpoint import GPT2Checkpoint


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize without assigning arbitrary directions to zero vectors."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("expected a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def stable_topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Descending scores, breaking even boundary ties by ascending token ID."""
    if scores.ndim != 2 or not 1 <= k <= scores.shape[1]:
        raise ValueError("invalid top-k dimensions")
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite vocabulary scores")
    ids = np.empty((len(scores), k), dtype=np.int64)
    for row, values in enumerate(scores):
        threshold = np.partition(values, len(values) - k)[len(values) - k]
        better = np.flatnonzero(values > threshold)
        tied = np.flatnonzero(values == threshold)[: k - len(better)]
        selected = np.concatenate((better, tied))
        ids[row] = selected[np.lexsort((selected, -values[selected]))]
    return ids, np.take_along_axis(scores, ids, axis=1)


def project_directions(embedding: np.ndarray, directions: np.ndarray, k: int,
                       chunk_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Exact vocabulary search in bounded-size matrix products; no prompts.

    Inputs and weights are independently unit-normalized. This removes vector
    magnitude from the ranking; it is a cosine probe, not LM logits. Its
    vocabulary is the entire real tokenizer vocabulary, never a corpus subset.
    """
    if chunk_size < 1 or directions.shape[1] != embedding.shape[1]:
        raise ValueError("invalid projection dimensions")
    vocabulary = unit_rows(embedding)
    rows = unit_rows(directions)
    ids, values = [], []
    for first in range(0, len(rows), chunk_size):
        selected, scores = stable_topk(rows[first:first + chunk_size] @ vocabulary.T, k)
        ids.append(selected)
        values.append(scores)
    return np.concatenate(ids), np.concatenate(values)


def unit_address(block: int, key: int, value: int, d: int, m: int) -> dict:
    """Byte-level certificate for one key column and its paired value row.

    FullyConnected stores [input_dim, output_dim] matrices. A key is therefore
    strided, whereas a value is contiguous. Token embedding rows are additional
    shared weights needed to interpret these vectors, not an external lexicon
    of Shakespeare passages.
    """
    return {
        "block": block,
        "key_neuron": key,
        "value_neuron": value,
        "key": {"tensor": f"blocks.{block}.mlp.input.weight",
                "file": f"weight_{2 + 12 * block + 8}.bin",
                "offset_bytes": key * 4, "count": d, "stride_bytes": m * 4},
        "value": {"tensor": f"blocks.{block}.mlp.output.weight",
                  "file": f"weight_{2 + 12 * block + 10}.bin",
                  "offset_bytes": value * d * 4, "count": d, "stride_bytes": 4},
        "vocabulary": {"tensor": "token_embedding.weight", "file": "weight_0.bin",
                       "row_bytes": d * 4},
    }


def build_edges(key_ids, key_scores, value_ids, value_scores, addresses):
    """Keep the strongest positive association for each directed token pair.

    A product of cosines is only an association score, not a probability.
    Both alignments must be positive, so two negative alignments cannot create
    a spurious positive edge. Self edges are kept for analysis but never used
    by path enumeration, since simple copying is not a recovered passage.
    """
    edges = {}
    for unit, address in enumerate(addresses):
        for source, a in zip(key_ids[unit], key_scores[unit]):
            for target, b in zip(value_ids[unit], value_scores[unit]):
                if a <= 0 or b <= 0:
                    continue
                pair = (int(source), int(target))
                score = float(a) * float(b)
                if pair not in edges or score > edges[pair]["score"]:
                    edges[pair] = {"score": score, "provenance": address}
    return edges


def decode_paths(edges: dict, *, length: int = 8, starts: int = 1000,
                 beam_width: int = 4):
    """Enumerate static graph paths; never reevaluate transformer activations.

    Starts come from strongest outgoing edges, without corpus frequencies or
    prompt seeds. The fixed heuristic allows a token at most twice and skips
    self edges to limit trivial repetition. A path is an unverified candidate;
    composing individually plausible edges does not prove sequence memory.
    """
    if length < 2 or starts < 0 or beam_width < 1:
        raise ValueError("invalid path search limits")
    adjacency = defaultdict(list)
    for (source, target), edge in edges.items():
        if source != target:
            adjacency[source].append((target, edge))
    for outgoing in adjacency.values():
        outgoing.sort(key=lambda item: (-item[1]["score"], item[0]))
    start_tokens = sorted(adjacency, key=lambda t: (-adjacency[t][0][1]["score"], t))[:starts]
    for source in start_tokens:
        beam = [((source,), 0.0, [])]
        for _ in range(length - 1):
            extended = []
            for tokens, score, provenance in beam:
                valid = [(token, edge) for token, edge in adjacency.get(tokens[-1], [])
                         if tokens.count(token) < 2][:beam_width]
                for token, edge in valid:
                    extended.append((tokens + (token,), score + edge["score"],
                                     provenance + [edge["provenance"]]))
            if not extended:
                break
            beam = sorted(extended, key=lambda item: (-item[1], item[0]))[:beam_width]
        tokens, score, provenance = beam[0]
        if len(tokens) >= 2:
            yield {"token_ids": list(tokens), "score": score / (len(tokens) - 1),
                   "provenance": {"edges": provenance, "search": "bounded static graph paths"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--path-length", type=int, default=8)
    parser.add_argument("--path-starts", type=int, default=1000)
    parser.add_argument("--include-control", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = GPT2Checkpoint(args.checkpoint, check_finite=True)
    config = checkpoint.config
    # Only the vocabulary labels are read. There is no encoder or corpus input.
    tokenizer_path = args.tokenizer_dir / "tokenizer.json"
    vocabulary = json.loads(tokenizer_path.read_text())["model"]["vocab"]
    if len(vocabulary) != config.vocab_size or set(vocabulary.values()) != set(range(config.vocab_size)):
        raise ValueError("tokenizer vocabulary does not match checkpoint")
    if not 1 <= args.top_k <= config.vocab_size or args.chunk_size < 1:
        parser.error("invalid top-k or chunk-size")
    if args.path_length < 2 or args.path_starts < 0:
        parser.error("invalid path limits")
    labels = {index: label for label, index in vocabulary.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite extraction artifacts")
    graphs = {"mlp_cosine": {}}
    if args.include_control:
        graphs["broken_mlp_cosine"] = {}
    counts = Counter()
    with args.output.open("x") as output:
        def emit(record):
            record["vocabulary_labels"] = [labels[t] for t in record["token_ids"]]
            output.write(json.dumps(record) + "\n")
            counts[record["method"]] += 1

        for block in range(config.n_layers):
            keys, key_scores = project_directions(checkpoint.token_embedding,
                checkpoint.mlp_keys(block), args.top_k, args.chunk_size)
            values, value_scores = project_directions(checkpoint.token_embedding,
                checkpoint.mlp_values(block), args.top_k, args.chunk_size)
            for method, graph in graphs.items():
                # Fixed derangement breaks associations while preserving all
                # learned key/value directions and their vocabulary marginals.
                pairing = np.arange(config.d_ff)
                if method.startswith("broken_"):
                    pairing = np.roll(pairing, 1)
                addresses = [unit_address(block, j, int(pairing[j]), config.d_model, config.d_ff)
                             for j in range(config.d_ff)]
                edges = build_edges(keys, key_scores, values[pairing], value_scores[pairing], addresses)
                for pair, edge in edges.items():
                    if pair not in graph or edge["score"] > graph[pair]["score"]:
                        graph[pair] = edge
                for j, address in enumerate(addresses):
                    paired = pairing[j]
                    # A zero direction has no preferred token. Stable tie
                    # breaking must not turn it into an invented association.
                    if key_scores[j, 0] <= 0 or value_scores[paired, 0] <= 0:
                        continue
                    emit({"candidate_id": f"{method}:block{block}:neuron{j}",
                          "method": method + "_pair",
                          "token_ids": [int(keys[j, 0]), int(values[paired, 0])],
                          "score": float(key_scores[j, 0]) * float(value_scores[paired, 0]),
                          "provenance": address,
                          "top_keys": keys[j].tolist(), "top_values": values[paired].tolist()})
            print(f"projected block {block + 1}/{config.n_layers}", file=sys.stderr, flush=True)
        for method, graph in graphs.items():
            for index, path in enumerate(decode_paths(graph, length=args.path_length,
                                                      starts=args.path_starts)):
                emit(dict(path, candidate_id=f"{method}:path{index}", method=method + "_path"))
    metadata = {"schema_version": 1, "stage": "corpus_blind_extraction",
                "method_counts": dict(counts), "parameters": {k: str(v) if isinstance(v, Path) else v
                                                             for k, v in vars(args).items()},
                "checkpoint": checkpoint.provenance(hash_weights=True),
                "limitations": ["No GELU, attention, LayerNorm, positions, or residual states.",
                                "Cosine associations and graph paths are not recovered text until independently verified.",
                                "Short matches may reflect generic language or BPE tokenization, not passage memorization."]}
    metadata["extractor_sources"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("mlp.py", "checkpoint.py")
    }
    metadata["tokenizer_sha256"] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    metadata["candidate_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(counts)), flush=True)


if __name__ == "__main__":
    main()
