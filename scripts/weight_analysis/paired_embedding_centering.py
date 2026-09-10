"""Separate a vocabulary-common embedding shift from row-specific changes.

For a FIXED final hidden vector h, adding the same vector mu to every output
dictionary row adds h.dot(mu) to every logical logit. In exact arithmetic this
cannot change softmax probabilities or token-to-token logit margins. The
decomposition here describes that common component; it does not edit weights.

This is NOT an invariance of the complete tied-embedding model. Changing its
input embeddings may change h, and native rounding/reductions need not preserve
exact real-arithmetic identities. Even centered row differences are not causal
importance or a prediction of actual probability changes. No forward is run.
"""

import argparse
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from . import checkpoint, historical_word_ablations as records
from . import paired_weight_diff
from . import phrase_reference

FORMAT = 'pluto-paired-embedding-centering-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decompose(original, replacement, *, vocabulary, selected_rows=(),
              precision='bf16', chunk_rows=256, top_k=20):
    """Two bounded passes over a physical [padded_vocabulary,width] matrix.

    Padding is finite-validated but excluded from mean/ranks/denominators.
    In BF16 mode, round EACH original operand first, not their FP32 difference.
    A second pass explicitly measures centered rows, avoiding subtraction of
    two nearly equal energies when the change is almost entirely common.
    Temporary storage is O(chunk_rows*width + vocabulary + width).
    """
    a, b = np.asarray(original), np.asarray(replacement)
    ids = list(selected_rows)
    require(a.dtype == np.dtype('<f4') and b.dtype == np.dtype('<f4')
            and a.ndim == 2 and b.shape == a.shape and a.shape[1] > 0,
            'expected equally shaped FP32 physical embedding matrices')
    require(type(vocabulary) is int and 1 <= vocabulary <= a.shape[0],
            'invalid logical vocabulary')
    require(precision in ('fp32', 'bf16'), 'precision must be fp32 or bf16')
    require(type(chunk_rows) is int and chunk_rows > 0
            and type(top_k) is int and top_k > 0, 'positive integer chunk/rank sizes required')
    require(all(type(row) is int and 0 <= row < vocabulary for row in ids)
            and ids == sorted(set(ids)), 'selected IDs must be sorted unique logical rows')

    def delta(first, last):
        left, right = a[first:last], b[first:last]
        require(np.isfinite(left).all() and np.isfinite(right).all(), 'nonfinite embedding parameter')
        if precision == 'bf16':
            left, right = phrase_reference.bf16(left), phrase_reference.bf16(right)
        return right.astype(np.float64) - left.astype(np.float64)

    raw_energy = np.empty(vocabulary, dtype=np.float64)
    total_vector = np.zeros(a.shape[1], dtype=np.float64)
    correction = np.zeros_like(total_vector)
    for first in range(0, vocabulary, chunk_rows):
        last = min(first + chunk_rows, vocabulary)
        values = delta(first, last)
        subtotal = np.sum(values, axis=0, dtype=np.float64)
        # Compensated vector summation keeps the shared direction stable when
        # chunk sizes differ or positive/negative row changes nearly cancel.
        updated = total_vector + subtotal
        correction += np.where(np.abs(total_vector) >= np.abs(subtotal),
            (total_vector - updated) + subtotal, (subtotal - updated) + total_vector)
        total_vector = updated
        raw_energy[first:last] = np.sum(values * values, axis=1)
    mean = (total_vector + correction) / vocabulary
    common_energy = float(vocabulary * np.dot(mean, mean))
    centered_energy = np.empty_like(raw_energy)
    centered_sum = np.zeros_like(mean)
    for first in range(0, vocabulary, chunk_rows):
        last = min(first + chunk_rows, vocabulary)
        values = delta(first, last) - mean
        centered_energy[first:last] = np.sum(values * values, axis=1)
        centered_sum += np.sum(values, axis=0)
    padding_subtotals = []
    for first in range(vocabulary, a.shape[0], chunk_rows):
        values = delta(first, min(first + chunk_rows, a.shape[0]))
        padding_subtotals.append(float(np.sum(values * values)))
    total = math.fsum(raw_energy)
    centered = math.fsum(centered_energy)
    residual = total - common_energy - centered
    require(math.isfinite(total) and math.isfinite(centered) and math.isfinite(common_energy),
            'nonfinite decomposition arithmetic')
    require(abs(residual) <= 1e-10 * max(total, np.finfo(np.float64).tiny),
            'orthogonal energy decomposition did not close')
    raw_order = np.lexsort((np.arange(vocabulary), -raw_energy))
    centered_order = np.lexsort((np.arange(vocabulary), -centered_energy))
    ranks = np.empty(vocabulary, dtype=np.int64)
    ranks[centered_order] = np.arange(1, vocabulary + 1)
    raw_ranks = np.empty_like(ranks)
    raw_ranks[raw_order] = np.arange(1, vocabulary + 1)

    def row_report(row):
        return dict(token_id=int(row), raw_delta_l2=math.sqrt(float(raw_energy[row])),
            centered_delta_l2=math.sqrt(float(centered_energy[row])),
            raw_rank=int(raw_ranks[row]), centered_rank=int(ranks[row]))

    return dict(precision=precision, vocabulary=vocabulary, width=a.shape[1],
        physical_rows=a.shape[0], delta_definition='replacement minus original',
        mean_delta=mean.tolist(), common_row_delta_l2=float(np.linalg.norm(mean)),
        total_delta_squared=total, common_delta_squared=common_energy,
        centered_delta_squared=centered, common_fraction=common_energy / total if total else None,
        energy_identity_residual=residual, centered_row_sum_l2=float(np.linalg.norm(centered_sum)),
        padding_delta_squared=math.fsum(padding_subtotals),
        selected_rows=[row_report(row) for row in ids],
        selected_raw_fraction=math.fsum(raw_energy[ids]) / total if total else None,
        selected_centered_fraction=math.fsum(centered_energy[ids]) / centered if centered else None,
        largest_raw_rows=[row_report(row) for row in raw_order[:top_k]],
        largest_centered_rows=[row_report(row) for row in centered_order[:top_k]])


def _stat(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def analyze(original, replacement, *, selected_rows=(), config=checkpoint.GPT2Config(),
            chunk_rows=256, top_k=20):
    """Validate complete checkpoint layouts; numerically inspect embeddings only.

    Embedding hashes/stats are checked before/after both precision passes.
    Authentication of training, other weight contents, and same optimizer step
    remains the responsibility of the separately recorded paired-run audit.
    """
    directories = [Path(path).absolute() for path in (original, replacement)]
    require(all(path.resolve() == path and not path.is_symlink() and path.is_dir()
                for path in directories), 'checkpoint paths must be canonical directories')
    models = [checkpoint.GPT2Checkpoint(path, config) for path in directories]
    paths = [path / 'weight_0.bin' for path in directories]
    sources = [records.record(module.__file__) for module in
               (checkpoint, phrase_reference, paired_weight_diff, records)]
    sources.append(records.record(__file__))
    before = [(records.record(path), _stat(path)) for path in paths]
    results = {precision: decompose(models[0]['token_embedding.weight'], models[1]['token_embedding.weight'],
        vocabulary=config.vocab_size, selected_rows=selected_rows,
        precision=precision, chunk_rows=chunk_rows, top_k=top_k) for precision in ('fp32', 'bf16')}
    require(before == [(records.record(path), _stat(path)) for path in paths],
            'embedding changed during analysis')
    for path in directories:
        checkpoint.GPT2Checkpoint(path, config)  # Recheck complete file layout.
    require(all(records.record(item['path']) == item for item in sources), 'analysis source changed')
    return dict(format=FORMAT, complete=True, config=asdict(config),
        original_embedding=before[0][0], replacement_embedding=before[1][0],
        sources=sources, by_precision=results, model_forward_performed=False,
        weights_edited=False, goal_completion_claimed=False,
        limitations=['Only embedding values are numerically inspected; other weight contents require the separate checkpoint audit.',
            'Common-row softmax invariance holds for a fixed hidden state in exact arithmetic, not the complete tied model.',
            'BF16 centering is a decomposition of rounded operands, not a representable native weight edit or a bitwise forward oracle.',
            'Squared weight differences are not causal importance, word probabilities, or unique storage fractions.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'replacement', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--row', action='append', default=[], type=int)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    output = args.output.resolve()
    require(all(path.resolve() not in (output, *output.parents) for path in (args.original, args.replacement)),
            'output must be outside both input checkpoints')
    result = analyze(args.original, args.replacement, selected_rows=args.row)
    paired_weight_diff.write_report(output, result)


if __name__ == '__main__':
    main()
