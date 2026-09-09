"""Prospective follow-through of the checkpoint-delta token-bag method.

The earlier boundary experiment motivated this fixed extension. Its method is
now applied to previously unexamined checkpoint intervals, one hundred steps
after each candidate restart. This is not independently held-out corpus data.
No corpus, tokenizer, model, optimizer, or GPU is accessed by this extractor.
Only unordered token IDs are frozen; a separate verifier tests the predicted
relationship to sampled windows without choosing a different extractor.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import time

import numpy as np

from . import restart_delta as delta
from .checkpoint import GPT2Config
from .checkpoint_archive import read_embedding
from .late_mlp_paths import file_record, write_exclusive


BOUNDARIES = delta.BOUNDARIES
VARIANTS = delta.VARIANTS
SEED = delta.SEED
TOP_K = delta.TOP_K
CHECKPOINT_OFFSETS = (90, 100, 110, 120)
CENTER_INTERVAL_OFFSETS = (100, 110)


def checkpoint_steps(boundary):
    if type(boundary) is not int or boundary < 0 or boundary % 10:
        raise ValueError("boundary must be a nonnegative integer multiple of ten")
    return [boundary + offset for offset in CHECKPOINT_OFFSETS]


def expected_names():
    return ({f"followthrough_{variant}_{boundary}" for boundary in BOUNDARIES for variant in VARIANTS}
            | {f"shared_followthrough_{variant}" for variant in VARIANTS})


def triplet_score(intervals):
    """Center N+100→N+110 against its immediate left/right intervals.

    The exact existing log/zero/floor rules are reused; no geometry parameter,
    weighting, or candidate count is fitted to this follow-through outcome.
    """
    intervals = np.asarray(intervals, dtype=np.float64)
    if intervals.ndim != 2 or intervals.shape[0] != 3:
        raise ValueError("need exactly three consecutive activity vectors")
    logs = np.asarray([delta.log_activity(row) for row in intervals])
    return logs[1] - (logs[0] + logs[2]) / 2


def shared_scores(components):
    """Keep the prior FP64 mean-percentile rule, including its rounding scope.

    Exact computed score ties use ascending token IDs downstream. Independent
    summation may order mathematically tied percentiles differently at FP64
    roundoff; this is not changed after observing candidates.
    """
    components = np.asarray(components, dtype=np.float64)
    if components.ndim != 2 or components.shape[0] != len(BOUNDARIES):
        raise ValueError("need one score vector for each fixed boundary")
    return np.mean([delta.percentile_midrank(row) for row in components], axis=0)


def run(checkpoint_root, output_dir, protocol, inventory):
    start = time.monotonic()
    checkpoint_root = Path(checkpoint_root).resolve()
    requested_output = Path(output_dir).absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(requested_output)
    output = requested_output.resolve()
    if output == checkpoint_root or checkpoint_root in output.parents:
        raise ValueError("outputs must be outside checkpoint root")
    source_dir = Path(__file__).resolve().parent
    sources = {name: file_record(source_dir / name) for name in
               ('delta_followthrough.py', 'restart_delta.py', 'checkpoint_archive.py',
                'checkpoint.py', 'late_mlp_paths.py')}
    sources['protocol'] = file_record(protocol)
    sources['history_inventory'] = file_record(inventory)
    paths = {}
    for boundary in BOUNDARIES:
        for step in checkpoint_steps(boundary):
            options = [checkpoint_root / f'step_{step}.tar.gz', checkpoint_root / f'step_{step}']
            present = [path for path in options if path.exists() or path.is_symlink()]
            if len(present) != 1:
                raise ValueError(f"missing or ambiguous checkpoint {step}")
            if step in paths:
                raise ValueError("follow-through neighborhoods unexpectedly overlap")
            paths[step] = present[0]
    config = GPT2Config()
    vocabulary = config.vocab_size
    output.mkdir()
    # The exclusive plan is durable before read_embedding can access any weight
    # payload. A failed/partial run never acquires a completion freeze record.
    write_exclusive(output / 'plan.json', {
        'created_utc_before_weight_reads': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'stage': 'delta_followthrough_plan_no_corpus_or_model', 'sources': sources,
        'boundaries': list(BOUNDARIES), 'checkpoint_offsets': list(CHECKPOINT_OFFSETS),
        'center_interval_offsets': list(CENTER_INTERVAL_OFFSETS),
        'checkpoint_paths': {str(step): str(path) for step, path in paths.items()},
        'logical_vocab_size': vocabulary, 'seed': SEED, 'top_k': TOP_K,
        'conditional_prediction_one_based_windows_at_sequence_batch_size_one': [101, 110],
        'motivation': 'Fixed follow-through after observing the earlier checkpoint-boundary token-bag signal.',
    })
    permutation = np.random.Generator(np.random.PCG64(SEED)).permutation(vocabulary)
    provenance, files, candidates, scores, evidence = {}, {}, {}, {}, []
    collected = {variant: [] for variant in VARIANTS}
    for boundary in BOUNDARIES:
        vectors = {variant: [] for variant in VARIANTS}
        previous = None
        previous_step = None
        for step in checkpoint_steps(boundary):
            embedding, record = read_embedding(paths[step], step)
            if (embedding.shape != (config.padded_vocab_size, config.d_model)
                    or embedding.dtype != np.dtype('<f4')):
                raise ValueError("checkpoint reader returned an unexpected physical embedding layout")
            provenance[str(step)] = record
            if previous is not None:
                interval, summary = delta.activities(previous[:vocabulary], embedding[:vocabulary])
                for variant in VARIANTS:
                    vectors[variant].append(interval[variant])
                key = f'interval_{previous_step}_{step}'
                files[key] = delta.write_npz(output / (key + '.npz'), **interval)
                evidence.append({'before_step': previous_step, 'after_step': step,
                                 **summary, 'activity_file': key})
            previous, previous_step = embedding, step
            print(f"Read step {step}; {len(provenance)}/{len(paths)} checkpoints, no corpus access", flush=True)
        for variant in VARIANTS:
            key = f'followthrough_{variant}_{boundary}'
            score = triplet_score(vectors[variant])
            scores[key] = score
            collected[variant].append(score)
            candidates[key] = delta.ranked_candidates(score, permutation, k=TOP_K)
    for variant in VARIANTS:
        components = np.asarray(collected[variant])
        score = shared_scores(components)
        key = f'shared_followthrough_{variant}'
        scores[key] = score
        ranking = delta.ranked_candidates(score, permutation, k=TOP_K)
        ids = ranking['token_ids']
        ranking['positive_component_count'] = (components[:, ids] > 0).sum(axis=0).tolist()
        ranking['component_scores'] = components[:, ids].tolist()
        candidates[key] = ranking
    if set(candidates) != expected_names():
        raise ValueError("missing or extra follow-through ranking")
    files['complete_scores'] = delta.write_npz(output / 'scores.npz', **scores,
                                               identity_permutation=permutation)
    write_exclusive(output / 'candidates.json', {
        'stage': 'frozen_unordered_followthrough_token_candidates',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'top_k': TOP_K, 'candidate_lists': candidates,
        'no_corpus_or_tokenizer_access': True, 'no_model_execution': True,
    })
    files['candidates'] = file_record(output / 'candidates.json')
    files['plan'] = file_record(output / 'plan.json')
    # Full source files and complete original archives (or every directory
    # weight) are rehashed at the end, not merely their embedding payloads.
    for record in provenance.values():
        if record.get('input_files_unchanged_during_read') is not True:
            raise ValueError("checkpoint reader did not verify input integrity")
        for item in record['input_files']:
            if file_record(item['path']) != item:
                raise ValueError("checkpoint input changed before completion")
    for record in [*sources.values(), *files.values()]:
        if file_record(record['path']) != record:
            raise ValueError("source or output evidence changed before freezing")
    result = {
        'schema_version': 1, 'complete': True,
        'stage': 'delta_followthrough_candidate_freeze_not_sequence_recovery',
        'frozen_utc_before_corpus_verification': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sources': sources, 'files': files, 'checkpoint_provenance': provenance,
        'intervals': evidence, 'elapsed_seconds': time.monotonic() - start,
        'runtime_environment': {'numpy': np.__version__,
                                'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
                                'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')},
        'limitations': [
            'Prospective checkpoint intervals, not independent held-out corpus data.',
            'Timestamp gaps and current sampling defaults do not authenticate historical restarts or batches.',
            'Uses the prior fixed activity method, not recovered raw gradients or optimizer state.',
            'Unordered token IDs do not reconstruct sequence order or complete passages.',
        ],
    }
    # This is deliberately the last evidence file. Verifiers must require it,
    # not treat an earlier candidates.json from a partial run as authenticated.
    write_exclusive(output / 'frozen.json', result)
    print(f"Frozen {len(candidates)} follow-through lists and their fixed identity shuffles.", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint-root', 'output-dir', 'protocol', 'inventory'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.checkpoint_root, args.output_dir, args.protocol, args.inventory)


if __name__ == '__main__':
    main()
