"""Separate relative-logit changes from softmax in embedding-factorial evidence.

This CPU-only postprocessor does not change any queued probe or readout. It
revalidates an existing factorial readout against its full native logits and
checkpoint/command provenance before deriving additional quantities. Synthetic
unit fixtures are not native execution evidence, even when they exercise that
validation protocol.

The same recipient-selected competitor is used in all four cells. This avoids
mistaking either a rival switch or an arbitrary common logit offset for an
input/output interaction. A softmax-normalizer contribution can itself include
genuine interactions on OTHER vocabulary rows; the full-vocabulary anchored
logit interaction is therefore also retained. No storage fractions are inferred.
"""

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from . import embedding_factorial_readout as base
from .checkpoint import GPT2Config

FORMAT = 'pluto-embedding-factorial-mechanism-v1'
CELLS = base.CELLS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decompose_logits(logits, target):
    """Analyze one target prediction, holding its AA-selected rival fixed.

    M = logit(target) - logit(rival), Q = logsumexp(logits - logit(rival)),
    so log P(target) = M - Q. All three are invariant in exact arithmetic to
    an arbitrary common offset in each cell. Effects are additive contrasts,
    never ratios. FP64 arithmetic acts on the actual saved FP32 logits, not
    on a reconstructed dot product or a substitute forward implementation.
    """
    require(isinstance(logits, dict) and set(logits) == set(CELLS),
            'expected exactly the four factorial cells')
    arrays = {cell: np.asarray(logits[cell]) for cell in CELLS}
    first = arrays['AA']
    require(first.ndim == 1 and first.size >= 2 and all(
        row.shape == first.shape and row.dtype == np.dtype('<f4')
        and np.isfinite(row).all() for row in arrays.values()),
        'expected equally sized finite one-dimensional FP32 logit rows')
    require(type(target) is int and 0 <= target < first.size, 'invalid target ID')
    candidates = np.arange(first.size)
    candidates = candidates[candidates != target]
    # np.argmax returns the first maximum: ascending IDs resolve exact ties.
    rival = int(candidates[np.argmax(first[candidates])])
    cells, anchored = {}, {}
    for cell, row in arrays.items():
        values = row.astype(np.float64)
        maximum = float(values.max())
        log_mass = math.log(float(np.exp(values - maximum).sum(dtype=np.float64)))
        target_logit, rival_logit = float(values[target]), float(values[rival])
        margin = target_logit - rival_logit
        normalizer = math.fsum((maximum - rival_logit, log_mass))
        # Compute log probability stably, without subtracting two large sums.
        log_probability = -math.fsum((maximum - target_logit, log_mass))
        cells[cell] = dict(target_logit=target_logit, rival_logit=rival_logit,
            margin=margin, log_partition_relative_to_rival=normalizer,
            log_probability=log_probability)
        anchored[cell] = values - rival_logit
    effects = {label: base.factorial_effects({cell: cells[cell][field] for cell in CELLS})
        for label, field in (('margin', 'margin'),
            ('normalizer', 'log_partition_relative_to_rival'), ('log_probability', 'log_probability'))}
    errors = [abs(math.fsum((effects['margin'][kind], -effects['normalizer'][kind],
                            -effects['log_probability'][kind])))
              for kind in effects['margin']]
    # Inspect EVERY relative output, not just the target/rival margin. A zero
    # target-margin interaction does not rule out a cross-effect elsewhere.
    interaction = ((anchored['JJ'] - anchored['JA']) -
                   (anchored['AJ'] - anchored['AA']))
    return dict(target_id=target, rival_id=rival, cells=cells, effects=effects,
        max_effect_identity_error=max(errors),
        anchored_logit_interaction=dict(max_abs=float(np.abs(interaction).max()),
            l2=float(np.linalg.norm(interaction)),
            target_margin_interaction=float(interaction[target])))


def _sequence(tokens):
    """Sum teacher-forced predictions; no averaging or event deduplication."""
    return {name: {kind: math.fsum(token['effects'][name][kind] for token in tokens)
                   for kind in ('input', 'output', 'interaction', 'joint')}
            for name in ('margin', 'normalizer', 'log_probability')}


def analyze(readout_path, output, *, config=GPT2Config()):
    """Revalidate a finished native readout, then write one exclusive new file.

    All saved per-case labels, target IDs, source hashes, scores and execution
    records must equal a fresh run of the existing frozen CPU reader. Missing
    execution provenance is rejected. This authenticates recorded execution
    evidence; it does not independently attest that a process really ran.
    """
    readout_path, output = Path(readout_path), Path(output)
    require(not output.exists() and not output.is_symlink(), 'output already exists')
    readout_record = base._record(readout_path)
    saved = base._json(readout_path)
    require(saved.get('format') == 'pluto-embedding-factorial-readout-v1',
            'not a factorial readout')
    require(saved.get('execution_provenance', {}).get('verified') is True,
            'execution provenance is required')
    metadata_path = Path(saved['native_metadata']['path'])
    forbidden = [Path(path).resolve() for path in saved['patch']['paths'].values()]
    forbidden.append(metadata_path.parent.resolve())
    destination = output.resolve()
    require(all(root != destination and root not in destination.parents for root in forbidden),
            'output must be outside source checkpoints and native score directory')
    own_source = base._record(__file__)
    with tempfile.TemporaryDirectory(prefix='pluto-factorial-mechanism-') as temporary:
        fresh = base.analyze(saved['cases']['path'], metadata_path.parent,
            saved['patch']['patch']['path'], Path(temporary) / 'revalidated.json',
            expected_rows=saved['patch']['selected_rows'],
            case_kind=saved['case_kind_selection'],
            execution_record=saved['execution_provenance']['record']['path'], config=config)
    require(fresh == saved, 'saved readout disagrees with independent revalidation')
    metadata = base._json(metadata_path)
    shape = tuple(metadata['logits_shape'])
    arrays = {cell: np.memmap(metadata_path.parent / (cell + '.logits.f32.bin'),
        dtype='<f4', mode='r', shape=shape) for cell in CELLS}
    per_case = []
    for index, item in enumerate(fresh['per_case']):
        require(item['native_case_index'] == index, 'unexpected native case ordering')
        tokens = []
        for position, target in enumerate(item['target_ids']):
            token = decompose_logits({cell: arrays[cell][index, position] for cell in CELLS}, target)
            for cell in CELLS:
                base._close(token['cells'][cell]['log_probability'],
                    item['cells'][cell]['token_log_probability'][position], 'token log probability')
            for kind, value in token['effects']['log_probability'].items():
                base._close(value, item['token_log_probability_effects'][position][kind],
                            'token log probability factorial contrast')
            token.update(target_position=position, **item['predictions'][position])
            tokens.append(token)
        result = {key: item[key] for key in ('native_case_index', 'source_case_index',
                  'kind', 'split', 'context_id', 'prefix_domain', 'target', 'target_ids')}
        for key in ('spelling_variant', 'piece_id', 'word_has_leading_space'):
            if key in item:
                result[key] = item[key]
        result.update(tokens=tokens, selected_sequence=_sequence(tokens))
        for kind, value in result['selected_sequence']['log_probability'].items():
            base._close(value, item['log_probability_effects'][kind], 'sequence log probability contrast')
        if item['kind'] in ('word', 'word_next_native'):
            result.update(first_piece=_sequence(tokens[:1]), suffix=_sequence(tokens[1:3]),
                          word_three=_sequence(tokens[:3]))
            if item['kind'] == 'word_next_native':
                result['exact_next_native_token'] = _sequence(tokens[3:4])
        per_case.append(result)
    records = [readout_record, own_source, *fresh['files']]
    for record in records:
        require(base._record(record['path']) == record, 'source changed during decomposition')
    report = dict(format=FORMAT, complete=True, source_readout=readout_record,
        execution_provenance=fresh['execution_provenance'], files=records, per_case=per_case,
        model_forward_performed=False, weights_edited=False, goal_completion_claimed=False,
        definitions=dict(rival='Highest recipient AA non-target logit, ties by lower ID; fixed in all four cells.',
            decomposition='log_probability effect = fixed-rival margin effect minus fixed-rival normalizer effect.',
            cells='AA recipient; AJ output dictionary only; JA input embedding only; JJ joint patch.',
            effects='input=JA-AA; output=AJ-AA; interaction=JJ-JA-AJ+AA; joint=JJ-AA.',
            event_weighting='Per-case evidence only: aliases are retained, not averaged or treated as independent samples.',
            full_vocabulary='Anchored logit interaction measures all logical columns relative to the same rival in FP64.'),
        limitations=[
            'A normalizer contribution can reflect genuine cross-effects on competing logits, not just softmax curvature.',
            'Zero chosen-margin interaction does not imply zero full-vocabulary interaction or no functional role.',
            'In exact-real arithmetic, margin interaction is (delta output_target minus delta output_rival) dot delta_hidden; native rounded logits are not an exact dot-product attribution.',
            'Candidate suffixes have different teacher-forced histories; their normalizers do not generally cancel in whole-word odds.',
            'Different targets can select different fixed rivals; their margin/normalizer attribution terms must not be pooled as one shared-rival comparison.',
            'Source hashes and recorded execution provenance are not independent attestation of native process execution.',
            'This is an interpretation of an embedding-row intervention, not a unique word-storage location or an all-parameter causal account.'])
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.readout, args.output)


if __name__ == '__main__':
    main()
