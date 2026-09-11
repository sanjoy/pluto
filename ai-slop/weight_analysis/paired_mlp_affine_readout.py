"""CPU numerical readout for the explicit E/N/F/M LN2-by-MLP experiment.

Every cell fixes the same selected donor embedding rows in both tied roles.
N adds early LN2 affine parameters, F adds early MLP projections and biases,
and M adds their union. These labels never stand in for another cube's axes.

The caller must authenticate checkpoint selection, native execution, cases,
and baseline controls before supplying the selected-score records here. This
module performs no file I/O or forward pass and does not certify those gates.
"""

from collections import defaultdict
import math
import re

from . import paired_complement_localization as core


FORMAT = 'pluto-paired-mlp-affine-readout-v1'
CELLS = ('E', 'N', 'F', 'M')
STRATA = ('suite', 'kind', 'split', 'spelling_variant', 'prefix_domain',
          'word_has_leading_space', 'piece_id', 'target')
_require = core.native._require


def _sum(values):
    """Reject an unrepresentable result instead of publishing infinite JSON."""
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError('nonfinite aggregate log score or contrast') from error
    _require(math.isfinite(result), 'nonfinite aggregate log score or contrast')
    return result


def _contrasts(values):
    """Signed log-score effects; M=N+F is a weight selection, not addition."""
    return {
        'N-E': _sum((values['N'], -values['E'])),
        'F-E': _sum((values['F'], -values['E'])),
        'M-F': _sum((values['M'], -values['F'])),
        'M-N': _sum((values['M'], -values['N'])),
        'M-N-F+E': _sum((values['M'], -values['N'], -values['F'], values['E'])),
    }


def aggregate(per_case):
    """Return stratified metrics for exactly four complete, paired cells.

    Each input uses the selected-case schema of the early-branch reader:
    case identity/stratum fields plus cells[cell].token_log_probability and
    cells[cell].argmax_ids. Extra raw measurements such as token_nll are not
    rewritten. The returned list contains the eight STRATA fields and metrics.

    Probability metrics sum token log probabilities before averaging over
    distinct causal events. Thus suffix is P(t1,t2 | prefix,t0), not a
    per-token average; word_three includes t0, and a supplemental fourth token
    is an exact native target. Argmax success is a conjunction within the same
    teacher-forced event, not a product of marginal rates or free generation.
    """
    _require(isinstance(per_case, (list, tuple)) and bool(per_case),
             'need nonempty selected cases for all four E/N/F/M cells')
    grouped = defaultdict(list)
    predictions, winners = {}, {}
    for item in per_case:
        _require(isinstance(item, dict), 'selected case must be a mapping')
        kind, ids = item.get('kind'), item.get('target_ids')
        cells = item.get('cells')
        _require(kind in ('word', 'word_next_native', 'control', 'shared_piece')
                 and item.get('suite') == ('supplemental' if kind in ('word_next_native', 'shared_piece') else 'main')
                 and isinstance(ids, (list, tuple))
                 and len(ids) == (4 if kind == 'word_next_native' else 3)
                 and all(type(value) is int and value >= 0 for value in ids)
                 and item.get('split') in ('training', 'test')
                 and item.get('prefix_domain') in ('original', 'replacement', 'shared')
                 and type(item.get('prefix_length')) is int and item['prefix_length'] > 0
                 and type(item.get('prefix_sha256')) is str
                 and re.fullmatch('[0-9a-f]{64}', item['prefix_sha256']) is not None
                 and type(item.get('target')) is str and bool(item['target'])
                 and isinstance(cells, dict) and set(cells) == set(CELLS),
                 'invalid case identity or incomplete E/N/F/M cell coverage')
        _require(item.get('spelling_variant') is None or type(item['spelling_variant']) is str,
                 'invalid spelling stratum')
        _require(item.get('word_has_leading_space') is None or type(item['word_has_leading_space']) is bool,
                 'invalid leading-space stratum')
        _require(item.get('piece_id') is None or
                 (type(item['piece_id']) is int and item['piece_id'] >= 0), 'invalid piece-ID stratum')
        for cell in CELLS:
            measurement = cells[cell]
            _require(isinstance(measurement, dict), 'selected cell must be a mapping')
            logs, argmax = measurement.get('token_log_probability'), measurement.get('argmax_ids')
            _require(isinstance(logs, (list, tuple)) and isinstance(argmax, (list, tuple))
                     and len(logs) == len(argmax) == len(ids)
                     and all(type(value) in (int, float) and math.isfinite(value) and value <= 0 for value in logs)
                     and all(type(value) is int and value >= 0 for value in argmax),
                     'invalid selected log probabilities or argmax IDs')
            # Enforce consistency even across different reporting strata. The
            # same complete causal prefix has one argmax, regardless of which
            # candidate target is scored; losses additionally depend on target.
            for position, target in enumerate(ids):
                prefix = (cell, item['prefix_sha256'], item['prefix_length'], tuple(ids[:position]))
                event = (*prefix, target)
                _require(predictions.setdefault(event, logs[position]) == logs[position]
                         and winners.setdefault(prefix, argmax[position]) == argmax[position],
                         'duplicate causal event has conflicting score or argmax')
        grouped[tuple(item.get(field) for field in STRATA)].append(item)

    result = []
    for key, cases in sorted(grouped.items(), key=lambda pair: repr(pair[0])):
        metrics = {}
        for name, positions in core._positions(cases[0]['kind'], len(cases[0]['target_ids'])).items():
            unique = {}
            for item in cases:
                ids = item['target_ids']
                # Include supplied earlier IDs through the last measured
                # target, not irrelevant later IDs. In particular, aliases
                # differing only in the fourth token count once for word_three.
                identity = (item['prefix_sha256'], item['prefix_length'], tuple(ids[:max(positions)+1]))
                values = {
                    cell: (_sum(item['cells'][cell]['token_log_probability'][position] for position in positions),
                           tuple(item['cells'][cell]['argmax_ids'][position] for position in positions),
                           all(item['cells'][cell]['argmax_ids'][position] == ids[position] for position in positions))
                    for cell in CELLS
                }
                _require(unique.setdefault(identity, values) == values,
                         'duplicate causal event has conflicting metric values')
            count = len(unique)
            # Divide before summing to avoid an unnecessary overflow in a mean
            # of finite same-sign scores. The per-event metric is never divided
            # by token count: this remains a sequence probability.
            means = {cell: _sum(value[cell][0]/count for value in unique.values()) for cell in CELLS}
            successes = {cell: sum(value[cell][2] for value in unique.values()) for cell in CELLS}
            metrics[name] = dict(
                unique_event_count=count, alias_case_count=len(cases),
                cells={cell: dict(mean_log_probability=means[cell],
                    geometric_mean_probability=math.exp(means[cell]),
                    joint_argmax_count=successes[cell], joint_argmax_rate=successes[cell]/count)
                    for cell in CELLS},
                contrasts=_contrasts(means))
        # There is deliberately no pooled "deduplicated_all" stratum. Different
        # domains, spellings, splits, leading spaces, and targets stay separate;
        # overlapping strata are not additional independent observations.
        result.append(dict(zip(STRATA, key), metrics=metrics))
    return result
