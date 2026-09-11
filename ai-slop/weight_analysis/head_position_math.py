"""CPU oracles for a head's query-local versus other-position contributions.

These functions do not execute a model or authenticate an experiment. A future
native reader must first bind each array to a real command, checkpoint, prefix,
selected site and unchanged-input record. Synthetic tests here are not measured
head effects. Keeping this numerical layer separate makes its assumptions small
enough to check independently of native suffix replay.
"""

import numpy as np

from .source_value_readout import _finite_bf16, scale_bf16, score_logits

SCOPES = ('query_only', 'other_queries', 'all_queries')


def selected_rows(total_rows, context_length, sequence, query, scope):
    """Partition ONE sequence's query positions, including any future rows.

    The query is a position within the selected sequence. Other sequences are
    never edited. With query zero, all 'other_queries' are in its causal future
    and cannot affect that query's eventual prediction. With the final query,
    all other queries precede it and can communicate through later attention.
    """
    if (any(type(x) is not int for x in (total_rows, context_length, sequence, query))
            or context_length < 1 or total_rows < 1 or total_rows % context_length
            or not 0 <= sequence < total_rows // context_length
            or not 0 <= query < context_length or scope not in SCOPES):
        raise ValueError('invalid head-position selection')
    local = np.arange(context_length)
    if scope == 'query_only':
        local = local[local == query]
    elif scope == 'other_queries':
        local = local[local != query]
    return sequence * context_length + local


def audit_context(original, changed, *, context_length, sequence, query, head,
                  head_dim, scope, scale):
    """Require the exact BF16 dose on precisely the selected context cells.

    This is an attention-CONTEXT intervention, not a source-V edit, key mask,
    or projection-weight scaling. Only at dose zero should it be compared with
    corresponding zero projection rows. Half-context and half-weight rounding
    are distinct floating-point interventions and must not be equated.
    """
    original, changed = np.asarray(original), np.asarray(changed)
    _finite_bf16(original)
    _finite_bf16(changed)
    if original.ndim != 2 or original.shape != changed.shape:
        raise ValueError('context shape mismatch')
    rows, width = original.shape
    if (type(head) is not int or type(head_dim) is not int or head_dim < 1
            or width < 1 or width % head_dim or not 0 <= head < width // head_dim):
        raise ValueError('invalid head dimension/index')
    positions = selected_rows(rows, context_length, sequence, query, scope)
    columns = np.arange(head * head_dim, (head + 1) * head_dim)
    expected = original.copy()
    expected[np.ix_(positions, columns)] = scale_bf16(
        original[np.ix_(positions, columns)], scale)
    if not np.array_equal(changed, expected):
        raise ValueError('context differs from exact position/head dose')
    return dict(selected_rows=positions.tolist(), selected_element_count=int(len(positions) * head_dim),
                changed_element_count=int(np.count_nonzero(changed != original)),
                scope=scope, scale=scale)


def audit_causal_logits(clean, changed, *, context_length, sequence, query,
                        scope, scale):
    """Check exact native FP32 parity wherever the causal graph forbids effects.

    This checks ALL physical output columns, including padded vocabulary, and
    every unaffected row. It does not establish correct softmax probabilities
    or tail replay at rows allowed to change; those need separate checks.
    """
    clean, changed = np.asarray(clean), np.asarray(changed)
    if (clean.dtype != np.dtype('<f4') or changed.dtype != np.dtype('<f4')
            or clean.ndim != 2 or clean.shape != changed.shape or clean.shape[1] < 1
            or not np.isfinite(clean).all() or not np.isfinite(changed).all()
            or type(scale) not in (int, float) or scale not in (0, 0.5, 1)):
        raise ValueError('invalid native logit arrays/dose')
    positions = selected_rows(len(clean), context_length, sequence, query, scope)
    unaffected = np.ones(len(clean), dtype=bool)
    if scale != 1 and len(positions):
        # Only rows in the SAME sequence at or after the first edit can change.
        unaffected[positions[0]:(sequence + 1) * context_length] = False
    if clean[unaffected].tobytes() != changed[unaffected].tobytes():
        raise ValueError('logits changed outside causal descendants of the edit')
    return dict(unchanged_row_count=int(unaffected.sum()),
                potentially_affected_row_count=int((~unaffected).sum()),
                all_physical_columns_checked=int(clean.shape[1]))


def partition_effects(logits, target, rival=None):
    """Compare query-only, other-query and all-query edits against clean logits.

    All arrays must be the complete LOGICAL vocabulary for one identical
    prediction event and one common dose. Choose the rival on the clean row
    once, then retain it even if the largest competitor changes. Positive
    log-probability/margin deltas favor the target; positive NLL deltas hurt it.

    The interaction is f(all)-f(query)-f(other)+f(clean). It measures failure
    of additive effects for the two disjoint sets of edited positions. It is
    not a unique pathway, a sum of neuron attributions, or a mediation fraction.
    Even linear logit additions can have nonlinear log-probability effects.
    """
    if set(logits) != {'clean', *SCOPES}:
        raise ValueError('need exactly clean and all three position scopes')
    arrays = {key: np.asarray(value) for key, value in logits.items()}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError('vocabulary shapes differ')
    clean = score_logits(arrays['clean'], target, rival)
    scores = {'clean': clean, **{scope: score_logits(arrays[scope], target, clean['fixed_rival_id'])
                               for scope in SCOPES}}
    effects = {scope: dict(delta_nll=scores[scope]['nll'] - clean['nll'],
        delta_log_probability=clean['nll'] - scores[scope]['nll'],
        delta_fixed_rival_margin=scores[scope]['fixed_rival_margin'] - clean['fixed_rival_margin'])
        for scope in SCOPES}
    interaction = {metric: effects['all_queries'][metric] - effects['query_only'][metric]
                   - effects['other_queries'][metric] for metric in effects['all_queries']}
    return dict(target_id=target, fixed_rival_id=clean['fixed_rival_id'], scores=scores,
                effects=effects, interaction=interaction,
                query_and_all_margin_effects_have_opposite_signs=(
                    effects['query_only']['delta_fixed_rival_margin'] *
                    effects['all_queries']['delta_fixed_rival_margin'] < 0),
                probability_is_temperature_one=True)
