"""Descriptive E/M computation accounting, never neuron deletion effects.

Inputs must come from authenticated, same-prefix native traces. This module
does not authenticate files or execute models. It separates target-logit and
normalizer changes, and separates changed feature activation from changed
decoder direction at one explicitly fixed readout. These decompositions are
hypothesis generators; subsequent interventions must test the hypotheses.
"""

import math

import numpy as np

from .phrase_reference import bf16
from .phrase_trace_analysis import validate_stages
from .token_path_math import neuron_operations, target_ledger


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def prediction(logits, target, competitor):
    """FP64 softmax statistics on the logical vocabulary, with stable ties."""
    values = np.asarray(logits, dtype=np.float64)
    _require(values.ndim == 1 and len(values) > 1 and np.isfinite(values).all(),
             'need finite logical-vocabulary logits')
    _require(all(type(i) is int and 0 <= i < len(values) for i in (target, competitor))
             and target != competitor, 'invalid target or competitor')
    maximum = float(values.max())
    shifted_normalizer = math.log(math.fsum(math.exp(float(x)-maximum) for x in values))
    normalizer = maximum + shifted_normalizer
    # Subtract the common offset before adding the small normalizer. Otherwise
    # e.g. equal logits at 1e20 would incorrectly give probability one.
    log_probability = (float(values[target])-maximum)-shifted_normalizer
    margin = float(values[target])-float(values[competitor])
    _require(all(math.isfinite(x) for x in (normalizer, log_probability, margin)),
             'nonfinite prediction statistics')
    return dict(target_logit=float(values[target]), competitor_logit=float(values[competitor]),
                logsumexp=normalizer, logit_maximum=maximum, shifted_logsumexp=shifted_normalizer,
                log_probability=log_probability,
                probability=math.exp(log_probability), margin=margin,
                argmax_id=int(np.argmax(values)))


def neuron_delta(z_e, z_m, decoder_e, decoder_m, direction):
    """Expand z_M D_M - z_E D_E in one fixed direction, feature by feature.

    The mixed term is retained: replacing only activations and replacing only
    decoder rows need not add to the combined change. This is an algebraic
    expansion of observed endpoints, NOT a native activation intervention or
    a prediction that these isolated changes would produce the same output.
    Decoder arrays are already the effective BF16 MMA operands promoted to
    FP64 by the caller; do not silently round arbitrary inputs here.
    """
    ze, zm, de, dm, q = [np.asarray(x, dtype=np.float64) for x in
                         (z_e, z_m, decoder_e, decoder_m, direction)]
    _require(ze.ndim == zm.ndim == q.ndim == 1 and de.ndim == dm.ndim == 2
             and ze.size > 0 and q.size > 0 and ze.shape == zm.shape
             and de.shape == dm.shape == (ze.size, q.size)
             and all(np.isfinite(x).all() for x in (ze, zm, de, dm, q)),
             'invalid neuron-delta shape or nonfinite input')
    with np.errstate(over='ignore', invalid='ignore'):
        qe, qm = de @ q, dm @ q
        dq, dz = (dm-de) @ q, zm-ze
        result = dict(delta=zm*qm-ze*qe, activation=dz*qe,
                      decoder=ze*dq, interaction=dz*dq)
    _require(all(np.isfinite(x).all() for x in result.values()), 'nonfinite neuron-delta result')
    # Each dot product is independently rounded in this FP64 diagnostic. Keep
    # the small residual explicit rather than pretending exact floating-point
    # distributivity. It is separate from actual native BF16/MMA remainders.
    try:
        result['analytical_remainder'] = np.asarray([
            math.fsum((float(result['delta'][i]), -float(result['activation'][i]),
                       -float(result['decoder'][i]), -float(result['interaction'][i])))
            for i in range(ze.size)], dtype=np.float64)
    except (ValueError, OverflowError) as error:
        raise ValueError('nonfinite analytical remainder') from error
    _require(np.isfinite(result['analytical_remainder']).all(), 'nonfinite analytical remainder')
    return result


def analyze_pair(checkpoint_e, stages_e, logits_e, checkpoint_m, stages_m,
                 logits_m, token_ids, target, top_k=3):
    """Explain a matched E/M observation without assigning a word to a neuron.

    The competitor is E's highest-logit non-target token, fixed across both
    traces (lowest ID breaks ties). The ordinary ledgers each use their own
    observed final-normalization statistics; their difference is accounting,
    not mediation. For activation/decoder separation below, FIX M's centered
    final-readout direction for BOTH terms. This prevents a changing final
    normalizer from being mislabeled as a changed MLP decoder row.
    """
    config = checkpoint_e.config
    _require(checkpoint_m.config == config and config.n_layers >= 2 and config.n_layers % 2 == 0,
             'paired traces require the same even model configuration')
    _require(isinstance(token_ids, (list, tuple)) and 0 < len(token_ids) <= config.context_length
             and all(type(i) is int and 0 <= i < config.vocab_size for i in token_ids),
             'invalid exact prefix IDs')
    _require(type(top_k) is int and 0 < top_k <= config.d_ff, 'invalid feature ranking count')
    _require(type(target) is int and 0 <= target < config.vocab_size, 'invalid target')
    early = tuple(f'blocks.{b}.{branch}.' for b in range(config.n_layers//2) for branch in ('ln2', 'mlp'))
    for spec in checkpoint_e.manifest:
        if not spec.name.startswith(early):
            a, b = checkpoint_e[spec.name], checkpoint_m[spec.name]
            _require(a.shape == b.shape and a.dtype == b.dtype
                     and a.tobytes() == b.tobytes(), 'non-M parameter differs: '+spec.name)
    for stages in (stages_e, stages_m):
        validate_stages(stages, len(token_ids), config)
    le, lm = [np.asarray(x, dtype=np.float64) for x in (logits_e, logits_m)]
    _require(le.shape == lm.shape == (config.vocab_size,)
             and np.isfinite(le).all() and np.isfinite(lm).all(), 'wrong logical-vocabulary logits')
    competitor = min((i for i in range(config.vocab_size) if i != target), key=lambda i: (-le[i], i))
    predictions = {'E': prediction(le, target, competitor), 'M': prediction(lm, target, competitor)}
    effects = {key+'_delta': predictions['M'][key]-predictions['E'][key]
               for key in ('target_logit', 'competitor_logit', 'logsumexp', 'log_probability', 'margin')}
    maximum_delta = predictions['M']['logit_maximum']-predictions['E']['logit_maximum']
    shifted_delta = predictions['M']['shifted_logsumexp']-predictions['E']['shifted_logsumexp']
    centered_target_delta = ((float(lm[target])-predictions['M']['logit_maximum'])
                             -(float(le[target])-predictions['E']['logit_maximum']))
    # Absolute logsumexp values can round away a meaningful normalization
    # change. Retain the small shifted term when differencing the endpoints.
    effects.update(logsumexp_delta=math.fsum((maximum_delta, shifted_delta)),
        logit_maximum_delta=maximum_delta, shifted_logsumexp_delta=shifted_delta,
        centered_target_logit_delta=centered_target_delta,
        centered_probability_closure_remainder=math.fsum((effects['log_probability_delta'],
                                                         -centered_target_delta, shifted_delta)))
    effects['absolute_probability_closure_remainder'] = math.fsum((effects['log_probability_delta'],
        -effects['target_logit_delta'], effects['logsumexp_delta']))
    _require(all(math.isfinite(value) for value in effects.values()), 'nonfinite prediction difference')
    row = len(token_ids)-1
    ledgers = {'E': target_ledger(checkpoint_e, stages_e, row, target, competitor, le),
               'M': target_ledger(checkpoint_m, stages_m, row, target, competitor, lm)}
    direction = ledgers['M']['direction'].copy()
    early_neurons = {}
    for block in range(config.n_layers//2):
        p = f'blocks.{block}'
        ze, zm = (stages[p+'.gelu'][row] for stages in (stages_e, stages_m))
        de, dm = (bf16(cp[p+'.mlp.output.weight']).astype(np.float64)
                  for cp in (checkpoint_e, checkpoint_m))
        changes = neuron_delta(ze, zm, de, dm, direction)
        positive = sorted((i for i, value in enumerate(changes['delta']) if value > 0),
                          key=lambda i: (-changes['delta'][i], i))[:top_k]
        negative = sorted((i for i, value in enumerate(changes['delta']) if value < 0),
                          key=lambda i: (changes['delta'][i], i))[:top_k]
        selected = [dict(neuron=i, delta=float(changes['delta'][i]),
            E=neuron_operations(checkpoint_e, stages_e, row, block, i, direction),
            M=neuron_operations(checkpoint_m, stages_m, row, block, i, direction)) for i in positive+negative]
        projected_delta = float(np.dot(np.asarray(stages_m[p+'.mlp_projected'][row], dtype=np.float64)
                                       -stages_e[p+'.mlp_projected'][row], direction))
        bias_delta = float(np.dot(np.asarray(checkpoint_m[p+'.mlp.output.bias'], dtype=np.float64)
                                  -checkpoint_e[p+'.mlp.output.bias'], direction))
        rounding_delta = projected_delta-math.fsum(changes['delta'])-bias_delta
        early_neurons[p] = dict(**changes, projected_delta=projected_delta,
            bias_delta=bias_delta, rounding_delta=rounding_delta, selected=selected)
    return dict(target_id=target, competitor_id=competitor, row=row, token_ids=list(token_ids),
        predictions=predictions, effects=effects, ledgers=ledgers,
        fixed_direction=direction, early_neurons=early_neurons,
        selection_rule='Per early block: top positive and negative observed feature deltas; lowest ID breaks ties.',
        limitations=['Descriptive accounting, not neuron ablation or unique word storage.',
                     'Each full ledger fixes its own observed final normalizer; neuron deltas instead fix M direction.',
                     'Activation changes include upstream effects and are not isolated neuron interventions.',
                     'Caller must establish the intended training-only case selection from its provenance.',
                     'Feature selection is discovery; independent held-out and collateral interventions are still needed.'])
