"""Fixed, weight-only late-MLP polynomial and whole-string discrete search.

This is a declared algebraic surrogate, NOT the Taylor polynomial or forward
computation of GPT-2. Compilation evaluates LayerNorm geometry and GELU
derivatives only at anchors made from the vocabulary mean and learned position
weights. These are not measured late-layer activations. Attention, LayerNorm
Hessians, output biases, and the final context-dependent RMS are absent.

After compilation, scoring and optimization contract fixed arrays and token
IDs only. There are no candidate-dependent nonlinear gates, normalization,
softmax, GPU calls, corpus inputs, or model calls. All earlier positions can
interact in the quadratic term; search does not stitch a stationary token graph.

Row-vector convention: direct[p,t] = centered_embedding[t] A6[p],
features[p,t] = direct[p,t] K7, and readout[t] = (centered_embedding[t] *
final_gamma) H. The final readout is the exact coefficient of the centered
LayerNorm NUMERATOR, not a normalized logit or the complete LM-head output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Promote the actual C++ float literals; FP64 arithmetic below is nevertheless
# an analytical calculation, not a replay of BF16 activations or CUDA rounding.
RECIPE_EPSILON = float(np.float32(1e-5))
GELU_C = float(np.float32(0.7978845608))
GELU_A = float(np.float32(0.044715))
SCORE_ATOL = 1e-10
SCORE_RTOL = 5e-10


def _finite_array(value, name, ndim=None):
    result = np.asarray(value, dtype=np.float64)
    if ((ndim is not None and result.ndim != ndim) or not result.size
            or not np.isfinite(result).all()):
        raise ValueError(f'{name} must be a nonempty finite FP64 array')
    return result


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _score_tolerance(left, right):
    # This tolerance checks algebraic replay only. It NEVER groups score ties
    # or changes the search objective. Exact computed ties use logical IDs.
    return SCORE_ATOL + SCORE_RTOL * np.maximum(np.abs(left), np.abs(right))


def layer_norm_and_jacobian(anchor, gamma, beta, epsilon=RECIPE_EPSILON):
    """Return (LN(anchor), row-vector Jacobian, RMS scale) in FP64.

    With H=I-11^T/D, c=anchor H and s=sqrt(c.c/D+epsilon), a row perturbation
    v maps to v J where J=(H/s-c^T c/(D s^3)) diag(gamma). The rank-one term
    is essential; H diag(gamma)/s alone is not the LayerNorm derivative.
    This helper operates only on the explicitly supplied fixed anchor.
    """
    anchor = _finite_array(anchor, 'anchor', 1)
    gamma, beta = (_finite_array(value, name, 1)
                   for value, name in ((gamma, 'gamma'), (beta, 'beta')))
    if gamma.shape != anchor.shape or beta.shape != anchor.shape:
        raise ValueError('LayerNorm vectors must have equal shapes')
    if isinstance(epsilon, (bool, np.bool_)) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('LayerNorm epsilon must be finite and positive')
    width = len(anchor)
    centered = anchor - anchor.mean()
    scale = float(np.sqrt(np.dot(centered, centered) / width + epsilon))
    h = np.eye(width) - np.ones((width, width)) / width
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = centered / scale * gamma + beta
        jacobian = (h / scale - np.outer(centered, centered) /
                    (width * scale**3)) * gamma[None, :]
    if (not np.isfinite(scale) or not np.isfinite(normalized).all()
            or not np.isfinite(jacobian).all()):
        raise ValueError('nonfinite LayerNorm anchor geometry')
    return normalized, jacobian, scale


def gelu_derivatives(values):
    """Analytic first/second derivatives of the recipe's tanh-GELU.

    For u=c(x+a*x^3), T=tanh(u), Q=1-T^2:
      g = (1+T)/2 + x Q u'/2
      h = Q (u' + x u''/2 - x T (u')^2).
    Negative derivatives are retained. This is used only while compiling
    fixed coefficient arrays, never to evaluate a candidate string's gates.
    """
    x = _finite_array(values, 'GELU arguments')
    with np.errstate(over='ignore', invalid='ignore'):
        u = GELU_C * (x + GELU_A * x**3)
        first = GELU_C * (1 + 3 * GELU_A * x**2)
        second = 6 * GELU_C * GELU_A * x
        t = np.tanh(u)
        q = 1 - t * t
        g = 0.5 * (1 + t) + 0.5 * x * q * first
        h = q * (first + 0.5 * x * second - x * t * first**2)
    if not all(np.isfinite(a).all() for a in (u, first, second, g, h)):
        raise ValueError('GELU derivative calculation exceeded finite FP64 range')
    return g, h


@dataclass(frozen=True)
class OptimizationResult:
    token_ids: np.ndarray
    scores: np.ndarray
    initial_scores: np.ndarray
    trace: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CompiledPolynomial:
    """Read-only coefficient arrays; public sequence inputs are logical IDs.

    Columns in replacement_scores follow token_ids, which MUST be ascending.
    Arbitrary finite coefficients can be supplied directly for independent
    algebraic tests. Compilation is the separate operation that reads weights.
    Array buffers are made read-only without duplicating multi-GB dictionaries;
    callers must not retain and mutate separate writable aliases of them.
    """
    token_ids: np.ndarray
    direct: np.ndarray
    features: np.ndarray
    readout: np.ndarray
    neuron_readout: np.ndarray
    g7: np.ndarray
    h7: np.ndarray
    alpha: float = 0.125
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        ids = np.asarray(self.token_ids)
        if (ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids[1:] <= ids[:-1])):
            raise ValueError('token_ids must be distinct ascending nonnegative integers')
        ids = ids.astype(np.int64, copy=False)
        object.__setattr__(self, 'token_ids', ids)
        for name, ndim in (('direct', 3), ('features', 3), ('readout', 2),
                           ('neuron_readout', 2), ('g7', 1), ('h7', 1)):
            object.__setattr__(self, name, _finite_array(getattr(self, name), name, ndim))
        length, vocabulary, width = self.direct.shape
        if (length < 2 or vocabulary != len(ids)
                or self.features.shape[:2] != (length, vocabulary)
                or self.readout.shape != (vocabulary, width)
                or self.neuron_readout.shape != (vocabulary, self.features.shape[2])
                or self.g7.shape != (self.features.shape[2],)
                or self.h7.shape != self.g7.shape):
            raise ValueError('inconsistent compiled polynomial coefficient shapes')
        if (isinstance(self.alpha, (bool, np.bool_)) or not np.isfinite(self.alpha)
                or self.alpha <= 0):
            raise ValueError('alpha must be finite and positive')
        if not isinstance(self.metadata, dict):
            raise ValueError('metadata must be a dictionary')
        object.__setattr__(self, 'alpha', float(self.alpha))
        object.__setattr__(self, 'metadata', dict(self.metadata))
        for name in ('token_ids', 'direct', 'features', 'readout',
                     'neuron_readout', 'g7', 'h7'):
            getattr(self, name).setflags(write=False)

    @property
    def length(self):
        return self.direct.shape[0]

    @property
    def vocabulary_size(self):
        return len(self.token_ids)

    @property
    def diagnostics(self):
        return self.metadata.get('diagnostics', {})

    def _indices(self, rows):
        rows = np.asarray(rows)
        if (rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != self.length
                or not np.issubdtype(rows.dtype, np.integer)):
            raise ValueError('sequences must be a nonempty integer [R,L] array')
        indices = np.searchsorted(self.token_ids, rows)
        if (np.any(indices >= self.vocabulary_size)
                or not np.array_equal(self.token_ids[indices], rows)):
            raise ValueError('sequence contains a token outside the compiled vocabulary')
        return indices

    def components(self, rows):
        """Full direct, affine-GELU and quadratic score decomposition.

        The synthetic prefix arrays x and z are returned for algebra auditing;
        they are sums of compiled dictionary rows, not GPT-2 hidden states.
        Position zero has no target term and its x/z entries are exactly zero.
        """
        indices = self._indices(rows)
        selected_direct = self.direct[np.arange(self.length)[None, :], indices]
        selected_features = self.features[np.arange(self.length)[None, :], indices]
        x = np.zeros_like(selected_direct)
        z = np.zeros_like(selected_features)
        scales = self.alpha / np.arange(1, self.length, dtype=np.float64)
        x[:, 1:] = np.cumsum(selected_direct[:, :-1], axis=1) * scales[None, :, None]
        z[:, 1:] = np.cumsum(selected_features[:, :-1], axis=1) * scales[None, :, None]
        destinations = self.readout[indices]
        neuron_destinations = self.neuron_readout[indices]
        with np.errstate(over='ignore', invalid='ignore'):
            direct = np.sum(x * destinations, axis=2)
            affine = np.sum((z * self.g7) * neuron_destinations, axis=2)
            quadratic = np.sum((0.5 * self.h7 * z**2) * neuron_destinations, axis=2)
            total = np.sum(direct + affine + quadratic, axis=1)
        if not all(np.isfinite(a).all() for a in (x, z, direct, affine, quadratic, total)):
            raise ValueError('nonfinite polynomial score or component')
        return {'total': total, 'direct': direct.sum(axis=1),
                'affine': affine.sum(axis=1), 'quadratic': quadratic.sum(axis=1),
                'linear': (direct + affine).sum(axis=1),
                'per_position_direct': direct, 'per_position_affine': affine,
                'per_position_quadratic': quadratic, 'x': x, 'z': z}

    def score_components(self, rows):
        parts = self.components(rows)
        return {name: parts[name] for name in ('linear', 'quadratic', 'total')}

    def score(self, rows):
        return self.components(rows)['total']

    def replacement_scores(self, rows, position, chunk_size=1024):
        """Exact algebraic scores for replacing one slot with EVERY token.

        A replacement changes (1) that slot's target readout, if position>0,
        and (2) all later direct/feature prefix sums. For each later r, put
        k=alpha/r and delta_f=f_candidate-f_current. Its quadratic change is
        h*z_r*k*delta_f + h*k^2*delta_f^2/2, times the fixed destination value.
        Summing these coefficients first makes each vocabulary chunk a few
        matrix products; no [R,V,M] or [V,V,M] tensor is constructed.
        """
        if type(position) is not int or not 0 <= position < self.length:
            raise ValueError('replacement position outside candidate length')
        _positive_integer(chunk_size, 'chunk_size')
        indices = self._indices(rows)
        parts = self.components(rows)
        base = parts['total']
        row_count = len(indices)
        current = indices[:, position]
        current_direct = self.direct[position, current]
        current_features = self.features[position, current]
        width, neurons = current_direct.shape[1], current_features.shape[1]
        direct_coefficient = np.zeros((row_count, width))
        feature_coefficient = np.zeros((row_count, neurons))
        square_coefficient = np.zeros((row_count, neurons))
        for later in range(position + 1, self.length):
            k = self.alpha / later
            destination = self.neuron_readout[indices[:, later]]
            direct_coefficient += k * self.readout[indices[:, later]]
            feature_coefficient += k * (self.g7 + self.h7 * parts['z'][:, later]) * destination
            square_coefficient += (0.5 * k * k * self.h7) * destination
        with np.errstate(over='ignore', invalid='ignore'):
            constant = (base - np.sum(current_direct * direct_coefficient, axis=1)
                        - np.sum(current_features * feature_coefficient, axis=1)
                        + np.sum(current_features**2 * square_coefficient, axis=1))
            linear_feature = feature_coefficient - 2 * current_features * square_coefficient
        if position:
            constant -= (parts['per_position_direct'][:, position]
                         + parts['per_position_affine'][:, position]
                         + parts['per_position_quadratic'][:, position])
            target_features = (self.g7 * parts['z'][:, position]
                               + 0.5 * self.h7 * parts['z'][:, position]**2)
        scores = np.empty((row_count, self.vocabulary_size))
        for start in range(0, self.vocabulary_size, chunk_size):
            stop = min(start + chunk_size, self.vocabulary_size)
            direct = self.direct[position, start:stop]
            features = self.features[position, start:stop]
            with np.errstate(over='ignore', invalid='ignore'):
                values = (constant[:, None] + direct_coefficient @ direct.T
                          + linear_feature @ features.T
                          + square_coefficient @ (features**2).T)
                if position:
                    values += (parts['x'][:, position] @ self.readout[start:stop].T
                               + target_features @ self.neuron_readout[start:stop].T)
            if not np.isfinite(values).all():
                raise ValueError('nonfinite replacement score')
            scores[:, start:stop] = values
        replay = scores[np.arange(row_count), current]
        if np.any(np.abs(replay - base) > _score_tolerance(replay, base)):
            raise ArithmeticError('current-token replacement disagrees with full polynomial')
        # The unchanged candidate is always an available option with its exact
        # fully recomputed score. This is not a positivity or magnitude filter.
        scores[np.arange(row_count), current] = base
        return scores

    def optimize(self, initial_rows, sweeps=8, chunk_size=1024):
        """Fixed-budget coordinate ascent, alternating forward/reverse sweeps.

        Each row is an independent start. Every slot considers the entire
        selected vocabulary, including its existing token and repeats. Ties
        in computed replacement scores choose the lowest logical token ID.
        Full-score recomputation checks each proposed change. A decrease or
        replay disagreement exceeding the fixed FP64 tolerance fails closed;
        a smaller negative rounding discrepancy retains the unchanged row.
        No convergence-based early stopping or corpus-dependent selection occurs.
        """
        _positive_integer(sweeps, 'sweeps')
        _positive_integer(chunk_size, 'chunk_size')
        indices = self._indices(initial_rows)
        rows = self.token_ids[indices].copy()
        initial_scores = self.score(rows)
        scores = initial_scores.copy()
        trace = []
        for sweep in range(sweeps):
            forward = sweep % 2 == 0
            positions = range(self.length) if forward else range(self.length - 1, -1, -1)
            for position in positions:
                alternatives = self.replacement_scores(rows, position, chunk_size)
                selected = np.argmax(alternatives, axis=1)
                expected = alternatives[np.arange(len(rows)), selected]
                proposed = rows.copy()
                proposed[:, position] = self.token_ids[selected]
                recomputed = self.score(proposed)
                discrepancy = np.abs(expected - recomputed)
                if np.any(discrepancy > _score_tolerance(expected, recomputed)):
                    raise ArithmeticError('chosen replacement disagrees with full polynomial')
                if np.any(recomputed < scores - _score_tolerance(recomputed, scores)):
                    raise ArithmeticError('coordinate update decreased the polynomial score')
                retained = recomputed < scores
                proposed[retained] = rows[retained]
                recomputed[retained] = scores[retained]
                changed = np.flatnonzero(np.any(proposed != rows, axis=1))
                trace.append({'sweep': sweep, 'position': position,
                              'direction': 'forward' if forward else 'reverse',
                              'scores_before': scores.tolist(), 'scores_after': recomputed.tolist(),
                              'selected_token_ids': proposed[:, position].tolist(),
                              'changed_rows': changed.tolist(),
                              'roundoff_retained_rows': np.flatnonzero(retained).tolist(),
                              'max_score_discrepancy': float(discrepancy.max())})
                rows, scores = proposed, recomputed
        for array in (rows, scores, initial_scores):
            array.setflags(write=False)
        return OptimizationResult(rows, scores, initial_scores, tuple(trace))


def compile_polynomial(checkpoint, token_ids, length, mode='full', blocks=None,
                       alpha=0.125):
    """Compile the declared polynomial without reading any text or activations.

    Modes: full; affine (curvature h7=0); no_cross (features=E_c K7 at every
    position, but direct paths remain unchanged); broken (roll first-MLP W2
    rows by +1 relative to its input keys). The last two model blocks are the
    defaults. Any block selection or amplitude is explicit in returned metadata.
    """
    _positive_integer(length, 'length')
    if length < 2 or length > checkpoint.config.context_length:
        raise ValueError('length must be between two and checkpoint context length')
    if mode not in ('full', 'affine', 'no_cross', 'broken'):
        raise ValueError('unknown polynomial mode')
    if (isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError('alpha must be finite and positive')
    ids = np.asarray(token_ids)
    if (ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= checkpoint.config.vocab_size)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('expected a distinct logical vocabulary subset')
    ids = np.sort(ids.astype(np.int64))
    if blocks is None:
        blocks = (checkpoint.config.n_layers - 2, checkpoint.config.n_layers - 1)
    if (not isinstance(blocks, (tuple, list)) or len(blocks) != 2
            or any(type(b) is not int for b in blocks)
            or not 0 <= blocks[0] < blocks[1] < checkpoint.config.n_layers):
        raise ValueError('expected two ascending valid block indices')
    first, second = blocks
    width, neurons = checkpoint.config.d_model, checkpoint.config.d_ff

    def weight(name, shape):
        value = _finite_array(checkpoint[name], name)
        if value.shape != shape:
            raise ValueError(f'wrong tensor shape: {name}')
        return value

    embedding = checkpoint.token_embedding
    if embedding.shape != (checkpoint.config.vocab_size, width):
        raise ValueError('embedding must contain logical vocabulary rows only')
    mean_embedding = np.mean(embedding, axis=0, dtype=np.float64)
    centered = _finite_array(np.asarray(embedding[ids], dtype=np.float64) - mean_embedding,
                             'centered embeddings', 2)
    positions = weight('position_embedding.weight', (checkpoint.config.context_length, width))[:length]
    anchors = mean_embedding + positions
    final_anchor = mean_embedding + positions.mean(axis=0)
    gamma6 = weight(f'blocks.{first}.ln2.scale', (width,))
    beta6 = weight(f'blocks.{first}.ln2.bias', (width,))
    w16 = weight(f'blocks.{first}.mlp.input.weight', (width, neurons))
    b16 = weight(f'blocks.{first}.mlp.input.bias', (neurons,))
    w26 = weight(f'blocks.{first}.mlp.output.weight', (neurons, width))
    if mode == 'broken':
        if neurons < 2:
            raise ValueError('broken pairing requires at least two first-MLP neurons')
        w26 = np.roll(w26, 1, axis=0)
    gamma7 = weight(f'blocks.{second}.ln2.scale', (width,))
    beta7 = weight(f'blocks.{second}.ln2.bias', (width,))
    w17 = weight(f'blocks.{second}.mlp.input.weight', (width, neurons))
    b17 = weight(f'blocks.{second}.mlp.input.bias', (neurons,))
    w27 = weight(f'blocks.{second}.mlp.output.weight', (neurons, width))
    gamma_final = weight('final_norm.scale', (width,))
    norm7, jacobian7, scale7 = layer_norm_and_jacobian(final_anchor, gamma7, beta7)
    arguments7 = norm7 @ w17 + b17
    g7, h7 = gelu_derivatives(arguments7)
    if mode == 'affine':
        h7 = np.zeros_like(h7)
    k7 = jacobian7 @ w17
    direct = np.empty((length, len(ids), width), dtype=np.float64)
    features = np.empty((length, len(ids), neurons), dtype=np.float64)
    scales6, argument_ranges6, gate_ranges6, operator_norms = [], [], [], []
    no_cross_features = centered @ k7 if mode == 'no_cross' else None
    for position in range(length):
        normalized6, jacobian6, scale6 = layer_norm_and_jacobian(anchors[position], gamma6, beta6)
        arguments6 = normalized6 @ w16 + b16
        g6, _ = gelu_derivatives(arguments6)
        a6 = np.eye(width) + ((jacobian6 @ w16) * g6[None, :]) @ w26
        direct[position] = centered @ a6
        features[position] = no_cross_features if mode == 'no_cross' else direct[position] @ k7
        scales6.append(scale6)
        argument_ranges6.append([float(arguments6.min()), float(arguments6.max())])
        gate_ranges6.append([float(g6.min()), float(g6.max())])
        operator_norms.append(float(np.linalg.norm(a6)))
    # H diag(gamma) E_column becomes E_row diag(gamma) H after transposing.
    # Center AFTER multiplying gamma; reversing these operations is incorrect.
    readout = centered * gamma_final
    readout -= readout.mean(axis=1, keepdims=True)
    neuron_readout = readout @ w27.T
    metadata = {
        'mode': mode, 'blocks': list(blocks), 'length': length, 'alpha': float(alpha),
        'epsilon': RECIPE_EPSILON, 'gelu_c': GELU_C, 'gelu_a': GELU_A,
        'mean_embedding': mean_embedding.tolist(),
        'anchors6': anchors.tolist(), 'anchor7': final_anchor.tolist(),
        'anchor_rule': 'mean of all logical embedding rows plus P[p]; second anchor averages P[0:L]',
        'final_readout': '(centered_embedding * final_gamma) H; no final_beta or context RMS',
        'broken_value_row_for_key': ('(key-1)%d_ff' if mode == 'broken' else 'key'),
        'feature_rule': ('E_c K7 independent of position; direct unchanged' if mode == 'no_cross'
                         else 'E_c A6[p] K7'),
        'score_tolerance': {'atol': SCORE_ATOL, 'rtol': SCORE_RTOL,
                            'scope': 'replay/monotonic integrity only; not tie grouping'},
        'diagnostics': {
            'anchor6_rms_scales': scales6, 'anchor7_rms_scale': scale7,
            'anchor6_gelu_argument_ranges': argument_ranges6,
            'anchor6_gelu_derivative_ranges': gate_ranges6,
            'anchor7_gelu_argument_range': [float(arguments7.min()), float(arguments7.max())],
            'g7_range': [float(g7.min()), float(g7.max())],
            'h7_range': [float(h7.min()), float(h7.max())],
            'A6_frobenius_norms': operator_norms,
            'K7_frobenius_norm': float(np.linalg.norm(k7)),
        },
        'limitations': [
            'This fixed polynomial is not GPT-2 inference or its full Taylor expansion.',
            'Anchors are weight-derived embedding coordinates, not true late-layer residual states.',
            'Uniform prefix pooling omits attention; LayerNorm Hessians and MLP output biases are omitted.',
            'Final LayerNorm context RMS and beta offset are absent; scores are not probabilities.',
            'Position-dependent coefficients only permit order sensitivity; they do not guarantee it.',
        ],
    }
    return CompiledPolynomial(ids, direct, features, readout, neuron_readout,
                              g7, h7, alpha, metadata)


def compile_probe(checkpoint, vocabulary_ids, length, mode='full', amplitude=0.125,
                  blocks=None):
    """Keyword-compatible wrapper for the research CLI."""
    return compile_polynomial(checkpoint, vocabulary_ids, length, mode, blocks, amplitude)


compile = compile_polynomial
