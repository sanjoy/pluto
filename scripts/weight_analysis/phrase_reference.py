"""Readable forward execution of Pluto GPT-2 on a short, causal token prefix.

This is an actual forward calculation, not the static weight contractions used
by the other analysis tools. It neither tokenizes text nor chooses a checkpoint.
Callers supply an already validated GPT2Checkpoint and literal token IDs. No
files are read at import time, and no checkpoint weights are changed.

``fp32`` is a mathematical diagnostic using the FP32 master parameters.
``bf16`` additionally follows the native activation/matrix rounding boundaries:
FP32 biases and LayerNorm parameters, BF16 MMA operands and stored activations,
and FP32 attention statistics and output logits. NumPy reductions/BLAS are NOT
the cuTile instructions: even this emulation is not a bitwise CUDA oracle. A
real native forward must establish parity before interpreting its predictions.
"""

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest


EPSILON = np.float32(1e-5)
GELU_C = np.float32(0.7978845608)
GELU_A = np.float32(0.044715)


def _mode(mode):
    if mode not in ('fp32', 'bf16'):
        raise ValueError("mode must be 'fp32' or 'bf16'")


def _float32(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must contain real numbers')
    with np.errstate(over='ignore', invalid='ignore'):
        array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite FP32 values')
    return array


def bf16(array):
    """Round finite values to BF16, returning their exact FP32 representation.

    BF16 retains the upper 16 bits of an IEEE FP32 value. Adding 0x7fff plus
    the retained low bit implements round-to-nearest, ties-to-even, including
    negative numbers and subnormals. Signed zeros survive. Nonfinite inputs or
    rounding overflow are errors rather than silent invalid diagnostics.
    """
    values = _float32(array, 'BF16 input')
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) +
                ((bits >> np.uint32(16)) & np.uint32(1))) &
               np.uint32(0xffff0000)).view(np.float32)
    if not np.isfinite(rounded).all():
        raise ValueError('BF16 rounding overflow')
    return rounded


def _store(array, mode):
    values = _float32(array, 'computed activation')
    return bf16(values) if mode == 'bf16' else values


def layer_norm(x, gamma, beta, mode='bf16'):
    """Row-wise population LayerNorm, epsilon 1e-5, then activation rounding.

    Input is [positions, width]. Gamma/beta remain FP32 in both modes, and we
    round only the final affine result, not the normalized values or moments.
    The input is already a stored activation: this helper does not silently
    pre-round it. Native GPU summation order and rsqrt may differ by FP32 ulps.
    """
    _mode(mode)
    x = _float32(x, 'LayerNorm input')
    gamma = _float32(gamma, 'LayerNorm gamma')
    beta = _float32(beta, 'LayerNorm beta')
    if (x.ndim != 2 or not all(x.shape) or
            gamma.shape != (x.shape[1],) or beta.shape != gamma.shape):
        raise ValueError('LayerNorm requires [rows,width] and two [width] arrays')
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    centered = x - mean
    variance = np.mean(centered * centered, axis=1, keepdims=True,
                       dtype=np.float32)
    inverse_stddev = np.float32(1) / np.sqrt(variance + EPSILON)
    return _store(centered * inverse_stddev * gamma + beta, mode)


def gelu(x, mode='bf16'):
    """The recipe's tanh GELU, not the erf-based exact GELU."""
    _mode(mode)
    x = _float32(x, 'GELU input')
    inner = GELU_C * (x + GELU_A * x * x * x)
    return _store(np.float32(.5) * x * (np.float32(1) + np.tanh(inner)),
                  mode)


def _dense_unrounded(x, matrix, bias, mode):
    # Native dense kernels cast matrix operands but NOT the FP32 bias. cuTile
    # starts its accumulator with bias; NumPy adds bias after its dot product.
    # That and different FMA/reduction ordering can change a final rounding bit.
    left = bf16(x) if mode == 'bf16' else x
    right = bf16(matrix) if mode == 'bf16' else matrix
    return _float32(left @ right + bias, 'dense result')


def _attention(qkv, config, mode):
    """Causal attention; probabilities are descriptive, never BF16-rounded.

    All Q/K/V values are already rounded by the packed projection. Each head
    maintains an online FP32 maximum, denominator, and weighted value sum, as
    the native kernel does. For inspection we also return the corresponding
    globally normalized probabilities; their separately calculated sum times V
    can differ from the online context by FP32 rounding, not by a new model.
    """
    rows = qkv.shape[0]
    q, k, v = [part.reshape(rows, config.n_heads, config.head_dim)
               for part in np.split(qkv, 3, axis=1)]
    probabilities = np.zeros((config.n_heads, rows, rows), dtype=np.float32)
    context = np.empty_like(q)
    scale = np.float32(1) / np.sqrt(np.float32(config.head_dim))
    for head in range(config.n_heads):
        for query in range(rows):
            # Explicitly stop at the query: future tokens cannot enter even a
            # temporary softmax denominator. No dropout or attention dropout.
            scores = (k[:query + 1, head] @ q[query, head]) * scale
            scores = _float32(scores, 'attention scores')
            exponentials = np.exp(scores - np.max(scores))
            probabilities[head, query, :query + 1] = (
                exponentials / np.sum(exponentials, dtype=np.float32))
            maximum = np.float32(-np.inf)
            denominator = np.float32(0)
            accumulator = np.zeros(config.head_dim, dtype=np.float32)
            for key, score in enumerate(scores):
                new_maximum = np.maximum(maximum, score)
                old_scale = np.exp(maximum - new_maximum)
                new_scale = np.exp(score - new_maximum)
                accumulator = accumulator * old_scale + v[key, head] * new_scale
                denominator = denominator * old_scale + new_scale
                maximum = new_maximum
            context[query, head] = accumulator / denominator
    context = _store(context, mode)
    return probabilities, context


def forward(checkpoint: GPT2Checkpoint, tokens, mode='bf16'):
    """Run a prefix and retain named states useful for a forward-pass trace.

    Tokens must be a nonempty 1-D integer sequence of logical IDs, with at most
    context_length entries. Position zero means absolute position zero; no BOS,
    EOS, padding, generation, or tokenizer special tokens are inserted. Row p
    of logits predicts the token AFTER tokens[p]. Future padding is unnecessary
    on CPU because attention is strictly causal. Small GPT2Config fixtures are
    supported even when their shapes could not be executed by cuTile.

    Return keys:
      stages: [rows,width] stored states; qkv has 3*width, fc1/gelu have d_ff.
      attention: one record per block, with probabilities [head,query,key],
        BF16/FP32 stored context [row,head,channel], and head_contributions
        [head,row,width]. Contributions exclude the shared FP32 output bias
        and the output projection's final activation rounding. They are
        algebraic terms, NOT measured causal effects. Their sum plus bias is
        close (not necessarily bitwise equal) to projection_before_rounding.
      logits: FP32 [rows,logical_vocab], with no padded vocabulary columns.

    Intermediate states are observations, not early predictions by themselves.
    Applying final LayerNorm and unembedding to them is a logit-lens heuristic.
    This function does not silently treat that heuristic as a causal mechanism.
    """
    _mode(mode)
    config = checkpoint.config
    if not isinstance(config, GPT2Config):
        raise ValueError('checkpoint.config must be GPT2Config')
    ids = np.asarray(tokens)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or
            not 1 <= ids.size <= config.context_length):
        raise ValueError('tokens must be 1..context_length integer token IDs')
    if np.any(ids < 0) or np.any(ids >= config.vocab_size):
        raise ValueError('token ID is outside the logical vocabulary')
    ids = np.array(ids, dtype=np.int64, copy=True)

    # Check exact named shapes and finite values even if the checkpoint loader
    # was created with check_finite=False. All access is read-only; the arrays
    # remain borrowed from the caller's immutable checkpoint mappings.
    weights = {}
    for spec in tensor_manifest(config):
        array = _float32(checkpoint[spec.name], spec.name)
        if array.shape != spec.shape:
            raise ValueError(f'{spec.name}: incorrect shape')
        weights[spec.name] = array

    embedding = weights['token_embedding.weight']
    stages = {}
    stages['embedding'] = _store(embedding[ids], mode)
    # Learned positions are FP32 parameters. Native code adds them to the
    # already-BF16 token embedding, then rounds the SUM, not positions alone.
    x = _store(stages['embedding'] +
               weights['position_embedding.weight'][:len(ids)], mode)
    stages['positioned'] = x
    attention_records = []
    for block in range(config.n_layers):
        prefix = f'blocks.{block}'
        norm = layer_norm(x, weights[f'{prefix}.ln1.scale'],
                          weights[f'{prefix}.ln1.bias'], mode)
        stages[f'{prefix}.ln1'] = norm
        qkv = _store(_dense_unrounded(
            norm, weights[f'{prefix}.attn.qkv.weight'],
            weights[f'{prefix}.attn.qkv.bias'], mode), mode)
        stages[f'{prefix}.qkv'] = qkv
        probabilities, context = _attention(qkv, config, mode)
        flat_context = context.reshape(len(ids), config.d_model)
        stages[f'{prefix}.attention'] = flat_context
        output_matrix = weights[f'{prefix}.attn.output.weight']
        output_bias = weights[f'{prefix}.attn.output.bias']
        projection = _dense_unrounded(flat_context, output_matrix,
                                      output_bias, mode)
        branch = _store(projection, mode)
        stages[f'{prefix}.attention_projected'] = branch
        effective_matrix = bf16(output_matrix) if mode == 'bf16' else output_matrix
        contributions = np.stack([
            context[:, head] @ effective_matrix[
                head * config.head_dim:(head + 1) * config.head_dim]
            for head in range(config.n_heads)])
        attention_records.append(dict(
            probabilities=probabilities, context=context,
            head_contributions=contributions, output_bias=output_bias.copy(),
            projection_before_rounding=projection))
        x = _store(x + branch, mode)
        stages[f'{prefix}.after_attention'] = x
        norm = layer_norm(x, weights[f'{prefix}.ln2.scale'],
                          weights[f'{prefix}.ln2.bias'], mode)
        stages[f'{prefix}.ln2'] = norm
        fc1 = _store(_dense_unrounded(
            norm, weights[f'{prefix}.mlp.input.weight'],
            weights[f'{prefix}.mlp.input.bias'], mode), mode)
        stages[f'{prefix}.fc1'] = fc1
        features = gelu(fc1, mode)
        stages[f'{prefix}.gelu'] = features
        branch = _store(_dense_unrounded(
            features, weights[f'{prefix}.mlp.output.weight'],
            weights[f'{prefix}.mlp.output.bias'], mode), mode)
        stages[f'{prefix}.mlp_projected'] = branch
        x = _store(x + branch, mode)
        stages[f'{prefix}.after_mlp'] = x

    final = layer_norm(x, weights['final_norm.scale'],
                       weights['final_norm.bias'], mode)
    stages['final_norm'] = final
    # Tied head: the logical rows of exactly the same master embedding. Unlike
    # other layer outputs, logits stay FP32 after the BF16-operand dot product.
    output_embedding = embedding[:config.vocab_size]
    if mode == 'bf16':
        output_embedding = bf16(output_embedding)
    logits = _float32(final @ output_embedding.T, 'output logits')
    return dict(mode=mode, tokens=ids, stages=stages,
                attention=attention_records, logits=logits)
