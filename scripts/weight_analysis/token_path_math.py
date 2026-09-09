"""Analytical accounting for a sampled token at one native context position.

Unlike a winner-only attribution, the target can be a low-ranked sampled token.
The competitor is explicit. This distinguishes the model's learned logit support
from the random-number/cumulative-probability operation that actually selects it.
All terms below are readout accounting, never asserted to be deletion effects.
"""

import math

import numpy as np

from .phrase_reference import bf16
from .phrase_trace_analysis import checked_matrix, validate_stages


def query_attention(qkv, heads, row):
    """Reconstruct one causal query from captured QKV, retaining source values.

    These are FP64 mathematical probabilities, not native FlashAttention's
    unstored online-softmax intermediates. Compare the reconstructed context
    and source totals to the captured BF16 context before interpreting them.
    """
    qkv = checked_matrix(qkv)
    if (type(heads) is not int or heads <= 0 or qkv.shape[1] % (3 * heads)
            or type(row) is not int or not 0 <= row < len(qkv)):
        raise ValueError('invalid query row or head geometry')
    width = qkv.shape[1] // 3
    dimension = width // heads
    q, k, v = [part.reshape(len(qkv), heads, dimension).transpose(1, 0, 2)
               for part in np.split(qkv, 3, axis=1)]
    scores = np.einsum('hd,hnd->hn', q[:, row], k[:, :row + 1]) / math.sqrt(dimension)
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    values = v[:, :row + 1]
    context = np.einsum('hn,hnd->hd', probabilities, values).reshape(width)
    return probabilities, values, context


def target_ledger(checkpoint, stages, row, target, competitor, logits):
    """Exactly account for a specified target-minus-competitor native margin.

    Freeze the actual final LayerNorm's scale and center the readout direction.
    This linearizes only the final readout, not the intervening transformer.
    Residual additions then telescope. Biases and every numerical remainder
    stay explicit; positive and negative feature terms must both be retained.
    Position embeddings use the selected *context row*, including window reset.
    """
    config, tensors = checkpoint.config, checkpoint.tensors
    if type(row) is not int or not 0 <= row < len(stages['positioned']):
        raise ValueError('invalid selected context row')
    for token in (target, competitor):
        if type(token) is not int or not 0 <= token < config.vocab_size:
            raise ValueError('invalid target or competitor')
    if target == competitor:
        raise ValueError('target and competitor must differ')
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (config.vocab_size,) or not np.isfinite(logits).all():
        raise ValueError('need one complete logical-vocabulary logit row')
    validate_stages(stages, len(stages['positioned']), config)

    def value(name):
        return np.asarray(stages[name][row], dtype=np.float64)

    def weight(name):
        return bf16(tensors[name]).astype(np.float64)

    embedding = bf16(tensors['token_embedding.weight'][[target, competitor]]).astype(np.float64)
    readout = embedding[0] - embedding[1]
    residual = value(f'blocks.{config.n_layers - 1}.after_mlp')
    gamma = np.asarray(tensors['final_norm.scale'], dtype=np.float64)
    beta = np.asarray(tensors['final_norm.bias'], dtype=np.float64)
    centered = residual - residual.mean()
    deviation = math.sqrt(float(np.mean(centered * centered)) + 1e-5)
    direction = readout * gamma / deviation
    direction -= direction.mean()
    ideal_norm = centered / deviation * gamma + beta
    terms, head_terms, neuron_terms, attention = {}, {}, {}, {}

    def add(name, vector):
        terms[name] = float(np.dot(vector, direction))

    base = value('embedding')
    position = np.asarray(tensors['position_embedding.weight'][row], dtype=np.float64)
    add('token_embedding', base)
    add('position_embedding', position)
    add('position_add_rounding', value('positioned') - base - position)
    residual = value('positioned')
    for block in range(config.n_layers):
        p = f'blocks.{block}'
        context = value(p + '.attention')
        wo = weight(p + '.attn.output.weight')
        bo = np.asarray(tensors[p + '.attn.output.bias'], dtype=np.float64)
        probabilities, source_values, reconstructed = query_attention(stages[p + '.qkv'], config.n_heads, row)
        heads = np.empty(config.n_heads, dtype=np.float64)
        sources = np.empty((config.n_heads, row + 1), dtype=np.float64)
        for head in range(config.n_heads):
            start, end = head * config.head_dim, (head + 1) * config.head_dim
            heads[head] = np.dot(context[start:end] @ wo[start:end], direction)
            sources[head] = probabilities[head] * (source_values[head] @ wo[start:end] @ direction)
        head_terms[p] = heads
        terms[p + '.attention_heads'] = float(heads.sum())
        add(p + '.attention_bias', bo)
        projected = value(p + '.attention_projected')
        add(p + '.attention_projection_rounding', projected - context @ wo - bo)
        after = value(p + '.after_attention')
        add(p + '.attention_residual_rounding', after - residual - projected)
        attention[p] = dict(probabilities=probabilities, source_margin_terms=sources,
                            reconstructed_context=reconstructed,
                            context_max_abs_error=float(np.max(np.abs(reconstructed - context))),
                            source_sum_vs_native_head_max_abs=float(np.max(np.abs(sources.sum(axis=1) - heads))))
        hidden = value(p + '.gelu')
        w2 = weight(p + '.mlp.output.weight')
        b2 = np.asarray(tensors[p + '.mlp.output.bias'], dtype=np.float64)
        neurons = hidden * (w2 @ direction)
        neuron_terms[p] = neurons
        terms[p + '.mlp_neurons'] = float(neurons.sum())
        add(p + '.mlp_bias', b2)
        projected = value(p + '.mlp_projected')
        add(p + '.mlp_projection_rounding', projected - hidden @ w2 - b2)
        residual = value(p + '.after_mlp')
        add(p + '.mlp_residual_rounding', residual - after - projected)
    terms['final_norm_bias'] = float(np.dot(beta, readout))
    native_norm = value('final_norm')
    terms['final_norm_rounding_and_reduction'] = float(np.dot(native_norm - ideal_norm, readout))
    margin = float(logits[target] - logits[competitor])
    terms['head_fp32_accumulation_remainder'] = margin - float(np.dot(native_norm, readout))
    closure = math.fsum(terms.values()) - margin
    if not math.isclose(closure, 0, abs_tol=2e-10):
        raise ValueError('target margin accounting failed to close')
    return dict(target=target, competitor=competitor, row=row, native_margin=margin,
                terms=terms, heads=head_terms, neurons=neuron_terms, attention=attention,
                direction=direction, final_std=deviation, closure_error=closure)


def neuron_operations(checkpoint, stages, row, block, neuron, direction):
    """Expose the addressed key-dot-input, bias, GELU, and decoder-row product.

    The FP64 key dot is an analytical reconstruction, not native accumulation
    order; actual captured pre/post-GELU values and discrepancies are retained.
    A neuron can support many tokens, so these numbers are not a semantic name.
    """
    config, tensors = checkpoint.config, checkpoint.tensors
    if (type(block) is not int or not 0 <= block < config.n_layers
            or type(neuron) is not int or not 0 <= neuron < config.d_ff
            or type(row) is not int or not 0 <= row < len(stages['positioned'])):
        raise ValueError('invalid neuron coordinate')
    validate_stages(stages, len(stages['positioned']), config)
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (config.d_model,) or not np.isfinite(direction).all():
        raise ValueError('invalid readout direction')
    p = f'blocks.{block}'
    normalized = np.asarray(stages[p + '.ln2'][row], dtype=np.float64)
    key = bf16(tensors[p + '.mlp.input.weight'][:, neuron]).astype(np.float64)
    bias = float(tensors[p + '.mlp.input.bias'][neuron])
    contributions = normalized * key
    analytical_pre = math.fsum(contributions) + bias
    native_pre = float(stages[p + '.fc1'][row, neuron])
    native_post = float(stages[p + '.gelu'][row, neuron])
    # Match the values of the C++ float constants; evaluate their expression
    # in FP64 here without pretending to reproduce native FP32 operation order.
    scale, cubic = float(np.float32(.7978845608)), float(np.float32(.044715))
    gelu = .5 * native_pre * (1 + math.tanh(scale * (native_pre + cubic * native_pre ** 3)))
    decoder = bf16(tensors[p + '.mlp.output.weight'][neuron]).astype(np.float64)
    aligned = float(np.dot(decoder, direction))
    return dict(block=block, neuron=neuron, row=row,
                input_weight_file=f'weight_{10 + 12 * block}.bin',
                input_column_first_byte=4 * neuron, input_column_byte_stride=4 * config.d_ff,
                input_bias_file=f'weight_{11 + 12 * block}.bin', input_bias_byte_offset=4 * neuron,
                output_weight_file=f'weight_{12 + 12 * block}.bin',
                output_row_byte_offset=4 * neuron * config.d_model,
                input_product_terms=contributions, input_bias=bias,
                analytical_pre_gelu=analytical_pre, native_pre_gelu=native_pre,
                native_pre_minus_analytical=native_pre - analytical_pre,
                analytical_gelu_of_native_pre=gelu, native_post_gelu=native_post,
                native_post_minus_analytical=native_post - gelu,
                output_readout_dot=aligned, margin_contribution=native_post * aligned)
