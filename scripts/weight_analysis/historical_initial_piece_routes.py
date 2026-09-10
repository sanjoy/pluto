"""Read-only routing audit of three heads before the historical ` Ex` token.

The heads were selected AFTER inspecting their saved native ablation effects.
This is a hypothesis-generating description of those same measurements, not a
new causal test. In particular, values at space-token positions in blocks 2/3
are contextualized: they may already contain earlier lexical information.

Attention probabilities and source contributions are independently reconstructed
in FP64 from captured BF16 QKV. They are not native FlashAttention's unstored
softmax statistics. The native head context and all reconstruction discrepancies
are retained. Clean fixed-final-LayerNorm margin accounting is separate from
the actual ablation's recomputed full-model probabilities and margin.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import tokenizers
from tokenizers import Tokenizer

from . import checkpoint as checkpoint_module
from . import historical_word_ablations as word_readout
from . import verify as token_byte_module
from .checkpoint import GPT2Config
from .historical_word_ablations import Evidence, record


HEADS = ((0, 5), (2, 2), (3, 6))


def decode_bf16(raw, shape):
    if len(raw) != math.prod(shape) * 2:
        raise ValueError('wrong BF16 tensor size')
    bits = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
    values = bits.view(np.float32).reshape(shape).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite native BF16 tensor')
    return values


def rounded_bf16(values):
    """FP32 -> BF16 round-to-nearest-even, then exact FP64 representation."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite FP32 weight')
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) &
               np.uint32(0xffff0000)).view(np.float32).astype(np.float64)
    if not np.isfinite(rounded).all():
        raise ValueError('BF16 weight conversion overflow')
    return rounded


def fixed_direction(embedding_pair, gamma, final_residual):
    """Linearize only the clean final readout, NOT the intervening network.

    The tied vocabulary rows are BF16 operands. LayerNorm gamma stays FP32.
    Centering the direction incorporates subtraction of the residual mean;
    final LayerNorm beta and arithmetic remainders are not head contributions.
    """
    pair = np.asarray(embedding_pair)
    gamma = np.asarray(gamma, dtype=np.float64)
    residual = np.asarray(final_residual, dtype=np.float64)
    if (residual.ndim != 1 or not len(residual) or pair.shape != (2, len(residual))
            or gamma.shape != residual.shape or not np.isfinite(gamma).all()
            or not np.isfinite(residual).all()):
        raise ValueError('incompatible final-readout tensors')
    centered = residual - residual.mean()
    deviation = math.sqrt(float(np.mean(centered * centered)) + 1e-5)
    rows = rounded_bf16(pair)
    direction = (rows[0] - rows[1]) * gamma / deviation
    direction -= direction.mean()
    return direction, deviation


def indentation_groups(ids, token_bytes):
    """Require the exact archived 63-space formatting state, never infer it."""
    if (len(ids) != 1024 or ids[944] != 198 or ids[960] != 198
            or ids[961:] != [220] * 63
            or token_bytes[198] != b'\n' or token_bytes[220] != b' '
            or any(b'\n' in token_bytes[token] for token in ids[945:960])):
        raise ValueError('historical indentation/line boundaries differ')
    return dict(current_indent=(961, 1024), last_newline=(960, 961),
                previous_line=(945, 960), rest=(0, 945))


def source_route(qkv, native_context, output_weight, direction, *, head,
                 n_heads, row, groups):
    """Independent one-query QK, softmax, V, and W_O contractions.

    A group partial vector is sum(p_s * v_s), with the original softmax weights
    unchanged. It is not a source deletion or a renormalized attention output.
    Norm ratios can exceed one through cancellation or BF16 rounding; they are
    not fractions of explained variance, storage, or causal responsibility.
    """
    qkv, native_context, output_weight, direction = [np.asarray(x, dtype=np.float64)
        for x in (qkv, native_context, output_weight, direction)]
    if (qkv.ndim != 2 or qkv.shape[1] % 3 or type(n_heads) is not int or n_heads < 1
            or type(row) is not int or not 0 <= row < len(qkv)):
        raise ValueError('invalid QKV/query geometry')
    width = qkv.shape[1] // 3
    if (width % n_heads or type(head) is not int or not 0 <= head < n_heads
            or native_context.shape != (width,) or output_weight.shape != (width, width)
            or direction.shape != (width,)
            or any(not np.isfinite(x).all() for x in (qkv, native_context, output_weight, direction))):
        raise ValueError('incompatible head tensors')
    coverage = np.zeros(row + 1, dtype=np.int32)
    for begin, end in groups.values():
        if type(begin) is not int or type(end) is not int or not 0 <= begin < end <= row + 1:
            raise ValueError('invalid source group interval')
        coverage[begin:end] += 1
    if not np.all(coverage == 1):
        raise ValueError('source groups must partition the causal context')
    dimension = width // n_heads
    q, k, v = [x.reshape(len(qkv), n_heads, dimension).transpose(1, 0, 2)
               for x in np.split(qkv, 3, axis=1)]
    scores = k[head, :row+1] @ q[head, row] / math.sqrt(dimension)
    unnormalized = np.exp(scores - scores.max())
    probabilities = unnormalized / math.fsum(map(float, unnormalized))
    values = v[head, :row+1]
    begin, end = head * dimension, (head + 1) * dimension
    head_weight = rounded_bf16(output_weight)[begin:end]
    native = native_context[begin:end]
    reconstructed = probabilities @ values
    source_terms = probabilities * (values @ head_weight @ direction)
    projected = native @ head_weight
    context_norm = float(np.linalg.norm(native))
    projected_norm = float(np.linalg.norm(projected))
    native_term = float(projected @ direction)
    grouped = {}
    for name, (first, last) in groups.items():
        partial = probabilities[first:last] @ values[first:last]
        partial_projection = partial @ head_weight
        partial_context_norm = float(np.linalg.norm(partial))
        partial_projected_norm = float(np.linalg.norm(partial_projection))
        grouped[name] = dict(interval=[first, last], source_count=last-first,
            reconstructed_attention_mass=float(probabilities[first:last].sum()),
            reconstructed_fixed_margin_term=float(source_terms[first:last].sum()),
            partial_context_norm=partial_context_norm,
            partial_projected_residual_norm=partial_projected_norm,
            ratio_to_native_context_norm=partial_context_norm/context_norm if context_norm else None,
            ratio_to_native_projected_norm=partial_projected_norm/projected_norm if projected_norm else None,
            scaled_qk_min=float(scores[first:last].min()), scaled_qk_max=float(scores[first:last].max()))
    return dict(head_dimension=dimension, scaled_qk_scores=scores.tolist(),
        reconstructed_attention_probabilities=probabilities.tolist(),
        reconstructed_fixed_margin_source_terms=source_terms.tolist(),
        scaled_qk_range=dict(min=float(scores.min()), max=float(scores.max()), span=float(np.ptp(scores))),
        native_context_norm=context_norm, native_context_projected_residual_norm=projected_norm,
        native_context_fixed_margin_term=native_term,
        reconstructed_context_norm=float(np.linalg.norm(reconstructed)),
        reconstructed_projected_residual_norm=float(np.linalg.norm(reconstructed @ head_weight)),
        reconstructed_source_term_sum=float(source_terms.sum()),
        reconstructed_source_sum_minus_native_term=float(source_terms.sum()-native_term),
        reconstructed_context_max_abs_error=float(np.max(np.abs(reconstructed-native))),
        groups=grouped)


def analyze(evidence_manifest, run_directory, word_ablations, repository):
    """Authenticate the selection report, then independently check its three arms."""
    evidence = Evidence(evidence_manifest)
    sources = [record(x) for x in (__file__, word_readout.__file__,
                                   checkpoint_module.__file__, token_byte_module.__file__)]
    reference_record = record(word_ablations)
    reference = json.loads(evidence.checked(word_ablations, reference_record).read_text())
    if (reference.get('complete') is not True
            or reference.get('format') != 'pluto-historical-word-ablations-v1'
            or reference.get('temperature') != 1
            or reference.get('config') != asdict(GPT2Config())):
        raise ValueError('incompatible historical word-selection report')
    for item in reference['provenance'] + reference['analysis_sources']:
        evidence.checked(item['path'], item)
    root = Path(run_directory).resolve(strict=True)
    plan = json.loads(evidence.checked(root / 'plan.json').read_text())
    inputs = word_readout._index(plan['inputs'], 'path')
    producer_path = str(Path(repository).resolve(strict=True) / word_readout.PRODUCER_SOURCE)
    producer = word_readout.recover_producer_source(repository, evidence.manifest['source_commit_sha'],
                                                   inputs[producer_path])
    if producer != reference['historical_producer_source']:
        raise ValueError('selection report identifies a different historical producer')
    checkpoint = Path(plan['checkpoint_directory'])
    if str(checkpoint) != reference['checkpoint_directory']:
        raise ValueError('selection report identifies a different checkpoint')
    selected = [word for word in reference['words'] if word['word'] == 'Exeunt']
    if len(selected) != 1 or selected[0]['generation_steps'] != [1340, 1341, 1342]:
        raise ValueError('need the exact historical Exeunt event')
    selected = selected[0]
    tokenizer_path = Path(plan['tokenizer_directory']) / 'tokenizer.json'
    evidence.checked(tokenizer_path, inputs[str(tokenizer_path)])
    token_bytes = token_byte_module.gpt2_token_bytes(Tokenizer.from_file(str(tokenizer_path)))
    metadata, raw_clean, raw_ablations = {}, {}, {head: [] for head in HEADS}
    runs = {run['step']: run for run in plan['runs']}
    generation_path = Path(plan['generation_directory']) / 'metadata.json'
    generation = json.loads(evidence.checked(generation_path, inputs[str(generation_path)]).read_text())
    original_path = Path(plan['generation_logits'])
    evidence.checked(original_path, inputs[str(original_path)])
    original = np.memmap(original_path, dtype='<f4', mode='r').reshape(-1, 50257)
    previous = None

    def logits(native, name):
        path = evidence.checked(word_readout._local(native, name))
        if path.stat().st_size != 50272*4:
            raise ValueError('wrong native logits size')
        row = np.fromfile(path, dtype='<f4')
        if not np.isfinite(row).all():
            raise ValueError('nonfinite native logits')
        return row

    for step, target in zip((1340, 1341, 1342), (1475, 68, 2797)):
        native = root / f'trace_step_{step}'
        md = json.loads(evidence.checked(native / 'metadata.json').read_text())
        metadata[step] = md
        previous = word_readout.validate_context(md, runs[step], generation['all_token_ids'],
            len(generation['initial_token_ids']), GPT2Config(), previous)
        if previous[1] != target or md.get('complete') is not True:
            raise ValueError('wrong or incomplete historical target trace')
        clean = logits(native, md['files']['logits']['file'])
        for name in ('clean_replay', 'alternate_padding'):
            if clean.tobytes() != logits(native, md['parity_files'][name]).tobytes():
                raise ValueError('full-padded native replay/padding parity failed')
        if clean[:50257].tobytes() != original[step].tobytes():
            raise ValueError('original generation logit parity failed')
        raw_clean[step] = clean
        arms = word_readout.validate_arms(md['interventions'], GPT2Config())
        for head in HEADS:
            block, number = head
            arm = arms[f'ablation.block{block}.head{number}']
            raw_ablations[head].append(logits(native, arm['logits_file']))
    md = metadata[1340]
    ids = md['token_ids']
    groups = indentation_groups(ids, token_bytes)
    native = root / 'trace_step_1340'
    old_analysis = json.loads(evidence.checked(root / 'analysis_step_1340/analysis.json').read_text())
    if (old_analysis['target_id'], old_analysis['competitor_id'], old_analysis['ledger']['row']) != (1475, 220, 1023):
        raise ValueError('historical fixed-rival ledger differs')

    def stage(name, width):
        spec = md['files'][name]
        if spec['dtype'] != 'bf16' or spec['shape'] != [1024, width]:
            raise ValueError('wrong native stage schema')
        return decode_bf16(evidence.checked(word_readout._local(native, spec['file'])).read_bytes(), spec['shape'])

    def weight(index, shape):
        path = checkpoint / f'weight_{index}.bin'
        evidence.checked(path, inputs[str(path)])
        if path.stat().st_size != math.prod(shape)*4:
            raise ValueError('wrong checkpoint weight size')
        return np.fromfile(path, dtype='<f4').reshape(shape)

    embedding = weight(0, (50272, 512))
    direction, deviation = fixed_direction(embedding[[1475, 220]], weight(98, (512,)),
                                           stage('blocks.7.after_mlp', 512)[1023])
    direction_error = float(np.max(np.abs(direction - np.asarray(old_analysis['ledger']['direction']))))
    if direction_error > 1e-11:
        raise ValueError('independent clean direction disagrees with archived ledger')
    clean_piece_readouts = [word_readout.token_readout(raw_clean[s], t, 50257)
                           for s, t in zip((1340, 1341, 1342), (1475, 68, 2797))]
    clean_nlls = [x['nll'] for x in clean_piece_readouts]
    clean_margin = float(raw_clean[1340][1475]) - float(raw_clean[1340][220])
    result_heads = []
    for block, head in HEADS:
        route = source_route(stage(f'blocks.{block}.qkv', 1536),
            stage(f'blocks.{block}.attention', 512)[1023], weight(6+12*block, (512, 512)),
            direction, head=head, n_heads=8, row=1023, groups=groups)
        old = old_analysis['ledger']['attention'][f'blocks.{block}']
        discrepancies = dict(
            probabilities_max_abs=float(np.max(np.abs(np.asarray(route['reconstructed_attention_probabilities']) - np.asarray(old['probabilities'])[head]))),
            source_terms_max_abs=float(np.max(np.abs(np.asarray(route['reconstructed_fixed_margin_source_terms']) - np.asarray(old['source_margin_terms'])[head]))),
            native_head_term_abs=abs(route['native_context_fixed_margin_term'] - old_analysis['ledger']['heads'][f'blocks.{block}'][head]))
        if max(discrepancies.values()) > 1e-10:
            raise ValueError('independent route differs from archived FP64 ledger')
        probabilities = np.asarray(route['reconstructed_attention_probabilities'])
        terms = np.asarray(route['reconstructed_fixed_margin_source_terms'])

        def source_info(row):
            row = int(row)
            return dict(local_row=row, original_absolute_index=row+322, token_id=ids[row],
                token_bytes_hex=token_bytes[ids[row]].hex(), token_bytes_repr=repr(token_bytes[ids[row]]),
                reconstructed_probability=float(probabilities[row]),
                reconstructed_fixed_margin_term=float(terms[row]),
                nearby_bytes_repr=repr(b''.join(token_bytes[t] for t in ids[max(0, row-4):min(1024, row+5)])))

        route.update(top_attention_sources=[source_info(i) for i in np.argsort(-probabilities, kind='stable')[:8]],
                     largest_positive_sources=[source_info(i) for i in np.argsort(-terms, kind='stable')[:5]],
                     most_negative_sources=[source_info(i) for i in np.argsort(terms, kind='stable')[:5]],
                     archived_FP64_ledger_discrepancies=discrepancies)
        per_piece = [word_readout.token_readout(x, t, 50257)
                     for x, t in zip(raw_ablations[(block, head)], (1475, 68, 2797))]
        effects = word_readout.sequence_readout([x['nll'] for x in per_piece], clean_nlls)
        name = f'ablation.block{block}.head{head}'
        old_arm = next(a for a in selected['interventions'] if a['name'] == name)
        for key in ('first_piece', 'conditional_suffix', 'joint'):
            for metric in ('nll', 'probability', 'delta_nll'):
                if not math.isclose(effects[key][metric], old_arm[key][metric], rel_tol=1e-12, abs_tol=1e-14):
                    raise ValueError('native probability recomputation differs from selection report')
        changed = raw_ablations[(block, head)][0]
        changed_margin = float(changed[1475]) - float(changed[220])
        first_effect = dict(clean_margin=clean_margin, ablated_margin=changed_margin,
            margin_change=changed_margin-clean_margin,
            clean_Ex_logit=float(raw_clean[1340][1475]), ablated_Ex_logit=float(changed[1475]),
            clean_space_logit=float(raw_clean[1340][220]), ablated_space_logit=float(changed[220]),
            clean_space_probability=word_readout.token_readout(raw_clean[1340], 220, 50257)['probability'],
            ablated_space_probability=word_readout.token_readout(changed, 220, 50257)['probability'])
        result_heads.append(dict(block=block, head=head, arm=name,
            physical_parameters=dict(qkv_weight=f'weight_{4+12*block}.bin',
                qkv_bias=f'weight_{5+12*block}.bin', q_columns=[64*head, 64*(head+1)],
                k_columns=[512+64*head, 512+64*(head+1)],
                v_columns=[1024+64*head, 1024+64*(head+1)],
                output_weight=f'weight_{6+12*block}.bin', output_rows=[64*head, 64*(head+1)],
                output_bias_unchanged=f'weight_{7+12*block}.bin'),
            routing=route, native_ablation=dict(per_piece=per_piece, **effects,
                                               first_piece_fixed_rival=first_effect)))
    evidence.recheck()
    for old in sources:
        if record(old['path']) != old:
            raise ValueError('analysis source changed during analysis')
    return dict(format='pluto-historical-initial-piece-routes-v1', complete=True,
        goal_completion_claimed=False, new_deterministic_experiment_result=False,
        selection='Post hoc: B0H5/B2H2 are the two largest Exeunt joint head effects; B3H6 contrasts with two other historical words.',
        checkpoint_directory=str(checkpoint), temperature=1,
        event=dict(generation_step=1340, target_id=1475, target_bytes_repr=repr(token_bytes[1475]),
            fixed_clean_rival_id=220, fixed_clean_rival_bytes_repr=repr(token_bytes[220]),
            query_row=1023, query_token_id=ids[1023], context_start=322,
            context_token_ids=ids, previous_line_bytes_repr=repr(b''.join(token_bytes[t] for t in ids[945:960])),
            groups={k: list(v) for k, v in groups.items()}),
        clean=dict(per_piece=clean_piece_readouts, **word_readout.sequence_readout(clean_nlls)),
        clean_fixed_readout=dict(direction=direction.tolist(), final_std=deviation,
                                archived_direction_max_abs_error=direction_error),
        heads=result_heads, historical_producer_source=producer,
        selection_report=reference_record, provenance=list(evidence.used.values()),
        analysis_sources=sources, runtime_versions=dict(numpy=np.__version__, tokenizers=tokenizers.__version__),
        limitations=[
            'One selected historical event with a long indentation prefix, not a corpus-wide result or a new paired-run result.',
            'Space-position values in later blocks are contextualized and may carry earlier lexical information; mass on spaces does not prove a pure-formatting head.',
            'FP64 attention probabilities are reconstructions from saved BF16 QKV, not native FlashAttention statistics.',
            'Source group sums are descriptive clean contributions with original attention weights; no source-specific intervention was run.',
            'Projected head norms use a FP64 contraction of native BF16 context with BF16 output-weight operands, not an independently captured native per-head projection.',
            'Norm ratios are not explained-variance fractions and can exceed one through vector cancellation or rounding.',
            'Fixed clean-final-LayerNorm margin terms are not ablation effects; later layers, normalization, all query positions, and competing logits may change after removal.',
            'Ablations zero head-output rows at every query, leave output bias unchanged, rerun the full model, and restore weights.',
            'Joint probabilities concern the three teacher-forced word pieces; the following word boundary was not intervened on.',
            'The supplied manifest and selection report anchor provenance; exact historical producer source is recovered from Git, with no current-binary equality or GPU rerun claimed.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-manifest', 'run-directory', 'word-ablations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.run_directory.resolve() in args.output.resolve().parents:
        raise ValueError('output must be outside the historical run')
    result = analyze(args.evidence_manifest, args.run_directory, args.word_ablations, args.repository)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('Audited three historical initial-piece attention routes using CPU only.')


if __name__ == '__main__':
    main()
