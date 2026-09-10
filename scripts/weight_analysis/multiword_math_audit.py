"""Independently audit frozen multiword native traces against checkpoint bytes.

The numerical body is adapted from the previously executed beseem audit, itself
adapted from token_path_math_independent_audit.py; it is not claimed as a wholly
new independent implementation. It imports no production-analysis math helpers,
runs no model, and never re-tokenizes generated text. Instead it reconstructs
every saved residual-margin term, attention source, neuron contribution, selected
gate, readout distribution, and declared removal arm from original saved arrays.

This supports this repository's eight-block, BF16, 512-wide GPT-2 checkpoint
format, not arbitrary architectures. A completed multiword native runner and
completed original generation are prerequisites. The output is exclusive-create.
"""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np

V, PV, D, HEADS, HD, FF, BLOCKS = 50257, 50272, 512, 8, 64, 2048, 8


def new_metrics():
    return {'ledgers': 0, 'head_terms': 0, 'neuron_terms': 0,
           'attention_source_terms': 0, 'selected_neuron_operations': 0,
           'input_key_products': 0, 'sampler_rows': 0, 'intervention_rows': 0,
           'lens_rows': 0, 'baseline_generation_byte_equal_rows': 0,
           'exact_generation_prefixes': 0,
           'max_ledger_closure_error': 0., 'max_term_error': 0.,
           'max_probability_error': 0., 'max_gate_error': 0.}


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def _remember(path, expected=None, *, records):
    key = str(Path(path).resolve())
    if key not in records:
        records[key] = identity(path)
    if expected is not None:
        assert records[key] == expected, ('file identity', key)
    return records[key]


def _read_json(path, *, records):
    _remember(path, records=records)
    return json.loads(Path(path).read_text())


def close(actual, expected, name, atol=3e-11, rtol=3e-11):
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape, (name, a.shape, e.shape)
    error = float(np.max(np.abs(a-e))) if a.size else 0.
    assert np.allclose(a, e, atol=atol, rtol=rtol), (name, error)
    return error


def f32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def quantized(array):
    bits = np.asarray(array, dtype=np.float32).view(np.uint32)
    high, low = bits >> 16, bits & 65535
    carry = (low > 32768) | ((low == 32768) & ((high & 1) == 1))
    return ((high+carry.astype(np.uint32)) << 16).view(np.float32).astype(float)


def _raw(native, record, *, records):
    path = (native/record['file']).resolve()
    assert path.parent == native.resolve(), 'array file escapes trace directory'
    _remember(path, records=records)
    dtype = {'bf16':'<u2', 'float32':'<f4', 'int32':'<i4'}[record['dtype']]
    assert path.stat().st_size == math.prod(record['shape']) * np.dtype(dtype).itemsize
    data = np.fromfile(path, dtype=dtype).reshape(record['shape'])
    if record['dtype'] == 'bf16':
        data = (data.astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(data).all()
    return data


def sampler(row, temperature, uniform, target):
    # Native logits are already FP32. Subtraction must round to FP32 BEFORE
    # promotion to double division/exp, unlike the FP64 attribution ledger.
    maximum = max(float(value) for value in row)
    weights = [math.exp(f32(float(value)-maximum)/temperature) for value in row]
    total = 0.
    for weight in weights:
        total += weight
    probabilities = [weight/total for weight in weights]
    accumulated, selected, lower, upper = 0., None, None, None
    for index, probability in enumerate(probabilities):
        previous = accumulated
        accumulated += probability
        if index+1 == len(probabilities):
            accumulated = 1.
        if selected is None and accumulated >= uniform:
            selected = index
        if index == target:
            lower, upper = previous, accumulated
    assert selected is not None
    return probabilities[target], lower, upper, selected


def _check_readout(row, saved, target, temperature, uniform, *, metrics):
    row = np.asarray(row[:V], dtype=np.float32)
    probability, lower, upper, selected = sampler(row, temperature, uniform, target)
    metrics['sampler_rows'] += 1
    error = abs(probability-saved['target_probability_at_generation_temperature'])
    metrics['max_probability_error'] = max(metrics['max_probability_error'], error)
    assert probability == saved['target_probability_at_generation_temperature']
    assert lower == saved['target_cdf_lower'] and upper == saved['target_cdf_upper']
    assert selected == saved['same_uniform_selected_id']
    assert saved['target']['id'] == target
    rank = 1+int(np.count_nonzero(row > row[target]))+int(np.count_nonzero(row[:target] == row[target]))
    assert saved['target']['rank'] == rank
    assert saved['target']['logit'] == float(row[target])
    order = np.lexsort((np.arange(V), -row))
    assert saved['winner']['id'] == int(order[0])
    assert saved['runner_up']['id'] == int(order[1])
    assert [item['id'] for item in saved['top5']] == order[:5].tolist()
    # T=1 display probabilities use the exact saved-logit differences in FP64,
    # not the production sampler's different float32 subtraction path.
    max64 = float(row[order[0]])
    exps = [math.exp(float(value)-max64) for value in row]
    denominator = math.fsum(exps)
    for entry in [saved['target'],saved['winner'],saved['runner_up'],*saved['top5']]:
        token = entry['id']
        assert entry['logit'] == float(row[token])
        close(entry['probability'],exps[token]/denominator,'T1 display probability',2e-14,2e-14)
    return probability


def selection_layout(selection):
    """Require distinct, contiguous word spans and one following boundary each.

    The plan is frozen before tracing. Events, rather than retokenized spellings,
    specify target IDs. A later gate compares each entire event with the original
    recorder. This helper does not make a lexical uncommonness claim.
    """
    words = selection['words']
    if not isinstance(words, list) or not words:
        raise ValueError('selection must contain words')
    steps, word_steps, names = [], [], set()
    for word in words:
        name = word['word']
        if not isinstance(name, str) or not name or name.casefold() in names:
            raise ValueError('word names must be distinct nonempty strings')
        names.add(name.casefold())
        pieces = word['word_steps']
        if (not isinstance(pieces, list) or len(pieces) < 2
                or any(type(step) is not int or step < 0 for step in pieces)
                or pieces != list(range(pieces[0], pieces[0]+len(pieces)))):
            raise ValueError('word steps must be contiguous and contain at least two pieces')
        boundary = word['boundary_step']
        if type(boundary) is not int or boundary != pieces[-1]+1:
            raise ValueError('boundary must be the immediately following generation event')
        word_steps.extend(pieces)
        steps.extend([*pieces, boundary])
    if len(set(steps)) != len(steps):
        raise ValueError('word and boundary trace events must not overlap')
    events = selection['events']
    if not isinstance(events, list) or len(events) != len(steps):
        raise ValueError('selection needs one frozen event per trace')
    event_steps = [event['step'] for event in events]
    if (any(type(step) is not int for step in event_steps)
            or len(set(event_steps)) != len(event_steps)
            or set(event_steps) != set(steps)):
        raise ValueError('frozen events must exactly cover selected trace steps')
    targets = {event['step']: event['token_id'] for event in events}
    if any(type(token) is not int or not 0 <= token < V for token in targets.values()):
        raise ValueError('selected token outside vocabulary')
    return tuple(sorted(steps)), frozenset(word_steps), targets


def normalize_runner_plan(plan):
    """Translate the runner's half-open spans without importing runner helpers."""
    if plan['schema_version'] != 1:
        raise ValueError('unsupported runner plan schema')
    words = []
    for selection in plan['selections']:
        first, end = selection['first'], selection['end']
        if type(first) is not int or type(end) is not int or not 0 <= first < end:
            raise ValueError('invalid half-open selected word span')
        words.append(dict(word=selection['word'], word_steps=list(range(first, end)),
                          boundary_step=selection['boundary_step']))
    normalized = dict(words=words, events=[item['event'] for item in plan['runs']],
                      inputs=plan['inputs'])
    steps, word_steps, _ = selection_layout(normalized)
    if [item['step'] for item in plan['runs']] != list(steps):
        raise ValueError('runner steps must exactly match sorted selected events')
    for item in plan['runs']:
        if (item['event']['step'] != item['step']
                or type(item['interventions']) is not bool
                or item['interventions'] != (item['step'] in word_steps)):
            raise ValueError('runner intervention policy differs from selected word pieces')
    return normalized


def generation_context(generation_tokens, initial_count, step, event, context_limit=1024):
    """Derive the exact sliding context independently of the event's offsets.

    In particular, absolute generation step 1340 must not be used as the local
    position-embedding row: only the final 1024 preceding IDs are supplied and
    positions restart at zero for each native forward.
    """
    if (type(step) is not int or step < 0 or type(initial_count) is not int
            or initial_count < 1 or initial_count+step >= len(generation_tokens)
            or type(context_limit) is not int or context_limit < 1):
        raise ValueError('invalid generation step or initial prompt length')
    end = initial_count+step
    start = max(0, end-context_limit)
    expected = generation_tokens[start:end]
    if (event['step'] != step or event['context_start'] != start
            or event['context_length'] != len(expected)
            or event['output_row'] != len(expected)-1
            or event['absolute_token_index'] != end):
        raise ValueError('recorded sliding context does not match generation history')
    return expected


def run(run_directory, generation_directory, generation_result_path,
        *, selection_plan=None, trace_result=None, output=None, adapted_from=None):
    if not __debug__:
        raise RuntimeError("audit assertions require Python without -O")
    BASE = Path(run_directory).resolve()
    generation_directory = Path(generation_directory).resolve()
    selection_plan = Path(selection_plan) if selection_plan else BASE/"plan.json"
    trace_result = Path(trace_result) if trace_result else BASE/"result.json"
    output_path = Path(output) if output else BASE/"multiword_math_independent_audit.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    START = time.time()
    records, metrics = {}, new_metrics()
    remember = partial(_remember, records=records)
    read_json = partial(_read_json, records=records)
    raw = partial(_raw, records=records)
    check_readout = partial(_check_readout, metrics=metrics)
    # This explicit completion gate prevents accidentally auditing partial streams.
    remember(__file__)
    if adapted_from is not None:
        remember(adapted_from)
    runner_plan = read_json(selection_plan)
    assert Path(runner_plan['generation_directory']).resolve() == generation_directory
    assert Path(runner_plan['output_directory']).resolve() == BASE
    selection = normalize_runner_plan(runner_plan)
    STEPS, WORD_STEPS, TARGETS = selection_layout(selection)
    completed = read_json(trace_result)
    assert completed['complete'] and completed['all_inputs_unchanged']
    remember(selection_plan, completed['plan'])
    assert len(completed['runs']) == len(STEPS)
    for expected_step, result_record in zip(STEPS, completed['runs']):
        remember(result_record['path'], result_record)
        item = read_json(result_record['path'])
        assert item['step'] == expected_step and item['complete'] and item['returncode'] == 0
        remember(selection_plan, item['plan'])
        assert item['baseline_logits_equal_original_generation']
        assert item['outputs']
        for record in item['outputs']:
            remember(record['path'], record)
    generation_result = read_json(generation_result_path)
    assert generation_result['returncode'] == 0 and generation_result['all_inputs_byte_unchanged']
    assert generation_result['changed_inputs'] == []
    for item in generation_result['outputs']:
        expected = dict(path=str(Path(item['path']).resolve()), bytes=item.get('bytes',item.get('size')), sha256=item['sha256'])
        remember(item['path'], expected)
    for item in selection['inputs']:
        remember(item['path'], item)
    generation = read_json(generation_directory/'metadata.json')
    generation_steps = generation['steps']
    assert generation['complete'] and generation_steps > max(STEPS)
    assert generation['logits_shape'] == [generation_steps, V]
    assert generation['no_bos'] and generation['requested_prompt'] == generation['prompt']
    assert not generation['empty_prompt_newline_fallback']
    generation_tokens = generation['initial_token_ids'] + generation['generated_token_ids']
    initial_count = len(generation['initial_token_ids'])
    assert initial_count > 0 and len(generation_tokens) == initial_count+generation_steps
    assert generation_tokens == generation['all_token_ids'] == generation['token_ids']
    remember(generation_directory/'tokens.i32')
    assert np.fromfile(generation_directory/'tokens.i32',dtype='<i4').tolist() == generation_tokens
    remember(generation_directory/'events.jsonl')
    events = [json.loads(line) for line in (generation_directory/'events.jsonl').read_text().splitlines()]
    assert len(events) == generation_steps
    for event in selection['events']:
        assert event == events[event['step']], 'frozen selection event changed in final recorder'
    generation_logits_path = generation_directory/'logits.f32'
    remember(generation_logits_path)
    assert generation_logits_path.stat().st_size == generation_steps*V*4
    reports = [read_json(BASE/f'analysis_step_{step}'/'analysis.json') for step in STEPS]
    for report in reports:
        assert report['complete']
        for record in report['inputs'].values():
            remember(record['path'],record)

    checkpoint = Path(read_json(BASE/f'trace_step_{STEPS[0]}'/'metadata.json')['checkpoint_directory'])
    assert checkpoint.resolve() == Path(runner_plan['checkpoint_directory']).resolve()
    cache = {}
    def weight(index,shape):
        if index not in cache:
            path = checkpoint/f'weight_{index}.bin'
            remember(path)
            cache[index] = np.fromfile(path,dtype='<f4').reshape(shape)
        return cache[index]

    embedding = quantized(weight(0,(PV,D)))
    position_table = weight(1,(1024,D)).astype(float)
    gamma = weight(98,(D,)).astype(float)
    beta = weight(99,(D,)).astype(float)
    weight_sets = []
    for block in range(BLOCKS):
        weight_sets.append(dict(
            wo=quantized(weight(6+12*block,(D,D))), bo=weight(7+12*block,(D,)).astype(float),
            w1=quantized(weight(10+12*block,(D,FF))), b1=weight(11+12*block,(FF,)).astype(float),
            w2=quantized(weight(12+12*block,(FF,D))), b2=weight(13+12*block,(D,)).astype(float)))

    summaries = []
    for step,report in zip(STEPS,reports):
        native = BASE/f'trace_step_{step}'
        meta = read_json(native/'metadata.json')
        assert meta['complete'] and meta['probe_kind'] == 'token_trace'
        assert meta['context_length'] == 1024
        assert meta['vocab_size'] == V and meta['padded_vocab_size'] == PV
        assert meta['checkpoint_unique_weight_count'] == 100
        assert Path(meta['checkpoint_directory']).resolve() == checkpoint.resolve()
        assert meta['original_forward'] == 'unmodified CreateGpt2'
        assert meta['optimizer_steps'] == meta['backward_calls'] == meta['checkpoint_writes'] == 0
        arrays = {name:raw(native,record) for name,record in meta['files'].items()}
        row = len(meta['token_ids'])-1
        assert row == report['ledger']['row']
        target,competitor = report['target_id'],report['competitor_id']
        assert target == TARGETS[step] == events[step]['token_id'] == meta['target_id']
        assert report['generation_step'] == step
        assert report['original_event'] == events[step]
        event = events[step]
        expected_prefix = generation_context(generation_tokens, initial_count, step, event)
        planned = next(item for item in runner_plan['runs'] if item['step'] == step)
        assert planned['context_token_ids'] == expected_prefix
        remember(planned['prefix']['path'], planned['prefix'])
        assert np.fromfile(planned['prefix']['path'], dtype='<i4').tolist() == expected_prefix
        assert meta['token_ids'] == report['context_token_ids'] == expected_prefix
        assert row == event['output_row'] == len(expected_prefix)-1
        metrics['exact_generation_prefixes'] += 1
        logits = arrays['logits'][0,:V]
        with generation_logits_path.open('rb') as original:
            original.seek(event['logits_byte_offset'])
            original_row = original.read(V*4)
        assert event['logits_byte_offset'] == step*V*4
        assert logits.astype('<f4',copy=False).tobytes() == original_row
        # Inspect parity arrays themselves; a metadata boolean is not evidence
        # that an unmodified replay really reproduced the original logits.
        for record in meta['parity_files'].values():
            parity_path = native/record
            remember(parity_path)
            parity = np.fromfile(parity_path, dtype='<f4').reshape(1, PV)[0, :V]
            assert parity.tobytes() == original_row
        metrics['baseline_generation_byte_equal_rows'] += 1
        order = np.lexsort((np.arange(V),-logits))
        assert competitor == int(order[0] if order[0] != target else order[1])
        native_margin = float(logits[target])-float(logits[competitor])
        assert native_margin == report['ledger']['native_margin']
        residual = arrays['blocks.7.after_mlp'][row].astype(float)
        mean = math.fsum(residual)/D
        centered = residual-mean
        std = math.sqrt(math.fsum(float(x)*float(x) for x in centered)/D+1e-5)
        readout = embedding[target]-embedding[competitor]
        direction = readout*gamma/std
        direction -= math.fsum(direction)/D
        metrics['max_term_error'] = max(metrics['max_term_error'],
            close(report['ledger']['direction'],direction,'direction'))
        close(report['ledger']['final_std'],std,'final std')

        def value(name):
            return arrays[name][row].astype(float)

        def project(vector):
            return math.fsum(float(x)*float(y) for x,y in zip(vector,direction))

        def add(name,vector):
            terms[name] = project(vector)

        terms = {}
        add('token_embedding',value('embedding'))
        add('position_embedding',position_table[row])
        add('position_add_rounding',value('positioned')-value('embedding')-position_table[row])
        prior = value('positioned')
        neuron_arrays = {}
        for block in range(BLOCKS):
            p = f'blocks.{block}'
            w = weight_sets[block]
            context = value(p+'.attention')
            q,k,v = [part.reshape(row+1,HEADS,HD).astype(float) for part in np.split(arrays[p+'.qkv'],3,axis=1)]
            probabilities = np.zeros((HEADS,row+1))
            source_terms = np.zeros_like(probabilities)
            reconstructed = np.zeros((HEADS,HD))
            heads = np.zeros(HEADS)
            for head in range(HEADS):
                lo,hi = head*HD,(head+1)*HD
                reverse = np.einsum('ij,j->i',w['wo'][lo:hi],direction,optimize=False)
                heads[head] = math.fsum(float(a)*float(b) for a,b in zip(context[lo:hi],reverse))
                scores = [math.fsum(float(a)*float(b) for a,b in zip(q[row,head],k[s,head]))/math.sqrt(HD)
                          for s in range(row+1)]
                maximum = max(scores)
                exp = [math.exp(score-maximum) for score in scores]
                denominator = math.fsum(exp)
                for s,e in enumerate(exp):
                    probability = e/denominator
                    probabilities[head,s] = probability
                    source_terms[head,s] = probability*math.fsum(float(a)*float(b) for a,b in zip(v[s,head],reverse))
                    reconstructed[head] += probability*v[s,head]
            for name,actual,expected in [('heads',report['ledger']['heads'][p],heads),
                                         ('attention probabilities',report['ledger']['attention'][p]['probabilities'],probabilities),
                                         ('sources',report['ledger']['attention'][p]['source_margin_terms'],source_terms),
                                         ('context',report['ledger']['attention'][p]['reconstructed_context'],reconstructed.reshape(D))]:
                metrics['max_term_error'] = max(metrics['max_term_error'],close(actual,expected,name))
            context_error=float(np.max(np.abs(reconstructed.reshape(D)-context)))
            source_error=float(np.max(np.abs(source_terms.sum(axis=1)-heads)))
            close(report['ledger']['attention'][p]['context_max_abs_error'],context_error,'context error')
            close(report['ledger']['attention'][p]['source_sum_vs_native_head_max_abs'],source_error,'source error')
            terms[p+'.attention_heads'] = math.fsum(heads)
            add(p+'.attention_bias',w['bo'])
            add(p+'.attention_projection_rounding',value(p+'.attention_projected')-context@w['wo']-w['bo'])
            add(p+'.attention_residual_rounding',value(p+'.after_attention')-prior-value(p+'.attention_projected'))
            features=value(p+'.gelu')
            neurons=features*np.einsum('jd,d->j',w['w2'],direction,optimize=False)
            neuron_arrays[p]=neurons
            metrics['max_term_error']=max(metrics['max_term_error'],close(report['ledger']['neurons'][p],neurons,'neurons'))
            terms[p+'.mlp_neurons']=math.fsum(neurons)
            add(p+'.mlp_bias',w['b2'])
            add(p+'.mlp_projection_rounding',value(p+'.mlp_projected')-features@w['w2']-w['b2'])
            add(p+'.mlp_residual_rounding',value(p+'.after_mlp')-value(p+'.after_attention')-value(p+'.mlp_projected'))
            prior=value(p+'.after_mlp')
            metrics['head_terms']+=HEADS
            metrics['neuron_terms']+=FF
            metrics['attention_source_terms']+=HEADS*(row+1)
        ideal=centered/std*gamma+beta
        terms['final_norm_bias']=math.fsum(float(a)*float(b) for a,b in zip(beta,readout))
        terms['final_norm_rounding_and_reduction']=math.fsum(float(a)*float(b) for a,b in zip(value('final_norm')-ideal,readout))
        terms['head_fp32_accumulation_remainder']=native_margin-math.fsum(float(a)*float(b) for a,b in zip(value('final_norm'),readout))
        assert set(terms)==set(report['ledger']['terms'])
        for name,amount in terms.items():
            metrics['max_term_error']=max(metrics['max_term_error'],close(report['ledger']['terms'][name],amount,name))
        closure=math.fsum(terms.values())-native_margin
        assert abs(closure)<3e-10
        metrics['max_ledger_closure_error']=max(metrics['max_ledger_closure_error'],abs(closure))
        metrics['ledgers']+=1

        gates=report['selected_neuron_operations']
        actual_pairs=[(entry['block'],entry['neuron']) for entry in gates]
        assert len(actual_pairs)==len(set(actual_pairs))
        for block in range(BLOCKS):
            values=np.asarray(report['ledger']['neurons'][f'blocks.{block}'])
            expected=set(np.argsort(values)[:3].tolist()+np.argsort(-values)[:3].tolist())
            assert {n for b,n in actual_pairs if b==block}==expected
        for gate in gates:
            block,neuron=gate['block'],gate['neuron']
            p=f'blocks.{block}'
            w=weight_sets[block]
            assert gate['row']==row
            assert gate['input_weight_file']==f'weight_{10+12*block}.bin'
            assert gate['input_column_first_byte']==4*neuron
            assert gate['input_column_byte_stride']==4*FF
            assert gate['input_bias_file']==f'weight_{11+12*block}.bin'
            assert gate['input_bias_byte_offset']==4*neuron
            assert gate['output_weight_file']==f'weight_{12+12*block}.bin'
            assert gate['output_row_byte_offset']==4*neuron*D
            # Explicit raw file addressing, independent of tensor/view conventions.
            with (checkpoint/gate['input_weight_file']).open('rb') as stream:
                addressed=[]
                for i in range(D):
                    stream.seek(gate['input_column_first_byte']+i*gate['input_column_byte_stride'])
                    addressed.append(struct.unpack('<f',stream.read(4))[0])
            close(quantized(addressed),w['w1'][:,neuron],'addressed input column',0,0)
            with (checkpoint/gate['output_weight_file']).open('rb') as stream:
                stream.seek(gate['output_row_byte_offset'])
                addressed=np.frombuffer(stream.read(4*D),dtype='<f4')
            close(quantized(addressed),w['w2'][neuron],'addressed output row',0,0)
            with (checkpoint/gate['input_bias_file']).open('rb') as stream:
                stream.seek(gate['input_bias_byte_offset'])
                bias=struct.unpack('<f',stream.read(4))[0]
            products=value(p+'.ln2')*w['w1'][:,neuron]
            pre=math.fsum(products)+bias
            native_pre=float(arrays[p+'.fc1'][row,neuron])
            native_post=float(arrays[p+'.gelu'][row,neuron])
            analytical_gelu=.5*native_pre*(1+math.tanh(f32(.7978845608)*(native_pre+f32(.044715)*native_pre**3)))
            aligned=math.fsum(float(a)*float(b) for a,b in zip(w['w2'][neuron],direction))
            expected=dict(input_product_terms=products,input_bias=bias,analytical_pre_gelu=pre,
                          native_pre_gelu=native_pre,native_pre_minus_analytical=native_pre-pre,
                          analytical_gelu_of_native_pre=analytical_gelu,native_post_gelu=native_post,
                          native_post_minus_analytical=native_post-analytical_gelu,
                          output_readout_dot=aligned,margin_contribution=native_post*aligned)
            for name,amount in expected.items():
                metrics['max_gate_error']=max(metrics['max_gate_error'],close(gate[name],amount,'gate '+name))
            close(gate['margin_contribution'],neuron_arrays[p][neuron],'gate matches neuron budget')
            metrics['selected_neuron_operations']+=1
            metrics['input_key_products']+=D

        temperature,uniform=report['temperature'],report['original_event']['uniform']
        base_probability=check_readout(logits,report['baseline'],target,temperature,uniform)
        assert report['baseline']['same_uniform_selected_id']==target
        expected_lens_names = ['positioned'] + [f'blocks.{block}.after_{kind}' for block in range(BLOCKS) for kind in ['attention','mlp']]
        assert list(meta['lens']) == list(report['lens']) == expected_lens_names
        for name,record in meta['lens'].items():
            assert record['shape'] == [1, PV] and record['role'] == 'selected_row'
            values=raw(native,record)[0,:V]
            check_readout(values,report['lens'][name],target,temperature,uniform)
            metrics['lens_rows'] += 1
        assert len(meta['interventions'])==len(report['interventions'])
        expected_arm_names = {f'ablation.block{block}.{kind}_branch' for block in range(BLOCKS) for kind in ['attention','mlp']}
        expected_arm_names |= {f'ablation.block{block}.head{head}' for block in range(BLOCKS) for head in range(HEADS)}
        actual_arm_names = [arm['name'] for arm in meta['interventions']]
        if step in WORD_STEPS:
            assert len(actual_arm_names) == 80 and set(actual_arm_names) == expected_arm_names
        else:
            assert not actual_arm_names
        for arm,saved in zip(meta['interventions'],report['interventions']):
            for name,value_ in arm.items():
                assert saved[name]==value_
            values=raw(native,{'file':arm['logits_file'],'dtype':'float32','shape':arm['shape']})[0,:V]
            probability=check_readout(values,saved['readout'],target,temperature,uniform)
            margin=float(values[target])-float(values[competitor])
            assert margin==saved['target_margin']
            assert margin-native_margin==saved['margin_change']
            expected=math.log(probability/base_probability) if probability else None
            if expected is None:
                assert saved['delta_log_probability'] is None
            else:
                assert expected==saved['delta_log_probability']
            metrics['intervention_rows']+=1
        # Summaries retain two separate notions: a diagnostic projected rank, and a
        # full native removal's measured effect. Neither is a unique introduction.
        lens_trajectory = [dict(stage=name, target_rank=entry['target']['rank'],
                                target_probability=entry['target_probability_at_generation_temperature'],
                                winner_id=entry['winner']['id'], winner_piece=entry['winner']['piece_escaped'])
                           for name,entry in report['lens'].items()]
        first_winner = next((entry['stage'] for entry in lens_trajectory if entry['target_rank']==1),None)

        def summarize_arm(arm):
            readout=arm['readout']
            return dict(name=arm['name'], kind=arm['kind'],
                        probability_after_removal=readout['target_probability_at_generation_temperature'],
                        probability_change=readout['target_probability_at_generation_temperature']-base_probability,
                        delta_log_probability=arm['delta_log_probability'], margin_change=arm['margin_change'],
                        target_rank_after=readout['target']['rank'],
                        same_uniform_selected_id=readout['same_uniform_selected_id'],
                        same_uniform_selected_piece=readout['same_uniform_selected_piece'])

        causal_summary={}
        for label, kinds in [('branches',{'attention_branch','mlp_branch'}),('heads',{'attention_head'})]:
            arms=[summarize_arm(arm) for arm in report['interventions'] if arm['kind'] in kinds]
            if step in WORD_STEPS:
                assert len(arms) == (16 if label=='branches' else 64)
            causal_summary[label] = dict(
                largest_probability_decreases=sorted(arms,key=lambda item:(item['probability_change'],item['name']))[:3],
                largest_probability_increases=sorted(arms,key=lambda item:(-item['probability_change'],item['name']))[:3])
        all_direct=[(float(value),block,neuron) for block in range(BLOCKS)
                    for neuron,value in enumerate(neuron_arrays[f'blocks.{block}'])]
        positive=sorted(all_direct,key=lambda item:(-item[0],item[1],item[2]))[:3]
        negative=sorted(all_direct,key=lambda item:(item[0],item[1],item[2]))[:3]
        gate_map={(gate['block'],gate['neuron']):gate for gate in gates}
        gate_summaries=[]
        for label,chosen in [('positive',positive),('negative',negative)]:
            for amount,block,neuron in chosen:
                gate=gate_map[(block,neuron)]
                gate_summaries.append(dict(sign_group=label,block=block,neuron=neuron,direct_margin=amount,
                    input_weight_file=gate['input_weight_file'],input_column_first_byte=gate['input_column_first_byte'],
                    input_column_byte_stride=gate['input_column_byte_stride'],input_bias=gate['input_bias'],
                    analytical_pre_gelu=gate['analytical_pre_gelu'],native_pre_gelu=gate['native_pre_gelu'],
                    native_post_gelu=gate['native_post_gelu'],output_readout_dot=gate['output_readout_dot'],
                    output_weight_file=gate['output_weight_file'],output_row_byte_offset=gate['output_row_byte_offset']))
        summaries.append(dict(generation_step=step,target_id=target,target_piece=report['target_piece'],
                              target_rank=report['baseline']['target']['rank'],native_margin=native_margin,
                              baseline_probability=base_probability,first_top_ranked_lens=first_winner,
                              lens_trajectory=lens_trajectory,causal_summary=causal_summary,
                              selected_key_gate_examples=gate_summaries,
                              independent_ledger_closure_error=closure))

    count = len(STEPS)
    assert metrics['ledgers'] == count and metrics['intervention_rows'] == len(WORD_STEPS)*80
    assert metrics['selected_neuron_operations'] == count*BLOCKS*6
    assert metrics['head_terms'] == count*BLOCKS*HEADS and metrics['neuron_terms'] == count*BLOCKS*FF
    assert metrics['lens_rows'] == count*(1+2*BLOCKS)
    assert metrics['baseline_generation_byte_equal_rows'] == metrics['exact_generation_prefixes'] == count
    for record in records.values():
        assert identity(record['path'])==record
    source=identity(Path(__file__))
    output={'complete':True,'read_only_no_model_forward_or_generation':True,
            'source':source,'adapted_from':identity(adapted_from) if adapted_from else None,
            'reuse_provenance':'Numerical body adapted from audit_beseem_math.py and its prior token_path_math_independent_audit.py; no production-analysis math helpers imported.',
            'elapsed_seconds':time.time()-START,'metrics':metrics,
            'selection_words':selection['words'],
            'steps':summaries,'all_recorded_inputs_unchanged':True,'records':list(records.values()),
            'limitations':['A target-first logit-lens match does not establish layer necessity.',
                          'The summaries rank native removal effects separately from selected-position direct accounting terms.',
                          'Target logit support and CDF/RNG token selection are different operations.',
                          'Frozen-final-normalizer accounting is not an intervention effect.',
                          f'Only the {count} frozen word/boundary contexts and {len(WORD_STEPS)*80} declared branch/head interventions were audited; no neuron-removal experiment or full-vocabulary affinity study is claimed.']}
    with output_path.open('x') as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'complete':True,'metrics':metrics,'elapsed_seconds':output['elapsed_seconds']}))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', required=True)
    parser.add_argument('--generation-directory', required=True)
    parser.add_argument('--generation-result', required=True)
    parser.add_argument('--selection-plan')
    parser.add_argument('--trace-result')
    parser.add_argument('--output')
    parser.add_argument('--adapted-from', help='Optional original audit source to record and verify unchanged')
    args = parser.parse_args()
    run(args.run_directory, args.generation_directory, args.generation_result,
        selection_plan=args.selection_plan, trace_result=args.trace_result,
        output=args.output, adapted_from=args.adapted_from)


if __name__ == '__main__':
    main()
