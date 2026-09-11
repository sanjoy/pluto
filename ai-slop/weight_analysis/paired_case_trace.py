"""Bind a native token trace to a frozen teacher-forced case, never a sample.

Callers first authenticate ``cases`` with ``core.branch._cases``, ``model``
with the appropriate patch validator, and ``reference`` with ``core.load_native``.
This adapter preserves those identities and checks the saved bytes again; it
does not independently establish their upstream experiment/selection provenance.
It neither tokenizes, samples, executes a model, nor claims unique word storage.

``write_case_prefix`` returns ``{record, plan, prefix}``; pass that object to
``validate_trace``. ``read_binding(record)`` reloads it from a pinned binding
file record. The native command must include the binding file, source cases,
packed batch, exact prefix, checkpoint weights/patch, and binary in its recorded
inputs. Native output and its execution log/ledger must be separate directories.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path

import numpy as np

from . import checkpoint
from . import historical_source_archive as history
from . import paired_complement_localization as core
from . import phrase_reference as rounding
from . import phrase_trace_analysis as trace

GPT2Config = checkpoint.GPT2Config
FORMAT = 'pluto-paired-case-trace-binding-v1'
_require = core.native._require
SOURCE_RECORDS = tuple(history.file_record(Path(path).resolve()) for path in (
    __file__, checkpoint.__file__, history.__file__, core.__file__,
    core.branch.__file__, core.native.__file__, trace.__file__, rounding.__file__))


def _verify(records):
    for record in core.outer._unique(records):
        history._record_value(record)
        _require(history.file_record(record['path']) == record, 'frozen trace evidence changed')


def _path(value):
    path = Path(value).absolute()
    _require(path.resolve(strict=True) == path, 'noncanonical or symlink path')
    return path


def _suite(plan):
    contract = core.branch.case_contract
    formats = ((contract.LEGACY[0], contract.AMENDED[0]),
               (contract.LEGACY[1], contract.AMENDED[1]))
    for suite, allowed in zip(('main', 'supplemental'), formats):
        if plan.get('format') in allowed:
            return suite
    raise ValueError('unknown frozen case suite')


def bind_case(cases, case_index, target_position, config=GPT2Config()):
    """Return exact little-endian int32 prefix IDs through one prediction row.

    Positions index the case's selected targets, not the full context. For
    position two, the prefix includes targets zero and one, but not target two.
    Positions/learned absolute embeddings remain those of the original packed
    assay, starting at zero. No new BOS token, truncation, or retokenizing occurs.
    Upstream case authenticity is a caller precondition; geometry, source-file
    identities, and even the supplied in-memory packed array are checked here.
    """
    _require(isinstance(config, GPT2Config), 'expected explicit GPT2Config')
    GPT2Config(**asdict(config))
    _verify([*SOURCE_RECORDS, cases['record'], cases['batch']])
    plan = history._json_record(cases['record'])
    _require(plan == cases['plan'] and plan['packed_batch'] == cases['batch'],
             'case plan or packed-batch identity differs')
    suite = _suite(plan)
    count = plan.get('case_count')
    _require(type(count) is int and count > 0 and count == len(plan['cases'])
             and plan.get('context_length') == config.context_length
             and plan.get('vocab_size') == config.vocab_size
             and type(case_index) is int and 0 <= case_index < count,
             'invalid frozen case index or geometry')
    shape = (2, count, config.context_length)
    _require(cases['batch']['bytes'] == math.prod(shape)*4, 'packed-batch byte size differs')
    supplied = np.asarray(cases['packed'])
    _require(supplied.shape == shape and supplied.dtype == np.dtype('<i4'),
             'packed cases must be exact int32 arrays')
    packed = np.fromfile(cases['batch']['path'], dtype='<i4').reshape(shape)
    _require(np.array_equal(packed, supplied) and np.all(packed >= 0)
             and np.all(packed < config.vocab_size)
             and np.array_equal(packed[0, :, 1:], packed[1, :, :-1]),
             'supplied packed bytes or teacher forcing differ')
    case = plan['cases'][case_index]
    ids, prefix = case['target_ids'], case['prefix']
    length = prefix['length']
    allowed = ('word', 'control') if suite == 'main' else ('word_next_native', 'shared_piece')
    _require(case.get('kind') in allowed and type(case.get('case_index')) is int
             and case['case_index'] == case_index
             and case.get('split') in ('training', 'test')
             and type(case.get('context_id')) is str and bool(case['context_id'])
             and type(case.get('target')) is str and bool(case['target'])
             and case.get('prefix_domain') in ('original', 'replacement', 'shared')
             and type(length) is int and length > 0 and type(ids) is list
             and len(ids) == (4 if case['kind'] == 'word_next_native' else 3)
             and all(type(token) is int and 0 <= token < config.vocab_size for token in ids)
             and type(target_position) is int and 0 <= target_position < len(ids),
             'invalid case identity or selected target position')
    rows = list(range(length-1, length-1+len(ids)))
    _require(all(type(row) is int for row in case['scored_rows'])
             and case['scored_rows'] == rows and rows[-1] < config.context_length
             and packed[1, case_index, rows].tolist() == ids,
             'selected prediction row or target geometry differs')
    original = packed[0, case_index, :length].tobytes()
    _require(hashlib.sha256(original).hexdigest() == prefix['token_ids_sha256'],
             'original case prefix hash differs')
    row = rows[target_position]
    tokens = packed[0, case_index, :row+1].copy()
    _require(tokens[length:].tolist() == ids[:target_position],
             'causal prefix does not contain exactly the preceding targets')
    result = dict(format=FORMAT, config=asdict(config), token_dtype='<i4',
        token_ids=tokens.tolist(), target_id=ids[target_position], selected_row=row,
        identity=dict(suite=suite, case_index=case_index, target_position=target_position,
            kind=case['kind'], split=case['split'], context_id=case['context_id'],
            target=case['target'], prefix_domain=case['prefix_domain'],
            original_prefix_length=length, original_prefix_sha256=prefix['token_ids_sha256'],
            causal_prefix_length=row+1, causal_prefix_sha256=hashlib.sha256(tokens.tobytes()).hexdigest()),
        cases=cases['record'], packed_batch=cases['batch'], implementation=list(SOURCE_RECORDS),
        caller_authenticated_case_provenance=True, generated_sampling_event=False,
        native_execution_performed=False)
    _verify([*SOURCE_RECORDS, cases['record'], cases['batch']])
    return result


def _check_output(output, protected):
    _require(output.parent.resolve(strict=True) == output.parent
             and not output.exists() and not output.is_symlink(), 'output must be a NEW canonical directory')
    for directory in protected:
        directory = _path(directory)
        _require(output != directory and directory not in output.parents
                 and output not in directory.parents, 'output overlaps protected inputs')
    # A caller may supply additional protected roots. Also reject any ordinary
    # Pluto checkpoint ancestor even if it was not among the case-file inputs.
    for parent in output.parents:
        _require(not (parent/'weight_0.bin').exists() and not (parent/'weight_0.bin').is_symlink(),
                 'output must not be inside a checkpoint')


def write_case_prefix(cases, case_index, target_position, output, config=GPT2Config(), *,
                      forbidden_directories=()):
    """Exclusively export ``prefix.i32`` and ``binding.json`` into a NEW directory.

    Pass any broader input/checkpoint roots in ``forbidden_directories``.
    The case/batch directories and detected checkpoint ancestors are always
    protected. Failure preserves partial output; nothing is removed or retried.
    """
    output = Path(output).absolute()
    _check_output(output, [Path(cases[key]['path']).parent for key in ('record', 'batch')]
                  + list(forbidden_directories))
    plan = bind_case(cases, case_index, target_position, config)
    output.mkdir()
    with (output/'prefix.i32').open('xb') as stream:
        stream.write(np.asarray(plan['token_ids'], dtype='<i4').tobytes())
    plan = dict(plan, prefix=history.file_record(output/'prefix.i32'))
    core.training.publish(output/'binding.json', plan)
    return read_binding(history.file_record(output/'binding.json'), config)


def read_binding(record, config=GPT2Config()):
    """Reload an export by its pinned binding-file record, checking source bytes.

    Source provenance remains the original caller's responsibility; reconstructing
    geometry from pinned files is not a substitute for the upstream case audit.
    """
    plan = history._json_record(record)
    directory = _path(Path(record['path']).parent)
    _require(Path(record['path']).name == 'binding.json'
             and {path.name for path in directory.iterdir()} == {'binding.json', 'prefix.i32'}
             and plan.get('prefix') == history.file_record(directory/'prefix.i32'),
             'prefix export inventory or binding differs')
    source = history._json_record(plan['cases'])
    packed = np.fromfile(plan['packed_batch']['path'], dtype='<i4').reshape(
        2, source['case_count'], config.context_length)
    cases = dict(record=plan['cases'], batch=plan['packed_batch'], plan=source, packed=packed)
    identity = plan['identity']
    expected = bind_case(cases, identity['case_index'], identity['target_position'], config)
    _require(plan == dict(expected, prefix=plan['prefix']), 'stored case binding differs from source case')
    _require((directory/'prefix.i32').read_bytes() == np.asarray(plan['token_ids'], dtype='<i4').tobytes(),
             'exported prefix differs from its causal case')
    return dict(record=record, plan=plan, prefix=plan['prefix'])


def _index(records):
    _require(type(records) is list, 'record inventory must be a list')
    result = {}
    for record in records:
        history._record_value(record)
        _require(record['path'] not in result, 'duplicate execution record')
        result[record['path']] = record
    _verify(records)
    return result


def _flags(command):
    _require(type(command) is list and command and all(type(x) is str and x for x in command),
             'invalid native command')
    result, i = {}, 1
    while i < len(command):
        item = command[i]
        _require(item.startswith('--'), 'unexpected native positional argument')
        if '=' in item:
            name, value = item[2:].split('=', 1)
        elif item == '--interventions':
            name, value = 'interventions', 'true'
        else:
            _require(i+1 < len(command), 'missing native flag value')
            name, value = item[2:], command[i+1]
            i += 1
        _require(name not in result, 'duplicate native flag')
        result[name] = value
        i += 1
    return result


def _execution(path, native_directory, metadata, binding, model, outputs, config):
    record = history.file_record(path)
    execution = history._json_record(record)
    _require(execution.get('format') == 'pluto-paired-probe-execution-v1'
             and type(execution.get('pid')) is int and execution['pid'] > 0
             and type(execution.get('returncode')) is int and execution['returncode'] == 0
             and execution.get('inputs_before') == execution.get('inputs_after'),
             'missing or failed native execution evidence')
    times = [datetime.fromisoformat(execution[k].replace('Z', '+00:00'))
             for k in ('started_utc', 'finished_utc')]
    _require(all(t.tzinfo is not None and t.utcoffset() == timezone.utc.utcoffset(t) for t in times)
             and times[0] <= times[1], 'native timestamps must be ordered UTC')
    inputs = _index(execution['inputs_before'])
    actual_outputs = _index(execution['outputs'])
    _require(actual_outputs == {r['path']: r for r in outputs}, 'native execution output inventory differs')
    flags = _flags(execution['command'])
    required_flags = dict(checkpoint=model['paths']['patched'], tokens_file=binding['prefix']['path'],
                          target_id=str(binding['plan']['target_id']), output_dir=str(native_directory))
    _require(set(required_flags) <= set(flags)
             and set(flags) <= set(required_flags) | {'interventions', 'ablate_neuron'}
             and all(flags[key] == value for key, value in required_flags.items()),
             'trace native command differs from the bound case/model')
    binary = history.file_record(metadata['binary_file'])
    _require(execution['command'][0] == binary['path'], 'native trace binary differs')
    standard = flags.get('interventions', 'false')
    _require(standard in ('true', 'false'), 'use explicit true/false intervention flag')
    neurons = []
    for item in filter(None, flags.get('ablate_neuron', '').split(',')):
        pieces = item.split(':')
        _require(len(pieces) == 2 and all(p.isdecimal() for p in pieces), 'invalid neuron command')
        neurons.append(tuple(map(int, pieces)))
    _require(len(neurons) == len(set(neurons)), 'duplicate requested neuron')
    arms = metadata['interventions']
    recorded_neurons = {(arm['block'], arm['head_or_neuron']) for arm in arms if arm['kind'] == 'mlp_neuron'}
    _require(recorded_neurons == set(neurons)
             and any(arm['kind'] != 'mlp_neuron' for arm in arms) == (standard == 'true'),
             'trace intervention inventory differs from command')
    required = [binding['record'], binding['prefix'], binding['plan']['cases'],
                binding['plan']['packed_batch'], binary, *model['weight_records']['patched']]
    if 'patch' in model:
        required.append(model['patch'])
    _require(all(inputs.get(r['path']) == r for r in required),
             'native execution omits required case/model input')
    log = history.file_record(execution['log']['path'])
    _require(execution['log'] == log, 'native process log differs')
    return [record, log, *inputs.values(), *actual_outputs.values()]


def validate_trace(native_directory, binding, model, execution_path, reference, config=GPT2Config()):
    """Validate one saved trace against a previously authenticated native case.

    ``model`` is the native-loader patched-model contract. ``reference`` is
    the complete ``core.load_native`` result for that same model/case suite,
    not a user-supplied scalar loss. Its ledger and arrays are reread. All dump
    files, replay controls, causal coordinates, and execution-time hashes are
    checked. FP64 log-sum-exp agrees within the existing fixed FP32 NLL bound;
    argmax agrees exactly. This is not an assertion of full-logit byte parity
    with the loss probe, which did not export full logits.

    Returns native stages, physical logits, diagnostic lens/intervention logits,
    the bound case, checked reference NLL/argmax, and frozen file records. No
    interpreter or report output is written, and no native process is launched.
    """
    native_directory = _path(native_directory)
    checked_binding = read_binding(binding['record'], config)
    _require(checked_binding == binding, 'supplied prefix binding changed')
    plan = binding['plan']
    model_path = _path(model['paths']['patched'])
    for protected in (model_path, Path(binding['record']['path']).parent,
                      Path(plan['cases']['path']).parent, Path(plan['packed_batch']['path']).parent):
        _require(native_directory != protected and protected not in native_directory.parents
                 and native_directory not in protected.parents, 'native output overlaps protected inputs')
    _verify(model['records'])
    records = [*model['records'], *SOURCE_RECORDS, binding['record'], binding['prefix'],
               plan['cases'], plan['packed_batch']]
    weight_records = core.branch._weights(model_path, model['hashes']['patched'], {}, config)
    _require(weight_records == model['weight_records']['patched'], 'trace checkpoint records differ')
    metadata_record = history.file_record(native_directory/'metadata.json')
    metadata = history._json_record(metadata_record)
    ids = plan['token_ids']
    expected = dict(schema_version=1, probe_kind='token_trace', complete=True,
        token_ids=ids, prompt_rows=len(ids), selected_row=plan['selected_row'], target_id=plan['target_id'],
        context_length=config.context_length, vocab_size=config.vocab_size,
        padded_vocab_size=config.padded_vocab_size, no_bos=True, autoregressive=False, retokenized=False,
        pad_token_id=ids[-1], alternate_pad_token_id=1 if ids[-1] == 0 else 0,
        checkpoint_directory=str(model_path), tokens_file=binding['prefix']['path'],
        checkpoint_unique_weight_count=len(checkpoint.tensor_manifest(config)),
        checkpoint_raw_weight_count=len(checkpoint.tensor_manifest(config))+1,
        byte_order='little', optimizer_steps=0, backward_calls=0, checkpoint_writes=0,
        original_forward='unmodified CreateGpt2', logit_parity_scope='selected_row', residual_parity_scope='full_prefix')
    _require(all(metadata.get(k) == v and type(metadata[k]) is type(v) for k, v in expected.items()),
             'trace metadata differs from bound case/checkpoint/geometry')
    _require(all(type(token) is int for token in metadata['token_ids']), 'native token IDs must be integers')
    checks = ('final_lens_logits_byte_equal', 'alternate_padding_logits_byte_equal',
              'clean_replay_logits_byte_equal', 'attention_residual_replay_byte_equal',
              'mlp_residual_replay_byte_equal', 'checkpoint_file_stats_unchanged',
              'token_file_stats_unchanged', 'all_exported_activations_finite')
    _require(all(metadata.get('checks', {}).get(k) is True for k in checks), 'native replay checks failed')
    evidence = {metadata_record['path']: metadata_record}

    def load(record, dtype, shape):
        _require(record.get('dtype') == dtype and record.get('shape') == shape, 'native tensor dtype/shape differs')
        values, identity = trace.load_native_array(native_directory, record)
        _require(history.file_record(identity['path']) == identity
                 and identity['path'] not in evidence, 'duplicate or changed native tensor')
        evidence[identity['path']] = identity
        return values

    widths = trace.stage_widths(config)
    _require(set(metadata['files']) == set(widths) | {'tokens', 'logits'}, 'incomplete native stage inventory')
    stages = {}
    for name, width in widths.items():
        descriptor = metadata['files'][name]
        replay = name == 'embedding' or name.endswith(('.attention_projected', '.mlp_projected'))
        _require(descriptor.get('source') == ('native_replay_saved_input' if replay else 'native_original_forward')
                 and descriptor.get('role') == 'full_prefix', 'native stage provenance/role differs')
        stages[name] = load(descriptor, 'bf16', [len(ids), width])
    tokens = load(metadata['files']['tokens'], 'int32', [len(ids), 1])
    _require(metadata['files']['tokens'].get('source') == 'exact_input_token_ids'
             and metadata['files']['tokens'].get('role') == 'full_prefix', 'native token descriptor differs')
    _require(tokens[:, 0].tolist() == ids, 'native token dump differs from causal prefix')
    logits = load(metadata['files']['logits'], 'float32', [1, config.padded_vocab_size])
    _require(metadata['files']['logits'].get('source') == 'native_original_forward'
             and metadata['files']['logits'].get('role') == 'selected_row', 'native logit descriptor differs')
    _require(set(metadata['parity_files']) == {'alternate_padding', 'clean_replay'}, 'incomplete parity inventory')
    for filename in metadata['parity_files'].values():
        values = load(dict(file=filename, dtype='float32', shape=list(logits.shape)), 'float32', list(logits.shape))
        _require(values.tobytes() == logits.tobytes(), 'native selected-row replay bytes differ')
    for block in range(config.n_layers):
        before = stages['positioned' if block == 0 else f'blocks.{block-1}.after_mlp']
        for branch, after in (('attention_projected', 'after_attention'), ('mlp_projected', 'after_mlp')):
            expected_after = rounding.bf16(np.add(before, stages[f'blocks.{block}.{branch}'], dtype=np.float32))
            before = stages[f'blocks.{block}.{after}']
            _require(expected_after.tobytes() == before.tobytes(), 'BF16 residual replay differs')
    names = trace.residual_stage_names(config.n_layers)
    _require(set(metadata['lens']) == set(names), 'incomplete native lens inventory')
    lens = {}
    for name in names:
        _require(metadata['lens'][name].get('source') == 'native_diagnostic_final_norm_and_tied_head'
                 and metadata['lens'][name].get('role') == 'selected_row', 'native lens descriptor differs')
        lens[name] = load(metadata['lens'][name], 'float32', list(logits.shape))
    _require(lens[names[-1]].tobytes() == logits.tobytes(), 'final lens differs from original logits')
    trace.validate_intervention_arms(metadata['interventions'], 1, config)
    interventions = {}
    for arm in metadata['interventions']:
        _require(arm.get('role') == 'selected_row', 'intervention role differs')
        interventions[arm['name']] = load(dict(file=arm['logits_file'], dtype='float32', shape=arm['shape']),
                                           'float32', list(logits.shape))
    entries = list(native_directory.iterdir())
    _require(all(path.is_file() and not path.is_symlink() for path in entries)
             and {str(path) for path in entries} == set(evidence), 'native output inventory differs')
    records.extend(_execution(execution_path, native_directory, metadata, binding, model, list(evidence.values()), config))

    # Re-read native loss-probe evidence rather than trusting an in-memory
    # scalar or silently comparing a different model, prefix, or target.
    _verify(reference['records'])
    _require(reference['cases']['record'] == plan['cases']
             and reference['cases']['batch'] == plan['packed_batch'], 'reference case suite differs')
    outputs = reference['outputs']
    paths = {Path(r['path']).name: Path(r['path']) for r in outputs}
    _require(set(paths) == {'metadata.json', 'losses.f32.bin', 'argmax.i32.bin'}
             and len({path.parent for path in paths.values()}) == 1, 'reference output inventory differs')
    reference_records = {}
    for record in reference['records']:
        core.native._register(reference_records, record['path'], record)
    adapter = dict(paths={k: Path(v) for k, v in model['paths'].items()},
                   hashes=model['hashes'], weight_records=model['weight_records'])
    reread = core.branch._scores(paths['metadata.json'].parent, reference['execution']['path'],
        reference['cases'], 'patched', adapter, reference_records, config)
    _require(reread['execution'] == reference['execution'] and reread['outputs'] == outputs
             and core.native._bits_equal(reread['losses'], reference['losses'])
             and core.native._bits_equal(reread['argmax'], reference['argmax']), 'reference arrays or execution differ')
    records.extend(reference_records.values())
    i, row, target = plan['identity']['case_index'], plan['selected_row'], plan['target_id']
    native_nll, native_argmax = float(reread['losses'][i, row]), int(reread['argmax'][i, row])
    logical = logits[0, :config.vocab_size].astype(np.float64)
    winner = int(np.argmax(logical))
    rank = 1+int(np.count_nonzero(logical > logical[target]))+int(np.count_nonzero(logical[:target] == logical[target]))
    maximum = float(np.max(logical))
    # Subtract the target before adding log(sum(exp)); adding the huge maximum
    # first can erase the entire loss when logits share a large common offset.
    nll = (maximum-float(logical[target]))+math.log(math.fsum(math.exp(float(x)-maximum) for x in logical))
    error = abs(nll-native_nll)
    _require(error <= core.FP32_ABSOLUTE_TOLERANCE+core.FP32_RELATIVE_TOLERANCE*abs(nll)
             and winner == native_argmax, 'trace logits disagree with native reference loss/argmax')
    _require(type(metadata.get('greedy_id')) is int and metadata['greedy_id'] == winner
             and type(metadata.get('target_rank')) is int and metadata['target_rank'] == rank
             and type(metadata.get('target_logit')) in (int, float)
             and metadata['target_logit'] == float(logical[target]), 'native selected-logit metadata differs')
    records = core.outer._unique([*records, *evidence.values()])
    _verify(records)
    return dict(binding=binding, stages=stages, logits=logits, lens=lens, interventions=interventions,
        reference=dict(native_nll=native_nll, native_argmax=native_argmax, fp64_nll=nll,
            absolute_nll_error=error, absolute_tolerance=core.FP32_ABSOLUTE_TOLERANCE,
            relative_tolerance=core.FP32_RELATIVE_TOLERANCE, full_logit_byte_parity_claimed=False),
        records=records, native_execution_performed=False, generated_sampling_event=False)
