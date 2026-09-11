"""Capture the eighteen fixed paired E/M discovery traces, serially.

This is a new, case-bound descriptive capture after the prior owner exits.
No training, checkpoint copying, sampling, neuron ranking, or ablation occurs.
Every prediction retains the existing packed case's absolute positions.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import stat
import sys

from . import native_runtime_bundle as runtime
from . import paired_case_trace as adapter
from . import paired_mlp_affine_run as predecessor

core = predecessor.core
history = predecessor.history
training = predecessor.training
native = predecessor.native
outer = predecessor.outer
screen = predecessor.screen
freeze_file = predecessor.previous_runner.execution.paired_training.freeze_file
FORMAT = 'pluto-paired-case-trace-run-v1'
CELLS = ('E', 'M')
_require = native._require
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'ai-slop/research/weight_memorization/EXEUNT_PAIRED_CASE_TRACE_PROTOCOL.md'
TEST_NAMES = ('paired_case_trace_test.py', 'paired_case_trace_run_test.py')
IMPORTED_SOURCES = outer._unique([history.file_record(Path(__file__).resolve()), *[
    history.file_record(Path(module.__file__).resolve())
    for name, module in tuple(sys.modules.items())
    if name.startswith('weight_analysis.') and getattr(module, '__file__', '').endswith('.py')]])
DIRECT_NATIVE_SOURCES = [history.file_record(Path(__file__).with_name(name)) for name in (
    'token_trace_probe.cc', 'token_trace_probe_lib.cc', 'token_trace_probe.h',
    'phrase_probe_lib.cc', 'phrase_probe.h', 'causal_probe.cc', 'causal_probe.h',
    'token_trace_probe_test.cc', 'BUILD.bazel')]
BUILD_EVIDENCE = [history.file_record(ROOT/name) for name in ('.bazelrc', 'MODULE.bazel', 'MODULE.bazel.lock')]

_FIXED = (
    ('original_to_replacement', 'main', 0, (0, 1, 2), 'Exeunt', (1475, 68, 2797),
     'word', 'original', 'training:token_start:79444:title',
     '3652da5542c4dad6f987776c330d8318fff9fa3ce6c7d1f9756bdc3ab0766599'),
    ('replacement_to_original', 'main', 1, (0, 1, 2), 'Nuveth', (21733, 303, 400),
     'word', 'original', 'training:token_start:79444:title',
     '3652da5542c4dad6f987776c330d8318fff9fa3ce6c7d1f9756bdc3ab0766599'),
    ('original_to_replacement', 'main', 160, (1, 2), 'exeunt', (409, 68, 2797),
     'word', 'original', 'training:token_start:238155:lowercase',
     'cc7a302e808136d4af89ef597d53eaedd471909fc27785608f55cdcc0c314e67'),
    ('original_to_replacement', 'supplemental', 176, (0,), 'control_next_3', (303, 428, 11),
     'shared_piece', 'shared', 'training:piece:303:start:1023605',
     '06ce1ad462295458976de0d214d79e0c317e6eed0e89b826659fd215beb9a50b'),
)


def fixed_specs(handoff):
    """Validate the protocol's fixed cases and return position-major E/M pairs."""
    result = []
    for direction, suite, index, positions, target, ids, kind, domain, context, digest in _FIXED:
        for position in positions:
            paired = []
            for cell in CELLS:
                cases = handoff['references'][direction][cell]['measured'][suite]['cases']
                _require(cases['record'] == handoff['request']['cases'][suite]
                         and cases['plan']['cases'][index]['target_ids'] == list(ids),
                         'fixed discovery case suite/targets differ')
                bound = adapter.bind_case(cases, index, position)
                expected = dict(suite=suite, case_index=index, target_position=position,
                    target=target, kind=kind, split='training', prefix_domain=domain,
                    context_id=context, original_prefix_length=128,
                    original_prefix_sha256=digest, causal_prefix_length=128+position)
                _require(all(bound['identity'].get(k) == v for k, v in expected.items())
                         and bound['selected_row'] == 127+position
                         and bound['target_id'] == ids[position], 'fixed discovery case identity differs')
                paired.append(bound)
                result.append(dict(direction=direction, cell=cell, suite=suite, case_index=index,
                    target_position=position, target_id=ids[position], target=target, kind=kind,
                    prefix_domain=domain, context_id=context, prefix_sha256=digest, binding_plan=bound))
            _require(paired[0] == paired[1], 'paired cells have different causal cases')
    _require(len(result) == 18 and len({_name(spec) for spec in result}) == 18,
             'fixed trace selection is incomplete or duplicated')
    return result


def _name(spec):
    return (f"{spec['direction']}/{spec['suite']}_case_{spec['case_index']:03d}"
            f"_pos_{spec['target_position']}/{spec['cell']}")


def _verify(records):
    for record in outer._unique(records):
        _require(history.file_record(record['path']) == record, 'frozen capture input changed')


def _probe(value):
    """Accept only this experiment's caller-built optimized Bazel executable."""
    path = Path(value).resolve(strict=True)
    parts = path.parts
    positions = [i for i, value in enumerate(parts) if value == 'bazel-out']
    _require(len(positions) == 1, 'trace probe must resolve to bazel-out/<platform>-opt/bin')
    offset = positions[0]
    _require(len(parts) >= offset+6 and parts[offset+1].endswith('-opt')
             and parts[offset+2:] == ('bin', 'ai-slop', 'weight_analysis', 'token_trace_probe')
             and stat.S_ISREG(path.lstat().st_mode) and os.access(path, os.X_OK),
             'trace probe must be the executable optimized Bazel token_trace_probe')
    return history.file_record(path)


def _state(output, completed, phase, **details):
    training.publish(output/'state.json', dict(format=FORMAT, phase=phase,
        runner_pid=os.getpid(), updated_utc=training.now(), completed=list(completed),
        native_trace_count=18, **details), exclusive=False)


def _owned_execution(path, command, inputs, log, directory, binding):
    """Require this request's exact command/input membership and captured stdout."""
    record = history.file_record(path)
    execution = history._json_record(record)
    expected = {item['path']: item for item in outer._unique(inputs)}
    for key in ('inputs_before', 'inputs_after'):
        values = execution[key]
        _require(len(values) == len({item['path'] for item in values})
                 and {item['path']: item for item in values} == expected,
                 'capture execution has different owning input membership')
    log_record = history.file_record(log)
    _require(execution['command'] == command and execution['log'] == log_record
             and type(execution.get('returncode')) is int and execution['returncode'] == 0,
             'capture command, log, or successful execution differs')
    metadata = history._json_record(history.file_record(directory/'metadata.json'))
    expected_start = (f"Native token trace: prefix={len(binding['plan']['token_ids'])}, "
        f"target={binding['plan']['target_id']}, target rank={metadata['target_rank']}, greedy={metadata['greedy_id']}")
    lines = Path(log).read_text().splitlines()
    _require(lines.count(expected_start) == 1
             and lines.count('Complete: '+json.dumps(str(directory))) == 1,
             'native stdout lacks the exact bound trace/completion messages')
    entries = list(directory.parent.rglob('*'))
    expected_files = {item['path'] for item in execution['outputs']} | {
        record['path'], log_record['path'], binding['record']['path'], binding['prefix']['path']}
    _require(not any(entry.is_symlink() for entry in entries)
             and {str(entry) for entry in entries if entry.is_file()} == expected_files
             and {entry for entry in entries if entry.is_dir()} == {directory, directory.parent/'input'},
             'unexpected file or directory in trace capture')
    _verify([record, log_record])
    return [record, log_record]


def run(previous, summary_sha256, output, trace_probe):
    """Capture exactly eighteen NEW traces; stop at the first failed gate."""
    previous, output = Path(previous).absolute(), Path(output).absolute()
    _require(previous.resolve(strict=True) == previous and previous.is_dir()
             and output.parent == previous.parent and output.parent.resolve(strict=True) == output.parent
             and not output.exists() and not output.is_symlink(), 'output must be a NEW canonical sibling directory')
    for name in ('LD_PRELOAD', 'LD_AUDIT'):
        _require(name not in os.environ, 'unset inherited '+name)
    tests = [history.file_record(Path(__file__).with_name(name)) for name in TEST_NAMES]
    protocol = history.file_record(PROTOCOL)
    sources = outer._unique([*IMPORTED_SOURCES, *DIRECT_NATIVE_SOURCES, *BUILD_EVIDENCE, *tests, protocol])
    probe = _probe(trace_probe)
    handoff = predecessor.validate_handoff(previous, summary_sha256)
    specifications = fixed_specs(handoff)
    _verify([*sources, probe, *handoff['records']])
    plan = handoff['request']
    training.require_idle_gpu(plan['gpu'])
    output.mkdir()
    completed, reports = [], []
    old_library_path = os.environ.get('LD_LIBRARY_PATH')
    try:
        (output/'source').mkdir()
        copies = [freeze_file(record['path'], output/'source'/f"{i:03d}_{Path(record['path']).name}")
                  for i, record in enumerate(sources)]
        _require(all(all(source[key] == copy[key] for key in ('bytes', 'sha256'))
                     for source, copy in zip(sources, copies)), 'source snapshot bytes differ')
        (output/'bin').mkdir()
        executable = freeze_file(probe['path'], output/'bin/token_trace_probe', executable=True)
        copied_stat, original_stat = Path(executable['path']).stat(), Path(probe['path']).stat()
        _require(all(executable[key] == probe[key] for key in ('bytes', 'sha256'))
                 and copied_stat.st_nlink == 1
                 and (copied_stat.st_dev, copied_stat.st_ino) != (original_stat.st_dev, original_stat.st_ino),
                 'trace executable is not an independent exact copy')
        bundle = runtime.freeze_runtime(probe['path'], executable['path'], output/'runtime')
        _require(bundle.get('complete') is True and bundle.get('runtime_dlopen_covered') is False
                 and bundle['reference_executable'] == probe and bundle['relocated_executable'] == executable
                 and bundle['environment'] == {'LD_LIBRARY_PATH': str(output/'runtime')},
                 'unexpected trace runtime bundle')
        os.environ.update(bundle['environment'])
        frozen = outer._unique([*handoff['records'], *sources, *copies, probe, executable,
                                *bundle['frozen_records']])
        _verify(frozen)
        request = dict(format=FORMAT, created_utc=training.now(),
            runner_identity=training.process_identity(os.getpid()), config=asdict(core.GPT2Config()),
            previous_summary=handoff['summary_record'], source_archive=plan['source_archive'],
            cases=plan['cases'], specifications=specifications, cells=list(CELLS), native_trace_count=18,
            implementation=IMPORTED_SOURCES, direct_native_producer_sources=DIRECT_NATIVE_SOURCES,
            build_configuration_evidence=BUILD_EVIDENCE, test_sources=tests, protocol=protocol,
            source_snapshots=[dict(original=a, copy=b) for a, b in zip(sources, copies)],
            producer_source_scope='Direct probe/helpers and build configuration; not a full transitive Bazel compile closure.',
            binary_reference=probe, binary=executable, runtime=bundle, gpu=plan['gpu'], frozen_inputs=frozen,
            interventions=False, generated_sampling_event=False, training_restarted=False,
            checkpoint_mutations=False, goal_completion_claimed=False,
            selection='Training discovery: lowest word-case index; shared ve case selected from prior collateral results, not independently.')
        training.publish(output/'request.json', request)
        owner_record = history.file_record(output/'request.json')
        active = outer._unique([*frozen, owner_record])
        for spec in specifications:
            name = _name(spec)
            _state(output, completed, 'native_trace', trace=name)
            print(f'{training.now()} {name}', flush=True)
            reference = handoff['references'][spec['direction']][spec['cell']]
            model, measured = reference['model'], reference['measured'][spec['suite']]
            directory = output/'traces'/name
            directory.mkdir(parents=True)
            binding = adapter.write_case_prefix(measured['cases'], spec['case_index'], spec['target_position'],
                directory/'input', forbidden_directories=(previous, *model['paths'].values()))
            _require(binding['plan'] == dict(spec['binding_plan'], prefix=binding['prefix']),
                     'export differs from frozen protocol case')
            inputs = outer._unique([*active, binding['record'], binding['prefix'],
                measured['cases']['record'], measured['cases']['batch'], *model['records'],
                *model['weight_records']['patched'], model['patch']])
            command = [executable['path'], '--checkpoint='+model['paths']['patched'],
                '--tokens_file='+binding['prefix']['path'], '--target_id='+str(spec['target_id']),
                '--output_dir='+str(directory/'native'), '--interventions=false']
            _verify(inputs)
            screen.run_native(command, [record['path'] for record in inputs], directory/'native',
                directory/'process.log', directory/'execution.json', plan['gpu'])
            checked = adapter.validate_trace(directory/'native', binding, model,
                directory/'execution.json', measured)
            _require(checked['interventions'] == {} and checked['generated_sampling_event'] is False
                     and checked['native_execution_performed'] is False, 'adapter scope differs')
            owned = _owned_execution(directory/'execution.json', command, inputs,
                directory/'process.log', directory/'native', binding)
            _verify([*inputs, *checked['records'], *owned])
            report = dict(format=FORMAT, complete=True, trace=name, specification=spec,
                binding=binding, checkpoint=model['paths']['patched'], patch=model['patch'],
                reference=checked['reference'], files=outer._unique([*checked['records'], *owned]),
                native_execution_performed=True, generated_sampling_event=False,
                interventions=False, checkpoint_mutations=False, goal_completion_claimed=False)
            training.publish(directory/'readout.json', report)
            report_record = history.file_record(directory/'readout.json')
            _require(history._json_record(report_record) == report, 'published trace report differs')
            evidence = screen.stage_result(directory, dict(format=FORMAT, complete=True,
                trace=name, case_reference_parity_certified=True, readout=report_record))
            active = outer._unique([*active, *evidence])
            _verify(active)
            reports.append(report_record)
            completed.append(name)
        expected = [_name(spec) for spec in specifications]
        _require(completed == expected and len(set(completed)) == 18 and len(reports) == 18,
                 'incomplete fixed capture coverage')
        _verify(active)
        _state(output, completed, 'complete')
        entries = sorted(output.rglob('*'))
        _require(not any(path.is_symlink() for path in entries), 'symlink in capture artifacts')
        artifacts = [history.file_record(path) for path in entries if path.is_file()]
        expected_artifacts = {item['path'] for item in active if Path(item['path']).is_relative_to(output)}
        _require({item['path'] for item in artifacts} == expected_artifacts | {str(output/'state.json')},
                 'unexpected file in final capture inventory')
        training.publish(output/'summary.json', dict(format=FORMAT, complete=True,
            completed_utc=training.now(), completed=completed, native_trace_count=18,
            case_reference_parity_certified=True, previous_summary=handoff['summary_record'],
            source_archive=plan['source_archive'], request=owner_record, reports=reports,
            artifacts=artifacts, frozen_inputs=outer._unique([*frozen, owner_record]),
            training_restarted=False, checkpoint_mutations=False, interventions=False,
            generated_sampling_event=False, goal_completion_claimed=False))
    except BaseException as error:
        training.publish(output/'failure.json', dict(format=FORMAT, failed_utc=training.now(),
            error_type=type(error).__name__, error=str(error), completed=completed,
            no_automatic_restart=True, training_restarted=False, checkpoint_mutations=False,
            goal_completion_claimed=False))
        raise
    finally:
        if old_library_path is None:
            os.environ.pop('LD_LIBRARY_PATH', None)
        else:
            os.environ['LD_LIBRARY_PATH'] = old_library_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-probe', type=Path, required=True)
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'case-trace runner received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args.previous, args.summary_sha256, args.output, args.trace_probe)


if __name__ == '__main__':
    main()
