"""CPU-only snapshots of completed leaves of a live early-branch experiment.

This reader never calls runner handoff, process-liveness, or GPU-idle gates.
Missing cells stay missing. A snapshot does not certify controller completion.
Historical identities are resolved explicitly; old ledgers are never rewritten.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import math
from pathlib import Path
import sys

from . import historical_source_archive as history
from . import paired_complement_partial_readout as previous_partial
from . import paired_early_branches as runner

core = runner.core
native = core.native
training = core.training
FORMAT = 'pluto-paired-early-branches-partial-readout-v1'
READER_SOURCE = history.file_record(Path(__file__).resolve())
READER_SOURCES = core.outer._unique([READER_SOURCE, *[
    history.file_record(Path(module.__file__).resolve())
    for name, module in tuple(sys.modules.items())
    if name.startswith('weight_analysis.') and getattr(module, '__file__', '').endswith('.py')]])
_require = native._require


def _contrasts(values):
    formulas = {
        'A-E': {'A': 1, 'E': -1}, 'M-E': {'M': 1, 'E': -1},
        'AM-E': {'AM': 1, 'E': -1}, 'AM-A': {'AM': 1, 'A': -1},
        'AM-M': {'AM': 1, 'M': -1}, 'AR-R': {'AR': 1, 'R': -1},
        'MR-R': {'MR': 1, 'R': -1}, 'EC-MR': {'EC': 1, 'MR': -1},
        'EC-AR': {'EC': 1, 'AR': -1},
        'AM-A-M+E': {'AM': 1, 'A': -1, 'M': -1, 'E': 1},
        'EC-AR-MR+R': {'EC': 1, 'AR': -1, 'MR': -1, 'R': 1},
        'three_way': {'EC': 1, 'AR': -1, 'MR': -1, 'R': 1,
                      'AM': -1, 'A': 1, 'M': 1, 'E': -1},
    }
    return {name: math.fsum(values[cell]*sign for cell, sign in terms.items())
            for name, terms in formulas.items() if set(terms) <= set(values)}


def aggregate(per_case, available_cells):
    """Per-stratum causal-event means and argmax conjunctions, never imputed."""
    cells = tuple(available_cells)
    _require(all(type(cell) is str for cell in cells) and len(cells) == len(set(cells))
             and {'E', 'EC'} <= set(cells)
             and set(cells) <= set(core.cube_subsets(layout=runner.LAYOUT)),
             'invalid partial-cell coverage')
    fields = ('suite', 'kind', 'split', 'spelling_variant', 'prefix_domain',
              'word_has_leading_space', 'piece_id', 'target')
    grouped, predictions, winners = defaultdict(list), {}, {}
    for item in per_case:
        ids = item['target_ids']
        kind = item['kind']
        _require(kind in ('word', 'word_next_native', 'control', 'shared_piece')
                 and item['suite'] == ('supplemental' if kind in ('word_next_native', 'shared_piece') else 'main')
                 and len(ids) == (4 if kind == 'word_next_native' else 3)
                 and all(type(value) is int and value >= 0 for value in ids)
                 and item['split'] in ('training', 'test')
                 and item['prefix_domain'] in ('original', 'replacement', 'shared')
                 and type(item['prefix_length']) is int and item['prefix_length'] > 0
                 and set(item['cells']) == set(cells), 'invalid case identity or partial-cell coverage')
        for cell in cells:
            values = item['cells'][cell]['token_log_probability']
            argmax = item['cells'][cell]['argmax_ids']
            _require(len(values) == len(argmax) == len(ids)
                     and all(type(value) in (float, int) and math.isfinite(value) and value <= 0 for value in values)
                     and all(type(value) is int and value >= 0 for value in argmax),
                     'invalid selected scores or argmax IDs')
            for position in range(len(ids)):
                prefix = (cell, item['prefix_sha256'], item['prefix_length'], tuple(ids[:position]))
                event = (*prefix, ids[position])
                _require(predictions.setdefault(event, values[position]) == values[position]
                         and winners.setdefault(prefix, argmax[position]) == argmax[position],
                         'duplicate causal event has conflicting scores or argmax')
        grouped[tuple(item.get(field) for field in fields)].append(item)
    result = []
    for key, items in sorted(grouped.items(), key=lambda pair: repr(pair[0])):
        metrics = {}
        for name, positions in core._positions(items[0]['kind'], len(items[0]['target_ids'])).items():
            unique = {}
            for item in items:
                ids = item['target_ids']
                identity = (item['prefix_sha256'], item['prefix_length'], tuple(ids[:max(positions)+1]))
                values = {cell: (math.fsum(item['cells'][cell]['token_log_probability'][p] for p in positions),
                                 tuple(item['cells'][cell]['argmax_ids'][p] for p in positions),
                                 all(item['cells'][cell]['argmax_ids'][p] == ids[p] for p in positions))
                          for cell in cells}
                _require(unique.setdefault(identity, values) == values,
                         'duplicate causal event has conflicting metric values')
            count = len(unique)
            means = {cell: math.fsum(value[cell][0] for value in unique.values())/count for cell in cells}
            counts = {cell: sum(value[cell][2] for value in unique.values()) for cell in cells}
            metric = dict(unique_event_count=count, alias_case_count=len(items),
                cells={cell: dict(mean_log_probability=means[cell],
                    geometric_mean_probability=math.exp(means[cell]),
                    joint_argmax_count=counts[cell], joint_argmax_rate=counts[cell]/count) for cell in cells},
                delta_log_probability_from_E={cell: value-means['E'] for cell, value in means.items()},
                gap_to_EC_log_probability={cell: means['EC']-value for cell, value in means.items()},
                available_contrasts=_contrasts(means))
            if len(cells) == 8:
                metric['effects'] = core.score_cube(means, layout=runner.LAYOUT)
            metrics[name] = metric
        result.append(dict(zip(fields, key), metrics=metrics))
    return result


def _records(values):
    _require(type(values) is list, 'file inventory must be a list')
    records = [history._record_value(value) for value in values]
    _require(len({record['path'] for record in records}) == len(records), 'duplicate file inventory path')
    return records


def _read_json(path, expected=None):
    record = history.file_record(Path(path).absolute())
    _require(expected is None or record == expected, 'JSON differs from pinned record')
    return history._json_record(record), record


def _member(index, record, message='required frozen record is absent or changed'):
    _require(index.get(record['path']) == record, message)


def validate_request(directory, direction):
    """Authenticate inline early-run provenance once, without a handoff replay."""
    directory = Path(directory).absolute()
    _require(directory.resolve(strict=True) == directory and directory.is_dir(), 'invalid run directory')
    request, request_record = _read_json(directory/'request.json')
    cells = {key: list(value) for key, value in core.cube_subsets(layout=runner.LAYOUT).items()}
    _require(request.get('format') == runner.FORMAT and request.get('layout') == runner.LAYOUT
             and request.get('config') == asdict(core.GPT2Config())
             and request.get('groups') == core.groups(layout=runner.LAYOUT)
             and request.get('cells') == cells and request.get('inherited_cells') == runner.ANCHORS
             and request.get('cell_order') == list(runner.CELL_ORDER)
             and request.get('suites') == list(runner.SUITES)
             and type(request.get('native_measurement_count')) is int and request['native_measurement_count'] == 32
             and request.get('training_restarted') is False and request.get('goal_completion_claimed') is False
             and set(request['directions']) == set(runner.DIRECTIONS) and direction in runner.DIRECTIONS
             and set(request['cases']) == set(runner.SUITES), 'wrong early-run request/scope')
    frozen = _records(request['frozen_inputs'])
    index = {record['path']: record for record in frozen}
    for record in (request['source_archive'], request['previous_summary']):
        _member(index, record)
    previous = Path(request['previous_summary']['path']).parent
    _require(previous.parent == directory.parent and previous != directory
             and request['previous_summary']['path'] == str(previous/'summary.json'), 'invalid historical run path')
    summary, summary_record = _read_json(previous/'summary.json', request['previous_summary'])
    old_frozen = {record['path']: record for record in _records(summary['frozen_inputs'])}
    owner = old_frozen.get(str(previous/'request.json'))
    _require(owner is not None, 'historical summary omits its owning request')
    _member(index, owner)
    archive = history.SourceArchive(request['source_archive']['path'],
        expected_manifest_record=request['source_archive'], owning_request_record=owner)
    old_request, _ = _read_json(owner['path'], owner)
    plan, plan_record = _read_json(old_request['plan']['path'], old_request['plan'])
    _member(old_frozen, plan_record, 'historical summary omits its plan')
    _member(index, plan_record)
    _require(summary.get('format') == runner.OLD_FORMAT and summary.get('complete') is True
             and summary.get('baseline_controls_certified') is True
             and summary.get('native_measurement_count') == 40
             and summary.get('plan') == plan_record and old_request.get('format') == runner.OLD_FORMAT
             and plan.get('format') == core.FORMAT and plan.get('phase') == 'planned'
             and plan.get('config') == request['config'] and plan.get('step') == 331
             and plan.get('amendment_root') == str(directory.parent)
             and plan.get('groups') == core.groups()
             and plan.get('cells') == {key: list(value) for key, value in core.cube_subsets().items()}
             and plan['loss_tolerance']['absolute'] == core.FP32_ABSOLUTE_TOLERANCE
             and plan['loss_tolerance']['relative'] == core.FP32_RELATIVE_TOLERANCE,
             'historical fixed cube differs')
    for key in ('directions', 'cases', 'binaries', 'runtime', 'gpu'):
        _require(request[key] == plan[key], 'early/historical scope binding differs: '+key)
    mapping, mapping_record = _read_json(directory/'historical_sources.json')
    _member(index, mapping_record)
    _require(mapping.get('format') == runner.FORMAT
             and mapping.get('source_archive') == request['source_archive']
             and mapping.get('previous_summary') == summary_record
             and mapping.get('original_identities_rewritten') is False,
             'historical source provenance differs')
    originals = _records([binding['original'] for binding in mapping['bindings']])
    bindings = archive.verify(originals)
    _require(bindings == mapping['bindings'], 'historical physical mapping differs')
    logical = {record['path']: record for record in originals}
    for record in [*summary['artifacts'], *summary['frozen_inputs'], summary_record, request['source_archive']]:
        _member(logical, record, 'historical source map omitted predecessor evidence')
    for binding in bindings:
        _member(index, binding['physical'])
    source_records = _records([*request['implementation'], *request['test_sources'], request['protocol']])
    implementation = {record['path']: record for record in request['implementation']}
    for module in (runner, core, history, native, core.branch, core.patcher, core.outer,
                   runner.screen, runner.execution, training, core.branch.case_contract):
        _member(implementation, history.file_record(Path(module.__file__).resolve()),
                'executing implementation differs from frozen runner')
    _require({Path(record['path']).name for record in request['test_sources']} == set(runner.TEST_NAMES)
             and request['protocol']['path'] == str(runner.PROTOCOL_PATH), 'runner test/protocol provenance differs')
    _require(len({Path(record['path']).name for record in source_records}) == len(source_records),
             'source snapshot basenames collide')
    snapshots = []
    for record in source_records:
        _member(index, record)
        copy = dict(record, path=str(directory/'source'/Path(record['path']).name))
        _member(index, copy, 'frozen source copy differs')
        snapshots.append(copy)
    _require((directory/'source').resolve(strict=True) == directory/'source'
             and {str(path) for path in (directory/'source').iterdir()} == {r['path'] for r in snapshots},
             'source snapshot inventory differs')
    for record in [*request['binaries'].values(), *request['runtime']['frozen_records'], *request['cases'].values()]:
        _member(index, record)
    identity = request['runner_identity']
    argv = identity['argv']
    expected_flags = {'-m': 'weight_analysis.paired_early_branches', '--previous': str(previous),
        '--source-archive': request['source_archive']['path'], '--summary-sha256': summary_record['sha256'],
        '--archive-sha256': request['source_archive']['sha256'], '--output': str(directory)}
    _require(type(identity['pid']) is int and identity['pid'] > 0
             and type(identity['start_ticks']) is int and identity['start_ticks'] > 0
             and type(argv) is list and all(type(value) is str for value in argv)
             and all(argv.count(flag) == 1 and argv.index(flag)+1 < len(argv)
                     and argv[argv.index(flag)+1] == value for flag, value in expected_flags.items()),
             'recorded early-run invocation differs')
    records = core.outer._unique([*frozen, request_record])
    training.verify_records(records)
    return dict(request=request, request_record=request_record, plan=plan, archive=archive,
                previous=previous, summary=summary, logical=logical, records=records)


def _execution_binding(measured, directory, request_record, models_record, request):
    """Keep current native ledgers bound to their owner and exact process log."""
    directory = Path(directory)
    execution, record = _read_json(directory/'execution.json')
    _require(record in measured['records'], 'native execution missing from reader ledger')
    _require(execution.get('format') == 'pluto-paired-probe-execution-v1'
             and type(execution.get('returncode')) is int and execution['returncode'] == 0
             and execution['command'][0] == request['binaries']['loss_probe']['path']
             and execution['inputs_before'] == execution['inputs_after'], 'wrong or unsuccessful native execution')
    before = {value['path']: value for value in _records(execution['inputs_before'])}
    for expected in [request_record, models_record, *request['frozen_inputs']]:
        _member(before, expected, 'native execution omitted owning-run input')
    log = history.file_record(directory/'process.log')
    _require(execution['log'] == log, 'native process-log binding differs')
    return [record, log]


def _anchor_controls(cell, loaded, preserved, marker, direction):
    expected = {}
    for suite in runner.SUITES:
        value = dict(previous_cell=runner.ANCHORS[cell],
                     exact_native_parity=core.assert_native_copy_parity(preserved[suite], loaded[suite]))
        if cell in ('E', 'EC'):
            names = ('main',) if suite == 'main' else ('word_next_native', 'shared_piece')
            value['fp64_reference'] = {name: core.compare_fp64_reference(loaded[suite],
                history._json_record(direction['references'][cell][name]), 'JJ') for name in names}
        expected[suite] = value
    _require(marker.get('anchor_controls') == expected, 'saved anchor controls differ from recomputed controls')
    return expected


def analyze(run_directory, direction, output):
    """Publish one NEW sibling snapshot; leave every live-run file unchanged."""
    directory, output = Path(run_directory).absolute(), Path(output).absolute()
    _require(not output.exists() and not output.is_symlink(), 'output must be a NEW file')
    _require(directory not in output.parents and output != directory, 'output must be outside the live run')
    _require(output.parent == directory.parent and output.parent.resolve(strict=True) == output.parent,
             'partial output must be a new sibling file')
    sources = core.outer._unique([*READER_SOURCES, *[history.file_record(Path(__file__).with_name(name))
        for name in ('paired_early_branches_partial_test.py', 'paired_early_branches_partial_numeric_test.py')]])
    training.verify_records(sources)
    handoff = validate_request(directory, direction)
    training.verify_records(sources)
    request, archive = handoff['request'], handoff['archive']
    stage = directory/direction
    available = [cell for cell in runner.CELL_ORDER if (stage/'cells'/cell/'complete.json').exists()
                 or (stage/'cells'/cell/'complete.json').is_symlink()]
    _require({'E', 'EC'} <= set(available), 'both completed E and EC anchors are required')
    models, models_record = _read_json(stage/'models.json')
    _require(set(models) == set(request['cells']), 'incomplete declared model map')
    previous_models, previous_models_record = _read_json(handoff['previous']/direction/'models.json')
    _member(handoff['logical'], previous_models_record, 'historical models absent from authenticated evidence')
    _require(set(previous_models['cube']) == set(core.cube_subsets()), 'incomplete historical model map')
    records = [*handoff['records'], *sources, models_record, previous_models_record]
    loaded, controls = {}, {}
    direction_plan = request['directions'][direction]
    for cell in available:
        cell_dir = stage/'cells'/cell
        _require(cell_dir.resolve(strict=True) == cell_dir, 'symlink in completed-cell ancestors')
        marker, evidence = previous_partial.completed_leaf(cell_dir)
        expected_scores = {suite: str(cell_dir/suite/'scores') for suite in runner.SUITES}
        expected_ledgers = {suite: str(cell_dir/suite/'execution.json') for suite in runner.SUITES}
        _require(marker.get('cell') == cell and marker.get('groups') == request['cells'][cell]
                 and marker.get('scores') == expected_scores and marker.get('executions') == expected_ledgers
                 and marker.get('first_three_cross_suite_byte_equal') is True
                 and (cell in runner.ANCHORS or marker.get('anchor_controls') == {}),
                 'completed-cell identity, suites, or selection differs')
        records.extend(evidence)
        model = core.validate_model(cell_dir/'step_331/patch.json', direction_plan['recipient']['path'],
            direction_plan['donor']['path'], request['cells'][cell], direction_plan['rows'], layout=runner.LAYOUT)
        _require(model == models[cell], 'cell model differs from declared construction')
        loaded[cell] = {}
        for suite in runner.SUITES:
            measured = core.load_native(model, request['cases'][suite]['path'], suite,
                expected_scores[suite], expected_ledgers[suite], case_archive=archive)
            records.extend(measured['records'])
            records.extend(_execution_binding(measured, cell_dir/suite,
                handoff['request_record'], models_record, request))
            loaded[cell][suite] = measured
        core._cross_suite(loaded[cell]['main'], loaded[cell]['supplemental'])
        if cell in runner.ANCHORS:
            old_cell = runner.ANCHORS[cell]
            reference = runner.historical_model(model, previous_models['cube'][old_cell], archive)
            preserved = {}
            for suite in runner.SUITES:
                old_dir = handoff['previous']/direction/'cells'/old_cell/suite
                measured = core.load_native(reference, request['cases'][suite]['path'], suite,
                    old_dir/'scores', old_dir/'execution.json', case_archive=archive, execution_archive=archive)
                records.extend(measured['records'])
                preserved[suite] = measured
            core._cross_suite(preserved['main'], preserved['supplemental'])
            controls[cell] = _anchor_controls(cell, loaded[cell], preserved, marker, direction_plan)
    per_case = previous_partial.selected_cases(loaded)
    groups = aggregate(per_case, available)
    records = core.outer._unique(records)
    training.verify_records(records)
    result = dict(format=FORMAT, layout=runner.LAYOUT, snapshot_validated=True, created_utc=training.now(),
        run_directory=str(directory), direction=direction, request=handoff['request_record'],
        source_archive=archive.manifest_record, previous_summary=request['previous_summary'],
        available_cells=available, missing_cells=[cell for cell in runner.CELL_ORDER if cell not in available],
        all_cube_cells_present=len(available) == 8, baseline_controls_certified=True, anchor_controls=controls,
        per_case=per_case, groups_readout=groups, files=records,
        controller_exit_observed=False, controller_completion_claimed=False, native_execution_performed=False,
        goal_completion_claimed=False,
        definitions=dict(snapshot='Only completed leaves present when this snapshot began; not controller completion.',
            primary_domain='Original-prefix cases; other domains and all spelling/split/control strata remain separate.',
            cells='Every cell fixes E; A includes early LN1/attention, M early LN2/MLP, R positions/late blocks/final LN.',
            metrics='Natural-log teacher-forced probability; suffix conditions on supplied first piece; fourth is exact next native token.',
            argmax='Conjunction over this metric positions, counted once per causal event; not general free-generation accuracy.',
            weighting='Prefix hash/length plus target IDs through the last metric position; overlapping strata are not independent.',
            contrasts='Signed mean-log-probability contrasts, omitted whenever any required cell is absent.'),
        limitations=['No GPU, process-liveness, or successful-controller-exit assertion.',
                     'Native FP32 losses do not separate target logits from softmax normalization.',
                     'Coordinate-defined, interleaved weight bundles are not unique word-storage locations.'])
    training.publish(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', required=True)
    parser.add_argument('--direction', choices=runner.DIRECTIONS, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    analyze(args.run_directory, args.direction, args.output)


if __name__ == '__main__':
    main()
