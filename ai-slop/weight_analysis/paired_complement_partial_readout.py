"""Authenticate and summarize completed cells while a fixed cube is running.

Only completed leaf cells are read. E and EC must first pass reconstructed
preserved-reference controls, including all-context native byte parity and
the separately bounded FP32-loss/FP64-logit comparison. Missing cells are not
zeros: no complete-cube factorial formula is applied to an incomplete cube.

This CPU tool neither launches native code nor asks a live experiment to stop.
It never calls the owning runner's handoff/idle checks. Completion below means
validation of this snapshot, not successful controller exit or goal completion.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import math
from pathlib import Path

from . import paired_complement_localization as core
from . import paired_complement_localization_run as runner

training = core.training
native = core.native
FORMAT = 'pluto-paired-complement-partial-readout-v1'
READER_SOURCE = training.record(__file__)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def completed_leaf(directory):
    """A directory of dumped scores is not a completed, authenticated cell."""
    directory = Path(directory).absolute()
    _require(directory.is_dir() and not directory.is_symlink(), 'invalid completed-cell directory')
    marker_path = directory/'complete.json'
    marker_record = training.record(marker_path)
    marker = native._json(marker_path)
    declared = core.outer._unique(marker['artifacts'])
    _require(len(declared) == len(marker['artifacts']), 'duplicate completed-cell artifact')
    paths = list(directory.rglob('*'))
    _require(not any(path.is_symlink() for path in paths), 'symlink in completed cell')
    actual = {str(path) for path in paths if path.is_file() and path != marker_path}
    _require(actual == {record['path'] for record in declared},
             'completed-cell inventory differs from marker')
    records = [marker_record, *declared]
    training.verify_records(records)
    return marker, records


def validate_request(directory, direction):
    """Verify frozen request/plan/source identities without touching GPU state."""
    directory = Path(directory).absolute()
    _require(directory.is_dir() and not directory.is_symlink(), 'invalid run directory')
    request_path = directory/'request.json'
    request_record = training.record(request_path)
    request = native._json(request_path)
    plan_record = request['plan']
    training.verify_records([request_record, plan_record])
    plan = native._json(plan_record['path'])
    root = directory.parent
    _require(request.get('format') == runner.FORMAT
             and request.get('directions') == list(runner.DIRECTIONS)
             and request.get('cell_order') == list(runner.CELL_ORDER)
             and request.get('suites') == list(runner.SUITES)
             and request.get('native_measurement_count') == 40
             and request.get('training_restarted') is False
             and request.get('goal_completion_claimed') is False
             and direction in runner.DIRECTIONS, 'wrong owning-runner request/scope')
    _require(plan.get('format') == core.FORMAT and plan.get('phase') == 'planned'
             and plan.get('config') == asdict(core.GPT2Config()) and plan.get('step') == 331
             and plan.get('amendment_root') == str(root)
             and Path(plan_record['path']).parent.parent == root
             and plan.get('cells') == {key:list(value) for key,value in core.cube_subsets().items()}
             and plan.get('groups') == core.groups()
             and set(plan['directions']) == set(runner.DIRECTIONS)
             and set(plan['cases']) == set(runner.SUITES)
             and plan.get('native_execution_performed') is False,
             'prepared plan differs from fixed endpoint cube')
    _require(plan['loss_tolerance']['absolute'] == core.FP32_ABSOLUTE_TOLERANCE
             and plan['loss_tolerance']['relative'] == core.FP32_RELATIVE_TOLERANCE,
             'FP32 reference tolerance changed')
    _require(request['binaries'] == plan['binaries'] and request['runtime'] == plan['runtime']
             and request['gpu'] == plan['gpu'], 'request runtime/binary/GPU binding differs')
    records = core.outer._unique([*request['frozen_inputs'], *plan['frozen_inputs'],
                                  request_record, plan_record])
    frozen = {record['path']:record for record in request['frozen_inputs']}
    _require(all(frozen.get(record['path']) == record for record in plan['frozen_inputs'])
             and frozen.get(plan_record['path']) == plan_record,
             'request omitted prepared plan inputs')
    for module in (runner, core):
        record = training.record(module.__file__)
        _require(frozen.get(record['path']) == record, 'executing source differs from frozen runner')
    for record in [*request['binaries'].values(), *request['runtime']['frozen_records'],
                   *plan['cases'].values()]:
        _require(frozen.get(record['path']) == record, 'request omitted required source record')
    identity = request['runner_identity']
    argv = identity['argv']
    _require(type(identity['pid']) is int and identity['pid'] > 0
             and type(identity['start_ticks']) is int and identity['start_ticks'] > 0
             and argv.count('-m') == argv.count('--plan') == argv.count('--output') == 1
             and argv[argv.index('-m')+1] == 'weight_analysis.paired_complement_localization_run'
             and argv[argv.index('--plan')+1] == plan_record['path']
             and argv[argv.index('--output')+1] == str(directory), 'recorded runner invocation differs')
    # This authenticates the recorded identity only, not its current liveness
    # or exit status. A live controller is expected and is not a failure.
    training.verify_records(records)
    return request, plan, records


def _execution_binding(measured, directory, request_record, plan_record, models_record,
                       binary, frozen):
    """Require the loaded scores to belong to this exact owning run."""
    path = Path(directory)/'execution.json'
    record = training.record(path)
    _require(record in measured['records'], 'native execution missing from reader ledger')
    execution = native._json(path)
    _require(execution.get('format') == 'pluto-paired-probe-execution-v1'
             and execution.get('returncode') == 0
             and execution['command'][0] == binary['path'], 'wrong or unsuccessful native execution')
    before = {record['path']:record for record in execution['inputs_before']}
    _require(len(before) == len(execution['inputs_before'])
             and execution['inputs_before'] == execution['inputs_after'],
             'native input ledger changed or repeats paths')
    for expected in [request_record, plan_record, models_record, binary, *frozen]:
        _require(before.get(expected['path']) == expected, 'native execution omitted owning-run input')
    return [record, execution['log']]


def certify_baselines(loaded, preserved, markers, reference_markers, direction):
    """Recompute controls; booleans or filenames alone cannot certify E/EC."""
    result = {}
    for cell in ('E','EC'):
        _require(cell in loaded and cell in preserved and cell in markers and cell in reference_markers,
                 'both completed E and EC baseline controls are required')
        ref_marker = reference_markers[cell]
        marker = markers[cell]
        _require(ref_marker.get('cell') == cell and marker.get('cell') == cell
                 and ref_marker.get('all_targets_checked') is True
                 and ref_marker.get('first_three_cross_suite_byte_equal') is True
                 and marker.get('first_three_cross_suite_byte_equal') is True
                 and set(ref_marker['fp64_reference_checks']) == set(runner.SUITES)
                 and set(marker['copy_controls']) == set(runner.SUITES),
                 'incomplete or false baseline control certification')
        result[cell] = {}
        for suite in runner.SUITES:
            reference_fp64 = runner.compare_references(preserved[cell][suite], direction, cell, suite)
            copied_fp64 = runner.compare_references(loaded[cell][suite], direction, cell, suite)
            parity = core.assert_native_copy_parity(preserved[cell][suite], loaded[cell][suite])
            expected = dict(same_native_copy=parity, fp64_reference=copied_fp64)
            _require(ref_marker['fp64_reference_checks'][suite] == reference_fp64
                     and marker['copy_controls'][suite] == expected,
                     'saved baseline controls differ from recomputed controls')
            result[cell][suite] = dict(preserved_fp64=reference_fp64, **expected)
    return result


def selected_cases(loaded):
    """Retain raw selected FP32 NLLs, including aliases and both prefix domains."""
    result = []
    for suite in runner.SUITES:
        cases = loaded['E'][suite]['cases']
        amended = cases['plan']['format'] in core.branch.case_contract.AMENDED
        for case in cases['plan']['cases']:
            item = {key:case[key] for key in ('kind','split','context_id','prefix_domain','target','target_ids')}
            item.update(core.branch.case_contract.metadata(case, amended=amended))
            item.update(suite=suite, case_index=case['case_index'], selected_rows=case['scored_rows'],
                        prefix_sha256=case['prefix']['token_ids_sha256'],
                        prefix_length=case['prefix']['length'], cells={})
            if case['kind'] in ('word','word_next_native'):
                item['word_has_leading_space'] = bytes.fromhex(
                    case['target_source']['native_piece_bytes_hex'][0]).startswith(b' ')
            if 'piece_id' in case:
                item['piece_id'] = case['piece_id']
            for cell in loaded:
                measured = loaded[cell][suite]
                _require(measured['cases']['record'] == cases['record'], 'cell case sources differ')
                nll = [float(value) for value in measured['losses'][case['case_index'],case['scored_rows']]]
                _require(all(math.isfinite(value) and value >= 0 for value in nll), 'invalid selected native loss')
                item['cells'][cell] = dict(token_nll=nll, token_log_probability=[-value for value in nll],
                    argmax_ids=[int(value) for value in measured['argmax'][case['case_index'],case['scored_rows']]])
            result.append(item)
    return result


def aggregate(per_case, available_cells):
    """Original-domain primary strata; never impute an unmeasured cube cell."""
    cells = tuple(available_cells)
    _require(len(cells) == len(set(cells)) and {'E','EC'} <= set(cells)
             and set(cells) <= set(core.cube_subsets()), 'invalid partial-cell coverage')
    grouped = defaultdict(list)
    for item in per_case:
        word = item['kind'] in ('word','word_next_native')
        if item['prefix_domain'] != ('original' if word else 'shared'):
            continue
        _require(set(item['cells']) == set(cells), 'case partial-cell coverage differs')
        key = (item['suite'],item['kind'],item['split'],item.get('spelling_variant'),
               item.get('word_has_leading_space'),item.get('piece_id'),item['target'])
        grouped[key].append(item)
    result = []
    fields = ('suite','kind','split','spelling_variant','word_has_leading_space','piece_id','target')
    for key, items in sorted(grouped.items(), key=lambda pair:repr(pair[0])):
        length = len(items[0]['target_ids'])
        _require(length in (3,4) and all(len(item['target_ids']) == length for item in items),
                 'unexpected scored target length')
        metrics = {}
        for metric, positions in core._positions(items[0]['kind'], length).items():
            unique = {}
            for item in items:
                identity = (item['prefix_sha256'],item['prefix_length'],
                            tuple(item['target_ids'][:max(positions)+1]))
                values = {}
                for cell in cells:
                    tokens = item['cells'][cell]['token_log_probability']
                    _require(len(tokens) == length and all(type(value) in (int,float)
                             and math.isfinite(value) and value <= 0 for value in tokens),
                             'nonfinite, positive, or wrong-length token log score')
                    values[cell] = math.fsum(tokens[position] for position in positions)
                _require(identity not in unique or unique[identity] == values,
                         'duplicate causal event has conflicting scores')
                unique[identity] = values
            means = {cell:math.fsum(values[cell] for values in unique.values())/len(unique) for cell in cells}
            metrics[metric] = dict(unique_event_count=len(unique), alias_case_count=len(items),
                cells={cell:dict(mean_log_probability=value, geometric_mean_probability=math.exp(value))
                       for cell,value in means.items()},
                delta_log_probability_from_E={cell:value-means['E'] for cell,value in means.items()},
                gap_to_EC_log_probability={cell:means['EC']-value for cell,value in means.items()})
        result.append(dict(zip(fields,key), prefix_domain='original' if key[1] in ('word','word_next_native') else 'shared',
                           metrics=metrics))
    return result


def analyze(run_directory, direction, output):
    """Exclusively publish one authenticated snapshot; no native execution."""
    directory, output = Path(run_directory).absolute(), Path(output).absolute()
    _require(not output.exists() and not output.is_symlink(), 'output must be a NEW file')
    _require(directory not in output.parents and output != directory,
             'output must be outside the live run directory')
    # Bind executing code before a potentially long evidence audit. Later
    # source bytes must not be silently relabeled as the implementation used.
    source_records = [READER_SOURCE,
        training.record(Path(__file__).with_name('paired_complement_partial_readout_test.py'))]
    training.verify_records(source_records)
    request, plan, records = validate_request(directory, direction)
    training.verify_records(source_records)
    _require(output.parent == Path(plan['amendment_root']),
             'partial output must be a new immediate-child file of the amendment')
    stage = directory/direction
    available = [cell for cell in runner.CELL_ORDER if (stage/'cells'/cell/'complete.json').exists()
                 or (stage/'cells'/cell/'complete.json').is_symlink()]
    _require({'E','EC'} <= set(available), 'both completed E and EC baseline controls are required')
    models_record = training.record(stage/'models.json')
    declared_models = native._json(models_record['path'])
    _require(set(declared_models['cube']) == set(core.cube_subsets())
             and set(declared_models['preserved_references']) == {'E','EC'}, 'incomplete declared model map')
    records.extend([models_record,*source_records])
    request_record = training.record(directory/'request.json')
    direction_plan = plan['directions'][direction]
    models, markers, loaded, preserved, reference_markers = {}, {}, {}, {}, {}
    for cell in available:
        cell_dir = stage/'cells'/cell
        marker, evidence = completed_leaf(cell_dir)
        expected_scores = {suite:str(cell_dir/suite/'scores') for suite in runner.SUITES}
        expected_ledgers = {suite:str(cell_dir/suite/'execution.json') for suite in runner.SUITES}
        _require(marker.get('cell') == cell and marker.get('groups') == plan['cells'][cell]
                 and marker.get('scores') == expected_scores and marker.get('executions') == expected_ledgers
                 and marker.get('first_three_cross_suite_byte_equal') is True
                 and (cell in ('E','EC') or marker.get('copy_controls') == {}),
                 'completed-cell identity, suites, or selection differs')
        records.extend(evidence); markers[cell] = marker
        model = core.validate_model(cell_dir/'step_331/patch.json', direction_plan['recipient']['path'],
            direction_plan['donor']['path'], plan['cells'][cell], direction_plan['rows'])
        _require(model == declared_models['cube'][cell], 'cell model differs from declared construction')
        models[cell], loaded[cell] = model, {}
        for suite in runner.SUITES:
            measured = core.load_native(model,plan['cases'][suite]['path'],suite,
                                         expected_scores[suite],expected_ledgers[suite])
            records.extend(measured['records'])
            records.extend(_execution_binding(measured,cell_dir/suite,request_record,request['plan'],
                models_record,plan['binaries']['loss_probe'],request['frozen_inputs']))
            loaded[cell][suite] = measured
        core._cross_suite(loaded[cell]['main'],loaded[cell]['supplemental'])
    for cell in ('E','EC'):
        ref_dir = stage/'references'/cell
        marker, evidence = completed_leaf(ref_dir)
        reference_markers[cell] = marker; records.extend(evidence)
        model = runner.reference_model(models[cell],direction_plan['reference_checkpoints'][cell])
        _require(model == declared_models['preserved_references'][cell], 'preserved reference differs from declared binding')
        preserved[cell] = {}
        for suite in runner.SUITES:
            measured = core.load_native(model,plan['cases'][suite]['path'],suite,
                                        ref_dir/suite/'scores',ref_dir/suite/'execution.json')
            preserved[cell][suite] = measured; records.extend(measured['records'])
            records.extend(_execution_binding(measured,ref_dir/suite,request_record,request['plan'],
                models_record,plan['binaries']['loss_probe'],request['frozen_inputs']))
        core._cross_suite(preserved[cell]['main'],preserved[cell]['supplemental'])
    controls = certify_baselines(loaded,preserved,markers,reference_markers,direction_plan)
    per_case = selected_cases(loaded)
    groups = aggregate(per_case,available)
    records = core.outer._unique(records)
    training.verify_records(records)
    result = dict(format=FORMAT, snapshot_validated=True, created_utc=training.now(),
        run_directory=str(directory),direction=direction,request=request_record,plan=request['plan'],
        available_cells=available,missing_cells=[cell for cell in runner.CELL_ORDER if cell not in available],
        all_cube_cells_present=len(available)==8,baseline_controls_certified=True,baseline_controls=controls,
        per_case=per_case,groups_readout=groups,files=records,
        controller_exit_observed=False,controller_completion_claimed=False,native_execution_performed=False,
        goal_completion_claimed=False,
        definitions=dict(snapshot='Only immutable completed leaf cells present when this snapshot began',
            primary_domain='Original-corpus causal prefixes, in each swap direction; shared controls retain shared domain',
            effects='Available-cell minus E and EC minus available-cell mean log probabilities; no missing-cell imputation or incomplete factorial',
            weighting='Per-stratum causal prefix hash+length and target IDs through the last metric position; no cross-stratum global-independence claim',
            raw_scores='Selected native FP32 NLLs retained per case, including aliases and both domains',
            suffix='Conditional two-piece suffix given the supplied candidate first token',
            fourth='One exact following native token, not any word boundary'),
        limitations=['No unique storage or universally necessary circuit claim.',
            'Native FP32 loss reductions can round near-ceiling probabilities; these are not FP64 full-logit scores.',
            'The E/EC FP64 comparison uses the frozen numerical tolerance, not exact equality between different loss computations.',
            'No controller liveness, exit, or GPU-idle assertion is made by this partial CPU reader.'])
    training.verify_records(records)
    training.publish(output,result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--direction',choices=runner.DIRECTIONS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    analyze(args.run_directory,args.direction,args.output)


if __name__ == '__main__':
    main()
