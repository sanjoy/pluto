"""A new, endpoint-only E x C experiment; never resume an interrupted screen.

E is the eleven predeclared token-embedding rows (input and tied output).
C is every other weight FILE: positions, eight transformer blocks, final LN.
All remaining token-embedding rows stay recipient in A/E/C/EC. Full donor D
therefore differs from EC only in those remaining embedding rows.

Existing A/E and reciprocal donor-AA logits are reauthenticated, not rerun or
relabeled. C is made by replacing the 99 nonembedding files; EC is a separate
ROW-ONLY C -> EC patch, so the unchanged inner embedding-factorial probe and
reader remain valid. Their AA/JJ become the outer C/EC cells. An independent
100-file donor no-op copy must reproduce every old donor-AA selected logit in
all three suites before interpreting the new models. The two mixed inner
cells remain available but are not confused with the outer E x C factorial.

Only a NEW immediate-child output is allowed. No training, old checkpoint,
old state, or existing source is modified. Failure leaves evidence and is not
automatically retried. Native execution is serial and guarded by actual process
exit and GPU-idle checks; this module must be explicitly launched by its owner.
"""

import argparse
import math
import os
from pathlib import Path
import signal

from . import paired_causal_recovery as recovery_module
from . import paired_lowercase_causal_followup as screen

training = screen.training
reader = screen.embedding_factorial_readout
checkpoint = screen.checkpoint
FORMAT = 'pluto-paired-outer-factorial-v1'
STEP = 331
SUITES = ('main', 'word_next_native', 'shared_piece')
CELL_NAMES = ('A', 'E', 'C', 'EC', 'D')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(records):
    result = {}
    for record in records:
        _require(record['path'] not in result or result[record['path']] == record,
                 'conflicting frozen records for ' + record['path'])
        result[record['path']] = record
    return [result[path] for path in sorted(result)]


def require_exited(identity):
    """Missing identity or PID reuse is not an invented successful exit code."""
    if training.process_live(identity):
        raise RuntimeError('previous analysis process is still live')


def require_new_output(root, recovery, output):
    root = Path(root).resolve(strict=True)
    recovery, output = Path(recovery).absolute(), Path(output).absolute()
    _require(recovery.parent == root and recovery.is_dir() and not recovery.is_symlink(),
             'recovery must be a real immediate child of the amendment')
    _require(output.parent == root and not output.exists() and not output.is_symlink(),
             'outer factorial requires a NEW immediate-child output')
    return root, recovery, output


def outer_effect(values):
    """Unnormalized finite log-score contrasts, not mediation percentages."""
    _require(set(values) == set(CELL_NAMES) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in values.values()), 'outer effects require five finite cells A,E,C,EC,D')
    a, e, c, ec, d = (values[k] for k in CELL_NAMES)
    return dict(E_minus_A=e-a, C_minus_A=c-a, EC_minus_A=ec-a,
                EC_minus_E=ec-e, EC_minus_C=ec-c,
                interaction=math.fsum((ec, -c, -e, a)), D_minus_EC=d-ec)


def select_endpoints(plan):
    rows = plan['word_piece_ids']
    _require(len(rows) == 11 and rows == sorted(set(rows)) and all(
        type(x) is int and 0 <= x < checkpoint.GPT2Config().vocab_size for x in rows),
        'expected the eleven predeclared native word rows')
    items = [item for item in plan['interventions']
             if item['kind'] == 'embedding_rows' and item['step'] == STEP]
    _require(len(items) == 2 and len({i['name'] for i in items}) == 2,
             'need exactly two endpoint embedding interventions')
    directions = set()
    for item in items:
        direction = (item['recipient_arm'], item['donor_arm'])
        _require(direction in (('original', 'replacement'), ('replacement', 'original'))
                 and item['tensors'] == [] and item['embedding_rows'] == rows
                 and all(item[k]['step'] == STEP for k in ('recipient_checkpoint', 'donor_checkpoint')),
                 'endpoint direction/rows/checkpoint differs')
        directions.add(direction)
    _require(len(directions) == 2, 'endpoint directions must be reciprocal')
    _require(items[0]['recipient_checkpoint'] == items[1]['donor_checkpoint']
             and items[0]['donor_checkpoint'] == items[1]['recipient_checkpoint'],
             'endpoint source checkpoints are not reciprocal')
    return sorted(items, key=lambda item: item['recipient_arm'])


def validate_recovery(root, previous):
    """Authenticate a retired explicit recovery and its successful GPU test.

    The interrupted branch screen need not have a completion marker. We do not
    infer its return code or authorize continuing its partial intervention.
    Later baseline validation requires actual completed endpoint artifacts.
    """
    request_path = previous / 'request.json'
    request = training.read_json(request_path)
    state = training.read_json(previous / 'state.json')
    _require(request.get('format') == recovery_module.FORMAT
             and request.get('amendment_root') == str(root)
             and request.get('stage') == 'embedding_and_branches'
             and request.get('training_restarted') is False
             and request.get('same_predeclared_plan') is True
             and state.get('runner_pid') == request['runner_identity']['pid']
             and state.get('phase') in ('interventions', 'complete'),
             'not the authenticated retired causal recovery')
    argv = request['runner_identity']['argv']
    _require('-m' in argv and argv[argv.index('-m')+1] == 'weight_analysis.paired_causal_recovery'
             and '--root' in argv and argv[argv.index('--root')+1] == str(root)
             and '--output' in argv and argv[argv.index('--output')+1] == str(previous),
             'retired runner argv differs from recovery paths')
    require_exited(request['runner_identity'])
    training.verify_records(request['frozen_inputs'])
    runtime = request['runtime']
    _require(runtime.get('complete') is True and runtime.get('runtime_dlopen_covered') is False
             and runtime.get('environment') == {'LD_LIBRARY_PATH': str(previous / 'runtime')}
             and runtime['relocated_executable'] == request['binaries']['gpu_test'],
             'recovery runtime does not bind its packaged test')
    for key in ('LD_PRELOAD', 'LD_AUDIT'):
        _require(key not in os.environ, 'unset inherited ' + key)
    training.verify_records([*runtime['frozen_records'], *request['binaries'].values()])
    _require(all(record in request['frozen_inputs'] for record in
                 [*runtime['frozen_records'], *request['binaries'].values()]),
             'runtime/binary absent from original frozen request')
    test_path = previous / 'gpu_test_execution.json'
    test = training.read_json(test_path)
    test_json = previous / 'gpu_validation/gtest.json'
    _require(test.get('format') == 'pluto-paired-probe-execution-v1'
             and test.get('returncode') == 0 and test['inputs_before'] == test['inputs_after']
             and test['command'] == [request['binaries']['gpu_test']['path'],
                                    '--gtest_output=json:' + str(test_json)]
             and all(record in test['inputs_before'] for record in runtime['frozen_records']),
             'packaged GPU validation execution is incomplete')
    screen.execution.require_gpu_test_success(test_json)
    training.verify_records([*test['inputs_before'], *test['outputs'], test['log']])
    _require(training.record(test_json) in test['outputs'], 'GPU test JSON is not bound to execution')
    records = [training.record(p) for p in (request_path, previous/'state.json',
               previous/'plan/plan.json', test_path, test_json)]
    records += [*request['frozen_inputs'], *test['inputs_before'], *test['outputs'], test['log']]
    return request, training.read_json(previous/'plan/plan.json'), _unique(records)


def _suite_inputs(root, plan, suite):
    source = root / ('word_cases/cases.json' if suite == 'main' else 'supplemental_cases/cases.json')
    exported = plan['exports'][suite]
    batch = (training.read_json(source)['packed_batch']['path'] if suite == 'main'
             else exported['packed_batch']['path'])
    return source, batch, exported['selected_rows']['path'], exported['rows_per_case']


def validate_baselines(root, previous, plan, output):
    """Regenerate all six endpoint readouts, including their native ledgers."""
    output.mkdir()
    reports, records = {}, []
    for item in select_endpoints(plan):
        stage = previous / item['name']
        complete = training.read_json(stage / 'complete.json')
        _require(complete['intervention'] == item and set(complete['factorial']) == set(SUITES),
                 'completed baseline differs from predeclared endpoint')
        declared = complete['artifacts']
        actual_paths = {str(p) for p in stage.rglob('*') if p.is_file() and p != stage/'complete.json'}
        _require(not any(p.is_symlink() for p in stage.rglob('*'))
                 and len(declared) == len({r['path'] for r in declared})
                 and actual_paths == {r['path'] for r in declared},
                 'baseline completed artifact inventory differs')
        training.verify_records(declared)
        records.extend([*declared, training.record(stage/'complete.json')])
        reports[item['recipient_arm']] = {}
        for suite in SUITES:
            saved_record = complete['factorial'][suite]
            _require(saved_record in declared, 'baseline readout absent from complete ledger')
            saved = training.read_json(saved_record['path'])
            source, _, _, _ = _suite_inputs(root, plan, suite)
            fresh_path = output / (item['name'] + '_' + suite + '.json')
            fresh = reader.analyze(source, stage/('factorial_'+suite), stage/'step_331/patch.json',
                fresh_path, expected_rows=item['embedding_rows'],
                case_kind=None if suite == 'main' else suite,
                execution_record=stage/('factorial_'+suite+'_execution.json'))
            _require(fresh == saved and fresh['execution_provenance'].get('verified') is True,
                     'saved baseline differs from complete native-logit regeneration')
            patch = training.read_json(stage/'step_331/patch.json')
            for side, role in (('original', 'recipient'), ('replacement', 'donor')):
                _require(patch['sources'][side]['path'] == item[role+'_checkpoint']['path']
                         and patch['sources'][side]['weights_sha256'] == item[role+'_checkpoint']['sha256'],
                         'baseline patch source differs from endpoint')
            records.extend([*fresh['files'], saved_record, training.record(fresh_path)])
            reports[item['recipient_arm']][suite] = fresh
            print(f'{training.now()} authenticated endpoint baseline {item["recipient_arm"]}/{suite}', flush=True)
    return reports, _unique(records)


def prepare_models(item, direction_dir):
    """Materialize the explicit patch chain without modifying source weights."""
    direction_dir = Path(direction_dir)
    specs = checkpoint.tensor_manifest()
    _require(len(specs) == 100 and specs[0].name == 'token_embedding.weight',
             'expected exact 100-file GPT-2 geometry')
    a, d = (item[k]['path'] for k in ('recipient_checkpoint', 'donor_checkpoint'))
    paths = {}
    for name in ('C', 'EC', 'donor_copy'):
        parent = direction_dir / name
        parent.mkdir()
        paths[name] = parent / 'step_331'
    create = screen.paired_weight_patch.create_patch
    models = {'C': create(a, d, paths['C'], tensors=[s.name for s in specs[1:]])}
    models['EC'] = create(paths['C'], d, paths['EC'], embedding_rows=item['embedding_rows'])
    models['donor_copy'] = create(d, d, paths['donor_copy'])
    _require(models['donor_copy']['output']['weights_sha256'] == item['donor_checkpoint']['sha256']
             and len(models['donor_copy']['output']['weights_sha256']) == 100,
             'all 100 donor-copy tensor hashes must equal donor')
    expected_c = dict(item['donor_checkpoint']['sha256'])
    expected_c['weight_0.bin'] = item['recipient_checkpoint']['sha256']['weight_0.bin']
    _require(models['C']['output']['weights_sha256'] == expected_c,
             'C does not have recipient embedding and donor nonembedding files')
    _require(all(models['EC']['output']['weights_sha256'][s.filename] == expected_c[s.filename]
                 for s in specs[1:]), 'EC changed a C nonembedding file')
    return models


def _model_records(models):
    return [training.record(path) for model in models.values()
            for path in screen.execution.checkpoint_inputs(model['output']['path'])]


def run_suite(root, plan, suite, recipient, patched, expected_rows, output,
              binary, gpu, frozen):
    """Use the unchanged probe/reader and their full execution-time evidence."""
    output.mkdir()
    source, batch, rows, rows_per_case = _suite_inputs(root, plan, suite)
    scores = output / 'scores'
    execution_path = output / 'execution.json'
    command = [binary, f'--recipient={recipient}', f'--patched={patched}', f'--batch={batch}',
               f'--rows={rows}', f'--rows_per_case={rows_per_case}', '--batch_sequences=1',
               f'--output_dir={scores}']
    training.verify_records(frozen)
    inputs = [binary, source, batch, rows, *screen.execution.checkpoint_inputs(recipient),
              *screen.execution.checkpoint_inputs(patched), *(r['path'] for r in frozen)]
    screen.run_native(command, inputs, scores, output/'process.log', execution_path, gpu)
    report = reader.analyze(source, scores, Path(patched)/'patch.json', output/'readout.json',
                            expected_rows=expected_rows,
                            case_kind=None if suite == 'main' else suite,
                            execution_record=execution_path)
    training.verify_records([*frozen, *report['files']])
    return report


def _logit_record(report, cell):
    metadata = training.read_json(report['native_metadata']['path'])
    path = Path(report['native_metadata']['path']).parent / metadata['cells'][cell]['logits_file']
    records = [r for r in report['files'] if Path(r['path']) == path]
    _require(len(records) == 1 and training.record(path) == records[0], 'unbound native logits')
    return records[0]


def assert_donor_parity(control, baseline):
    """Compare ALL logical-vocabulary logits, not just target probabilities."""
    old = _logit_record(baseline, 'AA')
    for cell in ('AA', 'AJ', 'JA', 'JJ'):
        new = _logit_record(control, cell)
        _require(all(new[key] == old[key] for key in ('bytes', 'sha256')),
                 'donor-copy native logits differ from reciprocal donor AA')
    _require(len(control['per_case']) == len(baseline['per_case']), 'donor-copy cases differ')
    for a, b in zip(control['per_case'], baseline['per_case']):
        _require(_case_identity(a) == _case_identity(b)
                 and all(a['cells'][cell] == b['cells']['AA'] for cell in ('AA','AJ','JA','JJ')),
                 'donor-copy derived scores differ')


def _case_identity(case):
    keys = ('native_case_index', 'source_case_index', 'context_id', 'kind', 'split',
            'prefix_domain', 'target', 'target_ids', 'spelling_variant',
            'word_has_leading_space', 'candidate_pair', 'target_source_domain', 'piece_id')
    identity = {key: case[key] for key in keys if key in case}
    identity['prediction_prefixes'] = [{key: p[key] for key in
        ('prediction_row', 'causal_prefix_length', 'causal_prefix_sha256')} for p in case['predictions']]
    return identity


def combine_outer(baseline, conditional, donor):
    """Preserve every case/variant and both absolute word scores; no pooling."""
    _require(len(baseline['per_case']) == len(conditional['per_case']) == len(donor['per_case']),
             'outer case counts differ')
    result = []
    for a, c, d in zip(baseline['per_case'], conditional['per_case'], donor['per_case']):
        identity = _case_identity(a)
        _require(identity == _case_identity(c) == _case_identity(d), 'outer case identities differ')
        cells = dict(A=a['cells']['AA'], E=a['cells']['JJ'], C=c['cells']['AA'],
                     EC=c['cells']['JJ'], D=d['cells']['AA'])
        item = dict(case=identity, cells=cells,
            sequence_log_probability_effects=outer_effect({k:v['sequence_log_probability'] for k,v in cells.items()}))
        count = len(a['target_ids'])
        item['token_log_probability_effects'] = [outer_effect({k:v['token_log_probability'][p]
                                                               for k,v in cells.items()}) for p in range(count)]
        if a['kind'] in ('word', 'word_next_native'):
            item['word_three_log_probability_effects'] = outer_effect({k:v['word_three_log_probability']
                                                                       for k,v in cells.items()})
            suffix = {k:math.fsum(v['token_log_probability'][1:3]) for k,v in cells.items()}
            item['conditional_word_suffix'] = {k:dict(log_probability=v, probability=math.exp(v),
                                                      failure_probability=-math.expm1(v)) for k,v in suffix.items()}
            item['conditional_word_suffix_effects'] = outer_effect(suffix)
        result.append(item)
    return dict(per_case=result, case_count=len(result),
        definitions=dict(cells='A=recipient,E=selected donor rows,C=99 donor nonembedding tensors,EC=both,D=full donor',
            remaining_rows='All remaining stored token embedding rows are recipient in A/E/C/EC; D-minus-EC changes only those rows.',
            aggregation='No pooled means or flip counts: per-case rows include aliases; deduplicate actual causal prefixes before aggregation.',
            interaction='EC-C-E+A is log-probability nonadditivity, not proof of representational interaction; inspect saved logits and normalizers.'),
        limitations=['Teacher-forced candidate suffix histories differ; word odds do not cancel suffix normalizers.',
            'No unique word-storage, complete memorization, or universal necessity claim.',
            'Ordinary/shared-piece controls and absolute probabilities must accompany preference changes.',
            'Learned position, transformer and final-normalization tensors are bundled in C; this assay does not localize within C.'])


def run(root, recovery, output):
    root, previous, output = require_new_output(root, recovery, output)
    prior, old_plan, records = validate_recovery(root, previous)
    output.mkdir()
    completed = []
    completed_artifacts = []
    try:
        print(f'{training.now()} revalidating complete training and fixed endpoint plan', flush=True)
        plan = screen.planner.prepare(root, root/'analysis_trajectory/summary.json',
                                      root/'causal_cases/exports.json', output/'plan')
        _require(plan == old_plan, 'fresh predeclared plan differs from recovery plan')
        items = select_endpoints(plan)
        baselines, baseline_records = validate_baselines(root, previous, plan, output/'baseline_revalidation')
        records.extend(baseline_records)
        sources = [training.record(path) for path in (__file__, screen.__file__, reader.__file__,
                   screen.paired_weight_patch.__file__, checkpoint.__file__, training.__file__,
                   screen.planner.__file__, recovery_module.__file__)]
        (output/'source').mkdir()
        copies = [screen.execution.paired_training.freeze_file(r['path'], output/'source'/Path(r['path']).name)
                  for r in sources]
        original_training = training.read_json(root/'request.json')
        frozen = _unique([*records, *sources, *copies, *plan['frozen_inputs'], *plan['implementation'],
                          *original_training['frozen_inputs'], training.record(output/'plan/plan.json')])
        training.verify_records(frozen)
        # Second actual exit observation, separated by complete CPU validation.
        require_exited(prior['runner_identity'])
        gpu = original_training['gpu']
        training.require_idle_gpu(gpu)
        runtime = prior['runtime']
        os.environ.update(runtime['environment'])
        identity = training.process_identity(os.getpid())
        training.publish(output/'request.json', dict(format=FORMAT, created_utc=training.now(),
            amendment_root=str(root), previous_recovery=str(previous), previous_runner=prior['runner_identity'],
            runner_identity=identity, fixed_step=STEP, directions=items, suites=list(SUITES),
            binaries=prior['binaries'], runtime=runtime, frozen_inputs=frozen,
            training_restarted=False, previous_runner_exit_code_observed=False, goal_completion_claimed=False))
        frozen = _unique([*frozen, training.record(output/'request.json')])
        for item in items:
            name = item['donor_arm'] + '_to_' + item['recipient_arm']
            direction = output/name
            direction.mkdir()
            training.verify_records(frozen)
            models = prepare_models(item, direction)
            model_records = _model_records(models)
            active = _unique([*frozen, *model_records])
            training.publish(direction/'models.json', models)
            active = _unique([*active, training.record(direction/'models.json')])
            measured = {}
            for suite in SUITES:
                training.publish(output/'state.json', dict(format=FORMAT, phase='donor_copy_validation',
                    runner_pid=os.getpid(), direction=name, suite=suite, completed=completed), exclusive=False)
                print(f'{training.now()} {name}/{suite}: validating independent donor copy', flush=True)
                control = run_suite(root, plan, suite, item['donor_checkpoint']['path'],
                    models['donor_copy']['output']['path'], [], direction/('donor_control_'+suite),
                    prior['binaries']['factorial_probe']['path'], gpu, active)
                assert_donor_parity(control, baselines[item['donor_arm']][suite])
                active = _unique([*active, *control['files'],
                    training.record(direction/('donor_control_'+suite)/'readout.json')])
                print(f'{training.now()} {name}/{suite}: donor parity passed; measuring C and EC', flush=True)
                training.publish(output/'state.json', dict(format=FORMAT, phase='outer_factorial',
                    runner_pid=os.getpid(), direction=name, suite=suite, completed=completed), exclusive=False)
                conditional = run_suite(root, plan, suite, models['C']['output']['path'],
                    models['EC']['output']['path'], item['embedding_rows'], direction/('outer_'+suite),
                    prior['binaries']['factorial_probe']['path'], gpu, active)
                active = _unique([*active, *conditional['files'],
                    training.record(direction/('outer_'+suite)/'readout.json')])
                combined = combine_outer(baselines[item['recipient_arm']][suite], conditional,
                                         baselines[item['donor_arm']][suite])
                combined.update(format=FORMAT, step=STEP, direction=name, suite=suite,
                                donor_copy_all_logits_byte_equal=True,
                                conditional_readout=training.record(direction/('outer_'+suite)/'readout.json'),
                                donor_control_readout=training.record(direction/('donor_control_'+suite)/'readout.json'))
                combined_path = direction/(suite+'_outer_readout.json')
                training.verify_records(active)
                training.publish(combined_path, combined)
                measured[suite] = training.record(combined_path)
                active = _unique([*active, measured[suite]])
                completed.append(name+'/'+suite)
                print(f'{training.now()} completed {name}/{suite}; {len(completed)}/6 outer suite results', flush=True)
            training.verify_records(active)
            completed_artifacts.extend(screen.stage_result(direction,
                dict(direction=name, models=training.record(direction/'models.json'),
                     suites=measured, donor_copy_validation=True)))
        training.verify_records([*frozen, *completed_artifacts])
        training.publish(output/'state.json', dict(format=FORMAT, phase='complete',
            runner_pid=os.getpid(), completed=completed), exclusive=False)
        artifacts = [training.record(p) for p in sorted(output.rglob('*')) if p.is_file()]
        training.publish(output/'summary.json', dict(format=FORMAT, complete=True, completed_utc=training.now(),
            amendment_root=str(root), fixed_step=STEP, completed=completed, frozen_inputs=frozen,
            artifacts=artifacts, training_restarted=False, goal_completion_claimed=False,
            completion_meaning='Both fixed endpoint E x C directions and all donor-copy controls measured; not completion of the interrupted branch screen or the research goal.'))
        return training.read_json(output/'summary.json')
    except BaseException as error:
        training.publish(output/'failure.json', dict(format=FORMAT, failed_utc=training.now(),
            type=type(error).__name__, error=str(error), completed=completed,
            no_automatic_restart=True, training_restarted=False, goal_completion_claimed=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'recovery', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'outer-factorial runner received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    run(args.root, args.recovery, args.output)


if __name__ == '__main__':
    main()
