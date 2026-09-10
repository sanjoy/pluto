"""Follow the amended paired run and score its full checkpoint trajectory.

The original run remains at its real legacy location and is authenticated by
the explicit handoff evidence. No manifests, commands, or paths are fabricated
to make the imported run look newly executed. Both complete four-hour budgets,
shared initialization, repeatability controls, actual process exits, and frozen
corpus/case hashes are prerequisites to GPU scoring. This observer cannot start
or restart training. An interrupted analysis preserves its partial output.
"""

import argparse
import os
from pathlib import Path
import signal
import time

from . import checkpoint, paired_analysis, paired_lowercase_cases, paired_lowercase_inputs
from . import paired_lowercase_scores as scoring
from . import paired_lowercase_training as training
from . import paired_supplemental_cases, paired_training, paired_weight_diff, paired_word_cases

FORMAT = 'pluto-paired-lowercase-analysis-v1'
MODULES = (checkpoint, paired_analysis, paired_lowercase_cases, paired_lowercase_inputs,
           scoring, training, paired_supplemental_cases, paired_training, paired_weight_diff,
           paired_word_cases)
SOURCES = [training.record(module.__file__) for module in MODULES] + [training.record(__file__)]


def observer_identity(root, request, pid, start_ticks):
    """Pin the actual amended runner, never a matching name or stale JSON PID."""
    actual = training.process_identity(pid)
    supervisor, original = request['upstream_processes']
    suffix = ['-m', 'scripts.weight_analysis.paired_lowercase_training', 'run',
              '--legacy-root', request['legacy_root'], '--amendment', str(root),
              '--supervisor-pid', str(supervisor['pid']), '--supervisor-start-ticks', str(supervisor['start_ticks']),
              '--original-pid', str(original['pid']), '--original-start-ticks', str(original['start_ticks'])]
    if (actual['start_ticks'] != start_ticks or actual['argv'][-len(suffix):] != suffix
            or actual['state'] in ('Z','X','x')):
        raise ValueError('amended runner identity differs from requested live process')
    return actual


def wait_for_training(root, identity, *, poll_seconds=30):
    """Timeouts/transient reads do not mean completion or authorize a restart."""
    if not 0 < poll_seconds <= 60:
        raise ValueError('poll_seconds must be in (0, 60]')
    last, failures, handles = None, 0, {}
    while True:
        try:
            state = training.read_json(root/'state.json')
            if state.get('format') != training.FORMAT or state['runner_pid'] != identity['pid']:
                raise ValueError('training state names a different runner')
            phase = state['phase']
            if phase == 'failed':
                raise RuntimeError('amended training failed: ' + state.get('error', 'unknown'))
            if phase == 'training_complete':
                # Publication can precede process exit by milliseconds. Neither
                # a terminal flag nor an absent child alone grants GPU access.
                training.wait_for_exit([identity], poll_seconds=poll_seconds)
                return state, list(handles.values())
            if phase not in ('waiting_original', 'control_replacement', 'replacement'):
                raise ValueError('unexpected amended training phase')
            if not training.process_live(identity):
                raise OSError('amended runner missing before verified completion')
            handles[identity['pid']] = identity
            current = state.get('runs', {}).get(phase)
            if current and 'pid' in current and 'returncode' not in current:
                # The runner may be hashing inventory after reaping its child;
                # its own live identity is sufficient for a legitimate wait.
                try:
                    child = training.process_identity(current['pid'])
                except FileNotFoundError:
                    child = None
                if child and child['state'] not in ('Z','X','x'):
                    if (child['argv'] != current['command'] or child['parent_pid'] != identity['pid']
                            or child['start_ticks'] != current['process_identity']['start_ticks']):
                        raise ValueError('training child identity changed')
                    handles[child['pid']] = child
            failures = 0
            if phase != last:
                print(f'{training.now()} waiting for amended training: {phase}; no GPU analysis', flush=True)
                last = phase
        except (OSError, ValueError, KeyError) as error:
            failures += 1
            if failures >= 4:
                raise RuntimeError('cannot verify amended training after repeated handle checks') from error
        time.sleep(poll_seconds)


def validate_completion(root, request, state):
    """Recompute terminal evidence at ACTUAL source paths; do not trust flags."""
    if (request.get('format') != training.FORMAT or request['amendment_root'] != str(root)
            or state.get('format') != training.FORMAT or state.get('phase') != 'training_complete'
            or state.get('complete') is not True or request['seconds_per_arm'] != 14400
            or set(state['runs']) != {'control_replacement','replacement'}):
        raise ValueError('both complete amended runs and full time budgets are required')
    training.verify_records(request['frozen_inputs'])
    legacy = Path(request['legacy_root'])
    manifest = training.read_json(request['legacy_manifest']['path'])
    amendment = training.read_json(request['amendment']['path'])
    if (training.record(legacy/'manifest.json') != request['legacy_manifest']
            or training.record(root/'amendments.json') != request['amendment']):
        raise ValueError('training request names changed experiment inputs')
    paired_training.verify_frozen_inputs(manifest)
    training.validate_amendment(amendment, manifest, request['legacy_manifest'])
    original = training.validate_handoff(legacy, request, manifest)
    reference_path = root/'original_reference.json'
    if (training.record(reference_path) != state.get('original_reference')
            or training.read_json(reference_path) != original):
        raise ValueError('reused original does not match fresh independent validation')
    records = [training.record(reference_path), *original['evidence'], training.record(root/'state.json'),
               training.record(legacy/'determinism_gate.json')]
    inventories = {'original': original['checkpoints']}
    for name in ('control_replacement', 'replacement'):
        directory, run = root/name, state['runs'][name]
        control = name == 'control_replacement'
        expected = training.command(manifest, amendment, directory, control=control)
        if (run.get('returncode') != 0 or run.get('initial_weights_match') is not True
                or run.get('command') != expected or request['commands'].get(name) != expected
                or training.read_json(directory/'command.json') != expected):
            raise ValueError('amended execution command or completion differs')
        if training.process_live(run['process_identity']):
            raise ValueError('amended GPU child still live')
        terminal = paired_training.parse_training_result((directory/'train.log').read_text(),
            seconds=None if control else request['seconds_per_arm'], expected_steps=2 if control else None)
        if control and terminal['stop_reason'] != 'step_limit':
            raise ValueError('two-step control did not stop at its step limit')
        if (any(run.get(key) != value for key,value in terminal.items())
                or run['elapsed_process_seconds'] < terminal['training_elapsed_seconds']):
            raise ValueError('amended terminal state disagrees with actual log')
        items = training.inventory(directory/'checkpoints')
        training.validate_inventory(items, terminal, request['initial_weights'], control=control)
        if (items != training.read_json(directory/'checkpoints.json')
                or run.get('final_checkpoint') != items[-1]['path']):
            raise ValueError('amended terminal checkpoint disagrees with inventory')
        evidence = [training.record(directory/n) for n in ('command.json','train.log','process.log','checkpoints.json')]
        if run.get('evidence') != evidence:
            raise ValueError('amended terminal evidence changed')
        records.extend(evidence)
        inventories[name] = items
    initial = training.inventory(legacy/'initial'/'checkpoints')
    if [i['step'] for i in initial] != [0] or initial[0]['sha256'] != request['initial_weights']:
        raise ValueError('shared saved initialization changed')
    a, b = ({i['step']:i for i in inventories[name]} for name in ('original','replacement'))
    plan = paired_analysis.plan_comparisons(a,b,original['terminal']['final_step'],
        state['runs']['replacement']['final_step'], all_matched_steps=True)
    plan.update(initial=initial[0], initial_copies=[a[0],b[0]],
                arm_roots={'original':str(legacy),'replacement':str(root)})
    checked = [*initial, *inventories['original'], *inventories['replacement'],
               *inventories['control_replacement'],
               *(i for items in request['determinism_gate']['checkpoints'].values() for i in items)]
    training.verify_records([*request['frozen_inputs'],*records])
    return dict(plan=plan, terminal_records=records, checked=checked,
                original_reference=original, determinism_verification=request['determinism_gate'],
                original_reused_not_rerun=True)


def analyze(root, cases_path, probe, output, *, runner_pid, runner_start_ticks, poll_seconds=30):
    root, cases_path, probe, output = (Path(p).absolute() for p in (root,cases_path,probe,output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    request_path = root/'request.json'
    request = training.read_json(request_path)
    if (request.get('format') != training.FORMAT or request['amendment_root'] != str(root)
            or root != Path(request['legacy_root'])/'lowercase_amendment'):
        raise ValueError('not the authenticated amended experiment root')
    forbidden = [cases_path.parent, root/'replacement',root/'control_replacement',
                 *(Path(request['legacy_root'])/a for a in ('initial','original','control_a','control_b'))]
    if any(p.resolve()==output.resolve() or p.resolve() in output.resolve().parents for p in forbidden):
        raise ValueError('analysis output must be outside all case/training inputs')
    plan_cases, case_records = scoring.load_cases(cases_path)
    if plan_cases['amendment'] != request['amendment'] or plan_cases['old_manifest'] != request['legacy_manifest']:
        raise ValueError('cases do not describe this amended training request')
    identity = observer_identity(root, request, runner_pid, runner_start_ticks)
    frozen = [training.record(request_path),training.record(probe),*case_records,*SOURCES]
    training.verify_records([*frozen,*request['frozen_inputs']])
    output.mkdir()
    source_dir = output/'sources'; source_dir.mkdir()
    for source in SOURCES:
        frozen.append(paired_training.freeze_file(source['path'],source_dir/Path(source['path']).name))
    observer = training.process_identity(os.getpid())
    training.publish(output/'request.json',dict(format=FORMAT, amendment_root=str(root),
        runner_identity=identity, observer_identity=observer, frozen_inputs=frozen,
        all_matched_steps=True, goal_completion_claimed=False, created_utc=training.now()))
    try:
        state, handles = wait_for_training(root,identity,poll_seconds=poll_seconds)
        training.verify_records(frozen)
        evidence = validate_completion(root,request,state)
        evidence['verified_process_identities']=handles
        training.publish(output/'analysis_plan.json',evidence)
        plan = evidence['plan']
        weights, behavior, comparisons = {}, {}, {}
        for pair in plan['pairs']:
            path = output/(pair['name']+'_weights.json')
            report = paired_weight_diff.compare_checkpoints(pair['original']['path'],pair['replacement']['path'],
                                                             initial=plan['initial']['path'])
            paired_weight_diff.write_report(path,report)
            weights[pair['name']]=dict(report=training.record(path),model=report['model'])
        raw_summaries = {}
        for index,item in enumerate(plan['checkpoints']):
            training.verify_records(frozen)
            training.require_idle_gpu(request['gpu'])
            paired_analysis._verify_checkpoint(item)
            score_dir=output/f'checkpoint_{index}_scores'
            execution=paired_analysis._run_probe(probe,item['path'],plan_cases['packed_batch']['path'],score_dir,
                                                   output/f'checkpoint_{index}_process.log')
            path=output/f'checkpoint_{index}_behavior.json'
            report=scoring.summarize(cases_path,score_dir,path)
            if Path(report['probe_metadata']['checkpoint_directory']).resolve()!=Path(item['path']).resolve():
                raise ValueError('native probe named a different checkpoint')
            paired_analysis._verify_checkpoint(item)
            behavior[item['path']]=dict(process=execution,report=training.record(path),groups=report['groups'])
            raw_summaries[item['path']]=report
            print(f'{training.now()} scored {index+1}/{len(plan["checkpoints"])} checkpoints',flush=True)
        for alias,source in plan['behavior_aliases'].items():
            behavior[alias]=dict(behavior[source],shared_score_from=source)
            raw_summaries[alias]=raw_summaries[source]
        for pair in plan['pairs']:
            path=output/(pair['name']+'_behavior_comparison.json')
            report=scoring.compare(raw_summaries[pair['original']['path']],raw_summaries[pair['replacement']['path']])
            report.update(pair=pair, endpoint_step_counts_differ=pair['original']['step']!=pair['replacement']['step'])
            training.publish(path,report)
            comparisons[pair['name']]=training.record(path)
        for item in evidence['checked']:
            paired_analysis._verify_checkpoint(item)
        training.verify_records([*frozen,*request['frozen_inputs'],*evidence['terminal_records']])
        result=dict(format=FORMAT,complete=True,completed_utc=training.now(),plan=plan,
            weights=weights,behavior=behavior,behavior_comparisons=comparisons,
            frozen_inputs=frozen,terminal_records=evidence['terminal_records'],
            checkpoints=[dict(path=i['path'],weight_sha256=i['sha256']) for i in evidence['checked']],
            determinism_verification=evidence['determinism_verification'],
            original_reference=evidence['original_reference'],goal_completion_claimed=False,
            limitations=['Unequal final step counts confound endpoint deltas; inspect all common steps.',
                'Weight-delta magnitude and behavior are associations, not a causal storage account.',
                'Title/lowercase results remain separate; no lowercase test occurrences exist.',
                'Exact three native IDs are scored without a following delimiter; supplemental controls remain required.'])
        training.publish(output/'summary.json',result)
        return result
    except BaseException as error:
        training.publish(output/'failure.json',dict(type=type(error).__name__,error=str(error),
                                                  no_automatic_restart=True))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','cases','probe','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('runner-pid','runner-start-ticks'):
        parser.add_argument('--'+name,type=int,required=True)
    args=parser.parse_args(argv)
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'amended analysis received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    analyze(args.root,args.cases,args.probe,args.output,runner_pid=args.runner_pid,
            runner_start_ticks=args.runner_start_ticks)


if __name__=='__main__':
    main()
