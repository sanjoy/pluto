"""Run the amended embedding and branch interventions after trajectory scoring.

This observer owns only its NEW analysis children. It cannot start or restart
training and never signals either training controller or the upstream scorer.
The known scorer must actually exit, completed training must independently
validate, and the recorded GPU must be idle before any test/probe is launched.

Copy controls precede embedding input/output factorials and the full predeclared
attention/MLP transfer screen. Results retain both words' absolute probabilities,
subtoken scores, exact-following-token and shared-piece controls. Every completed
intervention is published separately so partial findings survive interruption.
No completion of this pipeline claims completion of the mechanistic goal.
"""

import argparse
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from . import checkpoint, embedding_factorial_readout, paired_branch_readout
from . import paired_causal_followup as execution
from . import paired_lowercase_analysis as trajectory
from . import paired_lowercase_intervention_plan as planner
from . import paired_lowercase_training as training
from . import paired_weight_patch

FORMAT='pluto-paired-lowercase-causal-followup-v1'
SOURCE_PATHS=sorted({Path(__file__).resolve(),*(Path(m.__file__).resolve() for name,m in tuple(sys.modules.items())
    if name.startswith('scripts.weight_analysis.') and getattr(m,'__file__','').endswith('.py'))})
SOURCES=[training.record(p) for p in SOURCE_PATHS]


def observer_identity(root,pid,start_ticks,upstream_request):
    actual=training.process_identity(pid)
    saved=upstream_request['observer_identity']
    argv=actual['argv']
    if (actual['start_ticks']!=start_ticks or actual['state'] in ('Z','X','x')
            or any(actual[k]!=saved[k] for k in ('pid','start_ticks','argv'))
            or upstream_request.get('format')!=trajectory.FORMAT
            or upstream_request.get('amendment_root')!=str(root)
            or argv.count('-m')!=1 or argv.index('-m')+1>=len(argv)
            or argv[argv.index('-m')+1]!='scripts.weight_analysis.paired_lowercase_analysis'):
        raise ValueError('upstream is not the exact recorded amended trajectory observer')
    return actual


def wait_for_trajectory(path,expected,*,poll_seconds=30):
    """Require two exit observations; transient read failures never grant GPU use."""
    if not 0<poll_seconds<=60: raise ValueError('poll interval must be in (0,60]')
    path=Path(path); missing=errors=0
    while True:
        if (path.parent/'failure.json').exists():
            raise RuntimeError('upstream trajectory failed; no GPU work started')
        try:
            live=training.process_live(expected)
            errors=0
        except (OSError,ValueError) as error:
            missing=0; errors+=1
            if errors>=4: raise RuntimeError('cannot verify upstream trajectory handle') from error
            time.sleep(poll_seconds); continue
        missing=0 if live else missing+1
        if missing>=2 and path.is_file():
            summary=training.read_json(path)
            if (summary.get('format')!=trajectory.FORMAT or summary.get('complete') is not True
                    or summary.get('goal_completion_claimed') is not False):
                raise ValueError('upstream trajectory summary is incomplete')
            return summary
        if missing>=4:
            raise RuntimeError('upstream exited without completed trajectory; no automatic restart')
        time.sleep(poll_seconds)


def run_native(command,inputs,directory,log,record_path,gpu):
    training.require_idle_gpu(gpu)
    return execution.execute_recorded(command,inputs,directory,log,record_path)


def score_loss(binary,checkpoint_dir,cases_path,directory,log,record_path,gpu):
    cases=training.read_json(cases_path); batch=cases['packed_batch']['path']
    command=[binary,f'--checkpoint={checkpoint_dir}',f'--batch={batch}',f'--output_dir={directory}','--batch_sequences=1']
    run_native(command,[binary,cases_path,batch,*execution.checkpoint_inputs(checkpoint_dir)],
               directory,log,record_path,gpu)
    # Validate complete native metadata, shapes and finite dumps immediately;
    # the full branch reader later verifies exact execution/weight provenance.
    metadata=training.read_json(directory/'metadata.json')
    count,length=cases['case_count'],cases['context_length']
    if (metadata.get('complete') is not True or metadata.get('kind')!='paired_loss_probe'
            or metadata.get('temperature')!=1 or metadata.get('case_count')!=count
            or metadata.get('byte_order')!='little' or metadata.get('loss_dtype')!='<f4'
            or metadata.get('argmax_dtype')!='<i4' or metadata.get('loss_file')!='losses.f32.bin'
            or metadata.get('argmax_file')!='argmax.i32.bin'
            or metadata.get('output_shape')!=[count,length]
            or Path(metadata['checkpoint_directory']).resolve()!=Path(checkpoint_dir).resolve()
            or Path(metadata['batch_file']).resolve()!=Path(batch).resolve()):
        raise ValueError('native loss probe metadata disagrees with its actual inputs')
    if any((directory/name).stat().st_size!=count*length*4 for name in ('losses.f32.bin','argmax.i32.bin')):
        raise ValueError('native loss/argmax file size mismatch')
    losses=np.fromfile(directory/'losses.f32.bin',dtype='<f4')
    winners=np.fromfile(directory/'argmax.i32.bin',dtype='<i4')
    if (losses.size!=count*length or winners.size!=count*length
            or not np.isfinite(losses).all() or np.any(losses<0)
            or np.any(winners<0) or np.any(winners>=cases['vocab_size'])):
        raise ValueError('invalid native loss/argmax dump')
    return dict(scores=str(directory),execution=training.record(record_path))


def verify_copy(main_scores,recipient,summary):
    """A weight-identical copy must reproduce ALL native rows, not rounded loss."""
    reference=summary['behavior'][recipient]['report']
    training.verify_records([reference])
    report=training.read_json(reference['path'])
    if Path(report['probe_metadata']['checkpoint_directory']).resolve()!=Path(recipient).resolve():
        raise ValueError('copy reference names another checkpoint')
    scores=report['scores']
    for key,name in (('losses','losses.f32.bin'),('argmax','argmax.i32.bin')):
        training.verify_records([scores[key]])
        if checkpoint.sha256_file(Path(main_scores)/name)!=scores[key]['sha256']:
            raise ValueError('copy control differs from trajectory native '+key)


def verify_copy_suite_prefixes(main_path,supplemental_path,results):
    """The added fourth target must not change any of the first three scores."""
    main,supplemental=(training.read_json(p) for p in (main_path,supplemental_path))
    arrays={}
    for suite,plan in (('main',main),('supplemental',supplemental)):
        arrays[suite]={name:np.fromfile(Path(results[suite]['scores'])/filename,dtype=dtype).reshape(
            plan['case_count'],plan['context_length']) for name,filename,dtype in
            (('losses','losses.f32.bin','<f4'),('argmax','argmax.i32.bin','<i4'))}
    for case in supplemental['cases']:
        if case['kind']!='word_next_native': continue
        source=main['cases'][case['source_case_index']]
        if source['target_ids']!=case['target_ids'][:3]: raise ValueError('word targets differ between suites')
        for name in ('losses','argmax'):
            a=arrays['main'][name][source['case_index'],source['scored_rows']]
            b=arrays['supplemental'][name][case['case_index'],case['scored_rows'][:3]]
            if a.tobytes()!=b.tobytes(): raise ValueError('copy main/supplemental first-three scores differ')


def stage_result(stage,evidence):
    paths=sorted(stage.rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('symlink in completed intervention output')
    evidence['artifacts']=[training.record(p) for p in paths if p.is_file()]
    training.publish(stage/'complete.json',evidence)
    return [*evidence['artifacts'],training.record(stage/'complete.json')]


def run_interventions(root,output,summary,plan,binaries,gpu,frozen):
    """Execute the already-authenticated plan in dependency order, serially."""
    main_path=root/'word_cases/cases.json'; supp_path=root/'supplemental_cases/cases.json'
    main=training.read_json(main_path)
    # Copy controls supply both source-arm baselines for every branch transfer.
    ordered=[i for i in plan['interventions'] if i['kind']=='copy_control']
    ordered += [i for i in plan['interventions'] if i['kind']=='embedding_rows']
    ordered += [i for i in plan['interventions'] if i['tensors']]
    if len(ordered)!=len(plan['interventions']) or len({i['name'] for i in ordered})!=len(ordered):
        raise ValueError('incomplete or duplicate intervention ordering')
    completed=[]; artifacts=[]; baselines={}
    state=dict(format=FORMAT,phase='interventions',runner_pid=os.getpid(),completed=[],total=len(ordered))
    for item in ordered:
        training.verify_records([*frozen,*plan['frozen_inputs'],*plan['implementation']])
        name=item['name']
        if Path(name).name!=name or name in ('','.','..'): raise ValueError('invalid intervention name')
        stage=output/name; stage.mkdir()
        recipient=item['recipient_checkpoint']['path']; donor=item['donor_checkpoint']['path']
        for key in ('recipient_checkpoint','donor_checkpoint'): trajectory.paired_analysis._verify_checkpoint(item[key])
        patch_dir=stage/f'step_{item["step"]}'
        paired_weight_patch.create_patch(recipient,donor,patch_dir,tensors=item['tensors'],embedding_rows=item['embedding_rows'])
        evidence=dict(intervention=item,patch=training.record(patch_dir/'patch.json'))
        state.update(current=name,updated_utc=training.now()); training.publish(output/'state.json',state,exclusive=False)
        if item['kind']=='copy_control' or item['tensors']:
            measured={}
            for suite,path in (('main',main_path),('supplemental',supp_path)):
                measured[suite]=score_loss(binaries['loss_probe']['path'],patch_dir,path,stage/(suite+'_scores'),
                    stage/(suite+'.log'),stage/(suite+'_execution.json'),gpu)
            evidence['native']=measured
            if item['kind']=='copy_control':
                verify_copy(measured['main']['scores'],recipient,summary)
                verify_copy_suite_prefixes(main_path,supp_path,measured)
                evidence['all_main_scores_equal_trajectory']=True
                evidence['main_supplemental_first_three_byte_equal']=True
                baselines[item['step'],item['recipient_arm']]=evidence
            else:
                baseline={role:baselines[item['step'],item[role+'_arm']]['native'] for role in ('recipient','donor')}
                baseline['patched']=measured
                scores={role:{suite:baseline[role][suite]['scores'] for suite in ('main','supplemental')} for role in baseline}
                executions={role:{suite:baseline[role][suite]['execution']['path'] for suite in ('main','supplemental')} for role in baseline}
                path=stage/'branch_readout.json'
                paired_branch_readout.analyze(item,patch_dir/'patch.json',main_path,supp_path,scores,path,executions=executions)
                evidence['readout']=training.record(path)
        else:
            evidence['factorial']={}
            for export_name,source_path,kind in (('main',main_path,None),
                ('word_next_native',supp_path,'word_next_native'),('shared_piece',supp_path,'shared_piece')):
                exported=plan['exports'][export_name]
                batch=main['packed_batch']['path'] if export_name=='main' else exported['packed_batch']['path']
                rows=exported['selected_rows']['path']; directory=stage/('factorial_'+export_name)
                execution_path=stage/('factorial_'+export_name+'_execution.json')
                binary=binaries['factorial_probe']['path']
                command=[binary,f'--recipient={recipient}',f'--patched={patch_dir}',f'--batch={batch}',
                    f'--rows={rows}',f'--rows_per_case={exported["rows_per_case"]}','--batch_sequences=1',f'--output_dir={directory}']
                run_native(command,[binary,source_path,batch,rows,*execution.checkpoint_inputs(recipient),
                    *execution.checkpoint_inputs(patch_dir)],directory,stage/('factorial_'+export_name+'.log'),execution_path,gpu)
                path=stage/('factorial_'+export_name+'_readout.json')
                embedding_factorial_readout.analyze(source_path,directory,patch_dir/'patch.json',path,
                    expected_rows=item['embedding_rows'],case_kind=kind,execution_record=execution_path)
                evidence['factorial'][export_name]=training.record(path)
        training.verify_records([*frozen,evidence['patch']])
        artifacts.extend(stage_result(stage,evidence)); completed.append(evidence)
        state['completed']=[r['intervention']['name'] for r in completed]
        training.publish(output/'state.json',state,exclusive=False)
        print(f'{training.now()} completed {name}; {len(completed)}/{len(ordered)} interventions',flush=True)
    training.verify_records([*frozen,*plan['frozen_inputs'],*plan['implementation'],*artifacts])
    unique={i[key]['path']:i[key] for i in ordered for key in ('recipient_checkpoint','donor_checkpoint')}
    for item in unique.values(): trajectory.paired_analysis._verify_checkpoint(item)
    return completed,artifacts


def run(args):
    root=Path(args.root).resolve(strict=True); output=Path(args.output).absolute()
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    if output.parent!=root: raise ValueError('new causal output must be an immediate child of the amended root')
    upstream_path=root/'analysis_trajectory/request.json'
    upstream=training.read_json(upstream_path)
    expected=observer_identity(root,args.observer_pid,args.observer_start_ticks,upstream)
    request=training.read_json(root/'request.json')
    main_path=root/'word_cases/cases.json'; supp_path=root/'supplemental_cases/cases.json'
    exports_path=root/'causal_cases/exports.json'
    _,_,_,case_records=planner.validate_exports(exports_path,main_path,supp_path)
    binary_sources={name:training.record(getattr(args,name)) for name in ('loss_probe','factorial_probe','gpu_test')}
    frozen=[training.record(upstream_path),training.record(root/'request.json'),*case_records,*SOURCES,*binary_sources.values()]
    training.verify_records([*frozen,*upstream['frozen_inputs'],*request['frozen_inputs']])
    output.mkdir(); (output/'bin').mkdir(); (output/'source').mkdir()
    binaries={name:execution.paired_training.freeze_file(Path(getattr(args,name)),output/'bin'/name,executable=True)
              for name in ('loss_probe','factorial_probe','gpu_test')}
    if any(binaries[name]['sha256']!=source['sha256'] or binaries[name]['bytes']!=source['bytes']
           for name,source in binary_sources.items()):
        raise ValueError('native executable changed during freeze')
    copies=[execution.paired_training.freeze_file(Path(r['path']),output/'source'/Path(r['path']).name) for r in SOURCES]
    frozen.extend([*binaries.values(),*copies])
    training.verify_records(frozen)
    training.publish(output/'request.json',dict(format=FORMAT,created_utc=training.now(),stage='embedding_and_branches',
        runner_identity=training.process_identity(os.getpid()),upstream_identity=expected,amendment_root=str(root),
        binaries=binaries,frozen_inputs=frozen,goal_completion_claimed=False))
    training.publish(output/'state.json',dict(format=FORMAT,phase='waiting_trajectory',runner_pid=os.getpid(),completed=[]))
    try:
        print(f'{training.now()} waiting for exact amended trajectory exit; no GPU work',flush=True)
        summary_path=root/'analysis_trajectory/summary.json'
        summary=wait_for_trajectory(summary_path,expected)
        training.verify_records(frozen)
        plan=planner.prepare(root,summary_path,exports_path,output/'plan')
        plan_record=training.record(output/'plan/plan.json')
        frozen.append(plan_record)
        tests=output/'gpu_validation'; tests.mkdir(); test_file=tests/'gtest.json'
        training.publish(output/'state.json',dict(format=FORMAT,phase='gpu_validation',runner_pid=os.getpid(),completed=[]),exclusive=False)
        test_run=run_native([binaries['gpu_test']['path'],f'--gtest_output=json:{test_file}'],
            [binaries['gpu_test']['path']],tests,output/'gpu_test.log',output/'gpu_test_execution.json',request['gpu'])
        execution.require_gpu_test_success(test_file)
        completed,artifacts=run_interventions(root,output,summary,plan,binaries,request['gpu'],frozen)
        training.verify_records([*frozen,*test_run['inputs_before'],*test_run['outputs'],test_run['log']])
        result=dict(format=FORMAT,complete=True,goal_completion_claimed=False,completed_utc=training.now(),
            plan=plan_record,gpu_test=test_run,completed=completed,completed_artifact_records=artifacts,
            frozen_inputs=frozen,remaining_research=['Interpret transfer effects together with checkpoint trajectories.',
                'Targeted head/source-value and neuron ablations for a computational account of each spelling piece.',
                'Confirmation controls and limitations; transfer is contribution, not unique storage.'])
        training.publish(output/'summary.json',result)
        training.publish(output/'state.json',dict(format=FORMAT,phase='complete',runner_pid=os.getpid(),
            completed=[c['intervention']['name'] for c in completed]),exclusive=False)
        return result
    except BaseException as error:
        training.publish(output/'failure.json',dict(type=type(error).__name__,error=str(error),
            no_automatic_restart=True,training_restarted=False,failed_utc=training.now()))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','loss-probe','factorial-probe','gpu-test'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('observer-pid','observer-start-ticks'):
        parser.add_argument('--'+name,type=int,required=True)
    args=parser.parse_args(argv)
    def interrupted(signum,frame): raise KeyboardInterrupt(f'amended causal observer received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    run(args)


if __name__=='__main__':
    main()
