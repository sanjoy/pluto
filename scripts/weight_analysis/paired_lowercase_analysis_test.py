"""CPU-only orchestration contracts; every process launch is mocked."""

from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import paired_lowercase_analysis as analysis
from . import paired_lowercase_training as training
from . import paired_lowercase_training_test as fixtures


class WaitingTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.identity=fixtures.identity(50)
        self.state=dict(format=training.FORMAT,phase='waiting_original',runner_pid=50,runs={})
        self.sleep=self.enterContext(mock.patch.object(analysis.time,'sleep'))
        self.launch=self.enterContext(mock.patch.object(analysis.paired_analysis,'_run_probe'))

    def write(self):
        training.publish(self.root/'state.json',self.state,exclusive=False)

    def test_transient_observation_failure_is_repolled_not_terminal(self):
        terminal=dict(self.state,phase='training_complete')
        with mock.patch.object(training,'read_json',side_effect=[OSError('timeout'),self.state,terminal]), \
             mock.patch.object(training,'process_live',return_value=True) as live, \
             mock.patch.object(training,'wait_for_exit') as exit_wait:
            state,handles=analysis.wait_for_training(self.root,self.identity,poll_seconds=1)
        self.assertEqual(state['phase'],'training_complete')
        self.assertEqual(handles,[self.identity]); live.assert_called_once()
        exit_wait.assert_called_once_with([self.identity],poll_seconds=1)
        self.assertEqual(self.sleep.call_count,2); self.launch.assert_not_called()

    def test_dead_handle_does_not_mean_training_complete(self):
        self.write()
        with mock.patch.object(training,'process_live',return_value=False) as live:
            with self.assertRaisesRegex(RuntimeError,'repeated handle'):
                analysis.wait_for_training(self.root,self.identity,poll_seconds=1)
        self.assertEqual(live.call_count,4); self.launch.assert_not_called()

    def test_failed_state_or_changed_pid_rejected(self):
        for changes in (dict(phase='failed',error='bad'),dict(runner_pid=51)):
            self.state.update(changes); self.write()
            with self.assertRaises(RuntimeError):
                analysis.wait_for_training(self.root,self.identity,poll_seconds=1)
        self.launch.assert_not_called()

    def test_invalid_poll_intervals(self):
        for seconds in (0,-1,61,float('nan')):
            with self.assertRaises(ValueError):
                analysis.wait_for_training(self.root,self.identity,poll_seconds=seconds)

    def test_live_runner_allows_reaped_child_inventory_phase(self):
        self.state.update(phase='replacement',runs={'replacement':dict(pid=60,command=['trainer'])})
        with mock.patch.object(training,'read_json',side_effect=[self.state,dict(self.state,phase='training_complete')]), \
             mock.patch.object(training,'process_live',return_value=True), \
             mock.patch.object(training,'process_identity',side_effect=FileNotFoundError), \
             mock.patch.object(training,'wait_for_exit'):
            _,handles=analysis.wait_for_training(self.root,self.identity,poll_seconds=1)
        self.assertEqual(handles,[self.identity])

    def test_live_child_must_match_parent_command_and_start(self):
        child=fixtures.identity(60,argv=['trainer']); child['parent_pid']=50
        self.state.update(phase='replacement',runs={'replacement':dict(pid=60,command=['trainer'],process_identity=child)})
        self.write()
        for changes in (dict(parent_pid=99),dict(start_ticks=21),dict(argv=['other'])):
            with mock.patch.object(training,'process_live',return_value=True), \
                 mock.patch.object(training,'process_identity',return_value=dict(child,**changes)):
                with self.assertRaises(RuntimeError):
                    analysis.wait_for_training(self.root,self.identity,poll_seconds=1)

    def test_exact_controller_argv_is_required(self):
        request=dict(legacy_root='/legacy',upstream_processes=[fixtures.identity(10,100),fixtures.identity(11,101)])
        suffix=['-m','scripts.weight_analysis.paired_lowercase_training','run','--legacy-root','/legacy',
                '--amendment',str(self.root),'--supervisor-pid','10','--supervisor-start-ticks','100',
                '--original-pid','11','--original-start-ticks','101']
        identity=fixtures.identity(50,argv=['python','-u',*suffix])
        with mock.patch.object(training,'process_identity',return_value=identity):
            self.assertEqual(analysis.observer_identity(self.root,request,50,20),identity)
            with self.assertRaises(ValueError): analysis.observer_identity(self.root,request,50,21)
        for changes in (dict(argv=['python','wrong']),dict(state='Z')):
            with mock.patch.object(training,'process_identity',return_value=dict(identity,**changes)):
                with self.assertRaises(ValueError): analysis.observer_identity(self.root,request,50,20)


class CompletionTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.legacy=Path(self.temporary.name)
        self.root=self.legacy/'lowercase_amendment'; self.root.mkdir()
        self.hashes={'weight_0.bin':'initial'}
        self.manifest=dict(root=str(self.legacy),flags={'seed':17,'batch_size':10},seconds_per_arm=14400.0,
                           binaries={'trainer':{'path':'/frozen/trainer'}})
        self.amendment=dict(inputs={'replacement.full':{'text':'/amended/corpus'}})
        training.publish(self.legacy/'manifest.json',self.manifest)
        training.publish(self.root/'amendments.json',self.amendment)
        self.request=dict(format=training.FORMAT,amendment_root=str(self.root),legacy_root=str(self.legacy),
            legacy_manifest=training.record(self.legacy/'manifest.json'),amendment=training.record(self.root/'amendments.json'),
            initial_weights=self.hashes,frozen_inputs=[],seconds_per_arm=14400.0,commands={},
            determinism_gate={'status':'verified','checkpoints':{}})
        training.publish(self.legacy/'determinism_gate.json',self.request['determinism_gate'])
        self.state=dict(format=training.FORMAT,phase='training_complete',complete=True,runner_pid=50,runs={})
        self.inventories={}
        for parent,arm,steps in ((self.legacy,'initial',[0]),(self.legacy,'original',[0,100,200,250]),
                                (self.root,'control_replacement',[0,1,2]),(self.root,'replacement',[0,100,200,240])):
            path=parent/arm; path.mkdir()
            items=[dict(step=s,path=str(path/f'checkpoints/step_{s}'),
                        sha256=self.hashes if s==0 else {'weight_0.bin':f'{arm}:{s}'}) for s in steps]
            self.inventories[str(path/'checkpoints')]=items
            if parent==self.legacy: continue
            control=arm=='control_replacement'
            command=training.command(self.manifest,self.amendment,path,control=control)
            self.request['commands'][arm]=command
            training.publish(path/'command.json',command)
            reason='step_limit' if control else 'time_limit'; elapsed=95. if control else 14401.
            log=(f'training stopped at step: {steps[-1]}\ntraining stop reason: {reason}\n'
                 f'training elapsed seconds: {elapsed}\nfinal training loss: 4\nfinal test loss: 4.1\n')
            (path/'train.log').write_text(log); (path/'process.log').write_text(log)
            training.publish(path/'checkpoints.json',items)
            terminal=analysis.paired_training.parse_training_result(log)
            self.state['runs'][arm]=dict(command=command,returncode=0,initial_weights_match=True,
                elapsed_process_seconds=elapsed+1,process_identity=fixtures.identity(60),
                final_checkpoint=items[-1]['path'],**terminal,
                evidence=[training.record(path/n) for n in ('command.json','train.log','process.log','checkpoints.json')])
        self.original=dict(format='pluto-reused-original-training-v1',complete=True,reused_not_rerun=True,
            checkpoints=self.inventories[str(self.legacy/'original/checkpoints')],terminal={'final_step':250},evidence=[])
        training.publish(self.root/'original_reference.json',self.original)
        self.state['original_reference']=training.record(self.root/'original_reference.json')
        training.publish(self.root/'state.json',self.state)
        # Corpus and legacy handoff internals have their own real-format tests.
        # Here we exercise the new bridge's command/log/evidence/step checks.
        self.enterContext(mock.patch.object(analysis.paired_training,'verify_frozen_inputs'))
        self.enterContext(mock.patch.object(training,'validate_amendment'))
        self.enterContext(mock.patch.object(training,'validate_handoff',return_value=self.original))
        self.enterContext(mock.patch.object(training,'inventory',side_effect=lambda p:self.inventories[str(p)]))
        self.live=self.enterContext(mock.patch.object(training,'process_live',return_value=False))

    def validate(self): return analysis.validate_completion(self.root,self.request,self.state)

    def test_actual_arm_roots_all_common_steps_and_unequal_endpoints(self):
        evidence=self.validate(); plan=evidence['plan']
        self.assertEqual(plan['arm_roots'],{'original':str(self.legacy),'replacement':str(self.root)})
        self.assertEqual(plan['matched_steps'],[0,100,200])
        self.assertEqual([p['name'] for p in plan['pairs']],['final','initial','matched_step_100','matched_step_200'])
        self.assertEqual(plan['pairs'][0]['original']['step'],250)
        self.assertEqual(plan['pairs'][0]['replacement']['step'],240)
        self.assertTrue(evidence['original_reused_not_rerun'])
        self.assertEqual(len(plan['behavior_aliases']),1)

    def test_incomplete_or_short_budget_rejected(self):
        for key,value in (('complete',False),('phase','replacement')):
            old=self.state[key]; self.state[key]=value
            with self.assertRaises(ValueError): self.validate()
            self.state[key]=old
        self.request['seconds_per_arm']=14399
        with self.assertRaises(ValueError): self.validate()

    def test_changed_original_reference_rejected(self):
        path=self.root/'original_reference.json'; path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'reused original'): self.validate()

    def test_command_cannot_revert_to_old_corpus_or_new_init(self):
        run=self.state['runs']['replacement']; run['command']=run['command']+['--corpus=/old']
        with self.assertRaisesRegex(ValueError,'command'): self.validate()

    def test_real_log_must_show_full_budget(self):
        path=self.root/'replacement/train.log'; path.write_text(path.read_text().replace('14401.0','14399.0'))
        with self.assertRaisesRegex(ValueError,'wall-clock'): self.validate()

    def test_terminal_values_must_match_log(self):
        self.state['runs']['replacement']['test_loss']=5
        with self.assertRaisesRegex(ValueError,'terminal state'): self.validate()

    def test_control_reason_must_be_step_limit(self):
        path=self.root/'control_replacement/train.log'; path.write_text(path.read_text().replace('step_limit','time_limit'))
        with self.assertRaisesRegex(ValueError,'step limit'): self.validate()

    def test_live_gpu_child_prevents_scoring(self):
        self.live.return_value=True
        with self.assertRaisesRegex(ValueError,'still live'): self.validate()

    def test_missing_periodic_checkpoint_or_wrong_initial_rejected(self):
        path=str(self.root/'replacement/checkpoints'); items=self.inventories[path]
        self.inventories[path]=items[:1]+items[2:]
        with self.assertRaisesRegex(ValueError,'periodic'): self.validate()
        self.inventories[path]=copy.deepcopy(items); self.inventories[path][0]['sha256']={}
        with self.assertRaisesRegex(ValueError,'initialization'): self.validate()

    def test_inventory_and_final_path_must_match_actual_evidence(self):
        self.state['runs']['replacement']['final_checkpoint']='/other/step_240'
        with self.assertRaisesRegex(ValueError,'checkpoint'): self.validate()

    def test_mutated_process_evidence_rejected(self):
        (self.root/'replacement/process.log').write_text('changed')
        with self.assertRaisesRegex(ValueError,'evidence changed'): self.validate()


class OrchestrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.legacy=Path(self.temporary.name)
        self.root=self.legacy/'lowercase_amendment'; self.root.mkdir()
        case_dir=self.root/'word_cases'; case_dir.mkdir()
        self.case_path=case_dir/'cases.json'; self.case_path.write_text('{}')
        self.probe=self.legacy/'probe'; self.probe.write_text('mock only, not executable')
        self.output=self.root/'analysis'
        self.request=dict(format=training.FORMAT,amendment_root=str(self.root),legacy_root=str(self.legacy),
            amendment={'id':'amended'},legacy_manifest={'id':'original'},frozen_inputs=[],gpu={'uuid':'test'})
        training.publish(self.root/'request.json',self.request)
        case_plan=dict(amendment=self.request['amendment'],old_manifest=self.request['legacy_manifest'],
                       packed_batch={'path':'/frozen/batch'})
        self.enterContext(mock.patch.object(analysis.scoring,'load_cases',return_value=(case_plan,[training.record(self.case_path)])))
        self.enterContext(mock.patch.object(analysis,'observer_identity',return_value=fixtures.identity(50)))
        self.wait=self.enterContext(mock.patch.object(analysis,'wait_for_training',return_value=({},[])))
        a=dict(step=100,path='/original/checkpoints/step_100',sha256={})
        b=dict(step=100,path='/amended/replacement/checkpoints/step_100',sha256={})
        pair=dict(name='final',original=a,replacement=b)
        self.evidence=dict(plan=dict(pairs=[pair],initial={'path':'/initial/step_0'},checkpoints=[a,b],
                                     behavior_aliases={}),checked=[],terminal_records=[],
                           determinism_verification={'status':'verified'},original_reference={'reused_not_rerun':True})
        self.validate=self.enterContext(mock.patch.object(analysis,'validate_completion',return_value=self.evidence))
        self.idle=self.enterContext(mock.patch.object(training,'require_idle_gpu'))
        self.enterContext(mock.patch.object(analysis.paired_analysis,'_verify_checkpoint'))
        self.enterContext(mock.patch.object(analysis.paired_weight_diff,'compare_checkpoints',return_value={'model':{}}))
        self.probes=self.enterContext(mock.patch.object(analysis.paired_analysis,'_run_probe',return_value={'returncode':0}))
        def summarize(cases,directory,path):
            index=int(directory.name.split('_')[1]); checkpoint_path=(a,b)[index]['path']
            report=dict(probe_metadata={'checkpoint_directory':checkpoint_path},groups=[])
            training.publish(path,report); return report
        self.enterContext(mock.patch.object(analysis.scoring,'summarize',side_effect=summarize))
        self.enterContext(mock.patch.object(analysis.scoring,'compare',return_value={'fixture':'comparison'}))

    def run_analysis(self):
        return analysis.analyze(self.root,self.case_path,self.probe,self.output,
                                runner_pid=50,runner_start_ticks=20,poll_seconds=1)

    def test_success_provenance_and_ordered_idle_gates(self):
        result=self.run_analysis()
        self.assertTrue(result['complete']); self.assertFalse(result['goal_completion_claimed'])
        self.assertEqual(self.probes.call_count,2); self.assertEqual(self.idle.call_count,2)
        self.wait.assert_called_once(); self.validate.assert_called_once()
        self.assertTrue((self.output/'request.json').is_file())
        self.assertTrue((self.output/'analysis_plan.json').is_file())
        self.assertEqual(len(list((self.output/'sources').iterdir())),len(analysis.SOURCES))
        self.assertEqual(set(result['behavior']),{'/original/checkpoints/step_100','/amended/replacement/checkpoints/step_100'})

    def test_incomplete_training_gate_prevents_gpu_work(self):
        self.validate.side_effect=ValueError('incomplete')
        with self.assertRaises(ValueError): self.run_analysis()
        self.probes.assert_not_called(); self.idle.assert_not_called()
        self.assertTrue((self.output/'failure.json').is_file())

    def test_busy_gpu_prevents_probe(self):
        self.idle.side_effect=RuntimeError('GPU busy')
        with self.assertRaises(RuntimeError): self.run_analysis()
        self.probes.assert_not_called()

    def test_frozen_probe_mutation_while_waiting_prevents_probe(self):
        def waited(*args,**kwargs):
            self.probe.write_text('changed'); return {},[]
        self.wait.side_effect=waited
        with self.assertRaisesRegex(ValueError,'frozen evidence'): self.run_analysis()
        self.probes.assert_not_called(); self.validate.assert_not_called()

    def test_existing_output_is_preserved(self):
        self.output.mkdir(); (self.output/'keep').write_text('keep')
        with self.assertRaises(FileExistsError): self.run_analysis()
        self.probes.assert_not_called()
        self.assertEqual((self.output/'keep').read_text(),'keep')


if __name__=='__main__':
    unittest.main()
