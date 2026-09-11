"""CPU-only process/ordering/provenance contracts; no trainer or GPU runs."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_lowercase_causal_followup as followup
from . import paired_lowercase_training as training
from . import paired_lowercase_training_test as process_fixtures


class WaitingTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name); self.summary=self.root/'summary.json'
        self.identity=process_fixtures.identity()
        self.sleep=self.enterContext(mock.patch.object(followup.time,'sleep'))
        self.native=self.enterContext(mock.patch.object(followup.execution,'execute_recorded'))

    def complete(self):
        value=dict(format=followup.trajectory.FORMAT,complete=True,goal_completion_claimed=False)
        training.publish(self.summary,value); return value

    def test_two_exits_required_even_when_summary_already_present(self):
        expected=self.complete()
        with mock.patch.object(training,'process_live',side_effect=[True,False,False]) as live:
            self.assertEqual(followup.wait_for_trajectory(self.summary,self.identity,poll_seconds=1),expected)
        self.assertEqual(live.call_count,3); self.native.assert_not_called()

    def test_transient_handle_error_never_counts_as_exit(self):
        self.complete()
        with mock.patch.object(training,'process_live',side_effect=[False,OSError('timeout'),False,False]) as live:
            followup.wait_for_trajectory(self.summary,self.identity,poll_seconds=1)
        self.assertEqual(live.call_count,4)

    def test_missing_or_failed_upstream_does_not_launch(self):
        with mock.patch.object(training,'process_live',return_value=False) as live:
            with self.assertRaisesRegex(RuntimeError,'without completed'): followup.wait_for_trajectory(self.summary,self.identity,poll_seconds=1)
        self.assertEqual(live.call_count,4)
        training.publish(self.root/'failure.json',{'error':'fixture'})
        with self.assertRaisesRegex(RuntimeError,'trajectory failed'): followup.wait_for_trajectory(self.summary,self.identity,poll_seconds=1)
        self.native.assert_not_called()

    def test_pid_reuse_and_repeated_read_errors_fail_closed(self):
        for error in (RuntimeError('PID reused'),OSError('read failed')):
            with mock.patch.object(training,'process_live',side_effect=error),self.assertRaises(RuntimeError):
                followup.wait_for_trajectory(self.summary,self.identity,poll_seconds=1)
        self.native.assert_not_called()

    def test_incomplete_or_wrong_format_summary_rejected(self):
        training.publish(self.summary,{'format':'old','complete':True})
        with mock.patch.object(training,'process_live',return_value=False),self.assertRaises(ValueError):
            followup.wait_for_trajectory(self.summary,self.identity,poll_seconds=1)

    def test_idle_gpu_gate_precedes_every_native_launch(self):
        with mock.patch.object(training,'require_idle_gpu',side_effect=RuntimeError('busy')):
            with self.assertRaises(RuntimeError): followup.run_native(['not-run'],[],self.root,self.root/'log',self.root/'run.json',{})
        self.native.assert_not_called()

    def test_observer_must_match_saved_identity_and_amended_module(self):
        identity=process_fixtures.identity(argv=['python','-m','weight_analysis.paired_lowercase_analysis'])
        request=dict(format=followup.trajectory.FORMAT,amendment_root=str(self.root),observer_identity=identity)
        with mock.patch.object(training,'process_identity',return_value=identity):
            self.assertEqual(followup.observer_identity(self.root,10,20,request),identity)
            with self.assertRaises(ValueError): followup.observer_identity(self.root,10,21,request)


class CopyTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name); self.scores=self.root/'scores'; self.scores.mkdir()

    def test_copy_compares_whole_dumps_and_correct_checkpoint(self):
        records={}
        for key,name in (('losses','losses.f32.bin'),('argmax','argmax.i32.bin')):
            (self.root/name).write_bytes(b'abcd'); (self.scores/name).write_bytes(b'abcd')
            records[key]=training.record(self.root/name)
        report=self.root/'report.json'
        training.publish(report,dict(scores=records,probe_metadata={'checkpoint_directory':'/model/step_100'}))
        summary={'behavior':{'/model/step_100':{'report':training.record(report)}}}
        followup.verify_copy(self.scores,'/model/step_100',summary)
        (self.scores/'losses.f32.bin').write_bytes(b'abce')
        with self.assertRaisesRegex(ValueError,'native losses'): followup.verify_copy(self.scores,'/model/step_100',summary)

    def test_fourth_target_must_not_change_first_three_scores(self):
        main=dict(case_count=1,context_length=5,cases=[dict(case_index=0,target_ids=[1,2,3],scored_rows=[0,1,2])])
        supp=dict(case_count=1,context_length=5,cases=[dict(kind='word_next_native',case_index=0,source_case_index=0,target_ids=[1,2,3,4],scored_rows=[0,1,2,3])])
        training.publish(self.root/'main.json',main); training.publish(self.root/'supp.json',supp)
        results={}
        for suite in ('main','supplemental'):
            directory=self.root/suite; directory.mkdir(); results[suite]={'scores':str(directory)}
            np.arange(5,dtype='<f4').tofile(directory/'losses.f32.bin')
            np.arange(5,dtype='<i4').tofile(directory/'argmax.i32.bin')
        followup.verify_copy_suite_prefixes(self.root/'main.json',self.root/'supp.json',results)
        np.ones(5,dtype='<f4').tofile(self.root/'supplemental/losses.f32.bin')
        with self.assertRaisesRegex(ValueError,'first-three'): followup.verify_copy_suite_prefixes(self.root/'main.json',self.root/'supp.json',results)

    def test_loss_probe_metadata_finite_values_and_exact_file_sizes(self):
        case_path=self.root/'cases.json'
        training.publish(case_path,dict(case_count=1,context_length=2,vocab_size=10,packed_batch={'path':'/batch'}))
        metadata=dict(complete=True,kind='paired_loss_probe',temperature=1,case_count=1,output_shape=[1,2],
            checkpoint_directory='/model',batch_file='/batch',byte_order='little',loss_dtype='<f4',argmax_dtype='<i4',
            loss_file='losses.f32.bin',argmax_file='argmax.i32.bin')
        def invoke():
            return followup.score_loss('/binary',Path('/model'),case_path,self.scores,self.root/'log',self.root/'execution.json',{})
        training.publish(self.scores/'metadata.json',metadata)
        np.ones(2,dtype='<f4').tofile(self.scores/'losses.f32.bin'); np.zeros(2,dtype='<i4').tofile(self.scores/'argmax.i32.bin')
        with mock.patch.object(followup,'run_native'):
            training.publish(self.root/'execution.json',{'mock':True})
            self.assertEqual(invoke()['scores'],str(self.scores))
            np.array([np.nan,1],dtype='<f4').tofile(self.scores/'losses.f32.bin')
            with self.assertRaisesRegex(ValueError,'invalid native'): invoke()
            (self.scores/'losses.f32.bin').write_bytes(b'123456789')
            with self.assertRaisesRegex(ValueError,'size mismatch'): invoke()
            metadata['temperature']=.7; training.publish(self.scores/'metadata.json',metadata,exclusive=False)
            with self.assertRaisesRegex(ValueError,'metadata'): invoke()


class EngineTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name); self.output=self.root/'output'; self.output.mkdir()
        for name in ('word_cases','supplemental_cases'):
            (self.root/name).mkdir(); training.publish(self.root/name/'cases.json',{'packed_batch':{'path':'/batch'}})
        sources={arm:dict(step=100,path='/'+arm+'/step_100',sha256={}) for arm in ('original','replacement')}
        def item(name,kind,recipient='original'):
            donor='replacement' if recipient=='original' else 'original'
            return dict(name=name,kind=kind,step=100,recipient_arm=recipient,donor_arm=donor,
                recipient_checkpoint=sources[recipient],donor_checkpoint=sources[donor],
                tensors=['branch'] if kind=='attention_whole_branch' else [],embedding_rows=[1,2] if kind=='embedding_rows' else [])
        self.plan=dict(interventions=[item('branch','attention_whole_branch'),item('rows','embedding_rows'),
                                     item('copy_replacement','copy_control','replacement'),item('copy_original','copy_control')],
                       frozen_inputs=[],implementation=[],exports={name:dict(packed_batch={'path':'/batch'},
                       selected_rows={'path':'/rows'},rows_per_case=4 if name=='word_next_native' else 3)
                       for name in ('main','word_next_native','shared_piece')})
        self.binaries={name:{'path':'/'+name} for name in ('loss_probe','factorial_probe')}
        self.verify=self.enterContext(mock.patch.object(followup.trajectory.paired_analysis,'_verify_checkpoint'))
        def patch(a,b,path,**kwargs): path.mkdir(); training.publish(path/'patch.json',kwargs)
        self.enterContext(mock.patch.object(followup.paired_weight_patch,'create_patch',side_effect=patch))
        self.enterContext(mock.patch.object(followup,'verify_copy'))
        self.enterContext(mock.patch.object(followup,'verify_copy_suite_prefixes'))
        self.order=[]
        def loss(binary,checkpoint,cases,directory,log,record_path,gpu):
            self.order.append(checkpoint.parent.name); directory.mkdir(); log.write_text('mock')
            training.publish(record_path,{'mock':True})
            return dict(scores=str(directory),execution=training.record(record_path))
        self.loss=self.enterContext(mock.patch.object(followup,'score_loss',side_effect=loss))
        def native(command,inputs,directory,log,record_path,gpu):
            self.order.append(directory.parent.name); directory.mkdir(); log.write_text('mock')
            training.publish(record_path,{'mock':True})
        self.native=self.enterContext(mock.patch.object(followup,'run_native',side_effect=native))
        def factorial(cases,directory,patch,path,**kwargs): training.publish(path,{'mock':True})
        self.factorial=self.enterContext(mock.patch.object(followup.embedding_factorial_readout,'analyze',side_effect=factorial))
        def branch(item,patch,main,supp,scores,path,**kwargs): training.publish(path,{'mock':True})
        self.branch=self.enterContext(mock.patch.object(followup.paired_branch_readout,'analyze',side_effect=branch))

    def run_engine(self): return followup.run_interventions(self.root,self.output,{},self.plan,self.binaries,{},[])

    def test_copy_then_all_factorial_suites_then_branch_with_both_baselines(self):
        completed,artifacts=self.run_engine()
        self.assertEqual([c['intervention']['name'] for c in completed],['copy_replacement','copy_original','rows','branch'])
        self.assertEqual(self.order,['copy_replacement']*2+['copy_original']*2+['rows']*3+['branch']*2)
        self.assertEqual(self.factorial.call_count,3); self.assertEqual(self.loss.call_count,6)
        self.assertEqual(set(completed[2]['factorial']),{'main','word_next_native','shared_piece'})
        self.assertEqual([c.kwargs['case_kind'] for c in self.factorial.call_args_list],[None,'word_next_native','shared_piece'])
        scores=self.branch.call_args.args[4]
        self.assertIn('copy_original',scores['recipient']['main'])
        self.assertIn('copy_replacement',scores['donor']['main'])
        self.assertIn('branch',scores['patched']['main'])
        training.verify_records(artifacts)
        self.assertEqual(len(training.read_json(self.output/'state.json')['completed']),4)

    def test_duplicate_plan_names_rejected_before_materialization(self):
        self.plan['interventions'][0]['name']='rows'
        with self.assertRaisesRegex(ValueError,'duplicate'): self.run_engine()
        self.loss.assert_not_called(); self.native.assert_not_called()

    def test_native_failure_preserves_completed_copy_results(self):
        self.native.side_effect=RuntimeError('fixture native failure')
        with self.assertRaisesRegex(RuntimeError,'native failure'): self.run_engine()
        self.assertTrue((self.output/'copy_original/complete.json').is_file())
        self.assertTrue((self.output/'copy_replacement/complete.json').is_file())
        self.assertFalse((self.output/'rows/complete.json').exists())
        self.branch.assert_not_called()

    def test_frozen_source_mutation_blocks_before_probe(self):
        source=self.root/'frozen'; source.write_text('before'); self.plan['implementation']=[training.record(source)]
        source.write_text('after')
        with self.assertRaisesRegex(ValueError,'frozen evidence'): self.run_engine()
        self.loss.assert_not_called(); self.native.assert_not_called()


class GateTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name); self.output=self.root/'causal'
        (self.root/'analysis_trajectory').mkdir()
        training.publish(self.root/'analysis_trajectory/request.json',{'frozen_inputs':[]})
        training.publish(self.root/'request.json',{'frozen_inputs':[],'gpu':{'uuid':'fixture'}})
        self.args=SimpleNamespace(root=self.root,output=self.output,observer_pid=10,observer_start_ticks=20)
        for name in ('loss_probe','factorial_probe','gpu_test'):
            path=self.root/name; path.write_bytes(b'not executable CPU fixture'); setattr(self.args,name,path)
        self.enterContext(mock.patch.object(followup,'SOURCES',[training.record(followup.__file__)]))
        self.enterContext(mock.patch.object(followup,'observer_identity',return_value=process_fixtures.identity()))
        self.enterContext(mock.patch.object(followup.planner,'validate_exports',return_value=({}, {}, {}, [])))
        self.wait=self.enterContext(mock.patch.object(followup,'wait_for_trajectory',return_value={}))
        self.plan={'frozen_inputs':[],'implementation':[],'interventions':[]}
        def prepare(root,summary,exports,out): out.mkdir(); training.publish(out/'plan.json',self.plan); return self.plan
        self.prepare=self.enterContext(mock.patch.object(followup.planner,'prepare',side_effect=prepare))
        self.native=self.enterContext(mock.patch.object(followup,'run_native',side_effect=self.gpu_result))
        self.engine=self.enterContext(mock.patch.object(followup,'run_interventions',return_value=([],[])))

    def gpu_result(self,command,inputs,directory,log,record_path,gpu):
        log.write_text('CPU mocked native test')
        training.publish(directory/'gtest.json',dict(tests=1,failures=0,errors=0,disabled=0,
            testsuites=[dict(testsuite=[dict(status='RUN',result='COMPLETED')])]))
        result=dict(inputs_before=[training.record(p) for p in inputs],outputs=[training.record(directory/'gtest.json')],log=training.record(log))
        result['inputs_after']=result['inputs_before']; training.publish(record_path,result)
        return result

    def test_request_freezes_source_binaries_and_success_is_not_goal_completion(self):
        result=followup.run(self.args)
        self.assertTrue(result['complete']); self.assertFalse(result['goal_completion_claimed'])
        self.prepare.assert_called_once(); self.native.assert_called_once(); self.engine.assert_called_once()
        request=training.read_json(self.output/'request.json')
        self.assertEqual(set(request['binaries']),{'loss_probe','factorial_probe','gpu_test'})
        training.verify_records(request['frozen_inputs'])
        for name,binary in request['binaries'].items():
            self.assertEqual(binary['sha256'],training.record(getattr(self.args,name))['sha256'])

    def test_failed_wait_prevents_plan_and_all_gpu_work(self):
        self.wait.side_effect=RuntimeError('upstream failed')
        with self.assertRaises(RuntimeError): followup.run(self.args)
        self.prepare.assert_not_called(); self.native.assert_not_called(); self.engine.assert_not_called()
        self.assertTrue((self.output/'failure.json').is_file())

    def test_incomplete_training_rejected_before_native_test(self):
        self.prepare.side_effect=ValueError('incomplete budget')
        with self.assertRaises(ValueError): followup.run(self.args)
        self.native.assert_not_called(); self.engine.assert_not_called()

    def test_changed_frozen_executable_while_waiting_prevents_launch(self):
        def changed(*args): self.args.loss_probe.write_bytes(b'changed'); return {}
        self.wait.side_effect=changed
        with self.assertRaisesRegex(ValueError,'frozen evidence'): followup.run(self.args)
        self.native.assert_not_called(); self.engine.assert_not_called()

    def test_failed_or_all_skipped_gpu_test_prevents_interventions(self):
        original=self.native.side_effect
        def skipped(*args):
            result=original(*args)
            training.publish(args[2]/'gtest.json',dict(tests=0,failures=0,errors=0,disabled=0,testsuites=[]),exclusive=False)
            return result
        self.native.side_effect=skipped
        with self.assertRaisesRegex(ValueError,'enabled test'): followup.run(self.args)
        self.engine.assert_not_called()

    def test_output_must_be_a_new_immediate_child(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError): followup.run(self.args)
        self.args.output=self.root/'nested/new'
        with self.assertRaisesRegex(ValueError,'immediate child'): followup.run(self.args)


if __name__=='__main__': unittest.main()
