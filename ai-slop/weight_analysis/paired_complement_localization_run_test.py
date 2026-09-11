"""CPU-only safety and orchestration contracts for the owning cube runner.

No native process or GPU operation is permitted by these fixtures. Tiny real
checkpoint files retain the production 100-tensor layout so reference checks
must inspect every tensor, rather than accepting a descriptive label.
"""

import copy
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_complement_localization as core
from . import paired_complement_localization_run as runner
from .checkpoint import GPT2Config, tensor_manifest


CONFIG = GPT2Config(vocab_size=16, padded_vocab_size=20, context_length=4,
                    n_layers=8, d_model=2, n_heads=1, d_ff=4)
ROWS = [45, 68, 303, 400, 409, 1475, 2797, 3109, 14364, 21733, 45177]
training = core.training


class PreservedReferenceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='localization-reference-test-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.recipient, self.donor = (self.root/name for name in ('recipient', 'donor'))
        self.specs = tensor_manifest(CONFIG)
        self.assertEqual(len(self.specs), 100)
        for directory, shift in ((self.recipient, 0), (self.donor, 10000)):
            directory.mkdir()
            for spec in self.specs:
                values = np.arange(np.prod(spec.shape), dtype='<f4') + np.float32(shift+100*spec.index)
                values.tofile(directory/spec.filename)
        self.model = core.build_model(self.recipient, self.donor, self.root/'copy',
                                      ('Q','H','L'), [1,2], config=CONFIG)
        self.conditioned = self.root/'C'
        core.patcher.create_patch(self.recipient, self.donor, self.conditioned,
            tensors=[spec.name for spec in self.specs[1:]], config=CONFIG)
        self.preserved = self.root/'preserved'
        # Existing EC metadata can describe C->EC, not the cube's direct A->D.
        # The reference adapter must not rewrite it or claim the schemas agree.
        core.patcher.create_patch(self.conditioned, self.donor, self.preserved,
                                  embedding_rows=[1,2], config=CONFIG)
        self.native = self.enterContext(mock.patch.object(runner.screen, 'run_native',
            side_effect=AssertionError('reference validation must never launch native work')))

    def reference(self):
        return runner.reference_model(self.model, self.preserved, config=CONFIG)

    def test_all_100_preserved_weights_are_bound_without_rewriting_inputs(self):
        before = copy.deepcopy(self.model)
        metadata = (self.preserved/'patch.json').read_bytes()
        result = self.reference()
        self.assertEqual(self.model, before)
        self.assertEqual((self.preserved/'patch.json').read_bytes(), metadata)
        self.assertEqual(result['paths']['patched'], str(self.preserved))
        self.assertEqual(result['paths']['recipient'], str(self.conditioned))
        self.assertEqual(result['paths']['donor'], self.model['paths']['donor'])
        self.assertEqual(result['hashes']['patched'], self.model['hashes']['patched'])
        self.assertEqual(result['hashes']['recipient'],
            json.loads(metadata)['sources']['original']['weights_sha256'])
        self.assertEqual({Path(r['path']).parent for r in result['weight_records']['recipient']},
                         {self.conditioned})
        self.assertIs(result['reference_only'], True)
        self.assertIs(result['preserved_patch_schema_not_reinterpreted'], True)
        self.assertEqual(result['expected_cube_copy_patch'], self.model['patch'])
        self.assertEqual(result['patch'], training.record(self.preserved/'patch.json'))
        records = result['weight_records']['patched']
        self.assertEqual(len(records), 100)
        self.assertEqual({record['path'] for record in records},
                         {str(self.preserved/spec.filename) for spec in self.specs})
        self.assertTrue(all(record in result['records'] for record in records))
        self.native.assert_not_called()

    def test_wrong_first_middle_or_last_tensor_is_rejected(self):
        for index in (0,50,99):
            path = self.preserved/f'weight_{index}.bin'
            original = path.read_bytes()
            values = np.fromfile(path, dtype='<f4'); values[0] += 1
            values.tofile(path)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.reference()
            path.write_bytes(original)

    def test_missing_tensor_is_rejected(self):
        (self.preserved/'weight_99.bin').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.reference()

    def test_wrong_tensor_size_is_rejected(self):
        path = self.preserved/'weight_50.bin'
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaises(ValueError):
            self.reference()

    def test_nonfinite_preserved_weight_is_rejected(self):
        path = self.preserved/'weight_0.bin'
        values = np.fromfile(path, dtype='<f4'); values[0] = np.nan; values.tofile(path)
        with self.assertRaises(ValueError):
            self.reference()

    def test_symlinked_reference_directory_is_rejected(self):
        alias = self.root/'alias'; alias.symlink_to(self.preserved, target_is_directory=True)
        with self.assertRaises(ValueError):
            runner.reference_model(self.model, alias, config=CONFIG)

    def test_symlinked_weight_is_rejected_even_with_identical_bytes(self):
        path = self.preserved/'weight_99.bin'; path.unlink()
        path.symlink_to(self.root/'copy/weight_99.bin')
        with self.assertRaises(ValueError):
            self.reference()

    def test_changed_copied_weight_invalidates_the_reference_binding(self):
        path = self.root/'copy/weight_99.bin'
        values = np.fromfile(path, dtype='<f4'); values[0] += 1; values.tofile(path)
        with self.assertRaises(ValueError):
            self.reference()

    def test_reference_cannot_be_its_own_copy_control(self):
        with self.assertRaises(ValueError):
            runner.reference_model(self.model, self.root/'copy', config=CONFIG)

    def test_reference_cannot_share_weight_inode_with_the_copy(self):
        path = self.preserved/'weight_99.bin'; path.unlink()
        os.link(self.root/'copy/weight_99.bin', path)
        with self.assertRaises(ValueError):
            self.reference()

    def test_preserved_metadata_must_be_real_completed_and_path_bound(self):
        path = self.preserved/'patch.json'; original=json.loads(path.read_text())
        for changed in (dict(original, complete=False), dict(original, format='invented'),
                        dict(original, output={'path':str(self.root/'copy')})):
            path.write_text(json.dumps(changed))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.reference()

    def test_changed_actual_preserved_source_is_not_hidden_by_copy_sources(self):
        path = self.conditioned/'weight_50.bin'
        values = np.fromfile(path, dtype='<f4'); values[0] += 1; values.tofile(path)
        with self.assertRaises(ValueError):
            self.reference()

    def test_forged_actual_source_hash_cannot_pass_from_metadata_only(self):
        path = self.preserved/'patch.json'; metadata=json.loads(path.read_text())
        metadata['sources']['original']['weights_sha256']['weight_50.bin'] = '0'*64
        path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            self.reference()

    def test_expected_cube_hashes_must_cover_all_tensor_files(self):
        broken = copy.deepcopy(self.model)
        del broken['hashes']['patched']['weight_99.bin']
        with self.assertRaises(ValueError):
            runner.reference_model(broken, self.preserved, config=CONFIG)


class PlanValidationTest(unittest.TestCase):
    """Use the actual planner to keep the synthetic frozen-plan schema honest."""
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='localization-plan-test-')
        self.addCleanup(directory.cleanup); self.root=Path(directory.name)
        self.outer=self.root/'outer'; self.outer.mkdir()
        self.output=self.root/'run'; self.plan_path=self.root/'plan/plan.json'
        self.source=self.root/'frozen-input'; self.source.write_text('immutable')
        self.binary=self.root/'trusted-probe'; self.binary.write_text('not executable')
        binary_record=training.record(self.binary)
        self.gpu={'uuid':'test-gpu-never-accessed'}
        self.identity=dict(pid=1234, start_ticks=9876, argv=['python','-m',
            'weight_analysis.paired_outer_factorial','--root',str(self.root),
            '--output',str(self.outer)])
        self.binaries={'loss_probe':binary_record,'factorial_probe':binary_record,
                       'gpu_test':binary_record}
        self.runtime=dict(complete=True,runtime_dlopen_covered=False,
            environment={'LD_LIBRARY_PATH':str(self.root/'runtime')},frozen_records=[binary_record])
        (self.outer/'summary.json').write_text('{}')
        self.case_records={}
        for suite, folder in (('main','word_cases'),('supplemental','supplemental_cases')):
            path=self.root/folder/'cases.json'; path.parent.mkdir(); path.write_text('{}')
            self.case_records[suite]=training.record(path)
        endpoints={arm:dict(step=331,path=str(self.root/arm/'step_331'),
            sha256={s.filename:('a' if arm=='original' else 'b')*64
                    for s in tensor_manifest()}) for arm in ('original','replacement')}
        self.items=[]
        for recipient,donor in (('original','replacement'),('replacement','original')):
            name=f'step_331_{donor}_to_{recipient}_word_rows'
            self.items.append(dict(name=name,kind='embedding_rows',step=331,
                recipient_arm=recipient,donor_arm=donor,
                recipient_checkpoint=endpoints[recipient],donor_checkpoint=endpoints[donor],
                tensors=[],embedding_rows=list(ROWS)))
            direction=donor+'_to_'+recipient
            for suite in core.outer.SUITES:
                for cell,path in (
                    ('E',self.outer/'baseline_revalidation'/(name+'_'+suite+'.json')),
                    ('EC',self.outer/direction/('outer_'+suite)/'readout.json')):
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps(dict(execution_provenance={'verified':True},
                        cases=self.case_records['main' if suite=='main' else 'supplemental'],
                        patch=dict(selected_rows=list(ROWS),paths={'J':str(self.outer/direction/cell/'step_331')}),
                        files=[])))
        self.handoff=dict(request=dict(runner_identity=self.identity,binaries=self.binaries,
                          runtime=self.runtime),summary={},plan={},items=self.items,gpu=self.gpu,
                          records=[training.record(self.source),binary_record])
        self.handoff_mock=self.enterContext(mock.patch.object(core,'validate_outer_handoff',
                                                             return_value=self.handoff))
        self.live=self.enterContext(mock.patch.object(training,'process_live',return_value=False))
        self.idle=self.enterContext(mock.patch.object(training,'require_idle_gpu'))
        self.native=self.enterContext(mock.patch.object(runner.screen,'run_native',
            side_effect=AssertionError('plan validation must not launch native work')))
        def checked_cases(path,suite,records,config):
            record=self.case_records[suite]; records[record['path']]=record
            return dict(record=record,plan={'case_count':188 if suite=='main' else 265})
        self.cases=self.enterContext(mock.patch.object(core.branch,'_cases',side_effect=checked_cases))
        self.plan=core.prepare_plan(self.root,self.outer,self.plan_path.parent)
        self.live.reset_mock(); self.idle.reset_mock(); self.handoff_mock.reset_mock()

    def write_plan(self, plan):
        self.plan_path.write_text(json.dumps(plan))

    def test_valid_frozen_plan_is_read_only_and_rechecks_real_handoff(self):
        result=runner.validate_plan(self.plan_path,self.output)
        self.assertEqual(result,self.plan)
        self.assertFalse(self.output.exists())
        self.handoff_mock.assert_called()
        self.idle.assert_called_with(self.gpu)
        self.native.assert_not_called()

    def test_existing_and_nested_outputs_are_rejected(self):
        occupied=self.root/'occupied'; occupied.mkdir()
        dangling=self.root/'dangling'; dangling.symlink_to(self.root/'missing')
        for output in (occupied,dangling,self.plan_path.parent/'nested',self.outer/'nested'):
            with self.subTest(output=output),self.assertRaises((ValueError,FileExistsError)):
                runner.validate_plan(self.plan_path,output)
        self.native.assert_not_called()

    def test_frozen_source_mutation_blocks_before_native(self):
        self.source.write_text('changed')
        with self.assertRaises(ValueError):
            runner.validate_plan(self.plan_path,self.output)
        self.native.assert_not_called()

    def test_live_or_reused_outer_identity_cannot_pass(self):
        for value in (True,RuntimeError('PID reused'),OSError('transient process read')):
            self.live.side_effect=value if isinstance(value,Exception) else None
            self.live.return_value=value if isinstance(value,bool) else False
            with self.subTest(value=value),self.assertRaises((ValueError,RuntimeError,OSError)):
                runner.validate_plan(self.plan_path,self.output)
        self.native.assert_not_called()

    def test_busy_gpu_blocks_before_native(self):
        self.idle.side_effect=RuntimeError('GPU busy')
        with self.assertRaisesRegex(RuntimeError,'GPU busy'):
            runner.validate_plan(self.plan_path,self.output)
        self.native.assert_not_called()

    def test_wrong_step_phase_config_or_native_claim_is_rejected(self):
        for field,value in (('step',100),('phase','complete'),('native_execution_performed',True),
                            ('config',asdict(CONFIG))):
            changed=copy.deepcopy(self.plan); changed[field]=value; self.write_plan(changed)
            with self.subTest(field=field),self.assertRaises(ValueError):
                runner.validate_plan(self.plan_path,self.output)
        self.native.assert_not_called()

    def test_fixed_cube_and_group_membership_cannot_change(self):
        mutations=[]
        changed=copy.deepcopy(self.plan); del changed['cells']['HL']; mutations.append(changed)
        changed=copy.deepcopy(self.plan); changed['cells']['QH']=['Q','L']; mutations.append(changed)
        changed=copy.deepcopy(self.plan); changed['groups']['Q'].append('token_embedding.weight'); mutations.append(changed)
        for changed in mutations:
            self.write_plan(changed)
            with self.assertRaises(ValueError):
                runner.validate_plan(self.plan_path,self.output)
        self.native.assert_not_called()

    def test_missing_direction_or_changed_selected_rows_is_rejected(self):
        mutations=[]
        changed=copy.deepcopy(self.plan); del changed['directions']['original_to_replacement']; mutations.append(changed)
        changed=copy.deepcopy(self.plan); changed['directions']['original_to_replacement']['rows']=ROWS[:-1]; mutations.append(changed)
        for changed in mutations:
            self.write_plan(changed)
            with self.assertRaises(ValueError):
                runner.validate_plan(self.plan_path,self.output)
        self.native.assert_not_called()

    def test_plan_changed_during_handoff_cannot_be_rebound_after_validation(self):
        def changed(*args):
            altered=copy.deepcopy(self.plan);altered['interpretation']='modified during audit'
            self.write_plan(altered)
            return self.handoff
        self.handoff_mock.side_effect=changed
        with self.assertRaisesRegex(ValueError,'frozen evidence changed'):
            runner.validate_plan(self.plan_path,self.output)
        self.assertFalse(self.output.exists());self.native.assert_not_called()

    def test_loaded_runner_source_identity_is_checked_before_publication(self):
        wrong=dict(runner.RUNNER_SOURCE,sha256='0'*64)
        with mock.patch.object(runner,'RUNNER_SOURCE',wrong):
            with self.assertRaisesRegex(ValueError,'frozen evidence changed'):
                runner.validate_plan(self.plan_path,self.output)
        self.assertFalse(self.output.exists());self.native.assert_not_called()


class OrchestrationTest(unittest.TestCase):
    """Real filesystem/ledgers, fake model building and measurements; never GPU."""
    def setUp(self):
        PlanValidationTest.setUp(self)
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(mock.patch.dict(os.environ, {}, clear=False))
        self.enterContext(mock.patch.object(runner,'validate_plan',return_value=self.plan))
        self.events=[]
        self.builds=self.enterContext(mock.patch.object(core,'build_model',side_effect=self.fake_model))
        self.references=self.enterContext(mock.patch.object(runner,'reference_model',side_effect=self.fake_reference))
        self.measures=self.enterContext(mock.patch.object(runner,'run_native',side_effect=self.fake_native))
        self.compare=self.enterContext(mock.patch.object(core,'compare_fp64_reference',
            return_value=dict(checked_predictions=3,maximum_absolute_nll_error=0,
                absolute_tolerance=core.FP32_ABSOLUTE_TOLERANCE,
                relative_tolerance=core.FP32_RELATIVE_TOLERANCE,exact_byte_equality_claimed=False)))
        self.parity=self.enterContext(mock.patch.object(core,'assert_native_copy_parity',
            return_value=dict(all_context_loss_and_argmax_byte_equal=True,predictions=1024)))
        self.cross=self.enterContext(mock.patch.object(core,'_cross_suite',return_value=156))
        self.analyze=self.enterContext(mock.patch.object(core,'analyze_cube',side_effect=self.fake_analysis))

    def fake_model(self,a,d,output,subset,rows):
        output=Path(output); output.mkdir()
        path=output/'patch.json'; path.write_text(json.dumps({'test_only':True}))
        record=training.record(path)
        return dict(paths={'recipient':a,'donor':d,'patched':str(output)},
            hashes={},weight_records={},patch=record,records=[record],subset=list(subset))

    def fake_reference(self,model,preserved):
        result=copy.deepcopy(model); result['paths']['patched']=str(preserved)
        result['reference_only']=True
        return result

    def fake_native(self,plan,model,suite,directory,frozen):
        training.verify_records(frozen)
        self.assertIn(training.record(self.plan_path),frozen)
        self.assertIn(training.record(self.output/'request.json'),frozen)
        directory=Path(directory); directory.mkdir()
        scores=directory/'scores'; scores.mkdir()
        path=scores/'measurement.json'; path.write_text(json.dumps({'test_only':True,'suite':suite}))
        log=directory/'process.log'; log.write_text('mock native completed')
        ledger=directory/'execution.json'; ledger.write_text(json.dumps({'test_only':True,'returncode':0}))
        self.events.append(str(directory.relative_to(self.output)))
        measured=dict(records=[training.record(path),training.record(ledger)],suite=suite)
        return measured,str(scores),str(ledger)

    def fake_analysis(self,models,cases,scores,ledgers,output,*,rows):
        self.assertEqual(set(models),set(core.cube_subsets()))
        self.assertEqual(set(scores),set(core.cube_subsets()))
        self.assertEqual(set(ledgers),set(core.cube_subsets()))
        self.assertTrue(all(set(value)=={'main','supplemental'} for value in scores.values()))
        self.assertEqual(rows,ROWS)
        controls=training.read_json(Path(output).parent/'baseline_controls.json')
        self.assertIs(controls['baseline_controls_certified'],True)
        self.assertEqual(set(controls['copied_model_checks']),{'E','EC'})
        for value in controls['copied_model_checks'].values():
            self.assertEqual(set(value),{'main','supplemental'})
        result=dict(complete=True,baseline_controls_certified=False,files=[])
        training.publish(output,result)
        return result

    def assert_failed_without_summary(self, error, message):
        with self.assertRaisesRegex(error,message):
            runner.run(self.plan_path,self.output)
        self.assertFalse((self.output/'summary.json').exists())
        failure=training.read_json(self.output/'failure.json')
        self.assertIs(failure['no_automatic_restart'],True)
        self.assertIs(failure['training_restarted'],False)
        self.assertIs(failure['goal_completion_claimed'],False)
        return failure

    def test_full_cube_has_exact_40_measurements_and_baselines_before_localization(self):
        result=runner.run(self.plan_path,self.output)
        expected=[]
        for direction in ('original_to_replacement','replacement_to_original'):
            for role,cells in (('references',('E','EC')),
                               ('cells',('E','EC','Q','H','L','QH','QL','HL'))):
                expected.extend(f'{direction}/{role}/{cell}/{suite}'
                    for cell in cells for suite in ('main','supplemental'))
        self.assertEqual(self.events,expected)
        self.assertEqual(self.builds.call_count,16)
        self.assertEqual(self.references.call_count,4)
        self.assertEqual(self.measures.call_count,40)
        self.assertEqual(self.compare.call_count,24)
        self.assertEqual(self.parity.call_count,8)
        self.assertEqual(self.analyze.call_count,2)
        self.assertEqual(self.cross.call_count,20)
        self.assertIs(result['complete'],True)
        self.assertEqual(result['native_measurement_count'],40)
        self.assertEqual(len(set(result['completed'])),40)
        self.assertIs(result['baseline_controls_certified'],True)
        self.assertIs(result['goal_completion_claimed'],False)
        self.assertFalse((self.output/'failure.json').exists())
        training.verify_records(result['frozen_inputs'])
        training.verify_records(result['artifacts'])
        self.native.assert_not_called()

    def test_native_failure_never_certifies_baselines_or_direction(self):
        self.measures.side_effect=RuntimeError('native failed')
        failure=self.assert_failed_without_summary(RuntimeError,'native failed')
        self.assertEqual(failure['completed'],[])
        self.compare.assert_not_called(); self.parity.assert_not_called(); self.analyze.assert_not_called()
        self.assertFalse((self.output/runner.DIRECTIONS[0]/'baseline_controls.json').exists())

    def test_preserved_fp64_reference_failure_prevents_cube_measurements(self):
        self.compare.side_effect=ValueError('reference mismatch')
        self.assert_failed_without_summary(ValueError,'reference mismatch')
        self.assertEqual(len(self.events),1)
        self.assertIn('/references/E/main',self.events[0])
        self.parity.assert_not_called(); self.analyze.assert_not_called()

    def test_copy_parity_failure_cannot_publish_direction_or_baseline_marker(self):
        self.parity.side_effect=ValueError('copy control differs')
        self.assert_failed_without_summary(ValueError,'copy control differs')
        self.assertEqual(len(self.events),5)
        self.assertTrue(all('/references/' in event for event in self.events[:4]))
        self.assertTrue(self.events[-1].endswith('/cells/E/main'))
        stage=self.output/runner.DIRECTIONS[0]
        self.assertFalse((stage/'baseline_controls.json').exists())
        self.assertFalse((stage/'complete.json').exists())
        self.analyze.assert_not_called()

    def test_source_change_after_native_blocks_following_native_call(self):
        def changed(*args):
            result=self.fake_native(*args); self.source.write_text('changed after native'); return result
        self.measures.side_effect=changed
        self.assert_failed_without_summary(ValueError,'frozen evidence changed')
        self.assertEqual(len(self.events),1)
        self.analyze.assert_not_called()

    def test_cube_readout_failure_retains_partial_evidence_but_not_direction_success(self):
        self.analyze.side_effect=ValueError('incomplete cube readout')
        failure=self.assert_failed_without_summary(ValueError,'incomplete cube readout')
        self.assertEqual(len(failure['completed']),20)
        stage=self.output/runner.DIRECTIONS[0]
        self.assertTrue((stage/'baseline_controls.json').exists())
        self.assertFalse((stage/'complete.json').exists())
        self.assertFalse((self.output/runner.DIRECTIONS[1]).exists())

    def test_plan_mutated_inside_validate_fails_before_output_creation(self):
        def changed(*args):
            self.plan_path.write_text(json.dumps(dict(self.plan,interpretation='changed')))
            return self.plan
        with mock.patch.object(runner,'validate_plan',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'frozen evidence changed'):
                runner.run(self.plan_path,self.output)
        self.assertFalse(self.output.exists())
        self.builds.assert_not_called();self.measures.assert_not_called()

    def test_loaded_runner_source_mismatch_fails_before_output_creation(self):
        with mock.patch.object(runner,'RUNNER_SOURCE',dict(runner.RUNNER_SOURCE,sha256='0'*64)):
            with self.assertRaisesRegex(ValueError,'frozen evidence changed'):
                runner.run(self.plan_path,self.output)
        self.assertFalse(self.output.exists())
        self.builds.assert_not_called();self.measures.assert_not_called()

    def test_mismatched_source_copy_cannot_launch_native_or_publish_summary(self):
        actual=runner.execution.paired_training.freeze_file
        def wrong_copy(source,destination):
            return dict(actual(source,destination),sha256='0'*64)
        with mock.patch.object(runner.execution.paired_training,'freeze_file',side_effect=wrong_copy):
            self.assert_failed_without_summary(ValueError,'source copy differs')
        self.assertFalse((self.output/'request.json').exists())
        self.builds.assert_not_called();self.measures.assert_not_called()


class NativeWrapperTest(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory(prefix='localization-native-wrapper-test-')
        self.addCleanup(directory.cleanup);self.root=Path(directory.name)
        self.source=self.root/'source';self.source.write_text('frozen')
        self.batch=self.root/'batch';self.batch.write_bytes(b'tiny CPU fixture')
        self.cases=self.root/'cases.json';self.cases.write_text(json.dumps({'packed_batch':{'path':str(self.batch)}}))
        self.frozen=[training.record(self.source)]
        self.plan=dict(cases={'main':training.record(self.cases)},
                       binaries={'loss_probe':{'path':'/never/run/native'}},gpu={'mock':True})
        self.model=dict(paths={'patched':'/mock/step_331'},records=[])
        self.native=self.enterContext(mock.patch.object(runner.screen,'run_native'))
        self.inputs=self.enterContext(mock.patch.object(runner.execution,'checkpoint_inputs',return_value=[]))
        self.load=self.enterContext(mock.patch.object(core,'load_native',return_value={'records':[]}))

    def invoke(self):
        return runner.run_native(self.plan,self.model,'main',self.root/'measurement',self.frozen)

    def test_source_changed_before_measurement_prevents_native_launch(self):
        self.source.write_text('modified')
        with self.assertRaises(ValueError):self.invoke()
        self.native.assert_not_called();self.load.assert_not_called()

    def test_source_changed_during_native_is_rejected_before_return(self):
        self.native.side_effect=lambda *args:self.source.write_text('modified')
        with self.assertRaises(ValueError):self.invoke()
        self.native.assert_called_once()

    def test_native_failure_does_not_load_or_certify_outputs(self):
        self.native.side_effect=RuntimeError('child failed')
        with self.assertRaisesRegex(RuntimeError,'child failed'):self.invoke()
        self.load.assert_not_called()

    def test_native_command_and_inputs_bind_frozen_model_case_and_batch(self):
        self.invoke()
        command,inputs,scores,log,ledger,gpu=self.native.call_args.args
        self.assertEqual(command,['/never/run/native','--checkpoint=/mock/step_331',
            '--batch='+str(self.batch),'--output_dir='+str(self.root/'measurement/scores'),
            '--batch_sequences=1'])
        self.assertIn(str(self.source),inputs);self.assertIn(str(self.cases),inputs)
        self.assertIn(str(self.batch),inputs)
        self.assertEqual(gpu,self.plan['gpu'])
        self.load.assert_called_once_with(self.model,str(self.cases),'main',scores,ledger)


if __name__ == '__main__':
    unittest.main()
