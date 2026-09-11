"""CPU fixtures for partial-cell provenance and non-imputed aggregation."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_complement_partial_readout as partial
from . import paired_complement_localization_run_test as run_fixtures

training=partial.training


def row(prefix='p', *, cells=('E','EC','Q'), kind='word', split='test',
        variant='title', leading=True, domain='original', target='Exeunt',
        ids=None, values=None, piece=None):
    ids=ids or ([1,2,3,4] if kind=='word_next_native' else [1,2,3])
    values=values or {cell:[-float(index+1)]*len(ids) for index,cell in enumerate(cells)}
    return dict(suite='supplemental' if kind in ('word_next_native','shared_piece') else 'main',
        kind=kind, split=split, spelling_variant=variant, word_has_leading_space=leading,
        prefix_domain=domain, target=target,target_ids=list(ids),piece_id=piece,
        prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),prefix_length=128,
        cells={cell:dict(token_log_probability=list(value)) for cell,value in values.items()})


class AggregateTest(unittest.TestCase):
    def aggregate(self,*rows):
        return partial.aggregate(list(rows),('E','EC','Q'))

    def test_only_completed_cells_exist_in_metrics_and_missing_cube_is_not_scored(self):
        with mock.patch.object(partial.core,'score_cube',side_effect=AssertionError('incomplete cube')):
            result=self.aggregate(row())
        metric=result[0]['metrics']['word_three']
        self.assertEqual(set(metric['cells']),{'E','EC','Q'})
        self.assertNotIn('H',metric['delta_log_probability_from_E'])
        self.assertEqual(metric['delta_log_probability_from_E']['Q'],-6)
        self.assertEqual(metric['gap_to_EC_log_probability']['Q'],3)

    def test_e_and_ec_are_required_to_publish_partial_effects(self):
        for cells in (('E','Q'),('EC','Q'),('Q',),('E','EC','unplanned'),('E','EC','E')):
            with self.subTest(cells=cells),self.assertRaises(ValueError):
                partial.aggregate([],cells)

    def test_domain_split_variant_spacing_and_target_are_not_pooled(self):
        rows=[row(prefix=f'{split}-{variant}-{leading}-{target}',split=split,
                  variant=variant,leading=leading,target=target)
              for split in ('training','test') for variant in ('title','lowercase')
              for leading in (True,False) for target in ('Exeunt','Nuveth')]
        rows.append(row(domain='replacement'))
        result=self.aggregate(*rows)
        self.assertEqual(len(result),16)
        self.assertTrue(all(g['prefix_domain']=='original' for g in result))

    def test_shared_controls_keep_piece_id_and_split_coverage(self):
        result=self.aggregate(row(kind='control',domain='shared',target='ordinary'),
            row(kind='shared_piece',domain='shared',piece=68),
            row(kind='shared_piece',domain='shared',piece=303),
            row(kind='shared_piece',domain='shared',piece=303,split='training'))
        self.assertEqual(len(result),4)
        self.assertTrue(all(g['prefix_domain']=='shared' for g in result))

    def test_aliases_deduplicate_actual_prefix_and_target_not_labels(self):
        result=self.aggregate(row(),row())
        metric=result[0]['metrics']['word_three']
        self.assertEqual(metric['unique_event_count'],1)
        self.assertEqual(metric['alias_case_count'],2)

    def test_prefix_length_is_part_of_identity(self):
        a,b=row(),row();b['prefix_length']+=1
        self.assertEqual(self.aggregate(a,b)[0]['metrics']['first_piece']['unique_event_count'],2)

    def test_exact_fourth_targets_do_not_reweight_first_or_suffix(self):
        first=row(kind='word_next_native',ids=[1,2,3,4])
        second=row(kind='word_next_native',ids=[1,2,3,5])
        for cell in second['cells']:
            second['cells'][cell]['token_log_probability'][3]-=1
        metrics=self.aggregate(first,second)[0]['metrics']
        for name in ('first_piece','suffix','word_three'):
            self.assertEqual(metrics[name]['unique_event_count'],1)
        for name in ('exact_next_native_token','selected_sequence'):
            self.assertEqual(metrics[name]['unique_event_count'],2)

    def test_conflicting_aliases_are_not_silently_averaged(self):
        a,b=row(),row();b['cells']['Q']['token_log_probability'][1]-=.5
        with self.assertRaisesRegex(ValueError,'conflicting'):
            self.aggregate(a,b)

    def test_nonfinite_positive_missing_and_wrong_length_scores_rejected(self):
        for value in (float('nan'),float('inf'),-float('inf'),True,.5):
            r=row();r['cells']['E']['token_log_probability'][0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.aggregate(r)
        r=row();del r['cells']['Q']
        with self.assertRaises(ValueError):self.aggregate(r)
        r=row();r['cells']['EC']['token_log_probability'].pop()
        with self.assertRaises(ValueError):self.aggregate(r)


class CompletedLeafTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='partial-leaf-test-')
        self.addCleanup(temporary.cleanup);self.stage=Path(temporary.name)/'E';self.stage.mkdir()
        self.file=self.stage/'losses';self.file.write_bytes(b'frozen scores')
        self.marker=self.stage/'complete.json'
        self.value=dict(cell='E',artifacts=[training.record(self.file)])

    def publish(self):
        self.marker.write_text(json.dumps(self.value))

    def test_dump_without_completion_marker_is_not_success(self):
        with self.assertRaises(ValueError):partial.completed_leaf(self.stage)

    def test_complete_exact_inventory_is_accepted(self):
        self.publish();marker,records=partial.completed_leaf(self.stage)
        self.assertEqual(marker,self.value);self.assertEqual(len(records),2)

    def test_changed_or_missing_output_is_rejected(self):
        self.publish();self.file.write_bytes(b'changed')
        with self.assertRaises(ValueError):partial.completed_leaf(self.stage)
        self.file.unlink()
        with self.assertRaises(ValueError):partial.completed_leaf(self.stage)

    def test_unlisted_output_or_duplicate_record_is_rejected(self):
        self.publish();extra=self.stage/'extra';extra.write_text('extra')
        with self.assertRaises(ValueError):partial.completed_leaf(self.stage)
        extra.unlink();self.value['artifacts']*=2;self.publish()
        with self.assertRaises(ValueError):partial.completed_leaf(self.stage)

    def test_symlink_output_or_marker_is_rejected(self):
        self.publish();self.file.unlink();self.file.symlink_to(self.marker)
        with self.assertRaises(ValueError):partial.completed_leaf(self.stage)
        self.file.unlink();self.marker.unlink();self.marker.symlink_to(self.stage/'missing')
        with self.assertRaises(ValueError):partial.completed_leaf(self.stage)


class BaselineGateTest(unittest.TestCase):
    def setUp(self):
        self.score=dict(cases={'record':{'path':'same'}},losses=np.zeros((2,4),dtype='<f4'),
                        argmax=np.zeros((2,4),dtype='<i4'))
        self.loaded={cell:{suite:copy.deepcopy(self.score) for suite in partial.runner.SUITES}
                     for cell in ('E','EC')}
        self.preserved=copy.deepcopy(self.loaded)
        self.fp64=dict(main=dict(checked_predictions=6,maximum_absolute_nll_error=0,
            absolute_tolerance=partial.core.FP32_ABSOLUTE_TOLERANCE,
            relative_tolerance=partial.core.FP32_RELATIVE_TOLERANCE,exact_byte_equality_claimed=False))
        self.compare=self.enterContext(mock.patch.object(partial.runner,'compare_references',return_value=self.fp64))
        parity=partial.core.assert_native_copy_parity(self.score,self.score)
        self.markers={cell:dict(cell=cell,first_three_cross_suite_byte_equal=True,
            copy_controls={suite:dict(same_native_copy=copy.deepcopy(parity),fp64_reference=copy.deepcopy(self.fp64))
                           for suite in partial.runner.SUITES}) for cell in ('E','EC')}
        self.references={cell:dict(cell=cell,all_targets_checked=True,first_three_cross_suite_byte_equal=True,
            fp64_reference_checks={suite:copy.deepcopy(self.fp64) for suite in partial.runner.SUITES})
            for cell in ('E','EC')}

    def certify(self):
        return partial.certify_baselines(self.loaded,self.preserved,self.markers,self.references,{})

    def test_true_controls_are_recomputed_for_both_cells_and_suites(self):
        result=self.certify();self.assertEqual(set(result),{'E','EC'})
        self.assertEqual(self.compare.call_count,8)

    def test_missing_ec_or_one_suite_is_rejected(self):
        del self.markers['EC']
        with self.assertRaises(ValueError):self.certify()

    def test_false_native_copy_flag_cannot_be_trusted(self):
        self.markers['E']['copy_controls']['main']['same_native_copy']['all_context_loss_and_argmax_byte_equal']=False
        with self.assertRaisesRegex(ValueError,'recomputed'):self.certify()

    def test_native_bytes_are_checked_even_with_true_flags(self):
        self.loaded['EC']['supplemental']['losses'][0,0]=1
        with self.assertRaisesRegex(ValueError,'byte|copy control'):self.certify()

    def test_fp64_control_fields_and_fixed_tolerances_must_match_exactly(self):
        control=self.markers['E']['copy_controls']['main']['fp64_reference']['main']
        original=copy.deepcopy(control)
        for key,value in (('absolute_tolerance',1),('relative_tolerance',1),
                          ('checked_predictions',0),('exact_byte_equality_claimed',True),
                          ('maximum_absolute_nll_error',1)):
            control.clear();control.update(original);control[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'recomputed'):self.certify()

    def test_false_cross_suite_or_target_coverage_is_rejected(self):
        for key in ('first_three_cross_suite_byte_equal','all_targets_checked'):
            self.references['EC'][key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):self.certify()
            self.references['EC'][key]=True

    def test_fp64_revalidation_error_is_not_replaced_with_saved_success(self):
        self.compare.side_effect=ValueError('native FP32 reference error')
        with self.assertRaisesRegex(ValueError,'FP32 reference'):self.certify()


class RequestTest(unittest.TestCase):
    def setUp(self):
        self.fixture=run_fixtures.PlanValidationTest()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;self.directory=f.output;self.directory.mkdir()
        self.plan_record=training.record(f.plan_path)
        frozen=partial.core.outer._unique([*f.plan['frozen_inputs'],self.plan_record,
                                           training.record(partial.runner.__file__)])
        self.request=dict(format=partial.runner.FORMAT,directions=list(partial.runner.DIRECTIONS),
            cell_order=list(partial.runner.CELL_ORDER),suites=list(partial.runner.SUITES),
            native_measurement_count=40,training_restarted=False,goal_completion_claimed=False,
            plan=self.plan_record,frozen_inputs=frozen,binaries=f.binaries,runtime=f.runtime,gpu=f.gpu,
            runner_identity=dict(pid=1234,start_ticks=5678,argv=['python','-m',
                'weight_analysis.paired_complement_localization_run','--plan',str(f.plan_path),
                '--output',str(self.directory)]))
        self.request_path=self.directory/'request.json';self.publish()
        f.live.side_effect=AssertionError('partial reader must not check process exit')
        f.idle.side_effect=AssertionError('partial reader must not check GPU idle')

    def publish(self):
        self.request_path.write_text(json.dumps(self.request))

    def validate(self):
        return partial.validate_request(self.directory,'original_to_replacement')

    def test_live_run_request_can_be_authenticated_without_exit_or_gpu_queries(self):
        request,plan,records=self.validate()
        self.assertEqual(request,self.request);self.assertEqual(plan,self.fixture.plan)
        self.assertIn(training.record(self.request_path),records)

    def test_changed_frozen_input_is_rejected(self):
        self.fixture.source.write_text('changed')
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_runner_argv_or_direction_is_rejected(self):
        self.request['runner_identity']['argv'][-1]='/different/run';self.publish()
        with self.assertRaises(ValueError):self.validate()

    def test_modified_fixed_scope_or_omitted_source_is_rejected(self):
        self.request['native_measurement_count']=39;self.publish()
        with self.assertRaises(ValueError):self.validate()
        self.request['native_measurement_count']=40
        self.request['frozen_inputs']=[r for r in self.request['frozen_inputs']
            if r['path']!=str(Path(partial.runner.__file__).resolve())];self.publish()
        with self.assertRaises(ValueError):self.validate()

    def test_missing_baseline_marker_blocks_effect_publication(self):
        output=self.fixture.root/'partial.json'
        with self.assertRaisesRegex(ValueError,'both completed E and EC'):
            partial.analyze(self.directory,'original_to_replacement',output)
        self.assertFalse(output.exists())

    def test_existing_output_and_live_tree_output_are_rejected(self):
        output=self.fixture.root/'existing.json';output.write_text('keep')
        with self.assertRaises(ValueError):partial.analyze(self.directory,'original_to_replacement',output)
        self.assertEqual(output.read_text(),'keep')
        with self.assertRaises(ValueError):
            partial.analyze(self.directory,'original_to_replacement',self.directory/'new.json')

    def test_reader_source_changed_during_request_validation_is_not_rebound(self):
        source=self.fixture.root/'loaded-reader.py';source.write_text('loaded bytes')
        record=training.record(source);original=partial.validate_request
        def changed(*args):
            result=original(*args);source.write_text('different bytes');return result
        with mock.patch.object(partial,'READER_SOURCE',record), \
             mock.patch.object(partial,'validate_request',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'frozen evidence changed'):
                partial.analyze(self.directory,'original_to_replacement',self.fixture.root/'result.json')
        self.assertFalse((self.fixture.root/'result.json').exists())


class PartialPipelineTest(unittest.TestCase):
    """Test orchestration using real completion files and tiny native arrays.

    Checkpoint/native loading is mocked here; its actual byte/command checks
    are already covered by the core tests. Partial completion and baseline
    checks, source rehashing, raw-score extraction, and grouping remain real.
    """
    def setUp(self):
        self.request_fixture=RequestTest();self.request_fixture.setUp()
        self.addCleanup(self.request_fixture.doCleanups)
        f=self.request_fixture.fixture
        self.run=self.request_fixture.directory;self.stage=self.run/'original_to_replacement'
        self.output=f.root/'partial.json';self.stage.mkdir()
        self.models={};self.loaded={};self.ref_models={};self.preserved={}
        for cell in partial.runner.CELL_ORDER:
            path=self.stage/'cells'/cell/'step_331';path.mkdir(parents=True)
            patch=path/'patch.json';patch.write_text('{}')
            self.models[cell]=dict(paths={'patched':str(path)},patch=training.record(patch),records=[])
        self.fp64={'main':dict(checked_predictions=3,maximum_absolute_nll_error=0,
            absolute_tolerance=partial.core.FP32_ABSOLUTE_TOLERANCE,
            relative_tolerance=partial.core.FP32_RELATIVE_TOLERANCE,exact_byte_equality_claimed=False)}
        self.enterContext(mock.patch.object(partial.runner,'compare_references',return_value=self.fp64))
        for cell in ('E','EC','Q'):
            self.loaded[cell]={}
            for suite in partial.runner.SUITES:
                directory=self.stage/'cells'/cell/suite;directory.mkdir()
                (directory/'scores').mkdir();log=directory/'process.log';log.write_text('fake native')
                execution=directory/'execution.json';execution.write_text('{}')
                shift={'E':1,'EC':2,'Q':3}[cell]
                losses=np.array([[shift,shift+1,shift+2,shift+3]],dtype='<f4')
                case=dict(kind='word' if suite=='main' else 'word_next_native',split='test',
                    context_id='one',prefix_domain='original',target='Exeunt',
                    target_ids=[1,2,3] if suite=='main' else [1,2,3,4],case_index=0,
                    scored_rows=[0,1,2] if suite=='main' else [0,1,2,3],
                    prefix={'token_ids_sha256':'a'*64,'length':1},
                    target_source={'native_piece_bytes_hex':['204578','65','756e74']})
                plan=dict(format='pluto-paired-word-cases-v1' if suite=='main' else
                    'pluto-paired-supplemental-cases-v1',cases=[case])
                if suite=='supplemental':
                    plan.update(source_word_cases=f.case_records['main'],source_word_packed_batch={'mock':'main batch'})
                measured=dict(cases={'record':f.case_records[suite],'plan':plan,'batch':{'mock':'main batch'}},
                    losses=losses,argmax=np.zeros_like(losses,dtype='<i4'),
                    records=[training.record(log),training.record(execution)])
                self.loaded[cell][suite]=measured
            controls={}
            if cell in ('E','EC'):
                self.ref_models[cell]=dict(reference_only=True,paths={'patched':f'preserved-{cell}'})
                self.preserved[cell]=copy.deepcopy(self.loaded[cell])
                ref_dir=self.stage/'references'/cell;ref_dir.mkdir(parents=True)
                for suite in partial.runner.SUITES:
                    d=ref_dir/suite;d.mkdir();(d/'scores').mkdir();(d/'execution.json').write_text('{}')
                    controls[suite]=dict(same_native_copy=partial.core.assert_native_copy_parity(
                        self.preserved[cell][suite],self.loaded[cell][suite]),fp64_reference=self.fp64)
                partial.runner.screen.stage_result(ref_dir,dict(cell=cell,all_targets_checked=True,
                    first_three_cross_suite_byte_equal=True,
                    fp64_reference_checks={suite:self.fp64 for suite in partial.runner.SUITES}))
            cell_dir=self.stage/'cells'/cell
            partial.runner.screen.stage_result(cell_dir,dict(cell=cell,groups=f.plan['cells'][cell],
                scores={suite:str(cell_dir/suite/'scores') for suite in partial.runner.SUITES},
                executions={suite:str(cell_dir/suite/'execution.json') for suite in partial.runner.SUITES},
                first_three_cross_suite_byte_equal=True,copy_controls=controls))
        training.publish(self.stage/'models.json',dict(cube=self.models,preserved_references=self.ref_models))
        self.validate=self.enterContext(mock.patch.object(partial.core,'validate_model',
            side_effect=lambda path,*args:self.models[Path(path).parent.parent.name]))
        self.reference=self.enterContext(mock.patch.object(partial.runner,'reference_model',
            side_effect=lambda model,path:self.ref_models[Path(model['paths']['patched']).parent.name]))
        def load(model,cases,suite,directory,execution):
            cell=Path(directory).parent.parent.name
            return self.preserved[cell][suite] if model.get('reference_only') else self.loaded[cell][suite]
        self.load=self.enterContext(mock.patch.object(partial.core,'load_native',side_effect=load))
        self.enterContext(mock.patch.object(partial,'_execution_binding',return_value=[]))
        self.enterContext(mock.patch.object(partial.core,'score_cube',side_effect=AssertionError('no incomplete factorial')))

    def analyze(self):
        return partial.analyze(self.run,'original_to_replacement',self.output)

    def test_partial_snapshot_succeeds_without_all8_or_global_baseline_marker(self):
        self.assertFalse((self.stage/'baseline_controls.json').exists())
        result=self.analyze()
        self.assertEqual(result['available_cells'],['E','EC','Q'])
        self.assertEqual(result['missing_cells'],['H','L','QH','QL','HL'])
        self.assertIs(result['all_cube_cells_present'],False)
        self.assertIs(result['baseline_controls_certified'],True)
        self.assertIs(result['controller_completion_claimed'],False)
        self.assertIs(result['native_execution_performed'],False)
        self.assertEqual(result['per_case'][0]['cells']['Q']['token_nll'],[3,4,5])
        self.assertEqual(self.validate.call_count,3);self.assertEqual(self.reference.call_count,2)
        self.assertEqual(self.load.call_count,10)
        training.verify_records(result['files'])

    def test_uncompleted_cell_with_dump_is_missing_not_zero(self):
        path=self.stage/'cells/H/unfinished-scores';path.write_text('in progress')
        result=self.analyze()
        self.assertIn('H',result['missing_cells'])
        self.assertNotIn('H',result['per_case'][0]['cells'])

    def test_false_per_cell_control_prevents_output(self):
        marker=self.stage/'cells/EC/complete.json';value=training.read_json(marker)
        value['copy_controls']['main']['same_native_copy']['all_context_loss_and_argmax_byte_equal']=False
        marker.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'recomputed'):self.analyze()
        self.assertFalse(self.output.exists())

    def test_mutation_during_partial_aggregation_prevents_publication(self):
        original=partial.aggregate
        def changed(*args):
            result=original(*args);self.request_fixture.fixture.source.write_text('changed');return result
        with mock.patch.object(partial,'aggregate',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'frozen evidence changed'):self.analyze()
        self.assertFalse(self.output.exists())


if __name__=='__main__':
    unittest.main()
