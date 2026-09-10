"""Real tiny weight/case files with mocked timed-run completion, never CUDA."""

import copy
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from . import paired_lowercase_intervention_plan as planner
from . import paired_lowercase_supplemental_test as case_fixtures
from . import paired_lowercase_training as training


class Fixture:
    def __init__(self,test):
        corpus=case_fixtures.SupplementalTest(); corpus.setUp()
        test.addCleanup(corpus.doCleanups)
        corpus.add_shared_e()
        self.root=corpus.fixture.amended_root; self.legacy=corpus.fixture.old_root
        corpus.output=self.root/'supplemental_cases'
        self.supplemental=corpus.prepare(); self.main=corpus.main
        self.main_path=corpus.path; self.supp_path=corpus.output/'cases.json'
        self.exports_dir=self.root/'causal_cases'; self.exports_dir.mkdir()
        exports={'main':planner.legacy.export_cases(self.main_path,self.exports_dir/'main',rows_per_case=3)}
        for kind,rows in (('word_next_native',4),('shared_piece',3)):
            indices=[i for i,c in enumerate(self.supplemental['cases']) if c['kind']==kind]
            exports[kind]=planner.legacy.export_cases(self.supp_path,self.exports_dir/kind,indices=indices,rows_per_case=rows)
        self.exports_path=self.exports_dir/'exports.json'
        self.exports=dict(format='pluto-paired-lowercase-causal-exports-v1',complete=True,exports=exports,
            exporter=training.record(planner.legacy.__file__),main=training.record(self.main_path),supplemental=training.record(self.supp_path))
        training.publish(self.exports_path,self.exports)
        self.config=planner.checkpoint.GPT2Config(vocab_size=50257,padded_vocab_size=50257,context_length=16,
                                                 n_layers=2,d_model=2,n_heads=1,d_ff=4)
        inventories={}
        for root,arm,steps in ((self.legacy,'initial',[0]),(self.legacy,'original',[0,100,200,300,331]),
                               (self.root,'replacement',[0,100,200,300,333])):
            parent=root/arm/'checkpoints'; parent.mkdir(parents=True)
            for step in steps:
                directory=parent/f'step_{step}'; directory.mkdir()
                for spec in planner.checkpoint.tensor_manifest(self.config):
                    values=np.full(spec.shape,0 if step==0 else step+(arm=='replacement'),dtype='<f4')
                    values.tofile(directory/spec.filename)
            inventories[arm]=training.inventory(parent,config=self.config)
        original,replacement=({i['step']:i for i in inventories[arm]} for arm in ('original','replacement'))
        plan=planner.trajectory.paired_analysis.plan_comparisons(original,replacement,331,333,all_matched_steps=True)
        plan.update(initial=inventories['initial'][0],initial_copies=[original[0],replacement[0]],
                    arm_roots={'original':str(self.legacy),'replacement':str(self.root)})
        self.request=dict(amendment_root=str(self.root),legacy_root=str(self.legacy),
                           amendment=self.main['amendment'],legacy_manifest=self.main['old_manifest'])
        training.publish(self.root/'request.json',self.request)
        training.publish(self.root/'state.json',{'phase':'training_complete','fixture_not_executed':True})
        self.fresh=dict(plan=plan,original_reference={'reused_not_rerun':True},
                       determinism_verification={'fixture_repeatable':True},
                       terminal_records=[training.record(self.root/'state.json')],
                       checked=[i for items in inventories.values() for i in items])
        self.summary=dict(format=planner.trajectory.FORMAT,complete=True,goal_completion_claimed=False,
            plan=planner.canonical(plan),original_reference=self.fresh['original_reference'],
            determinism_verification=self.fresh['determinism_verification'],
            frozen_inputs=[training.record(self.root/'request.json'),training.record(self.main_path),self.main['packed_batch']],
            terminal_records=self.fresh['terminal_records'],
            checkpoints=[dict(path=i['path'],weight_sha256=i['sha256']) for i in self.fresh['checked']])
        (self.root/'analysis_trajectory').mkdir()
        self.summary_path=self.root/'analysis_trajectory/summary.json'
        training.publish(self.summary_path,self.summary)
        self.output=self.root/'interventions'
        self.completion=test.enterContext(mock.patch.object(planner.trajectory,'validate_completion',return_value=self.fresh))

    def prepare(self,**kwargs):
        return planner.prepare(self.root,self.summary_path,self.exports_path,self.output,config=kwargs.get('config',self.config))

    def save_summary(self): training.publish(self.summary_path,self.summary,exclusive=False)
    def save_exports(self): training.publish(self.exports_path,self.exports,exclusive=False)


class AmendedPlanTest(unittest.TestCase):
    def setUp(self): self.fixture=Fixture(self)

    def test_actual_roots_common_steps_and_no_patch_materialization(self):
        f=self.fixture
        before=set(f.root.parent.rglob('weight_*.bin'))
        plan=f.prepare()
        self.assertEqual(plan['selected_steps'],[100,300])
        self.assertEqual(plan['endpoint_context']['original']['step'],331)
        self.assertEqual(plan['endpoint_context']['replacement']['step'],333)
        self.assertEqual(plan['arm_roots'],{'original':str(f.legacy),'replacement':str(f.root)})
        self.assertEqual(plan['word_piece_ids'],[45,68,303,400,409,1475,2797,3109,14364,21733,45177])
        self.assertEqual(len(plan['interventions']),40)  # Two-block toy geometry, both directions, two steps.
        self.assertFalse(plan['patches_materialized']); self.assertFalse(plan['goal_completion_claimed'])
        self.assertEqual(before,set(f.root.parent.rglob('weight_*.bin')))
        self.assertEqual(list(f.output.iterdir()),[f.output/'plan.json'])
        for item in plan['interventions']:
            self.assertEqual(item['recipient_checkpoint']['step'],item['donor_checkpoint']['step'])
            if item['embedding_rows']: self.assertEqual(len(item['embedding_rows']),11)

    def test_eight_block_screen_still_contains_all_branch_and_direction_combinations(self):
        f=self.fixture
        pair=next(p for p in f.summary['plan']['pairs'] if p['name']=='matched_step_100')
        entries=planner.legacy._interventions(pair,planner.checkpoint.tensor_manifest(),list(range(11)))
        self.assertEqual(len(entries),68)
        kinds={i['kind'] for i in entries}
        self.assertEqual(kinds,{'copy_control','embedding_rows','attention_whole_branch','attention_output_write','mlp_whole_branch','mlp_output_write'})

    def test_incomplete_trajectory_and_completion_failure_never_publish(self):
        f=self.fixture; f.summary['complete']=False; f.save_summary()
        with self.assertRaises(ValueError): f.prepare()
        self.assertFalse(f.output.exists())
        f.summary['complete']=True; f.save_summary(); f.completion.side_effect=ValueError('short training')
        with self.assertRaisesRegex(ValueError,'short training'): f.prepare()
        self.assertFalse(f.output.exists())

    def test_summary_must_match_independent_completion(self):
        f=self.fixture; f.summary['plan']['pairs'][0]['replacement']['step']=331; f.save_summary()
        with self.assertRaisesRegex(ValueError,'independently revalidated'): f.prepare()

    def test_original_cannot_be_relocated_in_an_amended_summary(self):
        f=self.fixture; f.fresh['plan']['arm_roots']['original']=str(f.root)
        f.summary['plan']=planner.canonical(f.fresh['plan']); f.save_summary()
        with self.assertRaisesRegex(ValueError,'relabeled'): f.prepare()

    def test_main_batch_must_be_one_that_trajectory_actually_scored(self):
        f=self.fixture; f.summary['frozen_inputs'].remove(f.main['packed_batch']); f.save_summary()
        with self.assertRaisesRegex(ValueError,'exact amended main'): f.prepare()

    def test_checkpoint_hash_mutation_is_rejected(self):
        f=self.fixture; file=f.root/'replacement/checkpoints/step_100/weight_0.bin'
        data=np.fromfile(file,dtype='<f4'); data[0]+=1; data.tofile(file)
        with self.assertRaisesRegex(ValueError,'hash changed'): f.prepare()

    def test_duplicate_or_conflicting_inventory_declarations_rejected(self):
        f=self.fixture; f.summary['checkpoints'].append(copy.deepcopy(f.summary['checkpoints'][0])); f.save_summary()
        with self.assertRaisesRegex(ValueError,'checkpoint hashes'): f.prepare()

    def test_export_subset_cannot_drop_a_case(self):
        f=self.fixture; f.exports['exports']['word_next_native']['source_indices'].pop(); f.save_exports()
        with self.assertRaisesRegex(ValueError,'export selection'): f.prepare()

    def test_exported_spelling_labels_cannot_change(self):
        f=self.fixture; path=Path(f.exports['exports']['main']['cases_json']['path'])
        document=training.read_json(path); document['cases'][0]['spelling_variant']='lowercase'
        training.publish(path,document,exclusive=False)
        with self.assertRaisesRegex(ValueError,'source labels'): f.prepare()

    def test_exported_rows_must_match_source_rows(self):
        f=self.fixture; path=Path(f.exports['exports']['shared_piece']['selected_rows']['path'])
        values=np.fromfile(path,dtype='<i4'); values[0]+=1; values.tofile(path)
        with self.assertRaisesRegex(ValueError,'scored rows'): f.prepare()

    def test_exported_batch_cannot_change_even_unscored_padding(self):
        f=self.fixture; path=Path(f.exports['exports']['main']['packed_batch']['path'])
        values=np.fromfile(path,dtype='<i4'); values[-1]=1; values.tofile(path)
        with self.assertRaisesRegex(ValueError,'source bytes'): f.prepare()

    def test_geometry_mismatch_rejected(self):
        f=self.fixture; config=planner.checkpoint.GPT2Config(vocab_size=50257,padded_vocab_size=50257,
            context_length=17,n_layers=2,d_model=2,n_heads=1,d_ff=4)
        with self.assertRaisesRegex(ValueError,'geometry'): f.prepare(config=config)

    def test_existing_output_is_preserved(self):
        f=self.fixture; f.output.mkdir(); (f.output/'keep').write_text('keep')
        with self.assertRaises(FileExistsError): f.prepare()
        self.assertEqual((f.output/'keep').read_text(),'keep')

    def test_output_cannot_be_under_case_or_checkpoint_inputs(self):
        f=self.fixture
        for path in (f.root/'causal_cases/plan',f.root/'replacement/checkpoints/plan'):
            f.output=path
            with self.assertRaisesRegex(ValueError,'overlaps'): f.prepare()
            self.assertFalse(path.exists())


if __name__=='__main__':
    unittest.main()
