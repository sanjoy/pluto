"""Corpus-only controls tested with synthetic native-format exports; no GPU."""

import json
from pathlib import Path
import unittest

import numpy as np

from . import paired_lowercase_supplemental as supplemental
from . import paired_lowercase_cases_test as fixtures
from . import paired_word_cases as old_cases
from . import paired_intervention_plan as exports


class SupplementalTest(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.LowercaseCasesTest(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        self.output=self.root/'supplemental'

    def prepare(self,**kwargs):
        self.main=self.fixture.prepare()
        self.path=self.fixture.output/'cases.json'
        return supplemental.prepare(self.path,self.output,prefix_tokens=4,**kwargs)

    def test_preserves_all_four_crossings_and_first_three_causal_inputs(self):
        result=self.prepare()
        words=[c for c in result['cases'] if c['kind']=='word_next_native']
        self.assertEqual(len(words),32)
        self.assertEqual(sum(c['spelling_variant']=='lowercase' for c in words),8)
        self.assertFalse(any(c['spelling_variant']=='lowercase' and c['split']=='test' for c in words))
        main=np.fromfile(self.main['packed_batch']['path'],dtype='<i4').reshape(2,36,16)
        packed=np.fromfile(result['packed_batch']['path'],dtype='<i4').reshape(2,len(words),16)
        for case in words:
            index=case['case_index']; source=case['source_case_index']; rows=case['scored_rows']
            self.assertEqual(case['target_ids'][:3],self.main['cases'][source]['target_ids'])
            self.assertTrue(np.array_equal(packed[0,index,:rows[2]+1],main[0,source,:rows[2]+1]))
            self.assertTrue(np.array_equal(packed[1,index,rows[:3]],main[1,source,rows[:3]]))
            data=self.fixture.updated[case['target_source_domain']+'.'+case['split']]
            start=case['prefix']['token_end']
            self.assertEqual(case['next_native_token']['id'],int(data['tokens'][start+3]))
            self.assertEqual(case['target_source']['token_start'],start)
        self.assertEqual(result['selection']['excluded_word_cases'],[])

    def test_piece_union_includes_lowercase_and_missing_coverage_explicit(self):
        result=self.prepare()
        self.assertEqual(result['selection']['union_piece_ids'],[45,68,303,400,409,1475,2797,3109,14364,21733,45177])
        for coverage in result['selection']['coverage'].values():
            self.assertEqual(len(coverage),11)
            self.assertTrue(all(c['missing_coverage'] for c in coverage))
            self.assertTrue(all(c['selected_count']==0 for c in coverage))
        self.assertFalse(result['selection']['model_outputs_used'])

    def add_shared_e(self):
        # Add the one-byte subword e outside all replacement spans. Re-export
        # source evidence honestly; no fake alignment metadata is injected.
        f=self.fixture
        for name,ids in f.old_streams.items():
            ids[100]=68
            if name.endswith('.full'): ids[240]=68
            f.old_inputs[name]=fixtures.native(ids)
            f.manifest['inputs'][name]=f.export(f.old_root,name,f.old_inputs[name])
        f.manifest_path.write_text(json.dumps(f.manifest))
        old_dir=f.old_root/'cases_with_shared_e'
        f.old_report=old_cases.prepare(f.manifest_path,old_dir,contexts_per_split=3,
                                      controls_per_split=0,prefix_tokens=4,context_length=16)
        f.old_case_path=old_dir/'cases.json'
        for name,data in f.updated.items():
            ids=data['tokens'].tolist(); ids[100]=68
            if name.endswith('.full'): ids[240]=68
            f.updated[name]=fixtures.native(ids)
            f.amendment['inputs'][name]=(f.manifest['inputs'][name] if name.startswith('original.')
                                         else f.export(f.amended_root,name,f.updated[name]))
        f.refresh_amendment()

    def test_shared_subword_controls_are_unchanged_disjoint_and_reproducible(self):
        self.add_shared_e()
        result=self.prepare(controls_per_piece=3)
        shared=[c for c in result['cases'] if c['kind']=='shared_piece']
        self.assertEqual(len(shared),2)
        for case in shared:
            self.assertEqual(case['piece_id'],68); self.assertEqual(case['prefix']['token_end'],100)
            self.assertEqual(case['target_ids'],[68,64,64]); self.assertIsNone(case['candidate_pair'])
            coverage=next(c for c in result['selection']['coverage'][case['split']] if c['piece_id']==68)
            self.assertEqual(coverage['selected_token_starts'],[100])
            self.assertEqual(coverage['unfilled_requested_count'],2)
        second=supplemental.prepare(self.path,self.root/'supplemental_two',prefix_tokens=4,controls_per_piece=3)
        self.assertEqual(result['selection'],second['selection'])
        self.assertEqual(result['packed_batch']['sha256'],second['packed_batch']['sha256'])

    def test_all_source_records_rehash(self):
        result=self.prepare()
        supplemental.training.verify_records([*result['provenance'],*result['sources'],result['packed_batch']])

    def test_native_case_export_preserves_amended_labels_and_bytes(self):
        self.add_shared_e(); result=self.prepare()
        main=exports.export_cases(self.path,self.root/'export_main',rows_per_case=3)
        self.assertEqual(main['packed_batch']['sha256'],self.main['packed_batch']['sha256'])
        for kind,nrows in (('word_next_native',4),('shared_piece',3)):
            indices=[i for i,c in enumerate(result['cases']) if c['kind']==kind]
            out=exports.export_cases(self.output/'cases.json',self.root/('export_'+kind),
                                     indices=indices,rows_per_case=nrows)
            plan=json.loads(Path(out['cases_json']['path']).read_text())
            self.assertEqual(plan['format'],supplemental.FORMAT)
            self.assertEqual([c['export_source_case_index'] for c in plan['cases']],indices)
            for c,i in zip(plan['cases'],indices):
                self.assertEqual(c['spelling_variant'],result['cases'][i]['spelling_variant'])
                self.assertEqual(c['candidate_pair'],result['cases'][i]['candidate_pair'])
            rows=np.fromfile(out['selected_rows']['path'],dtype='<i4').reshape(-1,nrows)
            self.assertEqual(rows.tolist(),[result['cases'][i]['scored_rows'] for i in indices])

    def test_native_source_mutation_rejected(self):
        self.fixture.prepare(); self.path=self.fixture.output/'cases.json'
        file=Path(self.fixture.amendment['inputs']['replacement.training']['text'])
        file.write_bytes(file.read_bytes().replace(b'nuveth',b'exeunt'))
        with self.assertRaises(ValueError):
            supplemental.prepare(self.path,self.output,prefix_tokens=4)
        self.assertFalse(self.output.exists())

    def test_invalid_geometry_and_counts_rejected(self):
        self.fixture.prepare(); self.path=self.fixture.output/'cases.json'
        for options in (dict(controls_per_piece=0),dict(controls_per_piece=True),dict(prefix_tokens=0),dict(prefix_tokens=16)):
            with self.subTest(options=options),self.assertRaises(ValueError):
                supplemental.prepare(self.path,self.output,**options)
        self.assertFalse(self.output.exists())

    def test_no_overwrite_or_output_inside_inputs(self):
        self.fixture.prepare(); self.path=self.fixture.output/'cases.json'
        self.output.mkdir(); (self.output/'keep').write_text('keep')
        with self.assertRaises(FileExistsError): supplemental.prepare(self.path,self.output,prefix_tokens=4)
        with self.assertRaises(ValueError): supplemental.prepare(self.path,self.path.parent/'new',prefix_tokens=4)
        self.assertEqual((self.output/'keep').read_text(),'keep')


if __name__=='__main__':
    unittest.main()
