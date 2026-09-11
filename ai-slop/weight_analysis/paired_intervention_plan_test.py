"""CPU-only immutable plan and exact selected-row packing tests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import checkpoint, paired_intervention_plan as planner
from . import paired_supplemental_cases, paired_word_cases
from .paired_supplemental_cases_test import add_piece_controls
from .paired_word_cases_test import make_manifest


def write_json(path, value):
    path.write_text(json.dumps(value))


class InterventionPlanTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = checkpoint.GPT2Config(vocab_size=64, padded_vocab_size=64,
                                           n_layers=8, d_model=2, n_heads=1, d_ff=4)
        manifest_path = make_manifest(self.root)
        add_piece_controls(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        manifest.update(root=str(self.root), seconds_per_arm=14400,
                        determinism={'required': True})
        # A third native toy variant adds the ninth union ID while preserving
        # exact bytes, equal counts, and the original replacement span mapping.
        for split in paired_word_cases.SPLITS:
            item = manifest['inputs'][f'original.{split}']
            tokens = np.fromfile(item['token_ids'], dtype='<u4')
            tokens[80] = 41
            tokens.tofile(item['token_ids'])
            item['token_ids_sha256'] = checkpoint.sha256_file(item['token_ids'])
            manifest['alignment'][split]['occurrences'][2]['original_ids'][0] = 41
        write_json(manifest_path, manifest)
        word_dir, supplemental_dir = self.root / 'word_cases', self.root / 'supplemental'
        self.word_path, self.supplemental_path = word_dir / 'cases.json', supplemental_dir / 'cases.json'
        self.word = paired_word_cases.prepare(manifest_path, word_dir, contexts_per_split=3,
                                              controls_per_split=1, prefix_tokens=4, context_length=16)
        self.supplemental = paired_supplemental_cases.prepare(manifest_path, self.word_path,
                                                              supplemental_dir, prefix_tokens=4)
        self.inventories = {}
        checkpoints = []
        for arm, steps in (('original', (0, 100, 200)), ('replacement', (0, 100, 200, 230))):
            self.inventories[arm] = {}
            directory = self.root / arm
            (directory / 'checkpoints').mkdir(parents=True)
            for step in steps:
                path = directory / 'checkpoints' / f'step_{step}'
                path.mkdir()
                hashes = {}
                for spec in checkpoint.tensor_manifest(self.config):
                    value = step + (1 if arm == 'replacement' and step else 0)
                    np.full(spec.shape, value, dtype='<f4').tofile(path / spec.filename)
                    hashes[spec.filename] = checkpoint.sha256_file(path / spec.filename)
                item = {'step': step, 'path': str(path), 'sha256': hashes}
                self.inventories[arm][step] = item
                checkpoints.append({'path': str(path), 'weight_sha256': hashes})
            write_json(directory / 'checkpoints.json', list(self.inventories[arm].values()))
        gate = {'status': 'verified', 'steps': [0, 1, 2]}
        write_json(self.root / 'gate.json', gate)
        state = {'phase': 'training_complete', 'determinism_gate': gate, 'runs': {}}
        for arm in planner.ARMS:
            last = max(self.inventories[arm])
            state['runs'][arm] = {'returncode': 0, 'initial_weights_match': True,
                                  'stop_reason': 'time_limit', 'training_elapsed_seconds': 14400.5,
                                  'final_step': last, 'final_checkpoint': self.inventories[arm][last]['path']}
        write_json(self.root / 'state.json', state)
        pairs = [{'name': 'final', 'original': self.inventories['original'][200],
                  'replacement': self.inventories['replacement'][230]}]
        pairs += [{'name': f'matched_step_{step}', **{arm: self.inventories[arm][step]
                                                     for arm in planner.ARMS}} for step in (0, 100, 200)]
        self.summary = {'format': 'pluto-paired-analysis-v1', 'complete': True,
                        'plan': {'pairs': pairs, 'all_matched_steps': True,
                                 'matched_steps': [0, 100, 200], 'matched_step': 200},
                        'frozen_inputs': [planner._record(manifest_path), planner._record(self.word_path),
                                           self.word['packed_batch']],
                        'terminal_records': [planner._record(self.root / 'state.json'),
                            *(planner._record(self.root / arm / 'checkpoints.json') for arm in planner.ARMS)],
                        'checkpoints': checkpoints,
                        'determinism_verification': {'evidence': gate,
                                                     'records': [planner._record(self.root / 'gate.json')]}}
        (self.root / 'analysis').mkdir()
        self.summary_path = self.root / 'analysis' / 'summary.json'
        write_json(self.summary_path, self.summary)
        self.output = self.root / 'plan'

    def prepare(self):
        return planner.prepare(self.summary_path, self.word_path, self.supplemental_path,
                               self.output, config=self.config)

    def test_first_latest_common_steps_not_unequal_endpoint_and_no_weight_copies(self):
        before = set(self.root.rglob('weight_*.bin'))
        plan = self.prepare()
        self.assertEqual(plan['selected_steps'], [100, 200])
        self.assertEqual(plan['endpoint_context']['replacement']['step'], 230)
        self.assertEqual(len(plan['interventions']), 136)
        self.assertFalse(plan['patches_materialized'])
        self.assertTrue(plan['complete'])
        self.assertEqual(before, set(self.root.rglob('weight_*.bin')))
        self.assertEqual(set(path.name for path in self.output.iterdir()),
                         {'plan.json', 'main', 'word_next_native', 'shared_piece'})
        for item in plan['interventions']:
            self.assertIn(item['step'], (100, 200))
            self.assertEqual(item['recipient_checkpoint']['step'], item['donor_checkpoint']['step'])
            self.assertNotEqual(item['recipient_arm'], item['donor_arm'])

    def test_branch_selections_use_manifest_and_separate_output_write_bias(self):
        plan = self.prepare()
        group = [item for item in plan['interventions'] if item['step'] == 100
                 and item['recipient_arm'] == 'original']
        self.assertEqual(len(group), 34)
        control = next(item for item in group if item['kind'] == 'copy_control')
        self.assertEqual(control['tensors'], [])
        self.assertEqual(control['embedding_rows'], [])
        rows = next(item for item in group if item['kind'] == 'embedding_rows')
        self.assertEqual(rows['embedding_rows'], [11, 12, 13, 21, 22, 23, 31, 32, 41])
        self.assertEqual(rows['tensors'], [])
        for block in range(8):
            for branch, norm, projection in (('attention', 'ln1', 'attn'), ('mlp', 'ln2', 'mlp')):
                whole = next(item for item in group if item['name'].endswith(f'block_{block}_{branch}_whole'))
                writes = next(item for item in group if item['name'].endswith(f'block_{block}_{branch}_output'))
                self.assertEqual(len(whole['tensors']), 6)
                self.assertIn(f'blocks.{block}.{norm}.scale', whole['tensors'])
                self.assertIn(f'blocks.{block}.{norm}.bias', whole['tensors'])
                self.assertEqual(writes['tensors'], [f'blocks.{block}.{projection}.output.weight',
                                                    f'blocks.{block}.{projection}.output.bias'])
                self.assertTrue(set(writes['tensors']) < set(whole['tensors']))
                self.assertEqual([spec['name'] for spec in whole['selection_specs']], whole['tensors'])

    def test_case_exports_preserve_selected_bytes_indices_and_three_four_row_packing(self):
        plan = self.prepare()
        for name, export in plan['exports'].items():
            source = self.word if name == 'main' else self.supplemental
            source_packed = np.fromfile(source['packed_batch']['path'], dtype='<i4').reshape(2, source['case_count'], 16)
            actual = np.fromfile(export['packed_batch']['path'], dtype='<i4').reshape(2, export['case_count'], 16)
            np.testing.assert_array_equal(actual, source_packed[:, export['source_indices'], :])
            expected_rows = 4 if name == 'word_next_native' else 3
            self.assertEqual(export['rows_per_case'], expected_rows)
            rows = np.fromfile(export['selected_rows']['path'], dtype='<i4').reshape(export['case_count'], expected_rows)
            exported = json.loads(Path(export['cases_json']['path']).read_text())
            for index, (original_index, row) in enumerate(zip(export['source_indices'], rows)):
                old = source['cases'][original_index]
                self.assertEqual(row.tolist(), old['scored_rows'])
                self.assertEqual(exported['cases'][index]['case_index'], index)
                self.assertEqual(exported['cases'][index]['export_source_case_index'], original_index)
                self.assertEqual(exported['cases'][index]['target_ids'], old['target_ids'])
                self.assertEqual(actual[1, index, row].tolist(), old['target_ids'])
            for key in ('cases_json', 'packed_batch', 'selected_rows'):
                self.assertEqual(planner._record(export[key]['path']), export[key])

    def test_export_rejects_mixed_rows_invalid_filters_and_overwrites(self):
        for indices in ([], [-1], [999], [1, 0], [1, 1], [True]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                planner.export_cases(self.word_path, self.output, indices=indices)
        with self.assertRaisesRegex(ValueError, 'mixed'):
            planner.export_cases(self.supplemental_path, self.output)
        with self.assertRaisesRegex(ValueError, 'rows_per_case'):
            planner.export_cases(self.word_path, self.output, rows_per_case=4)
        planner.export_cases(self.word_path, self.output, indices=[0, 2], rows_per_case=3)
        with self.assertRaises(FileExistsError):
            planner.export_cases(self.word_path, self.output)

    def test_single_positive_common_step_is_deduplicated(self):
        summary = json.loads(json.dumps(self.summary))
        summary['plan']['pairs'] = [pair for pair in summary['plan']['pairs']
                                    if pair['name'] != 'matched_step_100']
        summary['plan']['matched_steps'] = [0, 200]
        pairs, endpoints = planner._selected_pairs(summary)
        self.assertEqual([pair['original']['step'] for pair in pairs], [200])
        self.assertNotEqual(endpoints['original']['step'], endpoints['replacement']['step'])

    def test_prepare_with_single_common_step_has_only_one_set_of_interventions(self):
        self.summary['plan']['pairs'] = [pair for pair in self.summary['plan']['pairs']
                                         if pair['name'] != 'matched_step_100']
        self.summary['plan']['matched_steps'] = [0, 200]
        for arm in planner.ARMS:
            path = self.root / arm / 'checkpoints.json'
            records = [item for item in self.inventories[arm].values() if item['step'] != 100]
            write_json(path, records)
            self.summary['terminal_records'] = [planner._record(path) if item['path'] == str(path)
                                                 else item for item in self.summary['terminal_records']]
        write_json(self.summary_path, self.summary)
        plan = self.prepare()
        self.assertEqual(plan['selected_steps'], [200])
        self.assertEqual(len(plan['interventions']), 68)

    def test_equal_endpoints_are_also_valid_latest_matched_step(self):
        summary = json.loads(json.dumps(self.summary))
        summary['plan']['pairs'][0]['replacement'] = self.inventories['replacement'][200]
        summary['plan']['pairs'] = [pair for pair in summary['plan']['pairs']
                                    if pair['name'] != 'matched_step_200']
        pairs, endpoints = planner._selected_pairs(summary)
        self.assertEqual([pair['original']['step'] for pair in pairs], [100, 200])
        self.assertEqual(pairs[-1], endpoints)

    def test_rejects_unfinished_missing_common_or_incomplete_trajectory(self):
        for change in (lambda value: value.update(complete=False),
                       lambda value: value['plan'].update(all_matched_steps=False),
                       lambda value: value['plan'].update(matched_steps=[0], matched_step=None,
                           pairs=[value['plan']['pairs'][0], value['plan']['pairs'][1]])):
            summary = json.loads(json.dumps(self.summary))
            change(summary)
            write_json(self.summary_path, summary)
            with self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output.exists())

    def test_missing_early_common_step_cannot_be_hidden_in_summary(self):
        self.summary['plan']['pairs'] = [pair for pair in self.summary['plan']['pairs']
                                         if pair['name'] != 'matched_step_100']
        self.summary['plan']['matched_steps'] = [0, 200]
        write_json(self.summary_path, self.summary)
        with self.assertRaisesRegex(ValueError, 'actual completed inventories'):
            self.prepare()

    def test_rejects_changed_provenance_checkpoint_and_symlinks(self):
        for path in (self.root / 'state.json', self.word_path,
                     Path(self.supplemental['packed_batch']['path'])):
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.prepare()
            path.write_bytes(original)
        weight = Path(self.inventories['original'][100]['path']) / 'weight_0.bin'
        original = weight.read_bytes()
        weight.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(ValueError, 'weight hash changed'):
            self.prepare()
        weight.write_bytes(original)
        saved = weight.with_name('saved.bin')
        weight.rename(saved)
        weight.symlink_to(saved)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_bad_checkpoint_size_or_inventory_hash_claim(self):
        weight = Path(self.inventories['replacement'][200]['path']) / 'weight_99.bin'
        weight.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'layout mismatch'):
            self.prepare()
        self.summary['checkpoints'][0]['weight_sha256'] = {}
        write_json(self.summary_path, self.summary)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_rejects_missing_determinism_or_nine_piece_identity(self):
        original = json.loads(json.dumps(self.summary))
        self.summary['determinism_verification'] = None
        write_json(self.summary_path, self.summary)
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            self.prepare()
        write_json(self.summary_path, original)
        self.supplemental['selection']['union_piece_ids'][-1] = 42
        write_json(self.supplemental_path, self.supplemental)
        with self.assertRaisesRegex(ValueError, 'nine native'):
            self.prepare()

    def test_export_preserves_original_preexisting_source_case_index(self):
        word_indices = [index for index, case in enumerate(self.supplemental['cases'])
                        if case['kind'] == 'word_next_native']
        selected = [word_indices[1], word_indices[-1]]
        result = planner.export_cases(self.supplemental_path, self.output, indices=selected)
        exported = json.loads(Path(result['cases_json']['path']).read_text())
        for index, source_index in enumerate(selected):
            self.assertEqual(exported['cases'][index]['source_case_index'],
                             self.supplemental['cases'][source_index]['source_case_index'])
            self.assertEqual(exported['cases'][index]['export_source_case_index'], source_index)

    def test_export_rejects_mutated_rows_and_prefix_hash(self):
        original = json.loads(self.word_path.read_text())
        for change in (lambda plan: plan['cases'][0].update(scored_rows=[0, 1, 2]),
                       lambda plan: plan['cases'][0]['prefix'].update(token_ids_sha256='bad')):
            plan = json.loads(json.dumps(original))
            change(plan)
            write_json(self.word_path, plan)
            with self.assertRaises(ValueError):
                planner.export_cases(self.word_path, self.output)
            self.assertFalse(self.output.exists())

    def test_rejects_outputs_inside_any_input_or_existing_output(self):
        for parent in (self.word_path.parent, self.supplemental_path.parent,
                       self.summary_path.parent, self.root / 'original' / 'checkpoints'):
            self.output = parent / 'new'
            with self.assertRaisesRegex(ValueError, 'outside input'):
                self.prepare()
        self.output = self.root / 'plan'
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_cli_dispatch(self):
        with mock.patch.object(planner, 'prepare') as prepare:
            planner.main(['--analysis-summary', 'a.json', '--word-cases', 'w.json',
                          '--supplemental-cases', 's.json', '--output', 'new'])
            prepare.assert_called_once_with(Path('a.json'), Path('w.json'), Path('s.json'), Path('new'))


if __name__ == '__main__':
    unittest.main()
