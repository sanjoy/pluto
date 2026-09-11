"""Synthetic file-to-report checks; no real corpus, weights, or CUDA access."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import neuron_mapping as nm
from .checkpoint import tensor_manifest


class ReportFilesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan_path, self.run_path = self.root / 'plan.json', self.root / 'run.json'
        self.output = self.root / 'report.json'
        zero = np.zeros((2048, 512), dtype=np.float32)
        groups = nm.group_records(nm.fixed_groups(), [zero, zero])
        self.plan = {'groups': groups, 'arms': nm.arm_specs(groups),
                     'group_file': {'path': str(self.root / 'groups.i32')}}
        # Each synthetic token occupies four bytes. Offsets include the end
        # sentinel, so report interval endpoints exercise the real indexing.
        self.base = {'inputs': {'binary': {'path': str(self.root / 'binary')}},
                     'checkpoint': {'checkpoint_directory': str(self.root / 'checkpoint')},
                     'passages': [{'passage_id': f'passage_{p:02d}',
                                   'target_byte_offsets': (p * 10000 + np.arange(1025) * 4).tolist(),
                                   'native_token_ids': (p * 1024 + np.arange(1025)).tolist()}
                                  for p in range(32)]}
        self.batch = np.stack([np.arange(32 * 1024).reshape(32, 1024) + shift
                               for shift in (0, 1)]).astype('<i4')
        self.run = {
            'schema_version': 1, 'complete': True, 'context_length': 1024,
            'passage_count': 32, 'vocab_size': 50257, 'padded_vocab_size': 50272,
            'batch_sequences': 4, 'checkpoint_raw_weight_count': 101,
            'checkpoint_unique_weight_count': 100,
            'checkpoint_weight_bytes': [spec.nbytes for spec in tensor_manifest()],
            'batch_tokens_bytes': 262144, 'neuron_groups_bytes': 16384,
            'neuron_groups_shape': [2, 32, 64], 'neuron_group_blocks': [6, 7],
            'neuron_groups_permutations_verified': True, 'non_default_executor': True,
            'optimizer_steps': 0, 'backward_calls': 0,
            'binary_file': self.base['inputs']['binary']['path'],
            'checkpoint_directory': self.base['checkpoint']['checkpoint_directory'],
            'batch_tokens_file': str(self.root / 'passages/batch_tokens.bin'),
            'neuron_groups_file': self.plan['group_file']['path'],
            'arms': copy.deepcopy(self.plan['arms'])}
        self.clean = (1 + np.arange(32)[:, None] / 32 + np.arange(1024)[None, :] / 2048).astype('<f4')
        self.clean[0, 0] = 0.0  # Distinguish exact-byte replay from numeric equality.
        rng = np.random.default_rng(981)
        self.losses, self.argmax = {}, {}
        for arm in self.run['arms']:
            name = arm['name']
            arm.update(finite_losses=True, argmax_valid=True, restoration_verified_bytes=True,
                       loss_file=name + '.losses.f32', argmax_file=name + '.argmax.i32')
            loss, ids = self.clean.copy(), self.batch[1].copy()
            if arm['kind'] != 'clean':
                # Exactly represented, different discovery/confirmation deltas.
                loss[:, 512:768] += rng.integers(32, 128, (32, 1)) / 1024
                loss[:, 768:] += rng.integers(-16, 64, (32, 1)) / 1024
                ids[:, 512:768:2] += 1
                ids[:, 768::4] += 1
            self.losses[name], self.argmax[name] = loss, ids
            (self.root / arm['loss_file']).write_bytes(loss.tobytes())
            (self.root / arm['argmax_file']).write_bytes(ids.tobytes())
        self.plan_path.write_text(json.dumps(self.plan))
        self.run_path.write_text(json.dumps(self.run))

    def report(self):
        # The sole mock substitutes expensive authenticated real inputs. Every
        # metadata gate, measurement read/hash, summary, and output write is real.
        with mock.patch.object(nm, 'authenticate', return_value=(self.plan, self.base, self.batch)) as auth:
            result = nm.report_files(self.plan_path, self.run_path, self.output)
            self.assertEqual(auth.call_count, 2)
            return result

    def test_all_69_arms_round_trip_with_exact_intervals_and_selection(self):
        report = self.report()
        self.assertEqual(json.loads(self.output.read_text()), report)
        self.assertEqual(len(report['raw_files']), 138)
        self.assertTrue(all(record['bytes'] == 131072 for record in report['raw_files'].values()))
        self.assertTrue(report['clean_replays_byte_identical'])
        for label, (start, end) in {'discovery': (512, 768), 'confirmation': (768, 1024),
                                   'last512': (512, 1024)}.items():
            for name, loss in self.losses.items():
                # An explicit per-passage scalar sum is independent of the
                # reporter's vectorized slicing/reduction and FP32 subtraction.
                means = [sum(float(x) for x in row[start:end]) / (end - start) for row in loss]
                delta = [sum(float(x) - float(y) for x, y in zip(row[start:end], clean[start:end])) /
                         (end - start) for row, clean in zip(loss, self.clean)]
                accuracy = [sum(int(x == y) for x, y in zip(row[start:end], target[start:end])) /
                            (end - start) for row, target in zip(self.argmax[name], self.batch[1])]
                self.assertEqual(report['per_passage_mean_nll'][label][name], means)
                self.assertEqual(report['per_passage_mean_delta'][label][name], delta)
                self.assertEqual(report['per_passage_teacher_forced_accuracy'][label][name], accuracy)
        mapping = report['mapping']
        self.assertGreater(mapping['summary']['selected_passages'], 0)
        deltas = report['per_passage_mean_delta']
        expected = nm.select_and_confirm(
            [deltas['discovery'][g['name']][:16] for g in self.plan['groups']],
            [deltas['confirmation'][g['name']][:16] for g in self.plan['groups']], self.plan['groups'])
        self.assertEqual(mapping['discovery'], expected['discovery'])
        self.assertEqual(mapping['summary'], expected['summary'])
        for p, entry in enumerate(mapping['selected']):
            self.assertEqual(entry['passage_id'], f'passage_{p:02d}')
            self.assertEqual(entry['discovery_target_bytes'], [p * 10000 + 2048, p * 10000 + 3072])
            self.assertEqual(entry['confirmation_target_bytes'], [p * 10000 + 3072, p * 10000 + 4096])
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError): self.report()
        self.assertEqual(self.output.read_bytes(), before)

    def test_replays_reject_signed_zero_and_valid_but_changed_argmax(self):
        for name, suffix, dtype, changed in [('clean_repeat', '.losses.f32', '<f4', -0.0),
                                            ('clean_after', '.argmax.i32', '<i4', 2)]:
            with self.subTest(name=name):
                path = self.root / (name + suffix)
                original = path.read_bytes()
                data = np.frombuffer(original, dtype=dtype).copy()
                data[0] = changed
                path.write_bytes(data.tobytes())
                with self.assertRaisesRegex(ValueError, 'byte identical'): self.report()
                self.assertFalse(self.output.exists())
                path.write_bytes(original)

    def test_malformed_measurements_fail_closed(self):
        name = self.plan['groups'][0]['name']
        cases = [('.losses.f32', '<f4', x) for x in (np.nan, np.inf, -1)]
        cases += [('.argmax.i32', '<i4', x) for x in (-1, 50257)]
        cases += [('.losses.f32', '<f4', 'short'), ('.argmax.i32', '<i4', 'long')]
        for suffix, dtype, changed in cases:
            with self.subTest(suffix=suffix, changed=changed):
                path = self.root / (name + suffix)
                original = path.read_bytes()
                data = np.frombuffer(original, dtype=dtype).copy()
                if isinstance(changed, str):
                    payload = original[:-4] if changed == 'short' else original + b'\0' * 4
                else:
                    data[-1] = changed
                    payload = data.tobytes()
                path.write_bytes(payload)
                with self.assertRaises(ValueError): self.report()
                self.assertFalse(self.output.exists())
                path.write_bytes(original)

    def test_file_report_obeys_actual_selected_row_and_bias_metadata_gate(self):
        for key, changed in [('selected_rows', [0]), ('group_weight_indices', [84, 85])]:
            broken = copy.deepcopy(self.run)
            broken['arms'][3][key] = changed
            self.run_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, 'frozen intervention'): self.report()
            self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
