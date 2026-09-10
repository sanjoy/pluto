"""CPU dependency checks for embedding-row interventions; no CUDA execution."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .paired_input_exposure import analyze, exposures
from .paired_word_cases import _case_sequence, _record, _write_json


class ExposureTest(unittest.TestCase):
    def test_future_word_does_not_affect_first_prediction(self):
        # Three prompt IDs, then Ex/e/unt. The first prediction uses rows 0..2;
        # the second sees Ex, and the third sees Ex/e but not the future unt.
        tokens = np.array([11, 12, 13, 3109, 68, 2797, 50256], dtype='<i4')
        result = exposures(tokens, [2, 3, 4], [3109, 68, 2797])
        self.assertEqual([x['patched_input_ids'] for x in result], [[], [3109], [68, 3109]])
        self.assertEqual([x['input_path_must_be_unchanged'] for x in result], [True, False, False])

    def test_shared_piece_in_prompt_removes_guarantee(self):
        result = exposures(np.array([400, 12, 13, 3109]), [2], [3109, 400])
        self.assertEqual(result[0]['patched_input_ids'], [400])
        self.assertFalse(result[0]['input_path_must_be_unchanged'])

    def test_future_candidate_and_padding_do_not_change_prefix_identity(self):
        a = exposures(np.array([11, 12, 13, 3109, 50256]), [2], [3109, 45])
        b = exposures(np.array([11, 12, 13, 45, 400]), [2], [3109, 45])
        self.assertEqual(a, b)

    def test_prefix_hash_is_dtype_independent_but_order_sensitive(self):
        a = exposures(np.array([11, 12], dtype='<i4'), [1], [45])[0]
        b = exposures(np.array([11, 12], dtype='<i8'), [1], [45])[0]
        c = exposures(np.array([12, 11], dtype='<i4'), [1], [45])[0]
        self.assertEqual(a, b)
        self.assertNotEqual(a['causal_prefix_sha256'], c['causal_prefix_sha256'])

    def test_invalid_ids_rows_and_shapes(self):
        for tokens, rows, ids in [([-1], [0], [1]), ([50257], [0], [1]),
                                  ([1], [-1], [1]), ([1], [1], [1]),
                                  ([1], [True], [1]), ([1], [0], []),
                                  ([1], [0], [True]), ([[1]], [0], [1])]:
            with self.subTest(tokens=tokens, rows=rows, ids=ids):
                with self.assertRaises(ValueError):
                    exposures(np.array(tokens), rows, ids)

    def fixture(self, root):
        cases, xs, ys = [], [], []
        prefix = np.array([11, 12, 13], dtype='<i4')
        for target in ([3109, 68, 2797], [45, 45177, 400]) * 2:
            x, y, rows = _case_sequence(prefix, target, 8)
            cases.append({'case_index': len(cases), 'kind': 'word', 'split': 'training',
                          'context_id': 'same-context', 'target': str(target),
                          'prefix': {'length': 3, 'token_ids_sha256': hashlib.sha256(
                              prefix.tobytes()).hexdigest()},
                          'target_ids': target, 'scored_rows': rows})
            xs.append(x)
            ys.append(y)
        packed = root / 'packed.bin'
        np.array([xs, ys], dtype='<i4').tofile(packed)
        path = root / 'cases.json'
        _write_json(path, {'format': 'pluto-paired-word-cases-v1', 'vocab_size': 50257,
                          'eos_token_id': 50256, 'case_count': 4, 'context_length': 8,
                          'cases': cases, 'packed_batch': _record(packed)})
        return path

    def test_full_report_deduplicates_domains_and_first_candidate_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.fixture(root)
            result = analyze(cases, root / 'report.json')
            self.assertEqual(len(result['per_case']), 4)
            self.assertEqual([g['unique_causal_prefix_count'] for g in result['groups']],
                             [1, 2, 2])
            self.assertEqual([g['unchanged_input_path_count'] for g in result['groups']],
                             [1, 0, 0])
            self.assertEqual(json.loads((root / 'report.json').read_text()), result)
            with self.assertRaises(FileExistsError):
                analyze(cases, root / 'report.json')

    def test_full_report_rejects_packed_hash_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.fixture(root)
            raw = bytearray((root / 'packed.bin').read_bytes())
            raw[0] ^= 1
            (root / 'packed.bin').write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'size/hash'):
                analyze(cases, root / 'report.json')
            self.assertFalse((root / 'report.json').exists())


if __name__ == '__main__':
    unittest.main()
