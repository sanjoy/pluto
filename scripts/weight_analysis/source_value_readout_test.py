"""CPU evidence-auditor tests; synthetic files are NOT native GPU measurements."""

from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import source_value_readout as readout
from .checkpoint import GPT2Config, tensor_manifest


class NumericalTest(unittest.TestCase):
    def test_bf16_half_ties_subnormals_normal_and_signed_zero(self):
        values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0x7f, 0x80, 0x81, 0xff,
                           0x100, 0x101, 0x3f80, 0x7f7f, 0x8000, 0x8001, 0x8003], dtype='<u2')
        expected = np.array([0, 0, 1, 2, 2, 2, 3, 4, 0x40, 0x40, 0x40, 0x80,
                             0x80, 0x81, 0x3f00, 0x7eff, 0x8000, 0x8000, 0x8002], dtype='<u2')
        np.testing.assert_array_equal(readout.scale_bf16(values, 0.5), expected)

    def test_bf16_exhaustive_finite_halving(self):
        # Cross-check the nearest-number-line implementation against integer
        # exponent/significand mathematics over EVERY finite BF16 bit pattern.
        values = np.arange(65536, dtype='<u2')
        values = values[(values & 0x7f80) != 0x7f80]
        expected = []
        for value in values.tolist():
            magnitude, sign = value & 0x7fff, value & 0x8000
            if magnitude >= 0x100:
                expected.append(value - 0x80)
            else:
                quotient, remainder = divmod(magnitude, 2)
                expected.append(sign | (quotient + (remainder == 1 and quotient % 2 == 1)))
        np.testing.assert_array_equal(readout.scale_bf16(values, 0.5), expected)

    def test_zero_erases_negative_zero_and_identity_preserves_all_bits(self):
        values = np.array([0x8000, 0xbf80, 1, 0, 0x7f7f], dtype='<u2')
        np.testing.assert_array_equal(readout.scale_bf16(values, 0), np.zeros_like(values))
        np.testing.assert_array_equal(readout.scale_bf16(values, 1), values)

    def test_bf16_rejects_nonfinite_wrong_dtype_and_dose(self):
        for bad in (0x7f80, 0xff80, 0x7fc1):
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                readout.scale_bf16(np.array([bad], dtype='<u2'), 1)
        for dose in (-1, 0.25, float('nan'), True):
            with self.assertRaises(ValueError):
                readout.scale_bf16(np.array([1], dtype='<u2'), dose)
        with self.assertRaisesRegex(ValueError, 'raw BF16'):
            readout.scale_bf16(np.array([1], dtype='<f4'), 0.5)

    def test_scores_ties_exact_small_vocabulary(self):
        score = readout.score_logits([0, 0, 0, 0], 2)
        self.assertAlmostEqual(score['nll'], math.log(4))
        self.assertAlmostEqual(score['probability'], 0.25)
        self.assertEqual((score['target_rank'], score['argmax'], score['fixed_rival_id']), (3, 0, 0))

    def test_scores_fixed_rival_not_reselected_after_intervention(self):
        baseline = readout.score_logits([3, 2, 1], 0)
        score = readout.score_logits([3, -5, 10], 0, baseline['fixed_rival_id'])
        self.assertEqual(score['fixed_rival_id'], 1)
        self.assertEqual(score['fixed_rival_margin'], 8)
        self.assertEqual(score['argmax'], 2)
        self.assertAlmostEqual(score['probability'], math.exp(3) / sum(map(math.exp, [3, -5, 10])))

    def test_scores_stable_extremes_and_shift_invariance(self):
        for values in ([10000, 9999, 9998], [-10000, -10001, -10002]):
            result = readout.score_logits(values, 0)
            self.assertAlmostEqual(result['nll'], math.log(1 + math.exp(-1) + math.exp(-2)))
        self.assertEqual(readout.score_logits([10000, -10000], 1)['probability'], 0)

    def test_scores_reject_invalid_values_target_and_rival(self):
        for values, target, rival in (([0, float('nan')], 0, None), ([0, float('inf')], 0, None),
                                     ([0], 0, None), ([1, 2], -1, None), ([1, 2], 0, 0),
                                     ([1, 2], True, None)):
            with self.assertRaises(ValueError):
                readout.score_logits(values, target, rival)

    def tensors(self):
        qkv = np.full((3, 12), 0x3f80, dtype='<u2')
        ctx = np.full((3, 4), 0x3e80, dtype='<u2')
        edited = qkv.copy()
        edited[1, 10:12] = 0x3f00
        replayed = ctx.copy()
        replayed[1:, 2:4] = 0x3e00
        spliced = ctx.copy()
        spliced[2, 2:4] = replayed[2, 2:4]
        return [qkv, ctx, edited, replayed, spliced]

    def audit(self, tensors, scale=0.5):
        return readout.audit_tensors(*tensors, head=1, head_dim=2, query=2, source=1, scale=scale)

    def test_tensor_isolation_and_replayed_slice(self):
        result = self.audit(self.tensors())
        self.assertEqual(result, {'qkv_changed_elements': 2, 'spliced_context_changed_elements': 2,
                                  'replayed_attention_changed_elements': 4})

    def test_tensor_rejects_q_k_wrong_source_or_value_dose(self):
        for position in ((1, 0), (1, 4), (0, 10), (1, 10)):
            tensors = self.tensors()
            tensors[2][position] ^= 1
            with self.assertRaisesRegex(ValueError, 'exact source-V dose'):
                self.audit(tensors)

    def test_tensor_rejects_other_query_head_and_unreplayed_slice(self):
        for position in ((0, 2), (2, 0), (2, 2)):
            tensors = self.tensors()
            tensors[4][position] ^= 1
            with self.assertRaisesRegex(ValueError, 'context splice'):
                self.audit(tensors)

    def test_identity_requires_full_replayed_attention_not_only_selected_slice(self):
        qkv, ctx, *_ = self.tensors()
        tensors = [qkv, ctx, qkv.copy(), ctx.copy(), ctx.copy()]
        self.audit(tensors, scale=1)
        tensors[3][0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'identity attention'):
            self.audit(tensors, scale=1)

    def test_tensor_rejects_any_nonfinite_even_outside_selected_site(self):
        for index in range(5):
            tensors = self.tensors()
            tensors[index][0, 0] = 0x7f80
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                self.audit(tensors)

    def test_tensor_future_source_cannot_change_selected_causal_query(self):
        qkv, ctx, *_ = self.tensors()
        changed = qkv.copy()
        changed[2, 10:12] = 0
        replayed = ctx.copy()
        replayed[2, 2:4] = 0
        readout.audit_tensors(qkv, ctx, changed, replayed, ctx.copy(),
                             head=1, head_dim=2, query=1, source=2, scale=0)
        replayed[1, 2:4] = 0
        spliced = ctx.copy()
        spliced[1, 2:4] = 0
        with self.assertRaisesRegex(ValueError, 'future source'):
            readout.audit_tensors(qkv, ctx, changed, replayed, spliced,
                                 head=1, head_dim=2, query=1, source=2, scale=0)

    def test_tensor_rejects_invalid_head_dimension_without_division_error(self):
        for head_dim in (0, -1, True):
            with self.assertRaisesRegex(ValueError, 'invalid tensor selection'):
                readout.audit_tensors(*self.tensors(), head=1, head_dim=head_dim,
                                     query=2, source=1, scale=0.5)


class EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.directory = cls.root / 'native'
        cls.directory.mkdir()
        cls.checkpoint = cls.root / 'step_13030'
        cls.checkpoint.mkdir()
        # Correct physical byte sizes but zero synthetic weights. No native
        # invocation is claimed; the fixture exercises file/provenance logic.
        for spec in tensor_manifest(GPT2Config()):
            with (cls.checkpoint / spec.filename).open('wb') as stream:
                stream.truncate(spec.nbytes)
        cls.binary = cls.root / 'synthetic_binary'
        cls.binary.write_bytes(b'CPU test fixture, not a real executable')
        cls.log = cls.root / 'native.log'
        cls.log.write_text('synthetic successful execution fixture\n')
        prefix = np.arange(1024, dtype='<i4')
        baseline = np.linspace(-5, 5, 50272, dtype='<f4')
        baseline[2797] = 5.5
        baseline[50257:] = 100  # Padding must NOT enter the logical softmax.
        qkv = np.full((1024, 1536), 0x3f80, dtype='<u2')
        ctx = np.full((1024, 512), 0x3e80, dtype='<u2')
        for name, values in (('prefix.i32', prefix), ('baseline.f32', baseline),
                             ('qkv.bf16', qkv), ('context.bf16', ctx)):
            values.tofile(cls.root / name)
        cls.assay = {**readout.GEOMETRY, 'block': 1, 'head': 2, 'query': 1023, 'target_id': 2797,
            'sources': [1022, 889, 1023, 1021, 512], 'historical_checkpoint': str(cls.checkpoint),
            **{key: readout.causal.record(cls.root / name) for key, name in
               (('prefix', 'prefix.i32'), ('expected_logits', 'baseline.f32'),
                ('expected_qkv', 'qkv.bf16'), ('expected_context', 'context.bf16'))}}
        for name, values in (('tokens.i32', prefix), ('padded_tokens.i32', prefix),
                             ('baseline.f32', baseline), ('original_qkv.bf16', qkv),
                             ('original_context.bf16', ctx)):
            values.tofile(cls.directory / name)
        base_score = readout.score_logits(baseline[:50257], 2797)
        metadata = {'format': 'pluto-source-value-probe-v1', 'complete': True, 'scope': readout.SCOPE,
            'checkpoint': str(cls.checkpoint), 'tokens_file': cls.assay['prefix']['path'],
            'binary': str(cls.binary), 'expected_logits': cls.assay['expected_logits']['path'],
            'prefix_rows': 1024, 'execution_rows': 1024, 'query_token_id': 1023,
            **{key: cls.assay[key] for key in ('block', 'head', 'query', 'target_id',
                                              'vocabulary', 'padded_vocabulary')},
            **{key: True for key in ('native_full_attention_identity', 'native_all_row_padded_logit_identity',
                'weight_disk_and_device_bytes_unchanged', 'original_full_forward_replay_equal')},
            **{key: base_score[value] for key, value in {'baseline_nll': 'nll',
                'baseline_probability': 'probability', 'baseline_rank': 'target_rank',
                'baseline_argmax': 'argmax', 'fixed_rival_id': 'fixed_rival_id',
                'baseline_fixed_rival_margin': 'fixed_rival_margin'}.items()}, 'arms': []}
        for source in cls.assay['sources']:
            for label, scale in readout.DOSES.items():
                name = f'source_{source}_{label}'
                path = cls.directory / name
                path.mkdir()
                edited, replayed, spliced, logits = qkv.copy(), ctx.copy(), ctx.copy(), baseline.copy()
                edited[source, 1152:1216] = readout.scale_bf16(qkv[source, 1152:1216], scale)
                if scale != 1:
                    replayed[1023, 128:192] = 0x3e00 if scale == 0.5 else 0
                    spliced[1023, 128:192] = replayed[1023, 128:192]
                    logits[2797] -= (1 - scale) * 0.5
                for file, values in (('qkv.bf16', edited), ('replayed_attention.bf16', replayed),
                                     ('spliced_context.bf16', spliced), ('logits.f32', logits)):
                    values.tofile(path / file)
                score = readout.score_logits(logits[:50257], 2797, base_score['fixed_rival_id'])
                metadata['arms'].append({'directory': name, 'source': source, 'source_token_id': source,
                                        'scale': scale, **{k: v for k, v in score.items() if k != 'fixed_rival_id'}})
        (cls.directory / 'metadata.json').write_text(json.dumps(metadata))
        flags = {'checkpoint': str(cls.checkpoint), 'tokens_file': cls.assay['prefix']['path'],
            'output_dir': str(cls.directory), 'block': '1', 'head': '2', 'query': '1023',
            'sources': ','.join(map(str, cls.assay['sources'])), 'target_id': '2797',
            'expected_logits': cls.assay['expected_logits']['path']}
        inputs = [readout.causal.record(path) for path in sorted(cls.checkpoint.iterdir())]
        inputs += [readout.causal.record(cls.binary)]
        inputs += [cls.assay[name] for name in ('prefix', 'expected_logits', 'expected_qkv', 'expected_context')]
        cls.execution = {'format': 'pluto-source-value-execution-v1', 'returncode': 0, 'pid': 12345,
            'started_utc': '2026-09-10T00:00:00+00:00', 'finished_utc': '2026-09-10T00:01:00+00:00',
            'command': [str(cls.binary)] + [f'--{key}={value}' for key, value in flags.items()],
            'inputs_before': inputs, 'inputs_after': copy.deepcopy(inputs),
            'outputs': [readout.causal.record(path) for path in sorted(cls.directory.rglob('*')) if path.is_file()],
            'log': readout.causal.record(cls.log)}
        cls.execution_path = cls.root / 'execution.json'
        cls.execution_path.write_text(json.dumps(cls.execution))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def summarize(self, assay=None):
        return readout.summarize(self.directory, self.execution_path, self.assay if assay is None else assay)

    @contextmanager
    def changed(self, path, replacement, *, rehash=True):
        original = path.read_bytes()
        execution_original = self.execution_path.read_bytes()
        try:
            path.write_bytes(replacement)
            if rehash:
                execution = readout._json(self.execution_path)
                for key in ('inputs_before', 'inputs_after', 'outputs'):
                    execution[key] = [readout.causal.record(path) if item['path'] == str(path) else item
                                      for item in execution[key]]
                self.execution_path.write_text(json.dumps(execution))
            yield
        finally:
            path.write_bytes(original)
            self.execution_path.write_bytes(execution_original)

    def mutate_json(self, path, edit):
        value = readout._json(path)
        edit(value)
        return self.changed(path, json.dumps(value).encode(), rehash=path != self.execution_path)

    def test_full_public_readout_accepts_twenty_arms_and_masks_padding(self):
        result = self.summarize()
        self.assertTrue(result['complete'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertFalse(result['new_paired_model_result'])
        self.assertEqual(len(result['arms']), 20)
        self.assertEqual(result['baseline']['argmax'], 2797)
        self.assertEqual(result['baseline']['fixed_rival_id'], 50256)
        self.assertEqual(len(result['outputs']), 86)
        for arm in result['arms']:
            if arm['scale'] == 1:
                self.assertEqual(arm['nll_change'], 0)
            else:
                self.assertGreater(arm['nll_change'], 0)

    def test_public_rejects_changed_historical_bytes_even_with_execution_rehashed(self):
        path = self.root / 'baseline.f32'
        values = np.fromfile(path, dtype='<f4')
        values[0] += 1
        with self.changed(path, values.tobytes()), self.assertRaisesRegex(ValueError, 'frozen evidence changed'):
            self.summarize()

    def test_public_rejects_native_baseline_or_prefix_disagreement_with_historical_evidence(self):
        for file, dtype in (('baseline.f32', '<f4'), ('original_qkv.bf16', '<u2'),
                            ('original_context.bf16', '<u2'), ('tokens.i32', '<i4'),
                            ('padded_tokens.i32', '<i4')):
            path = self.directory / file
            values = np.fromfile(path, dtype=dtype)
            values[0] += 1
            with self.subTest(file=file), self.changed(path, values.tobytes()):
                with self.assertRaisesRegex(ValueError, 'historical|prefix'):
                    self.summarize()

    def test_public_recomputes_all_vocabulary_logits_not_just_target_and_rival(self):
        path = self.directory / 'source_1022_half' / 'logits.f32'
        values = np.fromfile(path, dtype='<f4')
        values[0] = 4  # Below target/rival; changes denominator but no reported raw logit.
        with self.changed(path, values.tobytes()), self.assertRaisesRegex(ValueError, 'declared score differs: nll'):
            self.summarize()

    def test_public_rejects_metadata_score_site_identity_and_arm_inventory(self):
        changes = [lambda x: x.__setitem__('baseline_probability', 0),
                   lambda x: x.__setitem__('fixed_rival_id', 0),
                   lambda x: x.__setitem__('native_full_attention_identity', False),
                   lambda x: x['arms'].pop(),
                   lambda x: x['arms'][0].__setitem__('source', 888),
                   lambda x: x['arms'][1].__setitem__('nll', 0),
                   lambda x: x['arms'][1].__setitem__('argmax', 0),
                   lambda x: x['arms'][1].__setitem__('target_rank', 0),
                   lambda x: x['arms'][1].__setitem__('fixed_rival_margin', -999)]
        for edit in changes:
            with self.subTest(edit=edit), self.mutate_json(self.directory / 'metadata.json', edit):
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_public_rejects_command_input_returncode_and_output_provenance_corruptions(self):
        changes = [lambda x: x.__setitem__('returncode', 1),
                   lambda x: x['command'].__setitem__(4, '--block=2'),
                   lambda x: x['command'].append('--block=1'),
                   lambda x: x['inputs_after'].pop(),
                   lambda x: (x['inputs_before'].pop(0), x['inputs_after'].pop(0)),
                   lambda x: x['outputs'].pop(),
                   lambda x: x['outputs'].append(x['outputs'][0]),
                   lambda x: x.__setitem__('finished_utc', '2026-09-09T00:00:00+00:00')]
        for edit in changes:
            with self.subTest(edit=edit), self.mutate_json(self.execution_path, edit):
                with self.assertRaises(ValueError):
                    self.summarize()

    def test_public_rejects_rehashed_tensor_corruptions_not_merely_hash_mismatches(self):
        changes = [('source_1022_half/qkv.bf16', '<u2', 0, 0x3f81, 'exact source-V dose'),
                   ('source_1022_half/qkv.bf16', '<u2', 1022 * 1536 + 1152, 0x3f01, 'exact source-V dose'),
                   ('source_1022_half/spliced_context.bf16', '<u2', 0, 0x3e81, 'context splice'),
                   ('source_1022_half/spliced_context.bf16', '<u2', 1023 * 512 + 128, 0x3e01, 'context splice'),
                   ('source_1022_one/replayed_attention.bf16', '<u2', 0, 0x3e81, 'identity attention'),
                   ('source_1022_one_after/logits.f32', '<f4', 0, -4, 'identity full selected logits'),
                   ('source_1022_half/logits.f32', '<f4', 50260, 0, 'padded logit bytes'),
                   ('source_1022_half/replayed_attention.bf16', '<u2', 0, 0x7f80, 'nonfinite'),
                   ('source_1022_half/logits.f32', '<f4', 0, float('nan'), 'nonfinite')]
        for file, dtype, index, value, message in changes:
            path = self.directory / file
            array = np.fromfile(path, dtype=dtype)
            array[index] = value
            with self.subTest(file=file, index=index), self.changed(path, array.tobytes()):
                with self.assertRaisesRegex(ValueError, message):
                    self.summarize()

    def test_public_rejects_extra_file_extra_directory_and_short_tensor(self):
        for name, is_dir in (('extra.bin', False), ('extra', True)):
            path = self.directory / name
            if is_dir:
                path.mkdir()
            else:
                path.write_bytes(b'unexpected')
            try:
                with self.assertRaisesRegex(ValueError, 'extra or missing'):
                    self.summarize()
            finally:
                path.rmdir() if is_dir else path.unlink()
        path = self.directory / 'source_1022_half' / 'qkv.bf16'
        with self.changed(path, b'\x00\x00'), self.assertRaisesRegex(ValueError, 'tensor byte count'):
            self.summarize()

    def test_public_rejects_unrecorded_symlink_and_fixed_geometry_changes(self):
        link = self.directory / 'extra_link'
        link.symlink_to(self.directory / 'baseline.f32')
        try:
            with self.assertRaisesRegex(ValueError, 'symlink'):
                self.summarize()
        finally:
            link.unlink()
        for key, value in (('context_length', 8), ('width', 8), ('sources', [1, 2, 3, 4, 4]),
                           ('head', True), ('query', 1024)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summarize({**self.assay, key: value})


if __name__ == '__main__':
    unittest.main()
