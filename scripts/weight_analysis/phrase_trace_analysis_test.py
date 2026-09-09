"""Independent, synthetic-only checks of native-trace analysis and accounting."""

import copy
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from . import phrase_reference
from . import phrase_trace_analysis as analysis


def independently_rounded_bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
    high, low = bits >> 16, bits & 65535
    if low > 32768 or (low == 32768 and high % 2):
        high += 1
    return struct.unpack('<f', struct.pack('<I', high << 16))[0]


class PhraseTraceAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def checkpoint_fixture(self):
        config = GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=5,
                            n_layers=2, d_model=4, n_heads=2, d_ff=6)
        rng = np.random.default_rng(941)
        for spec in tensor_manifest(config):
            value = rng.normal(0, .25, spec.shape).astype('<f4')
            if spec.name.endswith('.scale'):
                value += 1
            value.tofile(self.directory / spec.filename)
        checkpoint = GPT2Checkpoint(self.directory, config, check_finite=True)
        result = phrase_reference.forward(checkpoint, [1, 5, 2, 6], 'bf16')
        return checkpoint, result

    def native_fixture(self, checkpoint=None, result=None):
        """Fabricated tiny files for validation, not claimed native execution."""
        if checkpoint is None:
            checkpoint, result = self.checkpoint_fixture()
        config = checkpoint.config
        native = self.directory/'native'
        native.mkdir()
        rows = len(result['tokens'])
        metadata = {'context_length': config.context_length,
                    'vocab_size': config.vocab_size,
                    'padded_vocab_size': config.padded_vocab_size,
                    'token_ids': result['tokens'].tolist(), 'prompt_rows': rows,
                    'files': {}, 'lens': {}, 'interventions': [],
                    'parity_files': {'alternate_padding': 'alternate_padding.logits.f32',
                                     'clean_replay': 'clean_replay.logits.f32'}}
        for name, values in result['stages'].items():
            path = native/(name+'.bf16')
            (values.view(np.uint32) >> 16).astype('<u2').tofile(path)
            metadata['files'][name] = {'file': path.name, 'dtype': 'bf16',
                                       'shape': list(values.shape)}
        logits = np.full((rows, config.padded_vocab_size), -np.finfo(np.float32).max,
                         dtype='<f4')
        logits[:, :config.vocab_size] = result['logits']
        logits.tofile(native/'logits.f32')
        metadata['files']['logits'] = {'file': 'logits.f32', 'dtype': 'float32',
                                        'shape': list(logits.shape)}
        result['tokens'].astype('<i4').tofile(native/'tokens.i32')
        metadata['files']['tokens'] = {'file': 'tokens.i32', 'dtype': 'int32',
                                       'shape': [rows, 1]}
        for name in analysis.residual_stage_names(config.n_layers):
            filename = 'lens.'+name+'.f32'
            # These equal-valued synthetic readouts exercise schema only.
            logits.tofile(native/filename)
            metadata['lens'][name] = {'file': filename, 'dtype': 'float32',
                                       'shape': list(logits.shape)}
        for filename in metadata['parity_files'].values():
            logits.tofile(native/filename)
        return native, metadata, checkpoint, result

    def add_synthetic_arms(self, native, metadata, config, *, standard=True, neurons=()):
        specs = []
        if standard:
            for block in range(config.n_layers):
                specs.extend([(kind, block, None) for kind in ('attention_branch', 'mlp_branch')])
                specs.extend(('attention_head', block, head) for head in range(config.n_heads))
        specs.extend(('mlp_neuron', block, neuron) for block, neuron in neurons)
        for kind, block, element in specs:
            suffix = (kind if element is None else
                      ('head' if kind == 'attention_head' else 'neuron') + str(element))
            name = f'ablation.block{block}.{suffix}'
            arm = {'name': name, 'kind': kind, 'block': block, 'scale': 0,
                   'logits_file': name+'.logits.f32',
                   'shape': metadata['files']['logits']['shape'],
                   'restoration_verified_bytes': True}
            if element is not None:
                arm['head_or_neuron'] = element
            (native/arm['logits_file']).write_bytes((native/'logits.f32').read_bytes())
            metadata['interventions'].append(arm)

    def test_full_reference_trace_closure_and_component_budgets(self):
        checkpoint, result = self.checkpoint_fixture()
        accounting = analysis.margin_accounting(checkpoint, result['stages'],
                                                result['logits'])
        total = np.sum(list(accounting['terms'].values()), axis=0)
        logits = result['logits'].astype(np.float64)
        ordered = np.argsort(-logits, axis=1, kind='stable')
        expected_margin = np.array([row[order[0]]-row[order[1]]
                                    for row, order in zip(logits, ordered)])
        np.testing.assert_allclose(total, expected_margin, atol=1e-12)
        self.assertLess(accounting['max_closure_error'], 1e-12)
        direction = accounting['direction']
        for row in range(4):
            embedding = checkpoint['token_embedding.weight']
            readout = [independently_rounded_bf16(a)-independently_rounded_bf16(b)
                       for a, b in zip(embedding[ordered[row, 0]], embedding[ordered[row, 1]])]
            gamma = checkpoint['final_norm.scale']
            residual = result['stages']['blocks.1.after_mlp'][row]
            mean = math.fsum(float(x) for x in residual)/4
            std = math.sqrt(math.fsum((float(x)-mean)**2 for x in residual)/4 + 1e-5)
            raw = [readout[i]*float(gamma[i])/std for i in range(4)]
            expected = [value-math.fsum(raw)/4 for value in raw]
            np.testing.assert_allclose(direction[row], expected, atol=1e-14)
        for block in range(checkpoint.config.n_layers):
            key = f'blocks.{block}'
            np.testing.assert_allclose(accounting['head_terms'][key].sum(axis=1),
                                       accounting['terms'][key+'.attention_heads'],
                                       atol=1e-14)
            np.testing.assert_allclose(accounting['neuron_terms'][key].sum(axis=1),
                                       accounting['terms'][key+'.mlp_neurons'],
                                       atol=1e-14)
            diagnostic = accounting['attention'][key]
            np.testing.assert_array_equal(np.triu(diagnostic['probabilities'], 1), 0)
            np.testing.assert_array_equal(np.triu(diagnostic['source_margin_terms'], 1), 0)
            source_error = (diagnostic['source_margin_terms'].sum(axis=2).T -
                            accounting['head_terms'][key])
            self.assertEqual(float(np.max(np.abs(source_error))),
                             diagnostic['source_sum_vs_native_head_max_abs'])
            # Scalar independently rounded weights: forgetting the BF16 cast
            # can still make the accounting total close through remainders,
            # but it must NOT silently alter the stated per-head/neuron terms.
            context = result['stages'][key+'.attention']
            output_weight = checkpoint[key+'.attn.output.weight']
            mlp_weight = checkpoint[key+'.mlp.output.weight']
            features = result['stages'][key+'.gelu']
            for row in range(4):
                for head in range(2):
                    expected = math.fsum(float(context[row, channel]) *
                        independently_rounded_bf16(output_weight[channel, out]) *
                        float(direction[row, out]) for channel in range(2*head, 2*head+2)
                        for out in range(4))
                    self.assertAlmostEqual(accounting['head_terms'][key][row, head], expected, places=13)
                for neuron in range(6):
                    expected = float(features[row, neuron]) * math.fsum(
                        independently_rounded_bf16(mlp_weight[neuron, out]) *
                        float(direction[row, out]) for out in range(4))
                    self.assertAlmostEqual(accounting['neuron_terms'][key][row, neuron], expected, places=13)

    def test_independent_hand_constructed_linear_accounting(self):
        # This is not a model execution. It is an independently constructed
        # residual ledger, with nonzero biases and explicit numerical errors,
        # checking every term instead of merely checking their total sum.
        config = GPT2Config(vocab_size=3, padded_vocab_size=4, context_length=2,
                            n_layers=1, d_model=2, n_heads=1, d_ff=2)
        tensors = {spec.name: np.zeros(spec.shape, dtype=np.float32)
                   for spec in tensor_manifest(config)}
        tensors['token_embedding.weight'][:3] = [[1, 0], [0, 1], [-1, 0]]
        tensors['position_embedding.weight'][:] = [[.25, .5], [.5, -.25]]
        tensors['blocks.0.attn.output.weight'][:] = [[1, .5], [-.25, 2]]
        tensors['blocks.0.attn.output.bias'][:] = [.25, -.5]
        tensors['blocks.0.mlp.output.weight'][:] = [[.5, 1], [1, -.5]]
        tensors['blocks.0.mlp.output.bias'][:] = [-.25, .125]
        tensors['final_norm.scale'][:] = [1.25, .75]
        tensors['final_norm.bias'][:] = [.125, -.25]
        checkpoint = SimpleNamespace(config=config, tensors=tensors)
        base = np.array([[.5, -.25], [1, .5]])
        positions = tensors['position_embedding.weight'].astype(float)
        position_error = np.array([[.125, -.125], [-.0625, .125]])
        positioned = base + positions + position_error
        qkv = np.array([[1, 0, 0, 1, 1, 2], [1, 0, 1, 0, 3, 4]], dtype=float)
        p = math.exp(1/math.sqrt(2))/(1+math.exp(1/math.sqrt(2)))
        context = np.array([[1, 2], [1+2*p, 2+2*p]])
        attn_weight = tensors['blocks.0.attn.output.weight'].astype(float)
        attn_bias = tensors['blocks.0.attn.output.bias'].astype(float)
        attn_projection_error = np.array([[.01, -.02], [.03, -.04]])
        projected = context @ attn_weight + attn_bias + attn_projection_error
        attn_residual_error = np.array([[.002, .001], [-.003, .005]])
        after_attention = positioned + projected + attn_residual_error
        features = np.array([[.5, -.125], [1, .25]])
        mlp_weight = tensors['blocks.0.mlp.output.weight'].astype(float)
        mlp_bias = tensors['blocks.0.mlp.output.bias'].astype(float)
        mlp_projection_error = np.array([[-.01, .02], [.025, -.035]])
        mlp_projected = features @ mlp_weight + mlp_bias + mlp_projection_error
        mlp_residual_error = np.array([[.0002, -.001], [.003, -.002]])
        final = after_attention + mlp_projected + mlp_residual_error
        gamma = tensors['final_norm.scale'].astype(float)
        beta = tensors['final_norm.bias'].astype(float)
        ideal = []
        scales = []
        for row in final:
            mean = math.fsum(row)/2
            scale = math.sqrt(math.fsum((float(x)-mean)**2 for x in row)/2 + 1e-5)
            scales.append(scale)
            ideal.append([(float(x)-mean)/scale*g+b for x, g, b in zip(row, gamma, beta)])
        norm_error = np.array([[.0003, -.0002], [.0005, .0001]])
        normalized = np.array(ideal) + norm_error
        output_error = np.array([[.0001, -.0002, .0003], [.0004, -.0002, -.0001]])
        logits = normalized @ tensors['token_embedding.weight'][:3].T + output_error
        stages = dict(embedding=base, positioned=positioned, final_norm=normalized)
        stages.update({'blocks.0.'+name: value for name, value in {
            'ln1': np.zeros((2, 2)), 'qkv': qkv, 'attention': context,
            'attention_projected': projected, 'after_attention': after_attention,
            'ln2': np.zeros((2, 2)), 'fc1': np.zeros((2, 2)), 'gelu': features,
            'mlp_projected': mlp_projected, 'after_mlp': final}.items()})
        actual = analysis.margin_accounting(checkpoint, stages, logits)
        rows = range(2)
        readouts, directions = [], []
        for row in rows:
            order = sorted(range(3), key=lambda token: (-logits[row, token], token))
            readout = (tensors['token_embedding.weight'][order[0]] -
                       tensors['token_embedding.weight'][order[1]]).astype(float)
            raw = [float(readout[i])*float(gamma[i])/scales[row] for i in range(2)]
            mean = math.fsum(raw)/2
            readouts.append(readout)
            directions.append([value-mean for value in raw])
        directions = np.array(directions)
        np.testing.assert_allclose(actual['direction'], directions, atol=1e-14)

        def project(values):
            values = np.broadcast_to(values, (2, 2))
            return np.array([math.fsum(float(values[r, i])*float(directions[r, i])
                                      for i in range(2)) for r in rows])

        expected = {'token_embedding': project(base), 'position_embedding': project(positions),
                    'position_add_rounding': project(position_error),
                    'blocks.0.attention_heads': project(context @ attn_weight),
                    'blocks.0.attention_bias': project(attn_bias),
                    'blocks.0.attention_projection_rounding': project(attn_projection_error),
                    'blocks.0.attention_residual_rounding': project(attn_residual_error),
                    'blocks.0.mlp_neurons': project(features @ mlp_weight),
                    'blocks.0.mlp_bias': project(mlp_bias),
                    'blocks.0.mlp_projection_rounding': project(mlp_projection_error),
                    'blocks.0.mlp_residual_rounding': project(mlp_residual_error),
                    'final_norm_bias': [math.fsum(float(beta[i])*float(readouts[r][i])
                                                 for i in range(2)) for r in rows],
                    'final_norm_rounding_and_reduction': [math.fsum(
                        float(norm_error[r, i])*float(readouts[r][i]) for i in range(2)) for r in rows],
                    'head_fp32_accumulation_remainder': [
                        output_error[r, actual['winners'][r]] - output_error[r, actual['runners'][r]]
                        for r in rows]}
        self.assertEqual(set(actual['terms']), set(expected))
        for name, values in expected.items():
            np.testing.assert_allclose(actual['terms'][name], values, atol=1e-13, rtol=1e-11,
                                       err_msg=name)
        # An individual neuron's term is activation × output-row projection.
        for neuron in range(2):
            np.testing.assert_allclose(actual['neuron_terms']['blocks.0'][:, neuron],
                                       features[:, neuron] * project(mlp_weight[neuron]), atol=1e-14)
        self.assertLess(actual['attention']['blocks.0']['source_sum_vs_native_head_max_abs'], 1e-13)

    def test_qkv_independent_scalar_probabilities_values_and_future_mask(self):
        qkv = np.array([[1, 0, 0, 1, 1, 2], [1, 0, 1, 0, 3, 4],
                        [0, 1, 2, 1, -1, 5]], dtype=float)
        probabilities, values, context = analysis.reconstructed_attention(qkv, 1)
        p = math.exp(1/math.sqrt(2))/(1+math.exp(1/math.sqrt(2)))
        np.testing.assert_allclose(probabilities[0, 1], [1-p, p, 0], atol=1e-15)
        np.testing.assert_array_equal(values[0], [[1, 2], [3, 4], [-1, 5]])
        np.testing.assert_allclose(context[1], [1+2*p, 2+2*p], atol=1e-15)
        changed = qkv.copy()
        changed[2] = [-10, 10, 100, -100, 1000, 2000]
        changed_probs, _, changed_context = analysis.reconstructed_attention(changed, 1)
        np.testing.assert_array_equal(probabilities[:, :2], changed_probs[:, :2])
        np.testing.assert_array_equal(context[:2], changed_context[:2])
        np.testing.assert_array_equal(np.triu(probabilities, 1), 0)
        for heads in [0, -1, True, 3]:
            with self.assertRaises(ValueError):
                analysis.reconstructed_attention(qkv, heads)
        with self.assertRaises(ValueError):
            analysis.reconstructed_attention(np.ones((2, 5)), 1)

    def test_probability_rank_ties_byte_display_and_teacher_forcing(self):
        logits = np.array([[1000, 1000, 999], [-1000, -999, -998]], dtype=np.float32)
        pieces = {0: b'to', 1: b' be', 2: b'\xff'}
        result = analysis.predictions(logits, pieces, [2, 0], [0, 1])
        self.assertEqual(result[0]['winner']['id'], 0)
        self.assertEqual(result[0]['runner_up']['id'], 1)
        self.assertEqual(result[0]['final_winner']['rank'], 3)
        self.assertEqual(result[1]['final_winner']['rank'], 3)
        self.assertEqual(result[0]['observed_next_input']['id'], 1)
        self.assertNotIn('observed_next_input', result[1])
        self.assertEqual(result[1]['winner']['bytes_hex'], 'ff')
        self.assertIn('\\xff', result[1]['winner']['piece_escaped'])
        expected = 1/(2+math.exp(-1))
        self.assertAlmostEqual(result[0]['winner']['probability'], expected, places=14)
        self.assertAlmostEqual(sum(row['probability'] for row in result[0]['top5']), 1)
        expected_entropy = -math.fsum(p*math.log(p) for p in [expected, expected, expected/math.e])
        self.assertAlmostEqual(result[0]['entropy_nats'], expected_entropy, places=13)

    def test_prediction_validation_rejects_bad_coverage_and_token_ids(self):
        logits = np.zeros((2, 3), dtype=np.float32)
        pieces = {0: b'a', 1: b'b', 2: b'c'}
        for bad in [{0: b'a'}, {0: b'a', 1: b'b', 3: b'c'},
                    {0: b'a', 1: b'b', 2: 'c'}]:
            with self.assertRaises(ValueError):
                analysis.predictions(logits, bad)
        with self.assertRaises(ValueError):
            analysis.predictions(np.ones((2, 1)), {0: b'a'})
        for bad in [[-1, 0], [0, 3], [True, False], [0.0, 1.0], [0], [[0, 1]],
                    np.array([0, 2**64-1], dtype=np.uint64)]:
            for name in ['fixed_winners', 'next_inputs']:
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    analysis.predictions(logits, pieces, **{name: bad})

    def test_emergence_is_first_vs_persistent_readout_match(self):
        lens = {name: [{'winner': {'id': token}}] for name, token in
                [('positioned', 1), ('a', 2), ('b', 1), ('c', 1)]}
        result = analysis.emergence(lens, [1])[0]
        self.assertEqual(result['first_readout_match'], 'positioned')
        self.assertEqual(result['first_persistent_readout_match'], 'b')
        self.assertEqual(result['matching_stage_count'], 3)
        self.assertIn('heuristic', result['warning'])
        none = analysis.emergence(lens, [9])[0]
        self.assertIsNone(none['first_readout_match'])
        self.assertIsNone(none['first_persistent_readout_match'])
        self.assertEqual(analysis.residual_stage_names(1),
                         ['positioned', 'blocks.0.after_attention', 'blocks.0.after_mlp'])

    def test_native_bf16_float32_and_int32_loaders(self):
        records = [
            ('b.bin', 'bf16', np.array([[0x3f80, 0x8000], [0xbf80, 0x0001]], dtype='<u2'),
             np.array([[1, -0.], [-1, 2**-133]], dtype=np.float32)),
            ('f.bin', 'float32', np.array([[1.25, -3], [4, 5]], dtype='<f4'),
             np.array([[1.25, -3], [4, 5]], dtype=np.float32)),
            ('i.bin', 'int32', np.array([[1, 2], [-3, 4]], dtype='<i4'),
             np.array([[1, 2], [-3, 4]], dtype=np.int32))]
        for name, dtype, stored, expected in records:
            stored.tofile(self.directory/name)
            result, identity = analysis.load_native_array(self.directory,
                {'file': name, 'dtype': dtype, 'shape': [2, 2]})
            np.testing.assert_array_equal(result, expected)
            if dtype == 'bf16':
                self.assertTrue(np.signbit(result[0, 1]))
            self.assertEqual(identity['bytes'], stored.nbytes)

    def test_native_loader_rejects_path_dtype_shape_and_truncation(self):
        data = np.array([0x3f80, 0xbf80], dtype='<u2')
        data.tofile(self.directory/'valid.bin')
        valid = {'file': 'valid.bin', 'dtype': 'bf16', 'shape': [1, 2]}
        invalid = [dict(valid, file='../valid.bin'), dict(valid, file='/tmp/valid.bin'),
                   dict(valid, file='nested/valid.bin'), dict(valid, file=''),
                   dict(valid, dtype='float16'), dict(valid, shape=[2, 2]),
                   dict(valid, shape=[True, 2]), dict(valid, shape=[1, -2]),
                   dict(valid, shape=[1, 2.0]), dict(valid, shape=[2]),
                   dict(valid, shape=[0, 2])]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                analysis.load_native_array(self.directory, record)
        (self.directory/'link.bin').symlink_to(self.directory/'valid.bin')
        with self.assertRaises(ValueError):
            analysis.load_native_array(self.directory, dict(valid, file='link.bin'))
        # Half of one BF16 value is never accepted as a smaller/truncated dump.
        (self.directory/'short.bin').write_bytes(b'\x80')
        with self.assertRaises(ValueError):
            analysis.load_native_array(self.directory, dict(valid, file='short.bin', shape=[1, 1]))
        for name, value in [('nan.bin', 0x7fc0), ('inf.bin', 0x7f80)]:
            np.array([value], dtype='<u2').tofile(self.directory/name)
            with self.assertRaises(ValueError):
                analysis.load_native_array(self.directory, dict(valid, file=name, shape=[1, 1]))

    def test_shape_validation_and_nonfinite_inputs(self):
        for bad in [[], [1, 2], [[np.inf]], [[np.nan]], np.empty((2, 0))]:
            with self.assertRaises(ValueError):
                analysis.checked_matrix(bad)
        with self.assertRaises(ValueError):
            analysis.compare_arrays(np.zeros((2, 2)), np.zeros((2, 3)))
        identical = analysis.compare_arrays(np.zeros((2, 2)), np.zeros((2, 2)))
        self.assertTrue(identical['values_equal'])
        self.assertIsNone(identical['relative_l2_error'])
        # Numeric equality deliberately treats signed zeros as equal. This is
        # not a claim that the saved binary representations are byte-identical.
        signed_zero = analysis.compare_arrays(np.array([[0.]], dtype=np.float32),
                                               np.array([[-0.]], dtype=np.float32))
        self.assertTrue(signed_zero['values_equal'])
        self.assertEqual(signed_zero['max_abs_error'], 0)
        known = analysis.compare_arrays(np.ones((2, 2)), np.ones((2, 2))*2)
        self.assertEqual(known['max_abs_error'], 1)
        self.assertEqual(known['rms_error'], 1)
        self.assertEqual(known['relative_l2_error'], 1)

    def test_required_stage_shapes_do_not_broadcast_or_hide_missing_logits(self):
        checkpoint, result = self.checkpoint_fixture()
        stages, logits = result['stages'], result['logits']
        analysis.validate_stages(stages, 4, checkpoint.config)
        for name in stages:
            for replacement in [np.zeros((4, 1)), np.zeros((1, stages[name].shape[1]))]:
                malformed = dict(stages, **{name: replacement})
                with self.subTest(name=name, shape=replacement.shape), self.assertRaises(ValueError):
                    analysis.margin_accounting(checkpoint, malformed, logits)
            missing = dict(stages)
            del missing[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                analysis.margin_accounting(checkpoint, missing, logits)
        for wrong_logits in [logits[:, :-1], np.pad(logits, ((0, 0), (0, 1)))]:
            with self.assertRaises(ValueError):
                analysis.margin_accounting(checkpoint, stages, wrong_logits)

    def test_native_loader_detects_input_identity_change(self):
        path = self.directory/'values.bin'
        np.zeros((1, 2), dtype='<f4').tofile(path)
        original = analysis.file_record(path)
        changed = dict(original, sha256='f'*64)
        with mock.patch.object(analysis, 'file_record', side_effect=[original, changed]):
            with self.assertRaisesRegex(ValueError, 'changed'):
                analysis.load_native_array(self.directory,
                    {'file': path.name, 'dtype': 'float32', 'shape': [1, 2]})

    def test_evidence_hashes_independent_parity_and_tokens(self):
        native, metadata, checkpoint, _ = self.native_fixture()
        result = analysis.load_native_evidence(native, metadata, checkpoint.config)
        self.assertEqual(set(result['parity_checks']), {
            'alternate_padding_logits_byte_equal', 'clean_replay_logits_byte_equal',
            'final_lens_logits_byte_equal'})
        for label in ['parity:alternate_padding', 'parity:clean_replay', 'stage:tokens']:
            record = result['records'][label]
            self.assertEqual(record, analysis.file_record(record['path']))
        np.testing.assert_array_equal(result['arrays']['tokens'][:, 0], metadata['token_ids'])
        self.assertEqual(len(result['records']),
                         len(metadata['files'])+len(metadata['lens'])+2)

    def test_parity_checks_do_not_trust_flags_or_equal_signed_zero_values(self):
        native, metadata, checkpoint, _ = self.native_fixture()
        baseline = np.zeros(metadata['files']['logits']['shape'], dtype='<f4')
        baseline.tofile(native/'logits.f32')
        for record in metadata['lens'].values():
            baseline.tofile(native/record['file'])
        for filename in metadata['parity_files'].values():
            baseline.tofile(native/filename)
        metadata['checks'] = {'alternate_padding_logits_byte_equal': True,
                              'clean_replay_logits_byte_equal': True}
        for target in [metadata['parity_files']['alternate_padding'],
                       metadata['parity_files']['clean_replay'],
                       metadata['lens'][analysis.residual_stage_names(checkpoint.config.n_layers)[-1]]['file']]:
            changed = baseline.copy()
            changed[0, 0] = -0.0
            self.assertTrue(np.array_equal(baseline, changed))
            changed.tofile(native/target)
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'byte-identical'):
                analysis.load_native_evidence(native, metadata, checkpoint.config)
            baseline.tofile(native/target)
        del metadata['parity_files']['clean_replay']
        with self.assertRaisesRegex(ValueError, 'parity files'):
            analysis.load_native_evidence(native, metadata, checkpoint.config)

    def test_evidence_rejects_token_mismatch_wrong_dtypes_dimensions_and_aliases(self):
        native, metadata, checkpoint, _ = self.native_fixture()
        for key in ['vocab_size', 'padded_vocab_size', 'context_length', 'prompt_rows']:
            for value in [metadata[key]+1, True, float(metadata[key])]:
                changed = dict(metadata, **{key: value})
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    analysis.load_native_evidence(native, changed, checkpoint.config)
        for name, dtype in [('tokens', 'float32'), ('embedding', 'float32'),
                             ('logits', 'bf16')]:
            changed = copy.deepcopy(metadata)
            changed['files'][name]['dtype'] = dtype
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'dtype'):
                analysis.load_native_evidence(native, changed, checkpoint.config)
        changed = copy.deepcopy(metadata)
        changed['lens']['positioned']['dtype'] = 'bf16'
        with self.assertRaisesRegex(ValueError, 'dtype'):
            analysis.load_native_evidence(native, changed, checkpoint.config)
        changed = copy.deepcopy(metadata)
        changed['files']['tokens']['shape'] = [1, metadata['prompt_rows']]
        with self.assertRaisesRegex(ValueError, 'shape'):
            analysis.load_native_evidence(native, changed, checkpoint.config)
        changed = copy.deepcopy(metadata)
        changed['parity_files']['clean_replay'] = 'logits.f32'
        with self.assertRaisesRegex(ValueError, 'independently named'):
            analysis.load_native_evidence(native, changed, checkpoint.config)
        changed_ids = np.array(metadata['token_ids'], dtype='<i4')
        changed_ids[0] = (changed_ids[0]+1) % checkpoint.config.vocab_size
        changed_ids.tofile(native/'tokens.i32')
        with self.assertRaisesRegex(ValueError, 'token dump'):
            analysis.load_native_evidence(native, metadata, checkpoint.config)

    def test_complete_standard_sweep_and_arbitrary_unique_neuron_subset(self):
        native, metadata, checkpoint, _ = self.native_fixture()
        self.add_synthetic_arms(native, metadata, checkpoint.config,
                                neurons=[(0, 5), (1, 0)])
        result = analysis.load_native_evidence(native, metadata, checkpoint.config)
        self.assertEqual(len(result['interventions']), 10)  # 8 toy standard + 2 neurons.
        self.assertEqual(len([key for key in result['records'] if key.startswith('intervention:')]), 10)
        metadata['interventions'] = [arm for arm in metadata['interventions']
                                      if arm['kind'] == 'mlp_neuron']
        result = analysis.load_native_evidence(native, metadata, checkpoint.config)
        self.assertEqual(len(result['interventions']), 2)

    def test_intervention_validation_rejects_partial_duplicate_or_misleading_arms(self):
        native, metadata, checkpoint, _ = self.native_fixture()
        self.add_synthetic_arms(native, metadata, checkpoint.config, neurons=[(1, 4)])
        arms = metadata['interventions']
        rows = metadata['prompt_rows']
        mutations = [arms[:-2], arms+[copy.deepcopy(arms[0])]]
        for key, value in [('scale', .5), ('scale', False), ('scale', float('nan')),
                           ('restoration_verified_bytes', False), ('restoration_verified_bytes', 1),
                           ('block', True), ('block', 2), ('block', -1), ('kind', 'invented'),
                           ('name', 'misleading'), ('head_or_neuron', 0),
                           ('shape', [rows, checkpoint.config.vocab_size]),
                           ('logits_file', 'logits.f32')]:
            changed = copy.deepcopy(arms)
            changed[0][key] = value
            mutations.append(changed)
        for index, value in [(2, -1), (2, checkpoint.config.n_heads),
                              (len(arms)-1, checkpoint.config.d_ff), (len(arms)-1, True)]:
            changed = copy.deepcopy(arms)
            changed[index]['head_or_neuron'] = value
            mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed[0]), self.assertRaises(ValueError):
                analysis.validate_intervention_arms(changed, rows, checkpoint.config)

    def test_small_synthetic_run_retains_parity_records(self):
        native, metadata, checkpoint, _ = self.native_fixture()
        self.add_synthetic_arms(native, metadata, checkpoint.config,
                                standard=False, neurons=[(0, 0)])
        arm_path = native/metadata['interventions'][0]['logits_file']
        arm_values = np.fromfile(arm_path, dtype='<f4').reshape(
            metadata['files']['logits']['shape'])
        baseline = arm_values.copy()
        order = np.argsort(-baseline[:, :checkpoint.config.vocab_size],
                           axis=1, kind='stable')
        for row in range(len(order)):
            arm_values[row, order[row, 0]] = np.float32(1)
            arm_values[row, order[row, 1]] = -np.float32(2**-24)
        arm_values.tofile(arm_path)
        tokenizer = self.directory/'tokenizer'
        tokenizer.mkdir()
        (tokenizer/'tokenizer.json').write_text('{}')
        pieces = {token: bytes([65+token]) for token in range(checkpoint.config.vocab_size)}
        metadata.update({'complete': True, 'autoregressive': False, 'no_bos': True,
                         'checkpoint_directory': str(self.directory),
                         'tokenizer_directory': str(tokenizer),
                         'prompt': b''.join(pieces[token] for token in metadata['token_ids']).decode(),
                         'checks': {name: True for name in (
                             'final_lens_logits_byte_equal', 'alternate_padding_logits_byte_equal',
                             'clean_replay_logits_byte_equal', 'attention_residual_replay_byte_equal',
                             'mlp_residual_replay_byte_equal')}})
        (native/'metadata.json').write_text(json.dumps(metadata))
        fake_tokenizers = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: object()))
        with mock.patch.dict('sys.modules', {'tokenizers': fake_tokenizers}), \
                mock.patch.object(analysis, 'GPT2Checkpoint', return_value=checkpoint), \
                mock.patch.object(analysis, 'gpt2_token_bytes', return_value=pieces), \
                mock.patch('builtins.print'):
            report = analysis.run(native, self.directory, tokenizer, self.directory/'analysis')
        self.assertTrue(report['complete'])
        self.assertEqual(report['independently_verified_parity'], {
            'alternate_padding_logits_byte_equal': True, 'clean_replay_logits_byte_equal': True,
            'final_lens_logits_byte_equal': True})
        self.assertIn('parity:clean_replay', report['native_files'])
        self.assertIn('parity:alternate_padding', report['native_files'])
        # Both source logits are exactly representable FP32, but their
        # difference is halfway between adjacent FP32 values. The analysis
        # must subtract the observed logits in FP64, not round this margin.
        expected_margin = np.full(len(order), 1+2**-24, dtype=np.float64)
        np.testing.assert_array_equal(
            report['interventions'][0]['fixed_winner_runner_margin'], expected_margin)
        clean_margin = (baseline[np.arange(len(order)), order[:, 0]].astype(float) -
                        baseline[np.arange(len(order)), order[:, 1]].astype(float))
        np.testing.assert_array_equal(
            report['interventions'][0]['margin_change_from_clean'], expected_margin-clean_margin)
        self.assertTrue((self.directory/'analysis'/'analysis.json').is_file())


if __name__ == '__main__':
    unittest.main()
