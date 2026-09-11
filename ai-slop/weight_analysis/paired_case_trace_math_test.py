"""Independent scalar and tiny CPU-forward tests for paired trace accounting.

These fixtures do not claim parity with a GPU kernel. They exercise the
algebra and validation with explicit values and small temporary checkpoints.
"""

import copy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import paired_case_trace_math as analysis
from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from .phrase_reference import forward
from .phrase_reference_test import scalar_bf16


class PredictionTest(unittest.TestCase):
    def test_probability_margin_and_partition_match_hand_computable_logits(self):
        values = np.log([1., 2., 4., 1.])
        result = analysis.prediction(values, 1, 2)
        self.assertAlmostEqual(result['target_logit'], math.log(2))
        self.assertAlmostEqual(result['competitor_logit'], math.log(4))
        self.assertAlmostEqual(result['logsumexp'], math.log(8))
        self.assertAlmostEqual(result['log_probability'], math.log(.25))
        self.assertAlmostEqual(result['probability'], .25)
        self.assertAlmostEqual(result['margin'], -math.log(2))
        self.assertEqual(result['argmax_id'], 2)

    def test_ties_use_lowest_id_not_target_or_competitor_order(self):
        values = np.array([0., 2., 2., -1.])
        self.assertEqual(analysis.prediction(values, 2, 1)['argmax_id'], 1)
        self.assertEqual(analysis.prediction(values, 1, 2)['argmax_id'], 1)
        self.assertEqual(analysis.prediction(values, 2, 1)['margin'], 0.)

    def test_large_common_offset_keeps_probabilities_and_margins(self):
        values = np.array([0., -1., 2., -3.])
        initial = analysis.prediction(values, 1, 2)
        shifted = analysis.prediction(values+1000., 1, 2)
        for key in ('probability', 'log_probability', 'margin'):
            self.assertAlmostEqual(shifted[key], initial[key], places=12)
        self.assertAlmostEqual(shifted['logsumexp']-initial['logsumexp'], 1000.)

    def test_offset_larger_than_log_normalizer_still_gives_half_probability(self):
        high = float(np.float32(1e20))
        result = analysis.prediction([high, high, 0.], 0, 1)
        self.assertEqual(result['logsumexp'], high)
        self.assertEqual(result['log_probability'], -math.log(2.))
        self.assertEqual(result['probability'], .5)

    def test_finite_logits_with_overflowing_margin_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite prediction statistics'):
            analysis.prediction([1e308, -1e308], 0, 1)

    def test_underflow_retains_finite_log_probability(self):
        result = analysis.prediction(np.array([0., -1000., -2.]), 1, 0)
        self.assertEqual(result['probability'], 0.)
        self.assertTrue(math.isfinite(result['log_probability']))
        self.assertAlmostEqual(result['log_probability'], -1000.-math.log1p(math.exp(-2.)))

    def test_target_logit_rise_can_be_overcome_by_larger_normalizer_rise(self):
        before = analysis.prediction(np.array([0., 0., -1.]), 0, 1)
        after = analysis.prediction(np.array([1., 4., -1.]), 0, 1)
        logit_delta = after['target_logit']-before['target_logit']
        partition_delta = after['logsumexp']-before['logsumexp']
        self.assertEqual(logit_delta, 1.)
        self.assertGreater(partition_delta, logit_delta)
        self.assertLess(after['probability'], before['probability'])
        self.assertAlmostEqual(after['log_probability']-before['log_probability'],
                               logit_delta-partition_delta)

    def test_invalid_logits_and_target_coordinates_are_rejected(self):
        for values in ([], [0.], [[0., 1.]], [0., math.nan], [math.inf, 0.], [-math.inf, 0.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                analysis.prediction(values, 0, 1)
        for target, competitor in ((-1, 1), (0, 3), (1, 1), (True, 2), (0, False), (1., 2)):
            with self.subTest(target=target, competitor=competitor), self.assertRaises(ValueError):
                analysis.prediction([0., 1., 2.], target, competitor)


class NeuronDeltaTest(unittest.TestCase):
    def setUp(self):
        self.ze = np.array([2., -3., .5])
        self.zm = np.array([-1., -2., 4.])
        self.de = np.array([[1., 2.], [-2., 1.], [3., -1.]])
        self.dm = np.array([[2., -1.], [-3., 2.], [1., 4.]])
        self.direction = np.array([.5, -2.])

    def call(self, **changes):
        values = dict(z_e=self.ze, z_m=self.zm, decoder_e=self.de, decoder_m=self.dm, direction=self.direction)
        values.update(changes)
        return analysis.neuron_delta(**values)

    def test_decomposition_matches_independent_scalar_loops(self):
        result = self.call()
        expected = {key: [] for key in ('delta', 'activation', 'decoder', 'interaction')}
        for neuron in range(len(self.ze)):
            qe = math.fsum(float(self.de[neuron, axis])*float(self.direction[axis]) for axis in range(2))
            qm = math.fsum(float(self.dm[neuron, axis])*float(self.direction[axis]) for axis in range(2))
            dz = float(self.zm[neuron]-self.ze[neuron])
            dq = math.fsum(float(self.dm[neuron, axis]-self.de[neuron, axis])*float(self.direction[axis])
                           for axis in range(2))
            expected['delta'].append(float(self.zm[neuron])*qm-float(self.ze[neuron])*qe)
            expected['activation'].append(dz*qe)
            expected['decoder'].append(float(self.ze[neuron])*dq)
            expected['interaction'].append(dz*dq)
        for key, values in expected.items():
            np.testing.assert_array_equal(result[key], values)
        np.testing.assert_array_equal(result['delta'], result['activation']+result['decoder']+result['interaction'])

    def test_cross_term_is_retained_when_first_order_terms_cancel(self):
        result = analysis.neuron_delta([2.], [3.], [[3.]], [[1.5]], [1.])
        np.testing.assert_array_equal(result['activation'], [3.])
        np.testing.assert_array_equal(result['decoder'], [-3.])
        np.testing.assert_array_equal(result['interaction'], [-1.5])
        np.testing.assert_array_equal(result['delta'], [-1.5])

    def test_activation_only_and_decoder_only_changes_have_no_interaction(self):
        activation = self.call(decoder_m=self.de.copy())
        np.testing.assert_array_equal(activation['delta'], activation['activation'])
        np.testing.assert_array_equal(activation['decoder'], np.zeros(3))
        np.testing.assert_array_equal(activation['interaction'], np.zeros(3))
        decoder = self.call(z_m=self.ze.copy())
        np.testing.assert_array_equal(decoder['delta'], decoder['decoder'])
        np.testing.assert_array_equal(decoder['activation'], np.zeros(3))
        np.testing.assert_array_equal(decoder['interaction'], np.zeros(3))

    def test_identical_models_and_zero_direction_have_zero_terms(self):
        for result in (self.call(z_m=self.ze.copy(), decoder_m=self.de.copy()),
                       self.call(direction=np.zeros(2))):
            for key in ('delta', 'activation', 'decoder', 'interaction'):
                np.testing.assert_array_equal(result[key], np.zeros(3))

    def test_swapping_endpoints_reverses_total_not_each_component(self):
        before = self.call()
        swapped = analysis.neuron_delta(self.zm, self.ze, self.dm, self.de, self.direction)
        np.testing.assert_array_equal(swapped['delta'], -before['delta'])
        self.assertFalse(np.array_equal(swapped['activation'], -before['activation']))
        np.testing.assert_array_equal(swapped['interaction'], before['interaction'])

    def test_opposing_neurons_cancel_without_disappearing(self):
        result = analysis.neuron_delta([1., 1.], [2., 2.], [[2.], [-2.]], [[2.], [-2.]], [1.])
        np.testing.assert_array_equal(result['delta'], [2., -2.])
        self.assertEqual(math.fsum(result['delta']), 0.)

    def test_broadcastable_but_wrong_shapes_and_empty_dimensions_are_rejected(self):
        changes = ({'z_e': [[2., -3., .5]]}, {'z_m': [1.]}, {'decoder_e': np.ones((3, 1))},
                   {'decoder_m': np.ones((2, 2))}, {'direction': [[.5, -2.]]}, {'direction': [1.]},
                   {'z_e': [], 'z_m': [], 'decoder_e': np.empty((0, 2)), 'decoder_m': np.empty((0, 2))},
                   {'decoder_e': np.empty((3, 0)), 'decoder_m': np.empty((3, 0)), 'direction': []})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.call(**change)

    def test_nonfinite_inputs_are_rejected_in_every_operand(self):
        originals = dict(z_e=self.ze, z_m=self.zm, decoder_e=self.de, decoder_m=self.dm, direction=self.direction)
        for name, values in originals.items():
            for invalid in (math.nan, math.inf, -math.inf):
                changed = values.copy(); changed.flat[0] = invalid
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    self.call(**{name: changed})

    def test_input_arrays_remain_unchanged(self):
        arrays = [self.ze, self.zm, self.de, self.dm, self.direction]
        saved = [value.copy() for value in arrays]
        self.call()
        for actual, expected in zip(arrays, saved):
            np.testing.assert_array_equal(actual, expected)

    def test_analytical_remainder_records_float_distributivity_error(self):
        result = analysis.neuron_delta([.1, 1e8], [.2, 1e8+1],
            [[.3, .4], [.7, -.2]], [[.9, -.8], [.7+1e-8, -.2]], [.3, -.6])
        for i in range(2):
            expected = math.fsum((float(result['delta'][i]), -float(result['activation'][i]),
                                  -float(result['decoder'][i]), -float(result['interaction'][i])))
            self.assertEqual(result['analytical_remainder'][i], expected)
        self.assertTrue(np.any(result['analytical_remainder'] != 0.))

    def test_extreme_cancellation_never_returns_an_infinite_remainder(self):
        # Every first-order term fits FP64, but summing positives first does not.
        # Either a finite retained remainder or a deliberate rejection is safe.
        try:
            result = analysis.neuron_delta([2.], [1.], [[-7e307]], [[1e307]], [1.])
        except ValueError:
            return
        for values in result.values():
            self.assertTrue(np.isfinite(values).all())

    def test_finite_inputs_with_overflowing_products_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite neuron-delta result'):
            analysis.neuron_delta([0.], [1e308], [[1.]], [[2.]], [2.])


class PairAccountingTest(unittest.TestCase):
    """Real tiny BF16 CPU forwards, plus explicitly synthetic edge-case ledgers.

    Synthetic stage/logit overrides below test diagnostic arithmetic only.
    File and native/reference authentication belong to the separate adapter.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=6,
                                 n_layers=2, d_model=4, n_heads=2, d_ff=6)
        self.tokens, self.target = [1, 5, 2, 6], 0
        rng = np.random.default_rng(771)
        weights = {}
        for spec in tensor_manifest(self.config):
            value = rng.normal(0, .2, spec.shape).astype('<f4')
            if spec.name.endswith('.scale'):
                value += 1
            weights[spec.name] = value
        # Padding is physically present yet is neither a target nor competitor.
        weights['token_embedding.weight'][7] = [0., -0., 200., -200.]
        self.e = self.checkpoint('E', weights)
        changed = {name: value.copy() for name, value in weights.items()}
        changed['blocks.0.ln2.scale'] += np.array([.3, -.1, .2, .4], dtype='<f4')
        changed['blocks.0.ln2.bias'] += np.array([.1, -.2, .05, .3], dtype='<f4')
        changed['blocks.0.mlp.input.weight'] += rng.normal(0, .15, (4, 6)).astype('<f4')
        changed['blocks.0.mlp.input.bias'] += np.linspace(-.2, .3, 6).astype('<f4')
        changed['blocks.0.mlp.output.weight'] += rng.normal(0, .15, (6, 4)).astype('<f4')
        changed['blocks.0.mlp.output.bias'] += np.array([.25, -.5, .75, -.125], dtype='<f4')
        self.m = self.checkpoint('M', changed)
        self.fe = forward(self.e, self.tokens, 'bf16')
        self.fm = forward(self.m, self.tokens, 'bf16')

    def checkpoint(self, name, weights):
        directory = self.root/name
        directory.mkdir()
        for spec in tensor_manifest(self.config):
            weights[spec.name].tofile(directory/spec.filename)
        return GPT2Checkpoint(directory, self.config, check_finite=True)

    def call(self, **changes):
        values = dict(checkpoint_e=self.e, stages_e=self.fe['stages'], logits_e=self.fe['logits'][-1],
                      checkpoint_m=self.m, stages_m=self.fm['stages'], logits_m=self.fm['logits'][-1],
                      token_ids=self.tokens, target=self.target, top_k=3)
        values.update(changes)
        return analysis.analyze_pair(**values)

    @staticmethod
    def changed_tensor(checkpoint, name, value):
        changed = copy.copy(checkpoint)
        changed.tensors = dict(checkpoint.tensors)
        changed.tensors[name] = value
        return changed

    def test_identical_endpoint_has_zero_effects_and_no_selected_features(self):
        result = self.call(checkpoint_m=self.e, stages_m=self.fe['stages'], logits_m=self.fe['logits'][-1])
        self.assertEqual(result['row'], len(self.tokens)-1)
        self.assertEqual(result['token_ids'], self.tokens)
        self.assertEqual(result['target_id'], self.target)
        self.assertEqual(result['predictions']['E'], result['predictions']['M'])
        for value in result['effects'].values():
            self.assertEqual(value, 0.)
        self.assertEqual(set(result['early_neurons']), {'blocks.0'})
        for key in ('delta', 'activation', 'decoder', 'interaction', 'analytical_remainder'):
            np.testing.assert_array_equal(result['early_neurons']['blocks.0'][key], np.zeros(6))
        for key in ('projected_delta', 'bias_delta', 'rounding_delta'):
            self.assertEqual(result['early_neurons']['blocks.0'][key], 0.)
        self.assertEqual(result['early_neurons']['blocks.0']['selected'], [])

    def test_actual_pair_predictions_and_margin_ledgers_close_independently(self):
        result = self.call()
        logits_e, logits_m = self.fe['logits'][-1], self.fm['logits'][-1]
        competitor = min((i for i in range(7) if i != self.target), key=lambda i: (-float(logits_e[i]), i))
        self.assertEqual(result['competitor_id'], competitor)
        for label, logits in (('E', logits_e), ('M', logits_m)):
            maximum = max(float(value) for value in logits)
            logp = float(logits[self.target])-maximum-math.log(math.fsum(math.exp(float(x)-maximum) for x in logits))
            self.assertAlmostEqual(result['predictions'][label]['log_probability'], logp, places=14)
            ledger = result['ledgers'][label]
            margin = float(logits[self.target])-float(logits[competitor])
            self.assertAlmostEqual(math.fsum(ledger['terms'].values()), margin, places=12)
            self.assertAlmostEqual(ledger['native_margin'], margin, places=12)
        effects = result['effects']
        self.assertAlmostEqual(effects['log_probability_delta'],
                               effects['target_logit_delta']-effects['logsumexp_delta'], places=13)
        self.assertAlmostEqual(effects['absolute_probability_closure_remainder'], 0., places=13)
        self.assertAlmostEqual(effects['centered_probability_closure_remainder'], 0., places=13)

    def test_feature_delta_uses_one_independently_computed_m_normalizer_direction(self):
        result = self.call()
        row, other = len(self.tokens)-1, result['competitor_id']
        residual = [float(x) for x in self.fm['stages']['blocks.1.after_mlp'][row]]
        mean = math.fsum(residual)/4
        deviation = math.sqrt(math.fsum((x-mean)**2 for x in residual)/4+1e-5)
        uncentered = [(scalar_bf16(self.m['token_embedding.weight'][self.target, j])-
                       scalar_bf16(self.m['token_embedding.weight'][other, j]))*
                      float(self.m['final_norm.scale'][j])/deviation for j in range(4)]
        center = math.fsum(uncentered)/4
        direction = [value-center for value in uncentered]
        np.testing.assert_allclose(result['fixed_direction'], direction, rtol=1e-14, atol=1e-15)
        np.testing.assert_array_equal(result['fixed_direction'], result['ledgers']['M']['direction'])
        self.assertGreater(np.max(np.abs(result['fixed_direction']-result['ledgers']['E']['direction'])), 1e-4)
        neurons = result['early_neurons']['blocks.0']
        expected = {key: [] for key in ('delta', 'activation', 'decoder', 'interaction')}
        for i in range(6):
            ze, zm = (float(forwarded['stages']['blocks.0.gelu'][row, i]) for forwarded in (self.fe, self.fm))
            de, dm = ([scalar_bf16(cp['blocks.0.mlp.output.weight'][i, j]) for j in range(4)]
                      for cp in (self.e, self.m))
            qe, qm = (math.fsum(weight[j]*direction[j] for j in range(4)) for weight in (de, dm))
            dq = math.fsum((dm[j]-de[j])*direction[j] for j in range(4))
            expected['delta'].append(zm*qm-ze*qe)
            expected['activation'].append((zm-ze)*qe)
            expected['decoder'].append(ze*dq)
            expected['interaction'].append((zm-ze)*dq)
        for key, values in expected.items():
            np.testing.assert_allclose(neurons[key], values, rtol=2e-14, atol=1e-15)
        # Using each endpoint's own normalizer would confound these terms.
        moving = result['ledgers']['M']['neurons']['blocks.0']-result['ledgers']['E']['neurons']['blocks.0']
        self.assertGreater(np.max(np.abs(moving-neurons['delta'])), 1e-5)

    def test_projected_delta_retains_bias_and_native_rounding_remainder(self):
        result = self.call()
        direction = result['fixed_direction']
        neurons = result['early_neurons']['blocks.0']
        expected_projection = math.fsum((float(self.fm['stages']['blocks.0.mlp_projected'][-1, j])-
                                         float(self.fe['stages']['blocks.0.mlp_projected'][-1, j]))*
                                        float(direction[j]) for j in range(4))
        expected_bias = math.fsum((float(self.m['blocks.0.mlp.output.bias'][j])-
                                   float(self.e['blocks.0.mlp.output.bias'][j]))*float(direction[j]) for j in range(4))
        self.assertAlmostEqual(neurons['projected_delta'], expected_projection, places=14)
        self.assertAlmostEqual(neurons['bias_delta'], expected_bias, places=14)
        self.assertGreater(abs(neurons['bias_delta']), 1e-5)
        self.assertGreater(abs(neurons['rounding_delta']), 1e-7)
        self.assertAlmostEqual(math.fsum(neurons['delta'])+neurons['bias_delta']+neurons['rounding_delta'],
                               neurons['projected_delta'], places=13)

    def test_selected_features_use_delta_sign_ranking_and_exact_detector_columns(self):
        result = self.call(top_k=2)
        features = result['early_neurons']['blocks.0']
        deltas = features['delta']
        positive = sorted((i for i in range(6) if deltas[i] > 0), key=lambda i: (-deltas[i], i))[:2]
        negative = sorted((i for i in range(6) if deltas[i] < 0), key=lambda i: (deltas[i], i))[:2]
        self.assertEqual([item['neuron'] for item in features['selected']], positive+negative)
        self.assertTrue(features['selected'])
        for item in features['selected']:
            i = item['neuron']
            self.assertEqual(item['delta'], deltas[i])
            for label, cp, values in (('E', self.e, self.fe), ('M', self.m, self.fm)):
                operation = item[label]
                expected = [float(values['stages']['blocks.0.ln2'][-1, j])*
                            scalar_bf16(cp['blocks.0.mlp.input.weight'][j, i]) for j in range(4)]
                np.testing.assert_array_equal(operation['input_product_terms'], expected)
                self.assertEqual(operation['input_column_first_byte'], 4*i)
                self.assertEqual(operation['input_column_byte_stride'], 4*6)
                self.assertEqual(operation['output_row_byte_offset'], 4*i*4)
                self.assertEqual(operation['native_post_gelu'], values['stages']['blocks.0.gelu'][-1, i])
            self.assertAlmostEqual(item['M']['margin_contribution']-item['E']['margin_contribution'], item['delta'], places=14)

    def test_fixed_competitor_uses_e_ties_and_does_not_follow_m_winner(self):
        e = np.array([8., 6., 6., 1., 0., -1., -2.])
        m = np.array([7., 3., 10., 1., 0., -1., -2.])
        result = self.call(logits_e=e, logits_m=m)
        self.assertEqual(result['competitor_id'], 1)
        self.assertEqual(result['predictions']['E']['argmax_id'], 0)
        self.assertEqual(result['predictions']['M']['argmax_id'], 2)
        self.assertEqual(result['predictions']['M']['competitor_logit'], 3.)
        self.assertEqual(result['effects']['margin_delta'], 2.)

    def test_equal_positive_and_negative_feature_deltas_use_lowest_indices(self):
        # Construct exact diagnostic ties. These overwritten hidden observations
        # are deliberately not advertised as a native forward or intervention.
        decoder = np.zeros((6, 4), dtype='<f4'); decoder[:, 0] = 1.
        cp = self.changed_tensor(self.e, 'blocks.0.mlp.output.weight', decoder)
        base = self.call(checkpoint_e=cp, checkpoint_m=cp,
                         stages_m=self.fe['stages'], logits_m=self.fe['logits'][-1])
        alignment = float(base['fixed_direction'][0])
        self.assertNotEqual(alignment, 0.)
        sign = 1. if alignment > 0 else -1.
        stages_e, stages_m = copy.deepcopy(self.fe['stages']), copy.deepcopy(self.fe['stages'])
        stages_e['blocks.0.gelu'][-1] = 0.
        stages_m['blocks.0.gelu'][-1] = np.array([sign, sign, -sign, -sign, 0., 0.])
        for count, expected in ((1, [0, 2]), (2, [0, 1, 2, 3])):
            result = self.call(checkpoint_e=cp, checkpoint_m=cp, stages_e=stages_e,
                               stages_m=stages_m, logits_m=self.fe['logits'][-1], top_k=count)
            features = result['early_neurons']['blocks.0']
            self.assertEqual([item['neuron'] for item in features['selected']], expected)
            np.testing.assert_array_equal(features['delta'],
                [abs(alignment), abs(alignment), -abs(alignment), -abs(alignment), 0., 0.])

    def test_huge_offset_preserves_small_normalizer_change(self):
        high = float(np.float32(1e20))
        e = np.array([high, high, 0., 0., 0., 0., 0.])
        m = np.array([high, 0., 0., 0., 0., 0., 0.])
        result = self.call(logits_e=e, logits_m=m)
        self.assertEqual(result['effects']['target_logit_delta'], 0.)
        self.assertEqual(result['effects']['log_probability_delta'], math.log(2.))
        self.assertEqual(result['effects']['logsumexp_delta'], -math.log(2.))
        self.assertEqual(result['effects']['centered_probability_closure_remainder'], 0.)
        self.assertEqual(result['effects']['absolute_probability_closure_remainder'], 0.)

    def test_target_logit_improves_but_probability_falls_when_other_logits_rise(self):
        result = self.call(logits_e=np.array([0., 0., -1., -2., -3., -4., -5.]),
                           logits_m=np.array([1., 4., -1., -2., -3., -4., -5.]))
        self.assertEqual(result['effects']['target_logit_delta'], 1.)
        self.assertGreater(result['effects']['logsumexp_delta'], 1.)
        self.assertLess(result['effects']['log_probability_delta'], 0.)

    def test_every_parameter_outside_early_ln2_mlp_is_byte_exact(self):
        for spec in self.e.manifest:
            if spec.name.startswith(('blocks.0.ln2.', 'blocks.0.mlp.')):
                continue
            value = np.array(self.m[spec.name]); value.flat[0] += np.float32(.125)
            changed = self.changed_tensor(self.m, spec.name, value)
            with self.subTest(parameter=spec.name), self.assertRaisesRegex(ValueError, 'non-M parameter differs'):
                self.call(checkpoint_m=changed)

    def test_unused_embedding_padding_including_signed_zero_must_be_unchanged(self):
        value = np.array(self.m['token_embedding.weight']); value[7, 0] = np.float32(-0.)
        np.testing.assert_array_equal(value, self.m['token_embedding.weight'])
        changed = self.changed_tensor(self.m, 'token_embedding.weight', value)
        with self.assertRaisesRegex(ValueError, 'non-M parameter differs'):
            self.call(checkpoint_m=changed)

    def test_invalid_ids_counts_geometry_and_logical_vocabulary_fail(self):
        for tokens in ([], [True], [-1], [7], [1.], [1]*7, np.array(self.tokens)):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.call(token_ids=tokens)
        for target in (-1, 7, True, 0.):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.call(target=target)
        for count in (0, -1, 7, True, 1.):
            with self.subTest(top_k=count), self.assertRaises(ValueError):
                self.call(top_k=count)
        for config in (GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=7,
                                   n_layers=2, d_model=4, n_heads=2, d_ff=6),
                       GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=6,
                                   n_layers=3, d_model=4, n_heads=2, d_ff=6)):
            changed = copy.copy(self.m); changed.config = config
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.call(checkpoint_m=changed)
        for values in (np.zeros(8), np.zeros((1, 7)), np.zeros(6), np.array([0.]*6+[math.nan])):
            for endpoint in ('logits_e', 'logits_m'):
                with self.subTest(endpoint=endpoint, shape=values.shape), self.assertRaises(ValueError):
                    self.call(**{endpoint: values})

    def test_stage_shape_missing_and_nonfinite_values_fail_for_both_endpoints(self):
        for endpoint, forward_result in (('stages_e', self.fe), ('stages_m', self.fm)):
            for name in ('embedding', 'blocks.0.ln2', 'blocks.0.fc1', 'blocks.0.gelu', 'blocks.1.after_mlp'):
                for mode in ('missing', 'shape', 'nonfinite'):
                    stages = dict(forward_result['stages'])
                    if mode == 'missing':
                        del stages[name]
                    elif mode == 'shape':
                        stages[name] = np.zeros((len(self.tokens), 1))
                    else:
                        stages[name] = stages[name].copy(); stages[name].flat[0] = math.nan
                    with self.subTest(endpoint=endpoint, name=name, mode=mode), self.assertRaises(ValueError):
                        self.call(**{endpoint: stages})

    def test_identical_but_odd_layer_configs_are_rejected_before_stage_accounting(self):
        odd = GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=6,
                         n_layers=3, d_model=4, n_heads=2, d_ff=6)
        e, m = copy.copy(self.e), copy.copy(self.m)
        e.config = m.config = odd
        with self.assertRaisesRegex(ValueError, 'same even model configuration'):
            self.call(checkpoint_e=e, checkpoint_m=m)


if __name__ == '__main__':
    unittest.main()
