"""Pure supplied-logit arithmetic tests; not evidence of any native forward.

AA/AJ/JA/JJ name original/replacement input and output dictionary choices,
respectively. Tiny synthetic arrays test decomposition semantics independently
of checkpoint ownership, provenance, GPU execution, or model replay.
"""

import json
import math
import unittest

import numpy as np

from . import embedding_factorial_mechanism as mechanism


CELLS = ('AA', 'AJ', 'JA', 'JJ')


def arrays(rows):
    return {cell: np.array(row, dtype='<f4') for cell, row in zip(CELLS, rows)}


def scalar_oracle(logits, target):
    """Python scalar math and fsum, with no production score helper calls."""
    clean = [float(x) for x in logits['AA']]
    rival = max((index for index in range(len(clean)) if index != target),
                key=lambda index: (clean[index], -index))
    cells = {}
    for name in CELLS:
        scores = [float(x) for x in logits[name]]
        shifted = [score - scores[rival] for score in scores]
        maximum = max(shifted)
        partition = maximum + math.log(math.fsum(math.exp(x - maximum) for x in shifted))
        margin = scores[target] - scores[rival]
        cells[name] = dict(target_logit=scores[target], rival_logit=scores[rival],
                           margin=margin, log_partition_relative_to_rival=partition,
                           log_probability=margin - partition)
    effects = {}
    for report_name, field in (('margin', 'margin'),
                               ('normalizer', 'log_partition_relative_to_rival'),
                               ('log_probability', 'log_probability')):
        aa, aj, ja, jj = (cells[name][field] for name in CELLS)
        effects[report_name] = dict(input=ja - aa, output=aj - aa,
            interaction=jj - ja - aj + aa, joint=jj - aa)
    return rival, cells, effects


class FactorialMechanismMathTest(unittest.TestCase):
    def setUp(self):
        self.logits = arrays([[2, 1, -.5, 0], [3, .5, 2, -1],
                              [1, 2, 0, .5], [4, 3, 1, 0]])

    def assert_matches_oracle(self, logits, target):
        rival, cells, effects = scalar_oracle(logits, target)
        actual = mechanism.decompose_logits(logits, target)
        self.assertEqual(actual['rival_id'], rival)
        self.assertEqual(set(actual['cells']), set(CELLS))
        for name in CELLS:
            for field, value in cells[name].items():
                self.assertAlmostEqual(actual['cells'][name][field], value, delta=2e-12)
        for name, terms in effects.items():
            self.assertEqual(set(actual['effects'][name]), {'input', 'output', 'interaction', 'joint'})
            for field, value in terms.items():
                self.assertAlmostEqual(actual['effects'][name][field], value, delta=5e-12)
        anchored = []
        for token in range(len(logits['AA'])):
            aa, aj, ja, jj = (float(logits[cell][token]) - float(logits[cell][rival]) for cell in CELLS)
            anchored.append(jj - ja - aj + aa)
        self.assertAlmostEqual(actual['anchored_logit_interaction']['max_abs'], max(map(abs, anchored)), delta=5e-12)
        self.assertAlmostEqual(actual['anchored_logit_interaction']['l2'],
                               math.sqrt(math.fsum(value * value for value in anchored)), delta=5e-12)
        self.assertAlmostEqual(actual['anchored_logit_interaction']['target_margin_interaction'],
                               anchored[target], delta=5e-12)
        self.assertLessEqual(actual['max_effect_identity_error'], 5e-12)
        json.dumps(actual, allow_nan=False)
        return actual

    def test_independent_scalar_math_on_all_four_cells_and_targets(self):
        for target in range(4):
            with self.subTest(target=target): self.assert_matches_oracle(self.logits, target)

    def test_random_modest_arrays_match_independent_scalar_math(self):
        rng = np.random.default_rng(181)
        for size in (2, 5, 17):
            values = {cell: rng.normal(size=size).astype('<f4') for cell in CELLS}
            self.assert_matches_oracle(values, size // 2)

    def test_factorial_letters_and_effect_definitions_are_not_transposed(self):
        logits = arrays([[1, 0], [3, 0], [5, 0], [12, 0]])
        actual = mechanism.decompose_logits(logits, 0)
        self.assertEqual(actual['effects']['margin'], dict(input=4, output=2, interaction=5, joint=11))
        for field in ('input', 'output', 'interaction', 'joint'):
            self.assertAlmostEqual(actual['effects']['log_probability'][field],
                actual['effects']['margin'][field] - actual['effects']['normalizer'][field], delta=1e-13)

    def test_each_cell_common_shift_leaves_relative_scores_and_effects_unchanged(self):
        original = mechanism.decompose_logits(self.logits, 0)
        shifted = {cell: row + np.float32(offset) for (cell, row), offset
                   in zip(self.logits.items(), (16, -32, 64, -128))}
        changed = mechanism.decompose_logits(shifted, 0)
        self.assertEqual(original['rival_id'], changed['rival_id'])
        for cell in CELLS:
            for field in ('margin', 'log_partition_relative_to_rival', 'log_probability'):
                self.assertAlmostEqual(original['cells'][cell][field], changed['cells'][cell][field], delta=1e-13)
        for group in original['effects']:
            for term in original['effects'][group]:
                self.assertAlmostEqual(original['effects'][group][term], changed['effects'][group][term], delta=1e-13)

    def test_target_winner_is_excluded_from_rival_selection(self):
        logits = arrays([[10, 1, 2], [2, 9, 8], [3, 0, 4], [4, 1, 0]])
        self.assertEqual(self.assert_matches_oracle(logits, 0)['rival_id'], 2)

    def test_rival_is_fixed_from_aa_even_when_other_cells_change_winner(self):
        logits = arrays([[4, 2, 3], [1, 8, -2], [0, 9, -3], [1, 10, -4]])
        actual = self.assert_matches_oracle(logits, 0)
        self.assertEqual(actual['rival_id'], 2)
        self.assertEqual(actual['cells']['JJ']['rival_logit'], -4)
        self.assertEqual(actual['cells']['JJ']['margin'], 5)

    def test_rival_ties_choose_lowest_non_target_id(self):
        for row, target, expected in (([9, 4, 4], 0, 1), ([4, 4, 9], 2, 0), ([4, 4], 0, 1)):
            logits = arrays([row] * 4)
            self.assertEqual(mechanism.decompose_logits(logits, target)['rival_id'], expected)

    def test_normalization_alone_can_create_nonzero_log_probability_interaction(self):
        # A third token moves additively; target and fixed-rival logits do not.
        # Log-sum-exp curvature creates an interaction without a margin effect.
        logits = arrays([[1, 0, 0], [1, 0, 1], [1, 0, 1], [1, 0, 2]])
        actual = self.assert_matches_oracle(logits, 0)
        self.assertTrue(all(value == 0 for value in actual['effects']['margin'].values()))
        self.assertEqual(actual['anchored_logit_interaction']['max_abs'], 0)
        self.assertEqual(actual['anchored_logit_interaction']['l2'], 0)
        interaction = actual['effects']['normalizer']['interaction']
        self.assertGreater(abs(interaction), .01)
        self.assertAlmostEqual(actual['effects']['log_probability']['interaction'], -interaction)

    def test_zero_target_margin_interaction_does_not_exclude_competitor_interaction(self):
        logits = arrays([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 2]])
        actual = self.assert_matches_oracle(logits, 0)
        self.assertEqual(actual['effects']['margin']['interaction'], 0)
        self.assertEqual(actual['anchored_logit_interaction'],
                         dict(max_abs=2, l2=2, target_margin_interaction=0))

    def test_bilinear_synthetic_head_margin_interaction_has_direct_formula(self):
        # Independent toy linear head, not a claim about a real tied model.
        h0, h1 = np.array([1, 2], dtype='<f4'), np.array([3, -1], dtype='<f4')
        w0 = np.array([[2, 0], [0, 1], [-1, 1]], dtype='<f4')
        w1 = np.array([[1, 2], [2, -1], [0, 3]], dtype='<f4')
        logits = dict(AA=w0 @ h0, AJ=w1 @ h0, JA=w0 @ h1, JJ=w1 @ h1)
        actual = self.assert_matches_oracle(logits, 0)
        rival = actual['rival_id']
        difference = (w1[0] - w1[rival]) - (w0[0] - w0[rival])
        self.assertEqual(actual['effects']['margin']['interaction'], float(difference @ (h1 - h0)))

    def test_identical_cells_have_zero_effects(self):
        logits = arrays([[0, 0, 0]] * 4)
        actual = self.assert_matches_oracle(logits, 1)
        for terms in actual['effects'].values(): self.assertTrue(all(value == 0 for value in terms.values()))
        self.assertAlmostEqual(actual['cells']['AA']['log_probability'], -math.log(3))

    def test_large_finite_logits_use_stable_logsumexp(self):
        logits = arrays([[1000, 999, -1000], [-1000, 1000, 999],
                          [1000, -1000, 998], [-1000, -999, -998]])
        actual = self.assert_matches_oracle(logits, 0)
        self.assertLess(actual['cells']['AJ']['log_probability'], -1900)
        self.assertTrue(all(math.isfinite(cell['log_probability']) for cell in actual['cells'].values()))

    def test_readonly_inputs_are_unchanged(self):
        before = {cell: row.tobytes() for cell, row in self.logits.items()}
        for row in self.logits.values(): row.flags.writeable = False
        self.assert_matches_oracle(self.logits, 0)
        self.assertEqual(before, {cell: row.tobytes() for cell, row in self.logits.items()})

    def test_rejects_missing_extra_or_wrong_cell_names(self):
        for changed in ({key: value for key, value in self.logits.items() if key != 'JJ'},
                        {**self.logits, 'extra': self.logits['AA']},
                        {key.lower(): value for key, value in self.logits.items()}):
            with self.subTest(keys=list(changed)), self.assertRaises(ValueError):
                mechanism.decompose_logits(changed, 0)

    def test_rejects_dtype_rank_length_and_shape_mismatch(self):
        for bad in (self.logits['AJ'].astype('f8'), self.logits['AJ'].astype('i4'),
                    self.logits['AJ'].astype('>f4'), self.logits['AJ'][None, :], self.logits['AJ'][:3]):
            with self.subTest(dtype=bad.dtype, shape=bad.shape), self.assertRaises(ValueError):
                mechanism.decompose_logits({**self.logits, 'AJ': bad}, 0)
        for size in (0, 1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                mechanism.decompose_logits({cell: np.zeros(size, dtype='<f4') for cell in CELLS}, 0)

    def test_rejects_nonfinite_values_in_every_cell(self):
        for cell in CELLS:
            for invalid in (np.nan, np.inf, -np.inf):
                changed = {name: row.copy() for name, row in self.logits.items()}
                changed[cell][-1] = invalid
                with self.subTest(cell=cell, invalid=invalid), self.assertRaises(ValueError):
                    mechanism.decompose_logits(changed, 0)

    def test_rejects_invalid_or_boolean_target_ids(self):
        for target in (-1, 4, True, False, 1.0, '1', None):
            with self.subTest(target=target), self.assertRaises(ValueError):
                mechanism.decompose_logits(self.logits, target)


if __name__ == '__main__':
    unittest.main()
