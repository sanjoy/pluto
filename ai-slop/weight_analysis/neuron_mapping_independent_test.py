"""Independent dense checks of the predeclared reliance-map arithmetic.

These tests never read checkpoints, tokens, or real intervention outcomes.
The oracle centers with explicit projectors and obtains the discovery mode
from a group-space symmetric eigensolve, rather than the implementation's
mean subtraction and rectangular SVD. Synthetic confirmation outcomes are
deliberately allowed to disagree with discovery: they must not choose groups.
"""

import copy
import unittest
from unittest import mock

import numpy as np

from . import neuron_mapping as mapping


def center_dense(matrix):
    """Two fixed orthogonal projections, not fitted passage effects."""
    rows, columns = matrix.shape
    left = np.eye(rows) - np.ones((rows, rows)) / rows
    right = np.eye(columns) - np.ones((columns, columns)) / columns
    return left @ matrix @ right


def centered_basis(length, count, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(length, count))
    values -= values.mean(axis=0)
    return np.linalg.qr(values)[0]


def fixture_groups():
    # Comparator identities are frozen, same-block, and independent of losses.
    return [{"name": f"block{6 + g // 32}_group{g % 32:02d}_half",
             "norm_comparators": [g // 32 * 32 + (g + shift) % 32
                                  for shift in range(1, 5)]}
            for g in range(64)]


class IndependentResidualTest(unittest.TestCase):
    def test_centering_agrees_with_dense_projectors(self):
        rng = np.random.default_rng(731)
        for shape in ((5, 3), (64, 16)):
            delta = rng.normal(size=shape)
            np.testing.assert_allclose(mapping.double_center(delta), center_dense(delta),
                                       atol=2e-15, rtol=2e-14)

    def test_original_other_passage_contrast_matches_scalar_exclusion(self):
        delta = np.random.default_rng(819).normal(size=(7, 5))
        expected = np.empty_like(delta)
        for group in range(7):
            for passage in range(5):
                other = [delta[group, q] for q in range(5) if q != passage]
                expected[group, passage] = delta[group, passage] - sum(other) / 4
        np.testing.assert_allclose(mapping.ordinary_contrasts(delta), expected, atol=5e-16)

    def test_discovery_projection_matches_group_covariance_eigen_oracle(self):
        left, right = centered_basis(64, 3, 2), centered_basis(16, 3, 3)
        delta = 10 + (left * [7., 2., .5]) @ right.T
        centered = center_dense(delta)
        eigenvalues, vectors = np.linalg.eigh(centered @ centered.T)
        direction = vectors[:, np.argmax(eigenvalues)]
        expected = (np.eye(64) - np.outer(direction, direction)) @ centered
        fitted = mapping.discover(delta)
        actual_direction = np.asarray(fitted["frozen_direction"])
        actual = np.asarray(fitted["residual"])
        self.assertEqual(fitted["status"], "unique_discovery_direction")
        np.testing.assert_allclose(np.outer(actual_direction, actual_direction),
                                   np.outer(direction, direction), atol=2e-14)
        np.testing.assert_allclose(actual, expected, atol=2e-13)
        self.assertAlmostEqual(fitted["removed_discovery_energy_fraction"], 49 / 53.25)
        self.assertLessEqual(np.linalg.norm(actual), np.linalg.norm(centered))
        np.testing.assert_allclose(actual_direction @ actual, 0, atol=2e-13)
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=5e-12)
        np.testing.assert_allclose(actual.sum(axis=1), 0, atol=5e-12)

    def test_product_plus_additive_effects_abstains(self):
        rng = np.random.default_rng(810)
        a, b = rng.normal(size=64), rng.normal(size=16)
        row, column = rng.normal(size=64), rng.normal(size=16)
        delta = 20 + np.outer(a, b) + row[:, None] + column[None, :]
        fitted = mapping.discover(delta)
        self.assertEqual(fitted["status"], "unique_discovery_direction")
        self.assertEqual(fitted["selected_group_indices"], [None] * 16)
        self.assertAlmostEqual(fitted["removed_discovery_energy_fraction"], 1)
        self.assertLess(np.max(np.abs(fitted["residual"])), fitted["numeric_floor"])

    def test_large_additive_cancellation_does_not_create_a_map(self):
        rng = np.random.default_rng(811)
        for magnitude in (1., 1e8, 1e14):
            delta = magnitude * (10 + rng.normal(size=(64, 1))
                                 + rng.normal(size=(1, 16)))
            # Residual arithmetic may be nonzero after these large subtractions.
            # The floor must use the original operands, not the tiny remainder.
            fitted = mapping.discover(delta)
            expected_floor = 64 * np.finfo(np.float64).eps * 64 * np.linalg.norm(delta)
            self.assertEqual(fitted["numeric_floor"], expected_floor)
            self.assertEqual(fitted["selected_group_indices"], [None] * 16)
            self.assertEqual(fitted["status"], "zero_centered_effect")

    def test_rank_one_signal_with_large_additive_offsets_still_abstains(self):
        left, right = centered_basis(64, 1, 912), centered_basis(16, 1, 914)
        rng = np.random.default_rng(818)
        delta = 1e9 * (10 + rng.normal(size=(64, 1)) + rng.normal(size=(1, 16)))
        delta += 1000 * left @ right.T
        fitted = mapping.discover(delta)
        self.assertEqual(fitted["status"], "unique_discovery_direction")
        self.assertEqual(fitted["selected_group_indices"], [None] * 16)

    def test_equal_leading_modes_abstain_instead_of_arbitrary_basis(self):
        # Exact integer centered directions make a genuinely two-dimensional
        # leading eigenspace. Neither basis vector is an identifiable primary.
        first = np.array([1., 1., -1., -1.]) / 2
        second = np.array([1., -1., 1., -1.]) / 2
        delta = 10 + 3 * (np.outer(first, first) + np.outer(second, second))
        fitted = mapping.discover(delta)
        self.assertEqual(fitted["status"], "ambiguous_leading_direction")
        self.assertEqual(fitted["selected_group_indices"], [None] * 4)
        np.testing.assert_array_equal(fitted["frozen_direction"], np.zeros(4))

    def test_confirmation_uses_discovery_direction_not_its_own_leading_mode(self):
        left, right = centered_basis(64, 3, 822), centered_basis(16, 3, 823)
        discovery_delta = 10 + (left[:, :2] * [7., 2.]) @ right[:, :2].T
        fitted = mapping.discover(discovery_delta)
        confirmation_delta = 10 + (left * [.5, 1., 20.]) @ right.T
        frozen = np.asarray(fitted["frozen_direction"])
        expected = (np.eye(64) - np.outer(frozen, frozen)) @ center_dense(confirmation_delta)
        # A mistaken refit would erase the new, dominant third direction.
        with mock.patch.object(np.linalg, "svd", side_effect=AssertionError("confirmation refitted SVD")):
            actual = mapping.confirm(confirmation_delta, fitted)
        np.testing.assert_allclose(actual, expected, atol=2e-13)
        self.assertAlmostEqual(float(left[:, 2] @ actual @ right[:, 2]), 20.)
        self.assertAlmostEqual(float(left[:, 0] @ actual @ right[:, 0]), 0., places=12)

    def test_frozen_direction_sign_does_not_change_confirmation(self):
        delta = 10 + np.random.default_rng(18).normal(size=(64, 16))
        fitted = mapping.discover(delta)
        reversed_sign = copy.deepcopy(fitted)
        reversed_sign["frozen_direction"] = [-x for x in fitted["frozen_direction"]]
        np.testing.assert_array_equal(mapping.confirm(delta, fitted),
                                      mapping.confirm(delta, reversed_sign))

    def test_confirmation_outcomes_cannot_change_selection(self):
        rng = np.random.default_rng(824)
        discovery = 10 + rng.normal(size=(64, 16))
        groups = fixture_groups()
        original_groups = copy.deepcopy(groups)
        first = mapping.select_and_confirm(discovery, np.zeros((64, 16)), groups)
        second = mapping.select_and_confirm(discovery, 1e8 * rng.normal(size=(64, 16)), groups)
        self.assertEqual(first["discovery"], second["discovery"])
        self.assertGreater(first["summary"]["selected_passages"], 0)
        self.assertEqual([x["selected_group_index"] for x in first["selected"]],
                         [x["selected_group_index"] for x in second["selected"]])
        self.assertEqual(groups, original_groups)
        self.assertEqual(first["summary"]["confirmation_positive_raw_and_residual"], 0)

    def test_exact_score_ties_report_competition_rank_and_all_ties(self):
        values = np.array([2., 2., -3., 5., 2.])
        self.assertEqual(mapping.score_rank(values, 0), {
            "rank": 2, "exact_tie_count_including_self": 3, "candidate_count": 5})
        self.assertEqual(mapping.score_rank(values, 2)["rank"], 5)
        self.assertEqual(mapping.score_rank(np.zeros(4), 2)["exact_tie_count_including_self"], 4)

    def test_exact_selection_tie_uses_lower_group_index(self):
        # Supply the known leading vector so unrelated eigensolver roundoff
        # cannot destroy the exact row tie that this policy test exercises.
        lead = np.array([0., 0., 1., -1.]) / np.sqrt(2)
        residual_left = np.array([1., 1., -1., -1.]) / 2
        delta = 10 + 3 * np.outer(lead, lead) + np.outer(residual_left, residual_left)
        vectors = np.zeros((4, 4))
        vectors[:, 0] = lead
        with mock.patch.object(np.linalg, "svd", return_value=(vectors, np.array([3., 1., 0., 0.]), np.eye(4))):
            fitted = mapping.discover(delta)
        residual = np.asarray(fitted["residual"])
        self.assertEqual(residual[0, 0], residual[1, 0])
        self.assertGreater(residual[0, 0], fitted["numeric_floor"])
        self.assertEqual(fitted["selected_group_indices"][0], 0)

    def test_negative_raw_damage_disqualifies_an_otherwise_positive_residual(self):
        rng = np.random.default_rng(827)
        delta = 10 + rng.normal(size=(64, 16))
        first = mapping.discover(delta)
        group = next(g for g in first["selected_group_indices"] if g is not None)
        delta[group] -= 1000  # Row-constant term disappears on double-centering.
        fitted = mapping.discover(delta)
        self.assertTrue(np.all(delta[group] < 0))
        self.assertNotIn(group, fitted["selected_group_indices"])
        np.testing.assert_allclose(fitted["residual"], first["residual"], atol=2e-12)


if __name__ == "__main__":
    unittest.main()
