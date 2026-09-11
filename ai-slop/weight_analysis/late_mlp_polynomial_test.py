"""Independent polynomial oracles on synthetic coefficients/checkpoints only.

No real corpus, real checkpoint, tokenizer, candidate model execution, or GPU
is used. The dense/scalar calculations below are deliberately independent of
the optimized prefix/replacement implementation. The planted AND interaction
cannot be solved by adding unary token preferences.
"""

import itertools
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from . import late_mlp_polynomial as polynomial


EPSILON = float(np.float32(1e-5))
GELU_C = float(np.float32(0.7978845608))
GELU_A = float(np.float32(0.044715))


def scalar_gelu(value):
    return .5 * value * (1 + math.tanh(GELU_C * (value + GELU_A * value**3)))


def scalar_derivatives(value):
    inner = GELU_C * (value + GELU_A * value**3)
    first = GELU_C * (1 + 3 * GELU_A * value**2)
    second = 6 * GELU_C * GELU_A * value
    tanh = math.tanh(inner)
    sech2 = 1 - tanh**2
    return (.5 * (1 + tanh) + .5 * value * sech2 * first,
            sech2 * first + .5 * value * sech2 * (second - 2 * tanh * first**2))


def scalar_norm(anchor, gamma, beta):
    """Row differential uses gamma of the OUTPUT coordinate, J[i,j]."""
    dimension = len(anchor)
    mean = math.fsum(float(x) for x in anchor) / dimension
    centered = [float(x) - mean for x in anchor]
    variance = math.fsum(x*x for x in centered) / dimension
    scale = math.sqrt(variance + EPSILON)
    values = np.array([centered[j] * float(gamma[j]) / scale + float(beta[j])
                       for j in range(dimension)])
    jacobian = np.array([
        [float(gamma[j]) / scale * ((1 if i == j else 0) - 1/dimension
                                    - centered[i]*centered[j]/(dimension*scale**2))
         for j in range(dimension)] for i in range(dimension)])
    return values, jacobian, scale


def coefficient_fixture(seed=23, length=4):
    rng = np.random.default_rng(seed)
    return dict(token_ids=np.array([1, 4, 7, 9], dtype=np.int64),
                direct=rng.normal(size=(length, 4, 3)),
                features=rng.normal(size=(length, 4, 5)),
                readout=rng.normal(size=(4, 3)),
                neuron_readout=rng.normal(size=(4, 5)),
                g7=rng.normal(size=5), h7=rng.normal(size=5), alpha=.125)


def scalar_components(compiled, rows):
    """Rebuild every prefix independently, with scalar coordinate sums."""
    index = {int(token): i for i, token in enumerate(compiled.token_ids)}
    rows = np.asarray(rows)
    result = {name: np.zeros(rows.shape) for name in ("direct", "affine", "quadratic")}
    for row, tokens in enumerate(rows):
        for r in range(1, len(tokens)):
            target = index[int(tokens[r])]
            x = [compiled.alpha/r * math.fsum(float(compiled.direct[p, index[int(tokens[p])], d])
                                              for p in range(r))
                 for d in range(compiled.direct.shape[2])]
            z = [compiled.alpha/r * math.fsum(float(compiled.features[p, index[int(tokens[p])], j])
                                              for p in range(r))
                 for j in range(compiled.features.shape[2])]
            result["direct"][row, r] = math.fsum(x[d]*float(compiled.readout[target, d]) for d in range(len(x)))
            result["affine"][row, r] = math.fsum(float(compiled.g7[j])*z[j]*float(compiled.neuron_readout[target, j]) for j in range(len(z)))
            result["quadratic"][row, r] = math.fsum(.5*float(compiled.h7[j])*z[j]**2*float(compiled.neuron_readout[target, j]) for j in range(len(z)))
    result["total"] = sum(result.values()).sum(axis=1)
    return result


def interaction_fixture():
    """Only A,B,C scores positively: .5[(a+b)^2-a^2-b^2] = a*b.

    Source A belongs in slot0 and B in slot1. Wrong order, either source alone,
    or another target has zero score. The prefix-length normalization makes
    the unique complete score 1/4 for alpha=1. Earlier r=1 self terms cancel.
    """
    features = np.zeros((3, 3, 3))
    features[0, 0] = [1, 1, 0]
    features[1, 1] = [1, 0, 1]
    neuron_readout = np.zeros((3, 3))
    neuron_readout[2] = 1
    return polynomial.CompiledPolynomial(
        token_ids=np.arange(3), direct=np.zeros((3, 3, 1)), features=features,
        readout=np.zeros((3, 1)), neuron_readout=neuron_readout,
        g7=np.zeros(3), h7=np.array([1., -1., -1.]), alpha=1.)


class DerivativesTest(unittest.TestCase):
    def test_tanh_gelu_first_and_second_derivatives_finite_difference(self):
        points = np.array([-3., -1.2, -.3, 0., .7, 2.5])
        first, second = polynomial.gelu_derivatives(points)
        for i, point in enumerate(points):
            epsilon = 1e-4
            f0, fm, fp = scalar_gelu(point), scalar_gelu(point-epsilon), scalar_gelu(point+epsilon)
            self.assertAlmostEqual(first[i], (fp-fm)/(2*epsilon), delta=2e-8)
            self.assertAlmostEqual(second[i], (fp-2*f0+fm)/epsilon**2, delta=2e-7)

    def test_gelu_zero_and_curvature_sign_are_not_relu(self):
        first, second = polynomial.gelu_derivatives(np.array([0., 2., -2.]))
        self.assertEqual(first[0], .5)
        self.assertAlmostEqual(second[0], GELU_C)
        self.assertLess(second[1], 0)
        self.assertLess(second[2], 0)

    def test_norm_jacobian_independent_scalar_and_directional_difference(self):
        anchor = np.array([.2, -.7, .9, 1.4])
        gamma = np.array([1.1, .4, -1.7, .8])
        beta = np.array([-.1, .5, .3, -.8])
        actual = polynomial.layer_norm_and_jacobian(anchor, gamma, beta)
        expected = scalar_norm(anchor, gamma, beta)
        for value, reference in zip(actual, expected):
            np.testing.assert_allclose(value, reference, rtol=2e-14, atol=2e-14)
        direction = np.array([.8, -.1, .4, -.3])
        epsilon = 1e-6
        derivative = (scalar_norm(anchor+epsilon*direction, gamma, beta)[0]
                      - scalar_norm(anchor-epsilon*direction, gamma, beta)[0])/(2*epsilon)
        np.testing.assert_allclose(direction @ actual[1], derivative, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(np.ones(4) @ actual[1], 0, atol=1e-14)

    def test_final_norm_numerator_not_jacobian_or_wrong_gamma_order(self):
        rng = np.random.default_rng(42)
        anchor, gamma, beta = rng.normal(size=(3, 4))
        embeddings = rng.normal(size=(5, 4)); embeddings -= embeddings.mean(axis=0)
        centered = np.eye(4) - np.ones((4, 4))/4
        readout = (embeddings * gamma) @ centered
        normalized, _, scale = scalar_norm(anchor, gamma, beta)
        np.testing.assert_allclose(anchor @ readout.T,
                                   scale * ((normalized-beta) @ embeddings.T), atol=1e-14)
        self.assertGreater(np.max(np.abs(readout - (embeddings @ centered)*gamma)), .01)

    def test_derivative_helpers_reject_nonfinite_and_wrong_shapes(self):
        with self.assertRaises(ValueError): polynomial.gelu_derivatives(np.array([np.nan]))
        for anchor, gamma, beta in (([1, 2], [1], [0, 0]),
                                    ([np.inf, 2], [1, 1], [0, 0]),
                                    ([[1, 2]], [1, 1], [0, 0])):
            with self.assertRaises(ValueError):
                polynomial.layer_norm_and_jacobian(anchor, gamma, beta)


class PolynomialOracleTest(unittest.TestCase):
    def test_all_prefix_positions_match_independent_scalar_polynomial(self):
        compiled = polynomial.CompiledPolynomial(**coefficient_fixture())
        rows = np.array([[1, 4, 7, 9], [9, 1, 9, 4], [7, 7, 7, 7]])
        expected = scalar_components(compiled, rows)
        actual = compiled.components(rows)
        np.testing.assert_allclose(compiled.score(rows), expected["total"], atol=1e-13, rtol=1e-13)
        for component in ("direct", "affine", "quadratic"):
            np.testing.assert_allclose(actual["per_position_"+component], expected[component], atol=1e-13)
            np.testing.assert_allclose(actual[component], expected[component].sum(axis=1), atol=1e-13)
            np.testing.assert_array_equal(actual["per_position_"+component][:, 0], 0)

    def test_all_replacement_columns_match_bruteforce_full_scores(self):
        compiled = polynomial.CompiledPolynomial(**coefficient_fixture(seed=31))
        rows = np.array([[1, 4, 7, 9], [9, 1, 9, 4]])
        original = rows.copy()
        for position in range(4):
            actual = compiled.replacement_scores(rows, position, chunk_size=2)
            expected = np.empty((2, 4))
            for row, token in itertools.product(range(2), range(4)):
                changed = rows[row:row+1].copy()
                changed[0, position] = compiled.token_ids[token]
                expected[row, token] = scalar_components(compiled, changed)["total"][0]
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(rows, original)

    def test_replacement_chunk_sizes_and_original_column_agree(self):
        compiled = polynomial.CompiledPolynomial(**coefficient_fixture())
        rows = np.array([[1, 4, 7, 9], [4, 4, 4, 4]])
        for position in range(4):
            a = compiled.replacement_scores(rows, position, chunk_size=1)
            b = compiled.replacement_scores(rows, position, chunk_size=99)
            np.testing.assert_allclose(a, b, atol=1e-13)
            columns = [list(compiled.token_ids).index(token) for token in rows[:, position]]
            np.testing.assert_array_equal(a[np.arange(2), columns], compiled.score(rows))

    def test_target_and_earlier_source_roles_both_change_full_objective(self):
        compiled = polynomial.CompiledPolynomial(**coefficient_fixture(seed=71))
        before = np.array([[1, 4, 7, 9]])
        after = np.array([[1, 9, 7, 9]])
        first = scalar_components(compiled, before)
        second = scalar_components(compiled, after)
        # Position1 is itself a target and contributes to BOTH later prefixes.
        differences = sum(second[k]-first[k] for k in ("direct", "affine", "quadratic"))
        self.assertTrue(np.all(np.abs(differences[0, 1:]) > 1e-6))
        np.testing.assert_allclose(compiled.score(after)-compiled.score(before), differences.sum(axis=1), atol=1e-13)

    def test_alpha_linear_and_quadratic_scaling(self):
        data = coefficient_fixture()
        first = polynomial.CompiledPolynomial(**data)
        second = polynomial.CompiledPolynomial(**{**data, "alpha": 2*data["alpha"]})
        rows = np.array([[1, 4, 7, 9]])
        a, b = first.components(rows), second.components(rows)
        for component, factor in (("direct", 2), ("affine", 2), ("quadratic", 4)):
            np.testing.assert_allclose(b[component], factor*a[component], atol=1e-13)

    def test_genuine_ordered_pair_interaction_has_unique_exhaustive_maximum(self):
        compiled = interaction_fixture()
        rows = np.array(list(itertools.product(range(3), repeat=3)))
        scores = compiled.score(rows)
        winners = rows[scores == scores.max()]
        np.testing.assert_array_equal(winners, [[0, 1, 2]])
        self.assertEqual(scores.max(), .25)
        # Neither unary source term alone can account for the positive score.
        np.testing.assert_array_equal(compiled.score(np.array([[0, 2, 2], [2, 1, 2], [1, 0, 2]])), 0)

    def test_source_order_antisymmetry_and_homogeneous_slot_null(self):
        a, b = np.array([1., 0.]), np.array([0., 1.])
        m0, m1 = np.eye(2), np.array([[0., 0.], [1., 1.]])
        curvature = np.diag([1., 0.])
        forward = a @ m0 @ curvature @ m1.T @ b
        reverse = b @ m0 @ curvature @ m1.T @ a
        self.assertEqual(forward, 1)
        self.assertEqual(reverse, 0)
        antisymmetric = m0 @ curvature @ m1.T - m1 @ curvature @ m0.T
        self.assertEqual(forward-reverse, a @ antisymmetric @ b)
        np.testing.assert_array_equal(antisymmetric, -antisymmetric.T)
        for shared in (m0, m1):
            self.assertEqual(a @ shared @ curvature @ shared.T @ b,
                             b @ shared @ curvature @ shared.T @ a)

    def test_invalid_coefficients_ids_and_nonfinite_rejected(self):
        for field, value in (("direct", np.zeros((4, 3, 3))), ("g7", np.zeros(3)),
                             ("alpha", np.nan), ("h7", np.full(5, np.inf)),
                             ("token_ids", np.array([1, 1, 7, 9]))):
            with self.assertRaises(ValueError):
                polynomial.CompiledPolynomial(**{**coefficient_fixture(), field: value})
        compiled = polynomial.CompiledPolynomial(**coefficient_fixture())
        for rows in (np.array([[1, 4, 7]]), np.array([[1, 4, 7, 2]]),
                     np.array([[1., 4., 7., 9.]]), np.array([1, 4, 7, 9])):
            with self.assertRaises(ValueError): compiled.score(rows)
        for position in (-1, 4):
            with self.assertRaises(ValueError):
                compiled.replacement_scores(np.array([[1, 4, 7, 9]]), position)


class SearchTest(unittest.TestCase):
    def test_planted_interaction_coordinate_search_matches_exhaustive_optimum(self):
        compiled = interaction_fixture()
        # These seeds expose an improving coordinate before a zero-score tie
        # can erase the correct source. Being merely one edit away does NOT
        # guarantee coordinate ascent succeeds on a pure interaction plateau.
        rows = np.array([[0, 1, 2], [0, 0, 2], [1, 1, 2]])
        result = compiled.optimize(rows, sweeps=2, chunk_size=2)
        np.testing.assert_array_equal(result.token_ids, np.tile([0, 1, 2], (3, 1)))
        np.testing.assert_array_equal(result.scores, [.25]*3)
        for step in result.trace:
            self.assertTrue(np.all(np.asarray(step["scores_after"]) >= np.asarray(step["scores_before"])))

    def test_coordinate_search_does_not_claim_global_optimum_on_plateau(self):
        compiled = interaction_fixture()
        result = compiled.optimize(np.array([[0, 1, 0]]), sweeps=2)
        # Updating the middle slot first selects lowest-ID A on a flat tie;
        # then changing target alone cannot expose the missing A*B interaction.
        self.assertEqual(result.scores[0], 0)
        self.assertLess(result.scores[0], compiled.score(np.array([[0, 1, 2]]))[0])
        for step in result.trace:
            self.assertTrue(np.all(np.asarray(step["scores_after"]) >= np.asarray(step["scores_before"])))

    def test_batched_search_equals_each_individual_and_full_recomputation(self):
        compiled = polynomial.CompiledPolynomial(**coefficient_fixture(seed=41))
        rows = np.array([[1, 4, 7, 9], [4, 4, 7, 7], [9, 7, 4, 1]])
        result = compiled.optimize(rows, sweeps=3, chunk_size=2)
        for i in range(len(rows)):
            one = compiled.optimize(rows[i:i+1], sweeps=3, chunk_size=1)
            np.testing.assert_array_equal(result.token_ids[i], one.token_ids[0])
            self.assertAlmostEqual(result.scores[i], one.scores[0], places=12)
        np.testing.assert_allclose(result.scores, scalar_components(compiled, result.token_ids)["total"], atol=1e-12)
        for step in result.trace:
            self.assertTrue(np.all(np.asarray(step["scores_after"]) >= np.asarray(step["scores_before"])))

    def test_flat_ties_choose_lowest_logical_id_without_epsilon_grouping(self):
        data = coefficient_fixture()
        for key in ("direct", "features", "readout", "neuron_readout", "g7", "h7"):
            data[key] = np.zeros_like(data[key])
        compiled = polynomial.CompiledPolynomial(**data)
        result = compiled.optimize(np.array([[9, 9, 9, 9]]), sweeps=1)
        np.testing.assert_array_equal(result.token_ids, [[1, 1, 1, 1]])
        np.testing.assert_array_equal(result.scores, [0.])


class CompilationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = GPT2Config(vocab_size=5, padded_vocab_size=8, context_length=5,
                                 n_layers=8, d_model=4, n_heads=2, d_ff=6)
        rng = np.random.default_rng(117)
        self.values = {}
        for spec in tensor_manifest(self.config):
            values = rng.integers(-8, 9, size=spec.shape).astype(np.float32)/32
            if spec.name.endswith(".scale"):
                values += 1
            self.values[spec.name] = values
        self.checkpoint = self.write_checkpoint("initial", self.values)

    def write_checkpoint(self, name, values):
        directory = self.root / name
        directory.mkdir()
        for spec in tensor_manifest(self.config):
            (directory / spec.filename).write_bytes(np.asarray(values[spec.name], dtype="<f4").tobytes())
        return GPT2Checkpoint(directory, self.config, check_finite=True)

    def compile(self, checkpoint=None, **kwargs):
        return polynomial.compile_polynomial(checkpoint or self.checkpoint, [4, 0, 2], 4, **kwargs)

    def test_compile_dense_independent_matrix_formula_and_output_centering(self):
        compiled = self.compile()
        cp = self.checkpoint
        embedding = np.asarray(cp.token_embedding, dtype=np.float64)
        mean = embedding.mean(axis=0)
        ec = embedding - mean
        selected = ec[[0, 2, 4]]
        position = np.asarray(cp["position_embedding.weight"], dtype=np.float64)
        anchor7 = mean + position[:4].mean(axis=0)
        n7, j7, _ = scalar_norm(anchor7, cp["blocks.7.ln2.scale"], cp["blocks.7.ln2.bias"])
        w17, w27 = np.asarray(cp["blocks.7.mlp.input.weight"], dtype=np.float64), np.asarray(cp["blocks.7.mlp.output.weight"], dtype=np.float64)
        pre7 = n7 @ w17 + cp["blocks.7.mlp.input.bias"]
        g7, h7 = np.array([scalar_derivatives(float(x)) for x in pre7]).T
        k7 = j7 @ w17
        for p in range(4):
            n6, j6, _ = scalar_norm(mean+position[p], cp["blocks.6.ln2.scale"], cp["blocks.6.ln2.bias"])
            w16 = np.asarray(cp["blocks.6.mlp.input.weight"], dtype=np.float64)
            pre6 = n6 @ w16 + cp["blocks.6.mlp.input.bias"]
            g6 = np.array([scalar_derivatives(float(x))[0] for x in pre6])
            a6 = np.eye(4) + ((j6 @ w16)*g6) @ cp["blocks.6.mlp.output.weight"]
            np.testing.assert_allclose(compiled.direct[p], selected @ a6, atol=2e-13)
            np.testing.assert_allclose(compiled.features[p], selected @ a6 @ k7, atol=2e-13)
        center = np.eye(4)-np.ones((4, 4))/4
        expected_readout = (selected*cp["final_norm.scale"]) @ center
        np.testing.assert_allclose(compiled.readout, expected_readout, atol=2e-13)
        np.testing.assert_allclose(compiled.neuron_readout, expected_readout @ w27.T, atol=2e-13)
        np.testing.assert_allclose(compiled.g7, g7, atol=2e-13)
        np.testing.assert_allclose(compiled.h7, h7, atol=2e-13)
        np.testing.assert_array_equal(compiled.token_ids, [0, 2, 4])

    def test_affine_mode_removes_only_curvature(self):
        full, affine = self.compile(), self.compile(mode="affine")
        rows = np.array([[0, 2, 4, 0], [2, 4, 0, 2]])
        expected = full.components(rows)
        np.testing.assert_allclose(affine.score(rows), expected["direct"]+expected["affine"], atol=1e-13)
        np.testing.assert_array_equal(affine.h7, np.zeros(6))
        np.testing.assert_array_equal(affine.direct, full.direct)

    def test_no_cross_removes_earlier_feature_routing_not_direct(self):
        full, no_cross = self.compile(), self.compile(mode="no_cross")
        np.testing.assert_array_equal(no_cross.direct, full.direct)
        for p in range(1, 4):
            np.testing.assert_array_equal(no_cross.features[p], no_cross.features[0])
        self.assertGreater(np.linalg.norm(full.features-no_cross.features), 1e-5)

    def test_equal_position_maps_remove_compiled_mixed_source_asymmetry(self):
        values = {name: value.copy() for name, value in self.values.items()}
        values["position_embedding.weight"][:] = values["position_embedding.weight"][0]
        homogeneous = self.compile(self.write_checkpoint("same_positions", values))
        for position in range(1, 4):
            np.testing.assert_array_equal(homogeneous.features[position], homogeneous.features[0])

        def mixed(compiled, source, other, target):
            return np.sum(compiled.h7 * compiled.features[0, source]
                          * compiled.features[1, other] * compiled.neuron_readout[target])

        full = self.compile()
        differences = []
        for source, other, target in itertools.product(range(3), repeat=3):
            self.assertAlmostEqual(mixed(homogeneous, source, other, target),
                                   mixed(homogeneous, other, source, target), delta=1e-15)
            differences.append(mixed(full, source, other, target)-mixed(full, other, source, target))
        self.assertGreater(max(abs(x) for x in differences), 1e-9)
        # This is a null for the mixed-source coefficient only. The complete
        # phrase objective still distinguishes source and target positions.

    def test_broken_matches_explicit_decoder_row_roll(self):
        values = {name: value.copy() for name, value in self.values.items()}
        key = "blocks.6.mlp.output.weight"
        values[key] = np.roll(values[key], 1, axis=0)
        explicit = self.compile(self.write_checkpoint("broken", values))
        broken = self.compile(mode="broken")
        for field in ("direct", "features", "readout", "neuron_readout", "g7", "h7"):
            np.testing.assert_allclose(getattr(broken, field), getattr(explicit, field), atol=1e-13)

    def test_consistent_hidden_neuron_permutation_preserves_scores(self):
        permutation = np.array([3, 0, 5, 2, 1, 4])
        baseline = self.compile()
        rows = np.array([[0, 2, 4, 0], [4, 4, 2, 2]])
        for block in (6, 7):
            values = {name: value.copy() for name, value in self.values.items()}
            prefix = f"blocks.{block}.mlp."
            values[prefix+"input.weight"] = values[prefix+"input.weight"][:, permutation]
            values[prefix+"input.bias"] = values[prefix+"input.bias"][permutation]
            values[prefix+"output.weight"] = values[prefix+"output.weight"][permutation]
            actual = self.compile(self.write_checkpoint(f"permuted{block}", values))
            np.testing.assert_allclose(actual.score(rows), baseline.score(rows), atol=2e-12, rtol=2e-12)

    def test_embedding_position_shift_gauge_preserves_compiled_polynomial(self):
        values = {name: value.copy() for name, value in self.values.items()}
        shift = np.array([.25, -.5, .125, -.25], dtype=np.float32)
        values["token_embedding.weight"] += shift
        values["position_embedding.weight"] -= shift
        before, after = self.compile(), self.compile(self.write_checkpoint("gauge", values))
        for field in ("direct", "features", "readout", "neuron_readout", "g7", "h7"):
            np.testing.assert_allclose(getattr(before, field), getattr(after, field), atol=2e-12, rtol=2e-12)

    def test_padding_rows_and_selected_subset_do_not_change_centering(self):
        before = self.compile()
        values = {name: value.copy() for name, value in self.values.items()}
        values["token_embedding.weight"][5:] = 1000
        after = self.compile(self.write_checkpoint("padding", values))
        np.testing.assert_array_equal(before.readout, after.readout)
        full = polynomial.compile_polynomial(self.checkpoint, list(range(5)), 4)
        np.testing.assert_allclose(before.readout, full.readout[[0, 2, 4]], atol=1e-14)
        np.testing.assert_allclose(before.direct, full.direct[:, [0, 2, 4]], atol=1e-14)

    def test_invalid_compile_mode_tokens_length_and_finite_weights(self):
        for ids in ([-1, 1], [0, 5], [True, 1], [0., 2.]):
            with self.assertRaises(ValueError):
                polynomial.compile_polynomial(self.checkpoint, ids, 4)
        for kwargs in ({"mode": "unknown"}, {"alpha": np.inf}):
            with self.assertRaises(ValueError): self.compile(**kwargs)
        with self.assertRaises(ValueError):
            polynomial.compile_polynomial(self.checkpoint, [0, 2], 6)
        # Corrupt only this temporary checkpoint after its initial load, so
        # this checks compile-time finite validation rather than loader checks.
        name = "blocks.6.mlp.input.weight"
        spec = next(spec for spec in self.checkpoint.manifest if spec.name == name)
        bad = self.values[name].copy(); bad[0, 0] = np.nan
        (self.checkpoint.directory / spec.filename).write_bytes(bad.astype("<f4").tobytes())
        with self.assertRaises(ValueError): self.compile()


if __name__ == "__main__":
    unittest.main()
