"""Synthetic ground-truth checks for the static weight-only extractor."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from .mlp import (build_edges, decode_paths, main, project_directions,
                  stable_topk, unit_address, unit_rows)


class StaticMlpTest(unittest.TestCase):
    def setUp(self):
        # The desired directed chain is literally written into paired basis
        # vectors: key=e_i, value=e_(i+1). No training examples or forward pass
        # are needed to read it. This establishes an identifiable positive
        # control, not that a real transformer has this same simple structure.
        self.embedding = np.eye(5, dtype=np.float32)
        self.keys = self.embedding[:4].copy()
        self.values = self.embedding[1:].copy()

    def edges(self, keys=None, values=None):
        ids, scores = project_directions(self.embedding,
            self.keys if keys is None else keys, 2, chunk_size=2)
        out_ids, out_scores = project_directions(self.embedding,
            self.values if values is None else values, 2, chunk_size=2)
        return build_edges(ids, scores, out_ids, out_scores,
                           [{"unit": i} for i in range(len(ids))])

    def test_extracts_known_directed_chain_not_ranked_token_bag(self):
        edges = self.edges()
        self.assertEqual(set(edges), {(0, 1), (1, 2), (2, 3), (3, 4)})
        paths = list(decode_paths(edges, length=5, starts=1))
        self.assertEqual(paths[0]["token_ids"], [0, 1, 2, 3, 4])
        self.assertEqual(paths[0]["score"], 1)
        self.assertEqual(paths[0]["provenance"]["edges"],
                         [{"unit": i} for i in range(4)])

    def test_cosine_scores_invariant_to_positive_vector_scales(self):
        ids, scores = project_directions(self.embedding, self.keys, 2)
        ids2, scores2 = project_directions(self.embedding * 17, self.keys * 3, 2)
        np.testing.assert_array_equal(ids, ids2)
        np.testing.assert_allclose(scores, scores2)

    def test_paired_neuron_permutation_preserves_associations(self):
        permutation = [2, 0, 3, 1]
        before = self.edges()
        after = self.edges(self.keys[permutation], self.values[permutation])
        self.assertEqual({k: v["score"] for k, v in before.items()},
                         {k: v["score"] for k, v in after.items()})
        for pair, edge in after.items():
            self.assertEqual(permutation[edge["provenance"]["unit"]],
                             before[pair]["provenance"]["unit"])

    def test_broken_pairing_preserves_marginals_but_changes_chain(self):
        broken = self.edges(values=np.roll(self.values, 1, axis=0))
        self.assertEqual({source for source, _ in broken}, {0, 1, 2, 3})
        self.assertEqual({target for _, target in broken}, {1, 2, 3, 4})
        self.assertFalse(any(path["token_ids"] == [0, 1, 2, 3, 4]
                             for path in decode_paths(broken, length=5)))

    def test_zero_and_negative_alignment_do_not_create_edges(self):
        edges = build_edges(np.array([[0, 1]]), np.array([[-1., 0.]]),
                            np.array([[2, 3]]), np.array([[-1., 0.]]), [{}])
        self.assertEqual(edges, {})
        np.testing.assert_array_equal(unit_rows(np.zeros((2, 3))), np.zeros((2, 3)))

    def test_stable_topk_resolves_boundary_ties_by_token_id(self):
        ids, scores = stable_topk(np.array([[1., 2., 2., 2., 0.]]), 2)
        self.assertEqual(ids.tolist(), [[1, 2]])
        self.assertEqual(scores.tolist(), [[2., 2.]])

    def test_chunk_size_does_not_change_results(self):
        for size in (1, 2, 19):
            ids, scores = project_directions(self.embedding, self.keys, 2, size)
            self.assertEqual(ids[:, 0].tolist(), [0, 1, 2, 3])
            self.assertEqual(scores[:, 0].tolist(), [1.] * 4)

    def test_invalid_scores_and_limits_rejected(self):
        for values in (np.array([[np.nan]]), np.array([[np.inf]])):
            with self.assertRaises(ValueError):
                stable_topk(values, 1)
            with self.assertRaises(ValueError):
                unit_rows(values)
        for k in (0, 6):
            with self.assertRaises(ValueError):
                stable_topk(self.embedding, k)
        for kwargs in ({"length": 1}, {"starts": -1}, {"beam_width": 0}):
            with self.assertRaises(ValueError):
                list(decode_paths(self.edges(), **kwargs))

    def test_self_edges_cannot_become_fake_repeated_passage(self):
        self.assertEqual(list(decode_paths({(1, 1): {"score": 1, "provenance": {}}})), [])

    def test_provenance_reconstructs_physical_key_and_value_views(self):
        config = GPT2Config(vocab_size=5, padded_vocab_size=5, context_length=3,
                            n_layers=2, d_model=5, n_heads=1, d_ff=4)
        specs = {spec.name: spec for spec in tensor_manifest(config)}
        for block in range(2):
            address = unit_address(block, 2, 3, config.d_model, config.d_ff)
            for side, coordinates in (("key", [(i, 2) for i in range(5)]),
                                      ("value", [(3, i) for i in range(5)])):
                view = address[side]
                spec = specs[view["tensor"]]
                self.assertEqual(view["file"], spec.filename)
                actual = [view["offset_bytes"] + i * view["stride_bytes"]
                          for i in range(view["count"])]
                expected = [spec.element_byte_range(*index)[0] for index in coordinates]
                self.assertEqual(actual, expected)

    def test_cli_artifacts_are_reproducible_without_any_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "step_1"
            weights.mkdir()
            config = GPT2Config(vocab_size=5, padded_vocab_size=5, context_length=3,
                                n_layers=1, d_model=5, n_heads=1, d_ff=4)
            for spec in tensor_manifest(config):
                values = np.zeros(spec.shape, dtype="<f4")
                if spec.name == "token_embedding.weight":
                    values[:] = self.embedding
                elif spec.name == "blocks.0.mlp.input.weight":
                    values[:] = self.keys.T
                elif spec.name == "blocks.0.mlp.output.weight":
                    values[:] = self.values
                values.tofile(weights / spec.filename)
            (root / "tokenizer.json").write_text(json.dumps(
                {"model": {"vocab": {str(i): i for i in range(5)}}}))
            checkpoint = GPT2Checkpoint(weights, config, check_finite=True)
            output = root / "candidates.jsonl"
            args = ["--checkpoint", str(weights), "--tokenizer-dir", str(root),
                    "--output", str(output), "--top-k", "2", "--path-length", "5",
                    "--path-starts", "1", "--include-control"]
            with mock.patch("scripts.weight_analysis.mlp.GPT2Checkpoint", return_value=checkpoint), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main(args)
                with self.assertRaises(FileExistsError):
                    main(args)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            real_path = next(r for r in records if r["method"] == "mlp_cosine_path")
            self.assertEqual(real_path["token_ids"], [0, 1, 2, 3, 4])
            metadata = json.loads(output.with_suffix(".jsonl.metadata.json").read_text())
            self.assertEqual(metadata["stage"], "corpus_blind_extraction")
            self.assertEqual(metadata["method_counts"]["mlp_cosine_pair"], 4)
            self.assertNotIn("corpus", metadata["parameters"])
            # Test pair emission as well as graph construction: zero directions
            # must not be reported as arbitrary token-zero associations.
            zero_output = root / "zero.jsonl"
            zero_args = list(args)
            zero_args[zero_args.index("--output") + 1] = str(zero_output)
            with mock.patch("scripts.weight_analysis.mlp.GPT2Checkpoint", return_value=checkpoint), \
                    mock.patch.object(checkpoint, "mlp_keys", return_value=np.zeros((4, 5))), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main(zero_args)
            self.assertEqual(zero_output.read_text(), "")


if __name__ == "__main__":
    unittest.main()
