"""Closed-coordinate probe tests with synthetic weights and no corpus inputs."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, tensor_manifest
from .closed_trigram import (ClosedTrigramProbe, RECIPE_EPSILON,
                             geometry_provenance, main, normalized_inputs,
                             select_vocabulary, validate_recipe_epsilon)
from .trigram import TrigramProbe, decode_trigram_paths


class ClosedTrigramTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=3,
                                 n_layers=1, d_model=4, n_heads=2, d_ff=8)
        self.final = self.checkpoint('final', 17)
        self.early = self.checkpoint('early', 88)
        (self.root / 'tokenizer.json').write_text(json.dumps(
            {'model': {'vocab': {str(i): i for i in range(7)}}}))

    def checkpoint(self, name, seed):
        random = np.random.default_rng(seed)
        directory = self.root / name
        directory.mkdir()
        for spec in tensor_manifest(self.config):
            values = random.normal(size=spec.shape).astype('<f4')
            values.tofile(directory / spec.filename)
        return GPT2Checkpoint(directory, self.config, check_finite=True)

    def test_normalized_coordinates_match_scalar_population_layernorm(self):
        for position in (0, 1):
            actual = normalized_inputs(self.final, position, np.array([1, 5]))
            expected = []
            for token in (1, 5):
                row = [float(a) + float(b) for a, b in zip(
                    self.final.token_embedding[token], self.final['position_embedding.weight'][position])]
                mean = sum(row) / 4
                variance = sum((value - mean) ** 2 for value in row) / 4
                expected.append([(value - mean) / np.sqrt(variance + RECIPE_EPSILON) * float(gamma) + float(beta)
                                 for value, gamma, beta in zip(row, self.final['blocks.0.ln1.scale'],
                                                                self.final['blocks.0.ln1.bias'])])
            self.assertEqual(actual.dtype, np.float64)
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        self.assertFalse(np.allclose(normalized_inputs(self.final, 0), normalized_inputs(self.final, 1)))
        with self.assertRaises(ValueError):
            normalized_inputs(self.final, 3)

    def test_epsilon_validation_rejects_source_drift(self):
        source = self.root / 'src/llm/recipes/gpt2.cc'
        source.parent.mkdir(parents=True)
        source.write_text('constexpr float kLayerNormEpsilon = 1e-5f;\n')
        metadata = validate_recipe_epsilon(self.root)
        self.assertEqual(metadata['fp64_promoted_value'], float(np.float32(1e-5)))
        self.assertEqual(metadata['cpp_float_literal'], '1e-5f')
        source.write_text('constexpr float kLayerNormEpsilon = 2e-5f;\n')
        with self.assertRaises(ValueError):
            validate_recipe_epsilon(self.root)

    def test_vocabulary_selection_matches_full_dense_normalized_query_norm(self):
        query = np.concatenate([self.final.qkv(0, 'q', head) for head in range(2)], axis=1).astype(np.float64)
        bias = np.concatenate([self.final.qkv_bias(0, 'q', head) for head in range(2)]).astype(np.float64)
        projected = normalized_inputs(self.final, 1) @ query + bias
        norms = np.sum(projected * projected, axis=1)
        ranked = sorted(range(7), key=lambda token: (-norms[token], token))[:5]
        for chunk in (1, 3, 20):
            selection = select_vocabulary(self.final, size=5, chunk_size=chunk)
            self.assertEqual(selection.ranked_ids.tolist(), ranked)
            self.assertEqual(selection.ids.tolist(), sorted(ranked))
            np.testing.assert_allclose(selection.ranked_norm_squared, norms[ranked])

    def test_all_ordered_pair_derivatives_match_dense_corrected_geometry(self):
        ids = np.array([0, 3, 6])
        for geometry in ('normalized', 'raw'):
            for pairing in ([0, 1], [1, 0]):
                probe = ClosedTrigramProbe(self.final, ids, geometry, pairing)
                x0, x1 = probe.inputs0, probe.inputs1
                for current in range(3):
                    expected = []
                    for previous in range(3):
                        difference = x0[previous] - x1[current]
                        write = np.zeros(4)
                        for route, content in enumerate(pairing):
                            query = x1[current] @ self.final.qkv(0, 'q', route).astype(np.float64)
                            query += self.final.qkv_bias(0, 'q', route).astype(np.float64)
                            key_difference = difference @ self.final.qkv(0, 'k', route).astype(np.float64)
                            values = difference @ self.final.qkv(0, 'v', content).astype(np.float64)
                            write += (query @ key_difference) * (values @ self.final.attention_output_head(0, content)) / (4 * np.sqrt(2))
                        expected.append(write)
                    np.testing.assert_allclose(probe.interaction_writes(current), expected, rtol=1e-12, atol=1e-12)

    def test_raw_geometry_exactly_matches_old_probe_restricted_to_same_ids(self):
        ids = np.array([0, 1, 4, 6])
        closed = ClosedTrigramProbe(self.final, ids, 'raw')
        q, k, v = [np.stack([self.final.qkv(0, kind, head) for head in range(2)]).astype(np.float64)
                   for kind in ('q', 'k', 'v')]
        output = np.stack([self.final.attention_output_head(0, head) for head in range(2)]).astype(np.float64)
        bias = np.stack([self.final.qkv_bias(0, 'q', head) for head in range(2)]).astype(np.float64)
        old = TrigramProbe(self.final.token_embedding.astype(np.float64), q, k, v, output, bias)
        for index, token in enumerate(ids):
            np.testing.assert_allclose(closed.interaction_writes(index), old.interaction_writes(int(token), ids), atol=1e-12)

    def test_raw_self_pairs_zero_normalized_same_token_different_positions_not_zero(self):
        ids = np.array([0, 1, 4, 6])
        raw = ClosedTrigramProbe(self.final, ids, 'raw')
        normalized = ClosedTrigramProbe(self.final, ids)
        for index in range(4):
            np.testing.assert_array_equal(raw.interaction_writes(index)[index], np.zeros(4))
            self.assertGreater(np.linalg.norm(normalized.interaction_writes(index)[index]), 0)
        raw_records, raw_stats = raw.extract(3)
        normal_records, normal_stats = normalized.extract(3)
        self.assertEqual(raw_stats['attempted_ordered_pairs'], 16)
        self.assertEqual(raw_stats['zero_write_pairs'], 4)
        self.assertEqual(len(raw_records), 12 * 3)
        self.assertEqual(normal_stats['zero_write_pairs'], 0)
        self.assertEqual(len(normal_records), 16 * 3)

    def test_destination_ranking_uses_original_output_embedding_not_normalized_input(self):
        ids = np.array([0, 1, 4, 6])
        probe = ClosedTrigramProbe(self.final, ids)
        records, _ = probe.extract(4)
        for record in records:
            previous, current, destination = [ids.tolist().index(t) for t in record['token_ids']]
            write = probe.interaction_writes(current)[previous]
            target = self.final.token_embedding[ids[destination]].astype(np.float64)
            score = (write @ target) / (np.linalg.norm(write) * np.linalg.norm(target))
            self.assertAlmostEqual(record['score'], score, places=12)
        self.assertFalse(np.allclose(probe.embedding, probe.inputs1))

    def test_closed_graph_has_real_overlaps_and_paths_only_use_recorded_triples(self):
        probe = ClosedTrigramProbe(self.final, np.arange(7))
        records, _ = probe.extract(4)
        for index, record in enumerate(records):
            record['candidate_id'] = str(index)
        source = {tuple(record['token_ids'][:2]) for record in records}
        target = {tuple(record['token_ids'][1:]) for record in records}
        self.assertEqual(source, {(a, b) for a in range(7) for b in range(7)})
        self.assertTrue(target.issubset(source))
        paths = list(decode_trigram_paths(records, length=12, starts=10))
        self.assertTrue(paths)
        edges = {tuple(record['token_ids']) for record in records}
        for path in paths:
            self.assertGreaterEqual(len(path['token_ids']), 4)
            for index in range(len(path['token_ids']) - 2):
                self.assertIn(tuple(path['token_ids'][index:index + 3]), edges)

    def test_top_destinations_keep_finite_negative_scores(self):
        probe = ClosedTrigramProbe(self.final, np.array([0, 1, 4]))
        probe.embedding = np.tile([-1., 0., 0., 0.], (3, 1))
        probe.target_norms = np.ones(3)
        with mock.patch.object(probe, 'interaction_writes', return_value=np.tile([1., 0., 0., 0.], (3, 1))):
            records, _ = probe.extract(2)
        self.assertEqual(len(records), 18)
        self.assertTrue(all(record['score'] == -1 for record in records))
        self.assertEqual([record['token_ids'][2] for record in records[:2]], [0, 1])

    def test_geometry_provenance_byte_ranges(self):
        ids = np.array([0, 3, 6])
        provenance = geometry_provenance(self.final, ids, 'normalized')
        self.assertEqual(provenance['embedding']['row_byte_ranges'], [[0, 16], [48, 64], [96, 112]])
        self.assertEqual(provenance['positions']['filename'], 'weight_1.bin')
        self.assertEqual(provenance['positions']['row_byte_ranges'], [[0, 16], [16, 32]])
        self.assertEqual(provenance['layer_norm']['scale']['filename'], 'weight_2.bin')
        self.assertEqual(provenance['layer_norm']['bias']['filename'], 'weight_3.bin')

    def test_invalid_geometry_and_selection_rejected(self):
        for ids in ([], [1, 0], [0, 0], [-1, 2], [0, 7], [0.0, 1.0]):
            with self.assertRaises(ValueError):
                ClosedTrigramProbe(self.final, ids)
        with self.assertRaises(ValueError):
            ClosedTrigramProbe(self.final, [0, 1], 'bad')
        with self.assertRaises(ValueError):
            select_vocabulary(self.final, size=0)

    def test_cli_all_four_arms_share_final_selected_vocabulary_and_refuse_overwrite(self):
        expected = select_vocabulary(self.final, 5).ids.tolist()
        for arm, checkpoint, geometry, broken in (
                ('normalized', self.final, 'normalized', False),
                ('raw', self.final, 'raw', False),
                ('broken', self.final, 'normalized', True),
                ('early', self.early, 'normalized', False)):
            output = self.root / (arm + '.jsonl')
            args = ['--checkpoint', str(checkpoint.directory),
                    '--vocabulary-checkpoint', str(self.final.directory),
                    '--tokenizer-dir', str(self.root), '--output', str(output),
                    '--input-geometry', geometry, '--vocabulary-size', '5', '--top-k', '4',
                    '--path-length', '6', '--path-starts', '8']
            if broken:
                args.append('--broken-routing-control')
            def load(path, **kwargs):
                return self.final if Path(path) == self.final.directory else self.early
            with mock.patch('scripts.weight_analysis.closed_trigram.GPT2Checkpoint', side_effect=load), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(args)
                before = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    main(args)
                self.assertEqual(output.read_bytes(), before)
            metadata = json.loads(output.with_suffix('.jsonl.metadata.json').read_text())
            self.assertEqual(metadata['vocabulary_selection']['ids_ascending'], expected)
            self.assertEqual(metadata['vocabulary_selection']['selection_checkpoint_differs_from_evaluation'], arm == 'early')
            self.assertEqual(metadata['vocabulary_selection_checkpoint']['checkpoint_directory'], str(self.final.directory))
            self.assertEqual(metadata['diagnostics']['attempted_ordered_pairs'], 25)
            self.assertEqual(sum(metadata['diagnostics']['candidate_length_histogram'].values()),
                             sum(metadata['method_counts'].values()))
            self.assertEqual(len(metadata['protocol_manifest']['sha256']), 64)
            self.assertGreater(metadata['diagnostics']['source_target_pair_intersection'], 0)
            self.assertNotIn('corpus', metadata['parameters'])
            output.unlink()
            with self.assertRaises(FileExistsError):
                main(args)


if __name__ == '__main__':
    unittest.main()
