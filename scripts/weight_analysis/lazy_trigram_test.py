"""Synthetic selector, cached-graph, numerical, and provenance regression tests.

The independent scalar attention/softmax oracle lives in
lazy_polynomial_oracle_test.py. These tests focus on the surrounding weight-only
search, and use planted WRITE VECTORS only for testing the graph decoder.
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest
from .lazy_trigram import (LazyTrigramProbe, decode_lazy_paths, load_centroid_selection,
                           main, stable_norm)
from .vocabulary import extract


class LazyTrigramTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=3,
                                 n_layers=1, d_model=4, n_heads=2, d_ff=8)
        self.final = self.make_checkpoint('final', 17)
        self.early = self.make_checkpoint('early', 29)
        self.artifact = extract(self.final, self.early)
        self.rankings = self.root / 'rankings.json'
        self.rankings.write_text(json.dumps(self.artifact))

    def make_checkpoint(self, name, seed):
        directory = self.root / name
        directory.mkdir()
        random = np.random.default_rng(seed)
        for spec in tensor_manifest(self.config):
            values = random.normal(size=spec.shape).astype('<f4')
            if spec.name == 'token_embedding.weight':
                values[-1] = 1e10  # Physical padding must not enter any centroid.
            values.tofile(directory / spec.filename)
        return GPT2Checkpoint(directory, config=self.config, check_finite=True)

    @staticmethod
    def planted_components(probe, pairs, choose):
        """Explicit vector oracle for graph tests, NOT a model formula oracle."""
        writes = np.stack([np.asarray(choose(*pair), dtype=np.float64) for pair in pairs])
        norm = stable_norm(writes)
        return {'uniform': writes, 'derivative': np.zeros_like(writes), 'chosen': writes,
                'gaps': np.zeros((len(pairs), probe.heads)),
                'projected_value_difference_norms': np.zeros((len(pairs), probe.heads)),
                'uniform_norms': norm, 'derivative_norms': np.zeros(len(pairs)),
                'chosen_norms': norm, 'head_norm_sum_plus_global_bias': norm}

    def test_stable_norm_preserves_tiny_directions_and_avoids_squaring_overflow(self):
        np.testing.assert_allclose(stable_norm([[3., 4.], [3e-300, 4e-300], [3e300, 4e300]]),
                                   [5., 5e-300, 5e300], rtol=1e-15, atol=0.)
        self.assertEqual(float(stable_norm([0., 0.])), 0.)
        self.assertEqual(float(stable_norm([np.nextafter(0., 1.), 0.])), np.nextafter(0., 1.))
        for values in ([], [np.inf], [np.nan], [1.7e308, 1.7e308]):
            with self.assertRaises(ValueError):
                stable_norm(values)

    def test_selector_recomputes_full_centroid_and_ranked_seeds_for_both_checkpoints(self):
        table = self.final.token_embedding.astype(np.float64)
        center = table.mean(axis=0)
        distances = np.sqrt(np.sum((table - center) ** 2, axis=1))
        expected = sorted(range(7), key=lambda t: (-distances[t], t))
        for checkpoint in (self.final, self.early):
            selection = load_centroid_selection(self.rankings, checkpoint, 5, 3)
            self.assertEqual(selection.ranked_ids.tolist(), expected[:5])
            self.assertEqual(selection.ids.tolist(), sorted(expected[:5]))
            self.assertEqual(selection.seed_ids.tolist(), expected[:3])
            self.assertEqual(selection.checkpoint.directory, self.final.directory)
            self.assertEqual(selection.rankings_sha256, sha256_file(self.rankings))
            self.assertFalse(selection.seed_ids.flags.writeable)

    def test_selector_rejects_bad_rankings_hashes_and_budgets(self):
        for size, seeds in [(0, 1), (8, 1), (5, 6), (True, 1), (5, 0)]:
            with self.assertRaises(ValueError):
                load_centroid_selection(self.rankings, self.final, size, seeds)
        corruptions = [
            lambda a: a.update(schema_version=True),
            lambda a: a['checkpoint']['final']['weight_sha256'].update({'weight_0.bin': 'bad'}),
            lambda a: a['checkpoint']['final']['config'].update(d_model=8),
            lambda a: a['protocol'].update(sha256='bad'),
            lambda a: next(e for e in a['rankings'] if e['method'] == 'final:centroid_distance')['scores'].__setitem__(0, 1e20),
            lambda a: next(e for e in a['rankings'] if e['method'] == 'final:centroid_distance')['token_ids'].reverse(),
        ]
        for change in corruptions:
            artifact = copy.deepcopy(self.artifact)
            change(artifact)
            path = self.root / 'changed.json'
            path.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):
                load_centroid_selection(path, self.final, 5, 3)

    def test_center_is_full_logical_vocabulary_not_subset_or_padding(self):
        ids = np.array([0, 2, 5])
        probe = LazyTrigramProbe(self.final, ids)
        expected = self.final.token_embedding.astype(np.float64).mean(axis=0)
        np.testing.assert_array_equal(probe.output_mean, expected)
        np.testing.assert_array_equal(probe.centered_targets, self.final.token_embedding[ids] - expected)
        self.assertFalse(np.allclose(expected, self.final.token_embedding[ids].mean(axis=0)))

    def test_lazy_cache_evaluates_only_unique_requested_pairs(self):
        probe = LazyTrigramProbe(self.final, [0, 2, 5])
        self.assertEqual(probe.write_evaluations, 0)
        with mock.patch.object(probe, 'components_for_pairs', wraps=probe.components_for_pairs) as calculate:
            probe.ensure_pairs([(0, 2), (0, 2), (5, 0)])
            first = probe.edges(0, 2)
            self.assertIs(first, probe.edges(0, 2))
            probe.ensure_pairs([(2, 5), (5, 0)])
        self.assertEqual(calculate.call_count, 2)
        self.assertEqual(probe.write_evaluations, 3)
        self.assertEqual(set(probe.cache), {(0, 2), (5, 0), (2, 5)})
        self.assertEqual(len(probe.edge_records()), 9)  # S has only3 destinations.
        self.assertLess(len(probe.cache), 3 ** 2)
        summary = probe.summary()
        self.assertEqual(summary['visited_routing_bounds']['stacked_heads']['head_contexts'], 6)

    def test_scores_match_scalar_centered_raw_oracle_and_preserve_rank(self):
        for component in ('uniform', 'derivative', 'first_order'):
            probe = LazyTrigramProbe(self.final, np.arange(7), component)
            for pair in ((0, 0), (2, 4)):
                write = probe.components_for_pair(*pair)['chosen']
                expected = [sum(float(a) * float(b) for a, b in zip(write, target)) / np.linalg.norm(write)
                            for target in probe.centered_targets]
                order = sorted(range(7), key=lambda t: (-expected[t], t))[:4]
                edges = probe.edges(*pair)
                self.assertEqual([edge['token_ids'][2] for edge in edges], order)
                np.testing.assert_allclose([edge['score'] for edge in edges], np.array(expected)[order], rtol=1e-13, atol=1e-13)
                raw_order = np.argsort(-(self.final.token_embedding @ write)).tolist()[:4]
                self.assertEqual(order, raw_order)

    def test_tiny_nonzero_write_is_scored_before_raw_product_underflow(self):
        probe = LazyTrigramProbe(self.final, [0, 1, 2], 'derivative')
        probe.centered_targets = np.array([[1e-100, 0, 0, 0], [2e-100, 0, 0, 0], [-1e-100, 0, 0, 0]])
        choose = lambda previous, current: [1e-300, 0, 0, 0]
        with mock.patch.object(probe, 'components_for_pairs', side_effect=lambda pairs: self.planted_components(probe, pairs, choose)):
            edges = probe.edges(0, 0)
        self.assertEqual(edges[0]['token_ids'], [0, 0, 1])
        self.assertEqual(edges[0]['centered_dot_product'], 0.)
        self.assertAlmostEqual(edges[0]['score'] / 1e-100, 2.)
        self.assertEqual(edges[0]['write_norm'], 1e-300)

    def test_planted_chain_and_repeated_tokens_survive_without_filters(self):
        probe = LazyTrigramProbe(self.final, np.arange(4))
        probe.centered_targets = np.eye(4) - .25
        choose = lambda previous, current: np.eye(4)[(current + 1) % 4]
        with mock.patch.object(probe, 'components_for_pairs', side_effect=lambda pairs: self.planted_components(probe, pairs, choose)):
            paths, stats = decode_lazy_paths(probe, [0], length=10, beam_width=4)
        self.assertEqual(len(paths), 4)
        self.assertEqual(paths[0]['token_ids'], [0, 0, 1, 2, 3, 0, 1, 2, 3, 0])
        self.assertGreater(stats['paths_with_adjacent_equal_tokens'], 0)
        self.assertGreater(stats['paths_with_any_token_more_than_twice'], 0)
        self.assertEqual(stats['early_terminated_paths'], 0)
        self.assertLessEqual(probe.write_evaluations, stats['maximum_pair_evaluations_without_cache_reuse'])

    def test_zero_direction_terminates_without_id_tie_invented_edges(self):
        probe = LazyTrigramProbe(self.final, [0, 1])
        with mock.patch.object(probe, 'components_for_pairs', side_effect=lambda pairs: self.planted_components(probe, pairs, lambda p, c: np.zeros(4))):
            paths, stats = decode_lazy_paths(probe, [0, 1], length=6)
        self.assertEqual(paths, [])
        self.assertEqual(stats['ordered_seed_pairs'], 4)
        self.assertEqual(stats['seed_pairs_without_extension'], 4)
        self.assertEqual(probe.edge_records(), [])
        self.assertEqual(probe.summary()['zero_write_pairs'], 4)

    def test_negative_scores_and_zero_centered_targets_are_valid(self):
        probe = LazyTrigramProbe(self.final, [0, 1, 2], 'derivative')
        probe.centered_targets = np.array([[-2., 0, 0, 0], [0., 0, 0, 0], [-1., 0, 0, 0]])
        with mock.patch.object(probe, 'components_for_pairs', side_effect=lambda pairs: self.planted_components(probe, pairs, lambda p, c: [1., 0, 0, 0])):
            edges = probe.edges(0, 0)
        self.assertEqual([edge['token_ids'][2] for edge in edges], [1, 2, 0])
        self.assertEqual([edge['score'] for edge in edges], [0., -1., -2.])

    def test_every_generated_path_replays_cached_edges_and_score_denominator(self):
        probe = LazyTrigramProbe(self.final, np.arange(7))
        paths, stats = decode_lazy_paths(probe, [2, 4], length=8)
        self.assertEqual(stats['ordered_seed_pairs'], 4)
        self.assertEqual(len(paths), 16)
        edges = {edge['candidate_id']: edge for edge in probe.edge_records()}
        for path in paths:
            references = path['provenance']['edge_candidate_ids']
            self.assertEqual(len(references), len(path['token_ids']) - 2)
            for index, edge_id in enumerate(references):
                self.assertEqual(edges[edge_id]['token_ids'], path['token_ids'][index:index + 3])
            self.assertAlmostEqual(path['score'], sum(edges[edge_id]['score'] for edge_id in references) / len(references))
        self.assertLessEqual(probe.write_evaluations, stats['maximum_pair_evaluations_without_cache_reuse'])

    def test_invalid_ids_components_and_search_budgets(self):
        for ids in ([], [2, 0], [0, 0], [-1, 1], [0., 1.], [0, 7]):
            with self.assertRaises(ValueError):
                LazyTrigramProbe(self.final, ids)
        for kwargs in ({'component': 'bad'}, {'pairing': [0, 0]}, {'pairing': [0., 1.]}):
            with self.assertRaises(ValueError):
                LazyTrigramProbe(self.final, [0, 1], **kwargs)
        probe = LazyTrigramProbe(self.final, [0, 1])
        for pair in ((0, 2), (True, 1), (0., 1)):
            with self.assertRaises(ValueError):
                probe.edges(*pair)
        for seeds, length in (([], 5), ([0, 0], 5), ([0.], 5), ([2], 5), ([0], 2)):
            with self.assertRaises(ValueError):
                decode_lazy_paths(probe, seeds, length)

    def test_five_cli_arms_share_selection_and_emit_physical_provenance(self):
        choices = [('final_first_order', self.final, 'first_order', False),
                   ('final_derivative', self.final, 'derivative', False),
                   ('final_uniform', self.final, 'uniform', False),
                   ('broken_first_order', self.final, 'first_order', True),
                   ('early_first_order', self.early, 'first_order', False)]
        selections = []
        for name, checkpoint, component, broken in choices:
            output = self.root / (name + '.jsonl')
            args = ['--checkpoint', str(checkpoint.directory), '--rankings', str(self.rankings),
                    '--output', str(output), '--component', component, '--vocabulary-size', '6',
                    '--seed-count', '2', '--path-length', '6']
            if broken:
                args.append('--broken-routing-control')
            def load(path, **kwargs):
                return self.final if Path(path).resolve() == self.final.directory else self.early
            with mock.patch('scripts.weight_analysis.lazy_trigram.GPT2Checkpoint', side_effect=load), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(args)
            metadata = json.loads(output.with_suffix('.jsonl.metadata.json').read_text())
            selections.append((metadata['vocabulary_selection']['ids_ascending'], metadata['vocabulary_selection']['seed_ids_by_rank']))
            self.assertEqual(metadata['vocabulary_selection']['selection_checkpoint_differs_from_evaluation'], name == 'early_first_order')
            self.assertEqual(metadata['candidate_sha256'], sha256_file(output))
            self.assertEqual(metadata['rankings']['sha256'], sha256_file(self.rankings))
            self.assertNotIn('corpus', metadata['parameters'])
            self.assertNotIn('tokenizer', metadata['parameters'])
            self.assertEqual(metadata['output_centering']['logical_byte_range'], [0, 7 * 4 * 4])
            self.assertEqual(metadata['output_centering']['embedding_shared_with'], ['lm_head.weight'])
            routing = metadata['routing_weight_provenance']
            self.assertEqual(routing['global_output_bias']['filename'], 'weight_7.bin')
            self.assertEqual(routing['global_output_bias']['multiplicity'], 1)
            self.assertEqual(routing['head_slices'][0]['value_bias']['byte_range'], [40, 48] if broken else [32, 40])
            self.assertEqual(routing['head_slices'][0]['value_bias']['used'], component != 'derivative')
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                main(args)
            self.assertEqual(output.read_bytes(), before)
        self.assertTrue(all(item == selections[0] for item in selections))

    def test_dangling_output_and_existing_sidecar_never_overwritten(self):
        output = self.root / 'blocked.jsonl'
        args = ['--checkpoint', '/missing', '--rankings', '/missing', '--output', str(output)]
        sidecar = output.with_suffix('.jsonl.metadata.json')
        sidecar.write_text('preserve')
        with self.assertRaises(FileExistsError):
            main(args)
        self.assertEqual(sidecar.read_text(), 'preserve')
        self.assertFalse(output.exists())
        sidecar.unlink()
        output.symlink_to(self.root / 'missing-target')
        with self.assertRaises(FileExistsError):
            main(args)
        self.assertTrue(output.is_symlink())


if __name__ == '__main__':
    unittest.main()
