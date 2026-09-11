"""Independent dense oracles for the analytical three-token probe.

Softmax appears only in the finite-difference TEST oracle. The production
extractor does not evaluate a softmax or any transformer layer.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from .checkpoint import GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest
from .trigram import TrigramProbe, _top, decode_trigram_paths, main, weight_provenance


class TrigramTest(unittest.TestCase):
    def setUp(self):
        random = np.random.default_rng(1234)
        self.embedding = random.normal(size=(7, 4))
        self.query, self.key, self.value = random.normal(size=(3, 2, 4, 2))
        self.output = random.normal(size=(2, 2, 4))
        self.bias = random.normal(size=(2, 2))

    def probe(self, pairing=None):
        return TrigramProbe(self.embedding, self.query, self.key, self.value,
                            self.output, self.bias, pairing)

    def dense_write(self, current, previous, pairing=(0, 1)):
        result = np.zeros(4)
        for head, content_head in enumerate(pairing):
            difference = self.embedding[previous] - self.embedding[current]
            q = self.embedding[current] @ self.query[head] + self.bias[head]
            k = difference @ self.key[head]
            score_difference = q @ k / np.sqrt(2)
            value_difference = difference @ self.value[content_head] @ self.output[content_head]
            result += score_difference * value_difference / 4
        return result

    def test_contracted_writes_equal_dense_signed_formula(self):
        for pairing in ((0, 1), (1, 0)):
            probe = self.probe(pairing)
            for current in range(7):
                actual = probe.interaction_writes(current, np.arange(7))
                expected = [self.dense_write(current, previous, pairing) for previous in range(7)]
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(actual[current], np.zeros(4))

    def test_matches_derivative_of_two_choice_softmax_at_uniform(self):
        random = np.random.default_rng(88)
        key_bias = random.normal(size=(2, 2))
        value_bias = random.normal(size=(2, 2))
        output_bias = random.normal(size=(2, 4))
        epsilon = 1e-5
        for pairing in ((0, 1), (1, 0)):
            probe = self.probe(pairing)
            for previous, current in ((0, 1), (4, 3), (2, 2)):
                def softmax_attention(scale):
                    result = np.zeros(4)
                    for head, content in enumerate(pairing):
                        q = self.embedding[current] @ self.query[head] + self.bias[head]
                        positions = self.embedding[[previous, current]]
                        keys = positions @ self.key[head] + key_bias[head]
                        scores = (keys @ q) / np.sqrt(2)
                        scaled = scale * scores
                        probabilities = np.exp(scaled - scaled.max())
                        probabilities /= probabilities.sum()
                        values = positions @ self.value[content] + value_bias[content]
                        result += probabilities @ values @ self.output[content] + output_bias[content]
                    return result

                expected = (softmax_attention(epsilon) - softmax_attention(-epsilon)) / (2 * epsilon)
                actual = probe.interaction_writes(current, np.array([previous]))[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)

    def test_search_equals_dense_full_vocab_oracle(self):
        probe = self.probe()
        query_norms = [sum(np.linalg.norm(self.embedding[token] @ self.query[head] +
                                         self.bias[head]) ** 2 for head in range(2))
                       for token in range(7)]
        expected_currents = sorted(range(7), key=lambda token: (-query_norms[token], token))[:3]
        selected = probe.select_previous(current_count=3, previous_count=2, chunk_size=2)
        self.assertEqual(list(dict.fromkeys(pair[1] for pair in selected)), expected_currents)
        expected_pairs = []
        for current in expected_currents:
            norms = [np.linalg.norm(self.dense_write(current, previous)) for previous in range(7)]
            previous_ids = sorted(range(7), key=lambda previous: (-norms[previous], previous))[:2]
            expected_pairs.extend((previous, current) for previous in previous_ids)
        self.assertEqual([(pair[0], pair[1]) for pair in selected], expected_pairs)
        actual = list(probe.destinations(selected, top_k=3, chunk_size=2))
        self.assertEqual(len(actual), 18)
        for pair_index, (previous, current) in enumerate(expected_pairs):
            write = self.dense_write(current, previous)
            scores = (self.embedding @ write) / (np.linalg.norm(self.embedding, axis=1) * np.linalg.norm(write))
            expected_targets = sorted(range(7), key=lambda target: (-scores[target], target))[:3]
            records = actual[3 * pair_index:3 * pair_index + 3]
            self.assertEqual([record['token_ids'] for record in records],
                             [[previous, current, target] for target in expected_targets])
            np.testing.assert_allclose([record['score'] for record in records], scores[expected_targets])

    def test_chunk_sizes_preserve_selection_and_scores(self):
        reference = None
        for size in (1, 2, 7, 11):
            probe = self.probe()
            pairs = probe.select_previous(7, 3, size)
            records = list(probe.destinations(pairs, 3, size))
            ids = [record['token_ids'] for record in records]
            scores = [record['score'] for record in records]
            if reference is None:
                reference = (ids, scores)
            self.assertEqual(ids, reference[0])
            np.testing.assert_allclose(scores, reference[1], rtol=1e-12, atol=1e-12)

    def test_signed_head_cancellation_happens_before_norm_selection(self):
        embedding = np.eye(2)
        query = np.ones((2, 2, 1))
        key = np.array([[[1.], [0.]], [[1.], [0.]]])
        value = key.copy()
        output = np.array([[[1., 2.]], [[-1., -2.]]])
        probe = TrigramProbe(embedding, query, key, value, output)
        np.testing.assert_array_equal(probe.interaction_writes(0, np.array([1])), [[0., 0.]])
        self.assertEqual(probe.select_previous(2, 2, 1), [])
        for head in range(2):
            single = TrigramProbe(embedding, query[head:head + 1], key[head:head + 1],
                                  value[head:head + 1], output[head:head + 1])
            self.assertGreater(np.linalg.norm(single.interaction_writes(0, np.array([1]))), 0)

    def test_broken_routing_preserves_intact_ov_and_query_selection(self):
        intact = self.probe()
        broken = self.probe((1, 0))
        np.testing.assert_array_equal(intact.queries, broken.queries)
        np.testing.assert_array_equal(intact.keys, broken.keys)
        np.testing.assert_array_equal(intact.query_norm_squared, broken.query_norm_squared)
        np.testing.assert_array_equal(broken.values, intact.values[:, [1, 0]])
        np.testing.assert_array_equal(broken.output,
                                       np.concatenate((self.output[1], self.output[0]), axis=0))
        self.assertFalse(np.allclose(intact.interaction_writes(0, np.arange(7)),
                                     broken.interaction_writes(0, np.arange(7))))
        self.assertEqual(list(dict.fromkeys(pair[1] for pair in intact.select_previous(3, 2, 3))),
                         list(dict.fromkeys(pair[1] for pair in broken.select_previous(3, 2, 3))))

    def test_permuting_all_heads_together_preserves_probe(self):
        order = [1, 0]
        permuted = TrigramProbe(self.embedding, self.query[order], self.key[order],
                                self.value[order], self.output[order], self.bias[order])
        for current in range(7):
            np.testing.assert_allclose(self.probe().interaction_writes(current, np.arange(7)),
                                       permuted.interaction_writes(current, np.arange(7)), atol=1e-12)

    def test_zero_directions_are_not_arbitrary_token_candidates(self):
        zero = np.zeros_like(self.query)
        probe = TrigramProbe(self.embedding, zero, self.key, self.value, self.output)
        self.assertEqual(probe.select_previous(), [])
        self.assertEqual(list(probe.destinations([])), [])
        self.embedding[2] = 0
        probe = self.probe()
        records = list(probe.destinations(probe.select_previous(7, 3), top_k=20))
        self.assertTrue(records)
        self.assertTrue(all(record['token_ids'][2] != 2 for record in records))

    def test_top_ties_resolve_by_supplied_id_not_chunk_order(self):
        self.assertEqual(_top(np.array([3., 3., 3., -np.inf]), 2,
                              np.array([9, 7, 8, 0])).tolist(), [1, 2])
        self.assertEqual(_top(np.array([-np.inf]), 8).tolist(), [])

    @staticmethod
    def edge(tokens, score=1.0):
        return {'token_ids': list(tokens), 'score': score,
                'candidate_id': ':'.join(map(str, tokens))}

    def test_paths_require_exact_two_token_overlap_and_recorded_edges(self):
        records = [self.edge((0, 1, 2), 1.0), self.edge((1, 2, 3), 0.8),
                   self.edge((2, 3, 4), 0.6), self.edge((3, 4, 5), 0.4),
                   self.edge((3, 6, 7), 0.8)]
        paths = list(decode_trigram_paths(records, length=12, starts=10))
        self.assertEqual(paths[0]['token_ids'], [0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(paths[0]['score'], (1.0 + 0.8 + 0.6 + 0.4) / 4)
        by_id = {record['candidate_id']: record for record in records}
        for path in paths:
            ids = path['token_ids']
            for index, edge_id in enumerate(path['provenance']['trigram_candidate_ids']):
                self.assertEqual(by_id[edge_id]['token_ids'], ids[index:index + 3])
            self.assertEqual(len(path['provenance']['trigram_candidate_ids']), len(ids) - 2)
        self.assertFalse(any(6 in path['token_ids'] for path in paths))
        # Sharing one token is insufficient to concatenate these contexts.
        self.assertEqual(list(decode_trigram_paths(
            [self.edge((0, 1, 2)), self.edge((1, 3, 4))])), [])

    def test_paths_reject_self_transitions_and_limit_repetition(self):
        self.assertEqual(list(decode_trigram_paths(
            [self.edge((0, 0, 1)), self.edge((0, 1, 1))])), [])
        paths = list(decode_trigram_paths([self.edge((0, 1, 0)), self.edge((1, 0, 1))],
                                          length=12, starts=2))
        self.assertTrue(paths)
        for path in paths:
            self.assertEqual(len(path['token_ids']), 4)
            self.assertLessEqual(max(path['token_ids'].count(token) for token in (0, 1)), 2)

    def test_invalid_weights_indices_and_budgets(self):
        with self.assertRaises(ValueError):
            TrigramProbe(self.embedding, self.query[:, :, :1], self.key, self.value, self.output)
        for pairing in ([0, 0], [1, 2], [0.0, 1.0]):
            with self.assertRaises(ValueError):
                self.probe(pairing)
        original = self.query.copy()
        self.query[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.probe()
        self.query = original
        probe = self.probe()
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                probe.select_previous(current_count=count)
        with self.assertRaises(ValueError):
            probe.interaction_writes(7, np.array([0]))
        with self.assertRaises(ValueError):
            probe.interaction_writes(0, np.array([-1]))
        with self.assertRaises(ValueError):
            list(decode_trigram_paths([], length=3))
        with self.assertRaises(ValueError):
            list(decode_trigram_paths([], starts=-1))

    def make_checkpoint(self, root):
        config = GPT2Config(vocab_size=7, padded_vocab_size=8, context_length=3,
                             n_layers=1, d_model=4, n_heads=2, d_ff=8)
        directory = root / 'step_1'
        directory.mkdir()
        for spec in tensor_manifest(config):
            values = np.zeros(spec.shape, dtype='<f4')
            if spec.name == 'token_embedding.weight':
                values[:7] = self.embedding
            elif spec.name == 'blocks.0.attn.qkv.weight':
                values[:] = np.concatenate([np.concatenate(list(projection), axis=1)
                                             for projection in (self.query, self.key, self.value)], axis=1)
            elif spec.name == 'blocks.0.attn.qkv.bias':
                values[:4] = self.bias.reshape(-1)
            elif spec.name == 'blocks.0.attn.output.weight':
                values[:] = np.concatenate(list(self.output), axis=0)
            values.tofile(directory / spec.filename)
        (root / 'tokenizer.json').write_text(json.dumps(
            {'model': {'vocab': {str(i): i for i in range(7)}}}))
        return GPT2Checkpoint(directory, config, check_finite=True)

    def test_provenance_byte_rectangles_match_checkpoint_views(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(Path(directory))
            address = weight_provenance(checkpoint, 0, [1, 0])
            for head in address['head_slices']:
                route, content = head['routing_head'], head['content_head']
                expected = {'query': checkpoint.qkv(0, 'q', route),
                            'key': checkpoint.qkv(0, 'k', route),
                            'value': checkpoint.qkv(0, 'v', content),
                            'output': checkpoint.attention_output_head(0, content)}
                for name, actual in expected.items():
                    view = head[name]
                    raw = np.fromfile(checkpoint.directory / view['filename'], dtype='<f4')
                    reconstructed = np.empty_like(actual)
                    for row in range(actual.shape[0]):
                        for col in range(actual.shape[1]):
                            byte = (view['offset_bytes'] + row * view['row_stride_bytes'] +
                                    col * view['column_stride_bytes'])
                            reconstructed[row, col] = raw[byte // 4]
                    np.testing.assert_array_equal(actual, reconstructed)
                bias = head['query_bias']
                raw = np.fromfile(checkpoint.directory / bias['filename'], dtype='<f4')
                start, end = bias['byte_range']
                np.testing.assert_array_equal(raw[start // 4:end // 4], checkpoint.qkv_bias(0, 'q', route))

    def test_cli_corpus_blind_frozen_artifacts_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            for broken in (False, True):
                output = root / f'candidates_{broken}.jsonl'
                args = ['--checkpoint', str(checkpoint.directory), '--tokenizer-dir', str(root),
                        '--output', str(output), '--current-count', '3', '--previous-count', '2',
                        '--top-k', '2', '--chunk-size', '3', '--path-length', '6']
                if broken:
                    args.append('--broken-routing-control')
                with mock.patch('weight_analysis.trigram.GPT2Checkpoint', return_value=checkpoint), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    main(args)
                    before = output.read_bytes()
                    with self.assertRaises(FileExistsError):
                        main(args)
                    self.assertEqual(output.read_bytes(), before)
                sidecar = output.with_suffix('.jsonl.metadata.json')
                metadata = json.loads(sidecar.read_text())
                records = [json.loads(line) for line in output.read_text().splitlines()]
                method = 'broken_qkov_trigram' if broken else 'static_qkov_trigram'
                self.assertEqual(metadata['method_counts'][method], 12)
                self.assertEqual(metadata['selected_pair_count'], 6)
                self.assertEqual(metadata['candidate_sha256'], sha256_file(output))
                self.assertEqual(metadata['extractor_sha256'], sha256_file(Path(__file__).with_name('trigram.py')))
                self.assertNotIn('corpus', metadata['parameters'])
                self.assertEqual(metadata['stage'], 'corpus_blind_extraction')
                expected_pairing = [1, 0] if broken else [0, 1]
                for record in records:
                    if record['method'] == method:
                        self.assertEqual(record['provenance']['ov_pairing'], expected_pairing)
                output.unlink()  # Sidecar alone also protects the frozen name.
                with self.assertRaises(FileExistsError):
                    main(args)


if __name__ == '__main__':
    unittest.main()
