"""Offset selection and static path tests; synthetic checkpoints, no corpus."""

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
from .closed_trigram import ClosedTrigramProbe
from .offset_paths import (SELECTOR_METHOD, load_offset_selection, main,
                           validate_rankings, write_artifacts)
from .vocabulary import extract, ranking


class OffsetPathsTest(unittest.TestCase):
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
        (self.root / 'tokenizer.json').write_text(json.dumps(
            {'model': {'vocab': {f'token-{i}': i for i in range(7)}}}))

    def make_checkpoint(self, name, seed):
        directory = self.root / name
        directory.mkdir()
        random = np.random.default_rng(seed)
        for spec in tensor_manifest(self.config):
            values = random.normal(size=spec.shape).astype('<f4')
            # Huge padding detects accidental use in logical-vocabulary ranks.
            if spec.name == 'token_embedding.weight':
                values[-1] = 1e10
            values.tofile(directory / spec.filename)
        return GPT2Checkpoint(directory, config=self.config, check_finite=True)

    def selected_entry(self, artifact=None):
        return next(entry for entry in (self.artifact if artifact is None else artifact)['rankings']
                    if entry['method'] == SELECTOR_METHOD)

    def load_modified(self, artifact):
        path = self.root / 'modified.json'
        path.write_text(json.dumps(artifact))
        return load_offset_selection(path, self.final, 5)

    def test_exact_offset_selection_full_scores_and_immutable_ids(self):
        selection = load_offset_selection(self.rankings, self.final, 5)
        expected_scores = [sum(float(a) * float(b) for a, b in zip(row, self.final['final_norm.bias']))
                           for row in self.final.token_embedding]
        expected = sorted(range(7), key=lambda token: (-expected_scores[token], token))[:5]
        self.assertEqual(selection.ranked_ids.tolist(), expected)
        self.assertEqual(selection.ids.tolist(), sorted(expected))
        np.testing.assert_allclose(selection.ranked_scores, np.array(expected_scores)[expected])
        self.assertEqual(selection.rankings_sha256, sha256_file(self.rankings))
        self.assertIs(selection.checkpoint, self.final)
        self.assertNotIn(7, selection.ids)
        for values in (selection.ids, selection.ranked_ids, selection.ranked_scores):
            self.assertFalse(values.flags.writeable)

    def test_same_final_selection_for_early_and_explicit_prior(self):
        final = load_offset_selection(self.rankings, self.final, 5)
        early = load_offset_selection(self.rankings, self.early, 5)
        np.testing.assert_array_equal(early.ids, final.ids)
        np.testing.assert_array_equal(early.ranked_scores, final.ranked_scores)
        self.assertEqual(early.checkpoint.directory, self.final.directory)
        self.assertNotEqual(early.checkpoint.directory, self.early.directory)

    def test_full_rankings_require_finite_sorted_permutations(self):
        corruptions = [
            lambda a: a.update(schema_version=True),
            lambda a: a.update(vocab_size=8),
            lambda a: a.update(vocab_size=True),
            lambda a: a.update(stage='corpus-derived'),
            lambda a: a.update(rankings=[]),
            lambda a: a['rankings'].append(copy.deepcopy(a['rankings'][0])),
            lambda a: a['rankings'][0]['token_ids'].__setitem__(0, 7),
            lambda a: a['rankings'][0]['token_ids'].__setitem__(0, True),
            lambda a: a['rankings'][0]['token_ids'].__setitem__(0, 1.0),
            lambda a: a['rankings'][0]['token_ids'].__setitem__(0, a['rankings'][0]['token_ids'][1]),
            lambda a: a['rankings'][0]['scores'].__setitem__(0, float('nan')),
            lambda a: a['rankings'][0]['scores'].__setitem__(0, float('inf')),
            lambda a: a['rankings'][0]['scores'].__setitem__(0, True),
            lambda a: a['rankings'][0]['scores'].__setitem__(0, '12'),
            lambda a: a['rankings'][0]['scores'].pop(),
            lambda a: a['rankings'][0]['scores'].reverse(),
            lambda a: a['rankings'].remove(self.selected_entry(a)),
        ]
        for corrupt in corruptions:
            artifact = copy.deepcopy(self.artifact)
            corrupt(artifact)
            with self.subTest(artifact=artifact['rankings'][0] if artifact['rankings'] else []), \
                    self.assertRaises(ValueError):
                validate_rankings(artifact, 7)

    def test_ties_are_ascending_ids_in_all_entries(self):
        artifact = copy.deepcopy(self.artifact)
        artifact['rankings'][0] = ranking('final:embedding_norm', np.zeros(7))
        validate_rankings(artifact, 7)
        artifact['rankings'][0]['token_ids'][0:2] = [1, 0]
        with self.assertRaisesRegex(ValueError, 'ascending token-ID ties'):
            validate_rankings(artifact, 7)

    def test_consistently_sorted_forged_offset_scores_are_rejected(self):
        artifact = copy.deepcopy(self.artifact)
        self.selected_entry(artifact)['scores'] = [score + 1 for score in self.selected_entry(artifact)['scores']]
        with self.assertRaisesRegex(ValueError, 'scores do not match'):
            self.load_modified(artifact)
        artifact = copy.deepcopy(self.artifact)
        entry = self.selected_entry(artifact)
        entry['token_ids'][0:2] = entry['token_ids'][1::-1]
        # Scores stay descending, so this falsified permutation is structurally
        # valid but is rejected by independent checkpoint recomputation.
        validate_rankings(artifact, 7)
        with self.assertRaisesRegex(ValueError, 'ordering does not match'):
            self.load_modified(artifact)

    def test_frozen_hashes_protocol_and_config_are_enforced(self):
        corruptions = [
            lambda a: a['checkpoint']['final']['weight_sha256'].__setitem__('weight_0.bin', '0' * 64),
            lambda a: a['checkpoint']['final']['weight_sha256'].__setitem__('weight_15.bin', '0' * 64),
            lambda a: a['checkpoint']['final']['weight_sha256'].__setitem__('weight_8.bin', '0' * 64),
            lambda a: a['checkpoint']['final']['config'].__setitem__('d_model', 8),
            lambda a: a['protocol'].__setitem__('sha256', '0' * 64),
        ]
        for corrupt in corruptions:
            artifact = copy.deepcopy(self.artifact)
            corrupt(artifact)
            with self.assertRaises(ValueError):
                self.load_modified(artifact)

    def test_changed_selector_file_rejected_even_when_ranking_json_unchanged(self):
        # Padding has no logical score, but belongs to the physical tensor hash.
        path = self.final.directory / 'weight_0.bin'
        with path.open('r+b') as stream:
            stream.seek(7 * 4 * 4)
            stream.write(np.array([1], dtype='<f4').tobytes())
        with self.assertRaisesRegex(ValueError, 'weight hashes differ'):
            load_offset_selection(self.rankings, self.final, 5)

    def test_invalid_size_rejected(self):
        for size in (0, -1, 8, True, 2.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                load_offset_selection(self.rankings, self.final, size)

    def run_cli(self, checkpoint, name, broken=False):
        output = self.root / (name + '.jsonl')
        args = ['--checkpoint', str(checkpoint.directory), '--rankings', str(self.rankings),
                '--tokenizer-dir', str(self.root), '--output', str(output), '--vocabulary-size', '7']
        if broken:
            args.append('--broken-routing-control')
        def load(path, **kwargs):
            return self.final if Path(path).resolve() == self.final.directory else self.early
        with mock.patch('scripts.weight_analysis.offset_paths.GPT2Checkpoint', side_effect=load), \
                contextlib.redirect_stdout(io.StringIO()):
            main(args)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        metadata = json.loads(output.with_suffix('.jsonl.metadata.json').read_text())
        return output, args, records, metadata

    def test_three_cli_arms_keep_identical_selector_and_exact_reused_contraction(self):
        selections = []
        for name, checkpoint, broken in [('final', self.final, False), ('early', self.early, False),
                                         ('broken', self.final, True)]:
            output, _, records, metadata = self.run_cli(checkpoint, name, broken)
            selections.append(metadata['vocabulary_selection']['ids_ascending'])
            self.assertEqual(metadata['vocabulary_selection_checkpoint']['checkpoint_directory'], str(self.final.directory))
            self.assertEqual(metadata['checkpoint']['checkpoint_directory'], str(checkpoint.directory))
            self.assertEqual(metadata['vocabulary_selection']['selection_checkpoint_differs_from_evaluation'], name == 'early')
            self.assertEqual(metadata['candidate_sha256'], sha256_file(output))
            self.assertEqual(metadata['rankings']['sha256'], sha256_file(self.rankings))
            self.assertEqual(metadata['parameters']['top_k'], 4)
            self.assertEqual(metadata['parameters']['path_length'], 12)
            self.assertEqual(metadata['parameters']['path_starts'], 256)
            self.assertEqual(metadata['parameters']['beam_width'], 4)
            self.assertNotIn('corpus', metadata['parameters'])
            expected, _ = ClosedTrigramProbe(checkpoint, np.arange(7), pairing=[1, 0] if broken else [0, 1]).extract(4)
            actual = [r for r in records if len(r['token_ids']) == 3]
            self.assertEqual(len(actual), 7 * 7 * 4)
            self.assertEqual([r['token_ids'] for r in actual], [r['token_ids'] for r in expected])
            np.testing.assert_array_equal([r['score'] for r in actual], [r['score'] for r in expected])
        self.assertEqual(selections, [list(range(7))] * 3)

    def test_path_closure_score_denominator_and_physical_provenance(self):
        _, _, records, metadata = self.run_cli(self.final, 'closure')
        triples = {r['candidate_id']: r for r in records if len(r['token_ids']) == 3}
        paths = [r for r in records if len(r['token_ids']) > 3]
        self.assertTrue(paths)
        for path in paths:
            tokens = path['token_ids']
            edges = [triples[key] for key in path['provenance']['trigram_candidate_ids']]
            self.assertEqual(len(edges), len(tokens) - 2)
            self.assertAlmostEqual(path['score'], sum(edge['score'] for edge in edges) / (len(tokens) - 2))
            for index, edge in enumerate(edges):
                self.assertEqual(edge['token_ids'], tokens[index:index + 3])
            self.assertTrue(all(a != b for a, b in zip(tokens, tokens[1:])))
            self.assertTrue(all(tokens.count(token) <= 2 for token in tokens))
        dependencies = metadata['vocabulary_selection']['weight_dependencies']
        self.assertEqual(dependencies['embedding_file'], 'weight_0.bin')
        self.assertEqual(dependencies['logical_byte_range'], [0, 7 * 4 * 4])
        self.assertEqual(dependencies['shared_beta_file'], 'weight_15.bin')
        self.assertEqual(dependencies['shared_beta_byte_range'], [0, 16])
        self.assertEqual(dependencies['embedding_shared_with'], ['lm_head.weight'])
        self.assertEqual(metadata['geometry_provenance']['positions']['row_byte_ranges'], [[0, 16], [16, 32]])
        self.assertEqual(metadata['routing_weight_provenance']['head_slices'][0]['value']['columns'], [8, 10])
        self.assertEqual(metadata['diagnostics']['source_pair_count'], 49)
        self.assertEqual(metadata['diagnostics']['overlap_edges'], 196)

    def test_no_overwrite_candidates_metadata_or_dangling_symlinks(self):
        output, args, _, _ = self.run_cli(self.final, 'existing')
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            main(args)
        self.assertEqual(output.read_bytes(), before)
        output.unlink()
        with self.assertRaises(FileExistsError):
            main(args)
        self.assertFalse(output.exists())
        output.with_suffix('.jsonl.metadata.json').unlink()
        output.symlink_to(self.root / 'absent')
        with self.assertRaises(FileExistsError):
            main(args)
        self.assertTrue(output.is_symlink())
        self.assertFalse((self.root / 'absent').exists())

    def test_exclusive_pair_writer_cleans_only_its_new_file_on_collision(self):
        output = self.root / 'atomic.jsonl'
        sidecar = output.with_suffix('.jsonl.metadata.json')
        sidecar.write_text('preserve this')
        with self.assertRaises(FileExistsError):
            write_artifacts(output, [{'a': 1}], {})
        self.assertFalse(output.exists())
        self.assertEqual(sidecar.read_text(), 'preserve this')
        sidecar.unlink()
        metadata = write_artifacts(output, [{'a': 1}], {'test': True})
        self.assertEqual(metadata['candidate_sha256'], sha256_file(output))
        with self.assertRaises(FileExistsError):
            write_artifacts(output, [], {})
        self.assertEqual(output.read_text(), '{"a": 1}\n')


if __name__ == '__main__':
    unittest.main()
