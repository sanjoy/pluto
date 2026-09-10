"""Synthetic, CPU-only tests of the frozen paired patch readout."""

from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import paired_patch_readout as readout


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def record(name, size=100):
    return {'path': '/synthetic/' + name, 'bytes': size, 'sha256': digest(name)}


def derive_scores(case, token_nll):
    case['token_nll'] = token_nll
    case['sequence_nll'] = math.fsum(token_nll)
    case['sequence_log_probability'] = -case['sequence_nll']
    case['sequence_probability'] = math.exp(-case['sequence_nll'])
    case['token_probabilities'] = [math.exp(-value) for value in token_nll]
    case['teacher_forced_argmax_ids'] = case['target_ids'][:]
    case['argmax_matches'] = [True, True, True]
    case['all_three_argmax_match'] = True


def derive_groups(summary):
    groups = defaultdict(list)
    for case in summary['per_case']:
        groups[tuple(case[key] for key in ('kind', 'split', 'prefix_domain', 'target'))].append(case)
    summary['groups'] = []
    for key, items in sorted(groups.items()):
        mean = math.fsum(item['sequence_nll'] for item in items) / len(items)
        summary['groups'].append({
            **dict(zip(('kind', 'split', 'prefix_domain', 'target'), key)),
            'case_count': len(items), 'mean_sequence_nll': mean, 'mean_token_nll': mean / 3,
            'geometric_mean_sequence_probability': math.exp(-mean),
            'arithmetic_mean_sequence_probability': math.fsum(
                item['sequence_probability'] for item in items) / len(items),
            'teacher_forced_argmax_token_accuracy': sum(
                sum(item['argmax_matches']) for item in items) / (3 * len(items)),
            'all_three_argmax_match_fraction': sum(
                item['all_three_argmax_match'] for item in items) / len(items)})


def synthetic_summaries():
    cases = []
    for split in readout.SPLITS:
        # Occurrences 0 and 2 have exactly the same prefix/candidate inputs.
        # Occurrence 1 has genuinely different original/replacement prefixes.
        for occurrence in range(3):
            for domain in readout.DOMAINS:
                for target_domain, word, ids, pieces in (
                        ('original', 'Exeunt', [11, 12, 13], [b'Ex', b'e', b'unt']),
                        ('replacement', 'Nuveth', [21, 22, 23], [b'N', b'uv', b'eth'])):
                    variant = 'shared' if occurrence != 1 else domain
                    start = 10 + 10 * occurrence
                    cases.append({
                        'case_index': len(cases), 'kind': 'word', 'split': split,
                        'context_id': f'{split}:occurrence:{occurrence}',
                        'occurrence_index': occurrence, 'prefix_domain': domain,
                        'prefix': {'token_start': start - 4, 'token_end': start, 'length': 4,
                                   'byte_start': start - 4, 'byte_end': start,
                                   'token_ids_sha256': digest(split + variant)},
                        'target': word, 'target_source_domain': target_domain, 'target_ids': ids,
                        'target_source': {'token_start': start, 'token_end': start + 3,
                                          'byte_start': start, 'byte_end': start + 6,
                                          'bytes_hex': b''.join(pieces).hex(), 'decoded_text': word,
                                          'native_piece_bytes_hex': [piece.hex() for piece in pieces]},
                        'scored_rows': [3, 4, 5]})
        for start in (70, 80):
            cases.append({
                'case_index': len(cases), 'kind': 'control', 'split': split,
                'context_id': f'{split}:control:{start}', 'occurrence_index': None,
                'prefix_domain': 'shared', 'prefix': {
                    'token_start': start - 4, 'token_end': start, 'length': 4,
                    'byte_start': start - 4, 'byte_end': start,
                    'token_ids_sha256': digest(split + 'control')},
                'target': 'control_next_3', 'target_source_domain': 'original',
                'target_ids': [1, 2, 3], 'target_source': {
                    'token_start': start, 'token_end': start + 3, 'byte_start': start,
                    'byte_end': start + 3, 'bytes_hex': b'abc'.hex(), 'decoded_text': 'abc',
                    'native_piece_bytes_hex': [b'a'.hex(), b'b'.hex(), b'c'.hex()]},
                'scored_rows': [3, 4, 5]})
    nlls = {'original': {'Exeunt': [1., 2., 3.], 'Nuveth': [2., 3., 4.]},
            'replacement': {'Exeunt': [3., 3., 3.], 'Nuveth': [1., 1., 1.]},
            'original_with_donor_rows': {'Exeunt': [2., 2., 3.], 'Nuveth': [1., 2., 3.]},
            'replacement_with_donor_rows': {'Exeunt': [2., 3., 3.], 'Nuveth': [2., 2., 2.]}}
    summaries = {}
    for model_index, model in enumerate(readout.MODELS):
        per_case = copy.deepcopy(cases)
        for case in per_case:
            if case['kind'] == 'word':
                factor = 1 if case['occurrence_index'] != 1 else (2 if case['prefix_domain'] == 'original' else 3)
                losses = [factor * value for value in nlls[model][case['target']]]
            else:
                losses = [1. + model_index * .1, 2., 3.]
            derive_scores(case, losses)
        count, length = len(cases), 16
        summary = {
            'format': 'pluto-paired-word-score-summary-v1', 'cases': record('cases.json'),
            'scores': {name: record(model + '/' + name, count * length * 4 if name != 'metadata' else 500)
                       for name in ('metadata', 'losses', 'argmax')},
            'probe_metadata': {
                'kind': 'paired_loss_probe', 'complete': True, 'temperature': 1,
                'byte_order': 'little', 'loss_dtype': '<f4', 'argmax_dtype': '<i4',
                'loss_file': 'losses.f32.bin', 'argmax_file': 'argmax.i32.bin',
                'case_count': count, 'passage_count': count, 'context_length': length,
                'output_shape': [count, length], 'batch_file': '/synthetic/packed_cases.bin',
                'batch_bytes': count * length * 8, 'checkpoint_directory': '/synthetic/' + model},
            'case_count': count, 'context_length': length,
            'probability_definition': 'Three target IDs, without following delimiter.',
            'prefix_positioning': 'Positions restart at zero.',
            'aggregation': 'Each frozen case has equal weight within its group.',
            'limitations': ['Synthetic fixture; no model was evaluated.'], 'per_case': per_case}
        derive_groups(summary)
        summaries[model] = summary
    return summaries


class PairedPatchReadoutTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.summaries = synthetic_summaries()
        self.paths = [self.root / f'{model}.json' for model in readout.MODELS]
        self.output = self.root / 'readout.json'

    def write_inputs(self):
        for path, model in zip(self.paths, readout.MODELS):
            path.write_text(json.dumps(self.summaries[model]))

    def compare(self, **options):
        self.write_inputs()
        return readout.compare_summaries(*self.paths, self.output, **options)

    def test_nll_log_odds_bidirectional_fractions_and_token_decomposition(self):
        result = self.compare()
        for split in readout.SPLITS:
            group = result['splits'][split]
            original = group['word_models']['original']
            self.assertEqual(original['word_nll'], {'Exeunt': 12., 'Nuveth': 18.})
            self.assertEqual(original['word_token_nll']['Exeunt'], [2., 4., 6.])
            self.assertEqual(original['nuveth_minus_exeunt_log_odds'], -6.)
            self.assertEqual(original['token_log_odds'], [-2., -2., -2.])
            forward = group['word_patch_effects']['original_with_donor_rows']
            reverse = group['word_patch_effects']['replacement_with_donor_rows']
            self.assertEqual(forward['log_odds_patch_change'], 8.)
            self.assertEqual(forward['full_donor_log_odds_shift'], 18.)
            self.assertAlmostEqual(forward['fraction_of_full_donor_shift'], 4 / 9)
            self.assertEqual(reverse['log_odds_patch_change'], -8.)
            self.assertEqual(reverse['full_donor_log_odds_shift'], -18.)
            self.assertAlmostEqual(reverse['fraction_of_full_donor_shift'], 4 / 9)
            for effect in (forward, reverse):
                self.assertAlmostEqual(sum(effect['token_log_odds_patch_changes']),
                                       effect['log_odds_patch_change'])
                for word in readout.WORDS:
                    self.assertAlmostEqual(sum(effect['word_token_nll_changes'][word]),
                                           effect['word_nll_changes'][word])

    def test_domain_and_occurrence_duplicates_count_once_but_values_are_preserved(self):
        result = self.compare()
        group = result['splits']['training']
        self.assertEqual(group['counts'], {
            'frozen_word_occurrences': 3, 'frozen_word_cases': 12,
            'unique_word_evaluation_contexts': 3, 'unique_word_prefixes': 3,
            'occurrences_with_identical_prefix_domains': 2,
            'occurrences_with_distinct_prefix_domains': 1,
            'frozen_control_cases': 2, 'unique_control_inputs': 1})
        self.assertEqual(group['duplicate_score_validation']['maximum_token_nll_difference'], 0)
        contexts = group['word_contexts']
        self.assertEqual(sum(len(context['cases']) for context in contexts), 12)
        shared = next(context for context in contexts if len(context['cases']) == 8)
        self.assertEqual(shared['prefix_domains'], ['original', 'replacement'])
        self.assertEqual(shared['prefix_token_ids_sha256'], digest('trainingshared'))
        for domain in readout.DOMAINS:
            self.assertEqual(group['by_prefix_domain'][domain]['unique_word_evaluation_contexts'], 2)
        for entry in shared['cases']:
            i = entry['case']['case_index']
            for model in readout.MODELS:
                self.assertEqual(entry['scores'][model]['token_nll'],
                                 self.summaries[model]['per_case'][i]['token_nll'])
        self.assertEqual(json.loads(self.output.read_text()), result)

    def test_control_changes_and_input_hashes(self):
        result = self.compare()
        changes = result['splits']['test']['control_changes']
        self.assertAlmostEqual(changes['replacement']['sequence_nll_change'], .1)
        self.assertAlmostEqual(changes['original_with_donor_rows']['sequence_nll_change'], .2)
        self.assertAlmostEqual(changes['replacement_with_donor_rows']['sequence_nll_change'], .2)
        self.assertEqual(changes['original_with_donor_rows']['token_nll_changes'][1:], [0., 0.])
        for model, path in zip(readout.MODELS, self.paths):
            self.assertEqual(result['input_summaries'][model]['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result['score_provenance'][model]['scores'], self.summaries[model]['scores'])
        self.assertEqual(result['frozen_cases'], self.summaries['original']['cases'])

    def test_identical_input_disagreement_is_rejected(self):
        summary = self.summaries['replacement_with_donor_rows']
        case = summary['per_case'][2]
        derive_scores(case, [value + .01 for value in case['token_nll']])
        derive_groups(summary)
        with self.assertRaisesRegex(ValueError, 'identical-input'):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_identical_input_argmax_disagreement_with_consistent_groups_is_rejected(self):
        summary = self.summaries['replacement_with_donor_rows']
        case = summary['per_case'][2]
        case['teacher_forced_argmax_ids'][0] = 99
        case['argmax_matches'] = [False, True, True]
        case['all_three_argmax_match'] = False
        derive_groups(summary)
        # The summary itself remains internally valid; only duplicate aliases disagree.
        readout._validate_summary(summary)
        with self.assertRaisesRegex(ValueError, 'identical-input argmax IDs disagree'):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_within_tolerance_duplicate_keeps_every_value_and_uses_first_case(self):
        summary = self.summaries['original']
        case = summary['per_case'][2]
        derive_scores(case, [case['token_nll'][0] + 5e-6, *case['token_nll'][1:]])
        derive_groups(summary)
        result = self.compare(absolute_tolerance=1e-5, relative_tolerance=0.)
        group = result['splits']['training']
        self.assertAlmostEqual(group['duplicate_score_validation']['maximum_token_nll_difference'], 5e-6)
        self.assertEqual(group['word_models']['original']['word_nll']['Exeunt'], 12.)
        shared = next(c for c in group['word_contexts'] if len(c['cases']) == 8)
        alias = next(c for c in shared['cases'] if c['case']['case_index'] == 2)
        self.assertAlmostEqual(alias['scores']['original']['token_nll'][0], 1.000005)

    def test_donor_shift_at_floor_is_null_without_hiding_patch_change(self):
        for case, original in zip(self.summaries['replacement']['per_case'],
                                  self.summaries['original']['per_case']):
            derive_scores(case, original['token_nll'][:])
        derive_groups(self.summaries['replacement'])
        result = self.compare()
        for effect in result['splits']['training']['word_patch_effects'].values():
            self.assertIsNone(effect['fraction_of_full_donor_shift'])
            self.assertEqual(effect['full_donor_log_odds_shift'], 0.)
        self.assertEqual(result['splits']['training']['word_patch_effects']
                         ['original_with_donor_rows']['log_odds_patch_change'], 8.)
        self.assertIsNone(readout._fraction(1., 1e-6, 1e-6))

    def test_fraction_retains_negative_and_overshoot_and_is_ratio_of_means(self):
        self.assertEqual(readout._fraction(-2., 1., 1e-6), -2.)
        self.assertEqual(readout._fraction(2., 1., 1e-6), 2.)
        summary = self.summaries['original_with_donor_rows']
        for case in summary['per_case']:
            if case['kind'] == 'word' and case['occurrence_index'] == 1 and case['prefix_domain'] == 'replacement':
                # Only the factor-three context returns to recipient scores.
                derive_scores(case, self.summaries['original']['per_case'][case['case_index']]['token_nll'][:])
        derive_groups(summary)
        group = self.compare()['splits']['training']
        fraction = group['word_patch_effects']['original_with_donor_rows']['fraction_of_full_donor_shift']
        self.assertAlmostEqual(fraction, 2 / 9)
        mean_fraction = sum(c['patch_effects']['original_with_donor_rows']['fraction_of_full_donor_shift']
                            for c in group['word_contexts']) / 3
        self.assertNotAlmostEqual(fraction, mean_fraction)

    def test_nonfinite_negative_and_inconsistent_scores_reject_before_write(self):
        baseline = copy.deepcopy(self.summaries)
        for mutation in ('nan', 'inf', 'negative', 'sum', 'probability', 'group', 'argmax'):
            with self.subTest(mutation=mutation):
                self.summaries = copy.deepcopy(baseline)
                summary = self.summaries['original']
                case = summary['per_case'][0]
                if mutation in ('nan', 'inf', 'negative'):
                    case['token_nll'][0] = {'nan': float('nan'), 'inf': float('inf'), 'negative': -1.}[mutation]
                elif mutation == 'sum':
                    case['sequence_nll'] += 1
                elif mutation == 'probability':
                    case['sequence_probability'] = .5
                elif mutation == 'group':
                    summary['groups'][0]['mean_sequence_nll'] += 1
                else:
                    case['argmax_matches'] = [False, False, False]
                with self.assertRaises(ValueError):
                    self.compare()
                self.assertFalse(self.output.exists())

    def test_misaligned_cases_hashes_and_probe_metadata_are_rejected(self):
        baseline = copy.deepcopy(self.summaries)
        for mutation in ('hash', 'batch', 'prefix', 'rows', 'order', 'count', 'temperature', 'missing_field'):
            with self.subTest(mutation=mutation):
                self.summaries = copy.deepcopy(baseline)
                summary = self.summaries['replacement']
                if mutation == 'hash':
                    summary['cases']['sha256'] = digest('different cases')
                elif mutation == 'batch':
                    summary['probe_metadata']['batch_file'] = '/another/batch.bin'
                elif mutation == 'prefix':
                    summary['per_case'][0]['prefix']['token_ids_sha256'] = digest('different prefix')
                elif mutation == 'rows':
                    summary['per_case'][0]['scored_rows'] = [4, 5, 6]
                elif mutation == 'order':
                    summary['per_case'].reverse()
                elif mutation == 'count':
                    summary['case_count'] += 1
                elif mutation == 'temperature':
                    summary['probe_metadata']['temperature'] = 2
                else:
                    del summary['per_case'][0]['token_nll']
                with self.assertRaises(ValueError):
                    self.compare()
                self.assertFalse(self.output.exists())

    def test_missing_or_misaligned_domain_candidate_pairs_are_rejected(self):
        for summary in self.summaries.values():
            # Remains aligned across models, but breaks within-context crossing.
            summary['per_case'][2]['context_id'] = 'training:missing-counterpart'
        with self.assertRaisesRegex(ValueError, 'missing a candidate'):
            self.compare()

    def test_cross_domain_occurrence_coordinates_must_match(self):
        for summary in self.summaries.values():
            for index in (2, 3):
                case = summary['per_case'][index]
                case['occurrence_index'] = 999
        with self.assertRaisesRegex(ValueError, 'misaligned occurrence coordinates'):
            self.compare()

    def test_existing_output_is_never_overwritten_including_input_and_symlink(self):
        self.write_inputs()
        self.output.write_text('keep this')
        with self.assertRaises(FileExistsError):
            readout.compare_summaries(*self.paths, self.output)
        self.assertEqual(self.output.read_text(), 'keep this')
        with self.assertRaises(FileExistsError):
            readout.compare_summaries(*self.paths, self.paths[0])
        dangling = self.root / 'dangling.json'
        dangling.symlink_to(self.root / 'missing.json')
        with self.assertRaises(FileExistsError):
            readout.compare_summaries(*self.paths, dangling)

    def test_no_controls_is_reported_explicitly(self):
        for summary in self.summaries.values():
            summary['per_case'] = [case for case in summary['per_case'] if case['kind'] == 'word']
            for index, case in enumerate(summary['per_case']):
                case['case_index'] = index
            count = len(summary['per_case'])
            summary['case_count'] = count
            summary['probe_metadata'].update(case_count=count, passage_count=count,
                                              output_shape=[count, 16], batch_bytes=count * 16 * 8)
            for name in ('losses', 'argmax'):
                summary['scores'][name]['bytes'] = count * 16 * 4
            derive_groups(summary)
        result = self.compare()
        self.assertIsNone(result['splits']['training']['control_models'])
        self.assertIsNone(result['splits']['training']['control_changes'])

    def test_input_mutation_rejected_and_no_output_created(self):
        self.write_inputs()
        original_load = readout._load
        calls = []

        def changing_load(path):
            value, provenance = original_load(path)
            calls.append(path)
            if len(calls) == 5:
                provenance['sha256'] = digest('changed')
            return value, provenance

        with mock.patch.object(readout, '_load', side_effect=changing_load):
            with self.assertRaisesRegex(ValueError, 'changed during readout'):
                readout.compare_summaries(*self.paths, self.output)
        self.assertFalse(self.output.exists())

    def test_cli_and_invalid_tolerances(self):
        self.write_inputs()
        args = []
        for model, path in zip(readout.MODELS, self.paths):
            args.extend(['--' + model.replace('_', '-'), str(path)])
        readout.main([*args, '--output', str(self.output)])
        self.assertEqual(json.loads(self.output.read_text())['format'], 'pluto-paired-patch-readout-v1')
        for value in (-1., float('inf'), float('nan'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                readout.compare_summaries(*self.paths, self.root / 'invalid.json', absolute_tolerance=value)

    def test_real_summarize_schema_with_synthetic_native_score_files(self):
        import numpy as np

        from . import paired_word_cases
        from .paired_word_cases_test import make_manifest

        manifest = make_manifest(self.root)
        prepared = self.root / 'prepared'
        plan = paired_word_cases.prepare(manifest, prepared, contexts_per_split=2,
                                          controls_per_split=2, prefix_tokens=4, context_length=16)
        shape = (plan['case_count'], plan['context_length'])
        for model_index, (model, path) in enumerate(zip(readout.MODELS, self.paths)):
            directory = self.root / model
            directory.mkdir()
            losses = np.full(shape, .5 + model_index * .1, dtype='<f4')
            argmax = np.zeros(shape, dtype='<i4')
            for case in plan['cases']:
                argmax[case['case_index'], case['scored_rows']] = case['target_ids']
            losses.tofile(directory / 'losses.f32.bin')
            argmax.tofile(directory / 'argmax.i32.bin')
            metadata = self.summaries[model]['probe_metadata'].copy()
            metadata.update(case_count=shape[0], passage_count=shape[0], context_length=shape[1],
                            output_shape=list(shape), batch_file=plan['packed_batch']['path'],
                            batch_bytes=plan['packed_batch']['bytes'])
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            paired_word_cases.summarize(prepared / 'cases.json', directory, path)
        result = readout.compare_summaries(*self.paths, self.output)
        self.assertEqual(result['splits']['training']['counts']['frozen_word_cases'], 8)
        self.assertEqual(result['splits']['training']['counts']['unique_word_evaluation_contexts'], 2)
        self.assertEqual(result['splits']['training']['counts']['occurrences_with_identical_prefix_domains'], 2)


if __name__ == '__main__':
    unittest.main()
