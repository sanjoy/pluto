"""CPU float fixtures for deduplicated, stratified outer-factorial summaries."""

import copy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import paired_outer_factorial_summary as summary


CELLS = ('A', 'E', 'C', 'EC', 'D')
PAIRS = {'title': {'original': 'Exeunt', 'replacement': 'Nuveth'},
         'lowercase': {'original': 'exeunt', 'replacement': 'nuveth'}}


def case_row(prefix='prefix', *, role='original', kind='word', split='test',
             domain='original', variant='title', leading=False, piece_id=None,
             target_ids=None, scores=None, alias=0):
    word = kind in ('word', 'word_next_native')
    if target_ids is None:
        target_ids = [1, 2, 3] if role == 'original' else [4, 5, 6]
        if kind == 'word_next_native':
            target_ids.append(7)
    identity = dict(native_case_index=alias, source_case_index=alias,
        context_id=f'context-{alias}', kind=kind, split=split, prefix_domain=domain,
        target=PAIRS[variant][role] if word else 'ordinary-control',
        target_ids=list(target_ids), spelling_variant=variant if word else None,
        candidate_pair=PAIRS[variant] if word else None, target_source_domain=role,
        prediction_prefixes=[dict(prediction_row=9 + position,
            causal_prefix_length=10 + position,
            causal_prefix_sha256=hashlib.sha256(
                repr((prefix, target_ids[:position])).encode()).hexdigest())
            for position in range(len(target_ids))])
    if word:
        identity['word_has_leading_space'] = leading
    if piece_id is not None:
        identity['piece_id'] = piece_id
    if scores is None:
        scores = {cell: [-float(index + 1)] * len(target_ids)
                  for index, cell in enumerate(CELLS)}
    return dict(case=identity, cells={cell: {'token_log_probability': list(scores[cell])}
                                      for cell in CELLS})


def readout(*rows):
    return dict(per_case=list(rows), case_count=len(rows))


class SummaryCasesTest(unittest.TestCase):
    def test_requested_prefix_domain_is_exact_and_must_be_supported(self):
        rows = readout(case_row('original', domain='original'),
                       case_row('replacement', domain='replacement'))
        for domain in ('original', 'replacement'):
            result = summary.summarize_cases(rows, prefix_domain=domain)
            self.assertEqual(result['prefix_domain'], domain)
            self.assertEqual(len(result['word_groups']), 1)
            self.assertEqual(result['word_groups'][0]['raw_case_count'], 1)
        for domain in ('shared', 'all', '', None):
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                summary.summarize_cases(rows, prefix_domain=domain)

    def test_kinds_splits_spelling_and_leading_space_are_never_pooled(self):
        rows = [case_row(f'{kind}-{split}-{variant}-{leading}', kind=kind,
                        split=split, variant=variant, leading=leading)
                for kind in ('word', 'word_next_native')
                for split in ('training', 'test')
                for variant in ('title', 'lowercase') for leading in (False, True)]
        groups = summary.summarize_cases(readout(*rows))['word_groups']
        self.assertEqual(len(groups), 16)
        self.assertEqual(len({(g['kind'], g['split'], g['spelling_variant'],
                              g['word_has_leading_space']) for g in groups}), 16)
        self.assertTrue(all(g['raw_case_count'] == g['event_count'] == 1 for g in groups))

    def test_original_and_replacement_absolute_word_scores_remain_separate(self):
        result = summary.summarize_cases(readout(case_row(role='original'),
                                                 case_row(role='replacement')))
        self.assertEqual({g['target'] for g in result['word_groups']}, {'Exeunt', 'Nuveth'})
        self.assertEqual(len(result['word_groups']), 2)

    def test_context_aliases_count_once_by_actual_prefix_and_target_ids(self):
        result = summary.summarize_cases(readout(case_row(alias=1), case_row(alias=999)))
        group = result['word_groups'][0]
        self.assertEqual(group['raw_case_count'], 2)
        self.assertEqual(group['event_count'], 1)
        self.assertTrue(all(metric['event_count'] == 1 for metric in group['metrics'].values()))

    def test_conflicting_duplicate_cell_evidence_is_rejected(self):
        original = case_row(alias=1)
        duplicate = case_row(alias=2)
        duplicate['cells']['EC']['token_log_probability'][2] -= 0.25
        with self.assertRaises(ValueError):
            summary.summarize_cases(readout(original, duplicate))

    def test_prefix_length_is_part_of_event_identity(self):
        short = case_row(alias=1)
        long = case_row(alias=2)
        for prefix in long['case']['prediction_prefixes']:
            prefix['causal_prefix_length'] += 1
            prefix['prediction_row'] += 1
        group = summary.summarize_cases(readout(short, long))['word_groups'][0]
        self.assertEqual(group['event_count'], 2)
        self.assertEqual(group['metrics']['first']['event_count'], 2)

    def test_shared_controls_are_retained_and_shared_pieces_stay_separate(self):
        rows = [case_row('control', kind='control', domain='shared'),
                case_row('piece-7', kind='shared_piece', domain='shared', piece_id=7),
                case_row('piece-8', kind='shared_piece', domain='shared', piece_id=8)]
        result = summary.summarize_cases(readout(*rows))
        self.assertEqual(result['word_groups'], [])
        groups = result['control_groups']
        self.assertEqual(len(groups), 3)
        self.assertEqual({g['piece_id'] for g in groups if g['kind'] == 'shared_piece'}, {7, 8})
        self.assertTrue(all(set(g['metrics']) == {'first', 'suffix', 'sequence'} for g in groups))

    def test_metrics_use_mean_log_scores_geometric_probabilities_and_outer_contrasts(self):
        scores = dict(A=[-1.0, -2.0, -3.0], E=[-1.0, -1.0, -1.0],
                      C=[-2.0, -2.0, -2.0], EC=[-.25, -.25, -.5], D=[-.25, -.5, -.75])
        scaled = {cell: [3 * value for value in row] for cell, row in scores.items()}
        document = readout(case_row('one', scores=scores), case_row('two', scores=scaled))
        before = copy.deepcopy(document)
        metrics = summary.summarize_cases(document)['word_groups'][0]['metrics']
        self.assertEqual(document, before)
        self.assertEqual(metrics['first']['cells']['A']['mean_log_probability'], -2.0)
        self.assertEqual(metrics['suffix']['cells']['A']['mean_log_probability'], -10.0)
        metric = metrics['word_three']
        expected = dict(A=-12.0, E=-6.0, C=-12.0, EC=-2.0, D=-3.0)
        for cell, value in expected.items():
            self.assertEqual(metric['cells'][cell]['mean_log_probability'], value)
            self.assertAlmostEqual(metric['cells'][cell]['geometric_probability'], math.exp(value))
        self.assertEqual(metric['effects'], dict(E_minus_A=6.0, C_minus_A=0.0,
            EC_minus_A=10.0, EC_minus_E=4.0, EC_minus_C=10.0,
            interaction=4.0, D_minus_EC=-1.0))
        self.assertEqual(metrics['sequence'], metrics['word_three'])

    def test_exact_fourth_token_is_separate_from_the_three_token_word(self):
        scores = {cell: [-1.0, -2.0, -3.0, -4.0] for cell in CELLS}
        row = case_row(kind='word_next_native', scores=scores)
        metrics = summary.summarize_cases(readout(row))['word_groups'][0]['metrics']
        for name, expected in (('first', -1.0), ('suffix', -5.0), ('word_three', -6.0),
                               ('next_fourth', -4.0), ('sequence', -10.0)):
            self.assertEqual(metrics[name]['cells']['A']['mean_log_probability'], expected)
        word_metrics = summary.summarize_cases(readout(case_row()))['word_groups'][0]['metrics']
        self.assertNotIn('next_fourth', word_metrics)

    def test_alternative_following_tokens_do_not_reweight_word_triple_metrics(self):
        first = case_row(kind='word_next_native', target_ids=[1, 2, 3, 7],
                         scores={cell: [-1.0, -2.0, -3.0, -4.0] for cell in CELLS})
        second = case_row(kind='word_next_native', target_ids=[1, 2, 3, 8],
                          scores={cell: [-1.0, -2.0, -3.0, -5.0] for cell in CELLS})
        group = summary.summarize_cases(readout(first, second))['word_groups'][0]
        self.assertEqual(group['event_count'], 2)
        for name in ('first', 'suffix', 'word_three'):
            self.assertEqual(group['metrics'][name]['event_count'], 1)
        for name in ('next_fourth', 'sequence'):
            self.assertEqual(group['metrics'][name]['event_count'], 2)
        self.assertEqual(group['metrics']['next_fourth']['cells']['A']['mean_log_probability'], -4.5)

    def test_preferences_count_ties_and_all_directions_of_flip_from_a(self):
        rows = []
        original_totals = (-3.0, -3.0, -6.0)
        replacement_totals = (dict(A=-6.0, E=-1.5, C=-3.0, EC=-6.0, D=-1.5),
                              dict(A=-3.0, E=-1.5, C=-6.0, EC=-3.0, D=-6.0),
                              dict(A=-3.0, E=-9.0, C=-6.0, EC=-3.0, D=-9.0))
        for index, (original, replacement) in enumerate(zip(original_totals, replacement_totals)):
            rows.append(case_row(str(index), role='original',
                scores={cell: [original / 3] * 3 for cell in CELLS}))
            rows.append(case_row(str(index), role='replacement',
                scores={cell: [replacement[cell] / 3] * 3 for cell in CELLS}))
        group = summary.summarize_cases(readout(*rows))['preference_groups'][0]
        self.assertEqual(group['context_count'], 3)
        self.assertEqual(group['unpaired_context_count'], 0)
        self.assertEqual(group['counts']['A'], dict(original_preferred=1, replacement_preferred=1, ties=1))
        self.assertEqual(group['counts']['E'], dict(original_preferred=1, replacement_preferred=2, ties=0))
        self.assertEqual(group['counts']['C'], dict(original_preferred=1, replacement_preferred=0, ties=2))
        self.assertEqual(group['flips_from_A']['E'], dict(original_to_replacement=1,
            replacement_to_original=1, tie_to_decisive=1, decisive_to_tie=0))
        self.assertEqual(group['flips_from_A']['C'], dict(original_to_replacement=0,
            replacement_to_original=0, tie_to_decisive=1, decisive_to_tie=2))
        self.assertEqual(group['mean_log_odds']['A'], 0.0)
        self.assertEqual(group['mean_log_odds']['E'], 0.0)
        self.assertEqual(group['mean_log_odds']['C'], -1.0)

    def test_candidate_matching_uses_actual_prefix_and_deduplicates_aliases(self):
        rows = readout(case_row(role='original', alias=1), case_row(role='original', alias=2),
                       case_row(role='replacement', alias=999))
        group = summary.summarize_cases(rows)['preference_groups'][0]
        self.assertEqual(group['context_count'], 1)
        self.assertEqual(group['unpaired_context_count'], 0)
        self.assertEqual(group['counts']['A']['ties'], 1)

    def test_unpaired_contexts_are_reported_and_excluded_from_preference_means(self):
        rows = readout(case_row('paired', role='original'), case_row('paired', role='replacement'),
                       case_row('unpaired', role='original',
                                scores={cell: [-100.0] * 3 for cell in CELLS}))
        group = summary.summarize_cases(rows)['preference_groups'][0]
        self.assertEqual(group['context_count'], 1)
        self.assertEqual(group['unpaired_context_count'], 1)
        self.assertEqual(group['mean_log_odds'], {cell: 0.0 for cell in CELLS})

    def test_fourth_token_never_changes_three_token_word_preference(self):
        rows = readout(case_row(role='original', kind='word_next_native',
                               scores={cell: [-2.0, -2.0, -2.0, -.1] for cell in CELLS}),
                       case_row(role='replacement', kind='word_next_native',
                                scores={cell: [-1.0, -1.0, -1.0, -100.0] for cell in CELLS}))
        group = summary.summarize_cases(rows)['preference_groups'][0]
        self.assertEqual(group['mean_log_odds'], {cell: 3.0 for cell in CELLS})
        self.assertTrue(all(group['counts'][cell]['replacement_preferred'] == 1 for cell in CELLS))

    def test_nonfinite_or_missing_outer_cells_cannot_produce_summary_metrics(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            row = case_row()
            row['cells']['EC']['token_log_probability'][1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summary.summarize_cases(readout(row))
        row = case_row()
        del row['cells']['D']
        with self.assertRaises((ValueError, KeyError)):
            summary.summarize_cases(readout(row))

    def test_same_prefix_candidate_with_different_word_ids_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            summary.summarize_cases(readout(case_row(target_ids=[1, 2, 3]),
                                            case_row(target_ids=[1, 2, 9])))

    def test_positive_log_probability_is_rejected(self):
        row = case_row()
        row['cells']['A']['token_log_probability'][0] = .001
        with self.assertRaises(ValueError):
            summary.summarize_cases(readout(row))


class ProvenanceGateTest(unittest.TestCase):
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)/'result.json'
            output.write_text('keep')
            with self.assertRaisesRegex(ValueError, 'must be new'):
                summary.analyze(Path(name)/'missing.json', output)
            self.assertEqual(output.read_text(), 'keep')

    def test_output_cannot_enter_live_runner(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(ValueError, 'outside the live'):
                summary.analyze(root/'run/direction/main_outer_readout.json', root/'run/new.json')

    def test_inner_readout_must_match_bound_hash(self):
        with mock.patch.object(summary.training, 'record', return_value={'sha256':'new'}):
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                summary._inner_records('unused', {'sha256':'old'}, {})

    def test_readout_does_not_imply_native_execution(self):
        record = {'path':'unused', 'sha256':'same'}
        for verified in (False, None):
            report = dict(format='pluto-embedding-factorial-readout-v1',
                          execution_provenance={'verified':verified})
            with mock.patch.object(summary.training, 'record', return_value=record), \
                 mock.patch.object(summary.reader, '_json', return_value=report):
                with self.assertRaisesRegex(ValueError, 'verified execution'):
                    summary._inner_records('unused', record, {})

    def test_incomplete_native_metadata_is_rejected(self):
        record = {'path':'unused', 'sha256':'same'}
        metadata_record = {'path':'metadata', 'sha256':'same'}
        report = dict(format='pluto-embedding-factorial-readout-v1',
                      execution_provenance={'verified':True},
                      files=[metadata_record], native_metadata=metadata_record)
        with mock.patch.object(summary.training, 'record', return_value=record), \
             mock.patch.object(summary.reader, '_json', side_effect=[report, {'complete':False}]):
            with self.assertRaisesRegex(ValueError, 'native completion'):
                summary._inner_records('unused', record, {})


if __name__ == '__main__':
    unittest.main()
