"""Synthetic segmentation tests; no real generation or lexical novelty claims."""

import copy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import generated_words as words
from . import verify


class GeneratedWordsTest(unittest.TestCase):
    def report(self, pieces, *, prompt=(), generated=None, extra=(), **kwargs):
        vocabulary = {i: piece for i, piece in enumerate([*pieces, *extra])}
        if generated is None:
            generated = list(range(len(pieces)))
        return words.analyze_generated_words(list(prompt), list(generated), vocabulary, **kwargs)

    def test_rare_structural_candidate_uses_actual_native_pieces(self):
        result = self.report([b' flur', b'bex', b'.'])
        self.assertFalse(result['retokenized'])
        self.assertEqual(result['candidate_word_indices'], [0])
        word = result['words'][0]
        self.assertEqual((word['word'], word['byte_start'], word['byte_end']), ('flurbex', 1, 8))
        self.assertEqual(word['generated_piece_count'], 2)
        self.assertEqual([o['bytes_hex'] for o in word['token_overlaps']], [b'flur'.hex(), b'bex'.hex()])
        self.assertTrue(word['token_overlaps'][0]['token_crosses_word_start'])
        self.assertEqual(word['token_overlaps'][0]['token_byte_start'], 1)
        self.assertTrue(word['lexical_commonness']['review_required'])
        self.assertFalse(result['lexical_review_complete'])

    def test_ordinary_multitoken_word_is_not_claimed_novel(self):
        result = self.report([b' micro', b'scope', b' '],
            lexicon={'microscope'}, lexicon_source='synthetic common-word fixture')
        word = result['words'][0]
        self.assertTrue(word['structural_candidate'])
        self.assertTrue(word['lexical_commonness']['whole_word_in_supplied_lexicon'])
        self.assertEqual(word['lexical_commonness']['status'], 'unreviewed')
        self.assertIn('do not establish novelty', word['lexical_commonness']['warning'])
        self.assertFalse(result['lexical_review_complete'])

    def test_simple_common_compound_is_separate_lexical_evidence(self):
        result = self.report([b' sun', b'flower', b' '],
            lexicon={'sun', 'flower'}, lexicon_source='synthetic word components')
        word = result['words'][0]
        self.assertTrue(word['structural_candidate'])
        self.assertFalse(word['lexical_commonness']['whole_word_in_supplied_lexicon'])
        self.assertEqual(word['lexical_commonness']['two_part_lexicon_splits'], [['sun', 'flower']])
        self.assertTrue(word['lexical_commonness']['review_required'])

    def test_case_and_optional_single_leading_space_whole_vocabulary_match(self):
        for spelling in [b'microscope', b' microscope', b'MICROSCOPE', b' MicroScope']:
            result = self.report([b' micro', b'scope', b'.'], extra=[spelling])
            word = result['words'][0]
            self.assertFalse(word['structural_candidate'])
            self.assertEqual(word['single_token_vocabulary_matches'], [{'id': 3, 'bytes_hex': spelling.hex()}])
        # The declared whole-token rule does not silently strip arbitrary
        # whitespace or punctuation surrounding a substring in a larger token.
        result = self.report([b' micro', b'scope', b'.'], extra=[b'  microscope', b'\nmicroscope', b'microscope!'])
        self.assertTrue(result['words'][0]['structural_candidate'])

    def test_single_token_words_not_resegmented_to_find_false_candidates(self):
        result = self.report([b' flurbex', b' '], extra=[b' flur', b'bex'])
        word = result['words'][0]
        self.assertEqual(word['generated_piece_count'], 1)
        self.assertFalse(word['structural_candidate'])
        self.assertIn('fewer_than_two_generated_pieces_contribute_letters', word['rejection_reasons'])
        # An alternate possible segmentation is irrelevant: only actual IDs
        # are retained, even when vocabulary components could spell the word.
        self.assertEqual([x['token_id'] for x in word['token_overlaps']], [0])

    def test_repeated_token_ids_are_distinct_occurrences_and_spaces_do_not_count(self):
        vocabulary = {0: b'zo', 1: b' ', 2: b'.'}
        result = words.analyze_generated_words([], [1, 0, 0, 2], vocabulary)
        word = result['words'][0]
        self.assertTrue(word['structural_candidate'])
        self.assertEqual(word['generated_piece_count'], 2)
        self.assertEqual([x['token_id'] for x in word['token_overlaps']], [0, 0])
        self.assertEqual([x['generated_index'] for x in word['token_overlaps']], [1, 2])

    def test_prompt_crossing_words_are_reported_but_not_eligible(self):
        result = self.report([b'flur', b'be', b'x ', b'jib', b'bet '], prompt=[0], generated=[1, 2, 3, 4])
        first, second = result['words']
        self.assertEqual(first['word'], 'flurbex')
        self.assertFalse(first['wholly_generated'])
        self.assertFalse(first['structural_candidate'])
        self.assertEqual(first['generated_piece_count'], 2)
        self.assertEqual([x['origin'] for x in first['token_overlaps']], ['prompt', 'generated', 'generated'])
        self.assertTrue(second['structural_candidate'])
        self.assertEqual(second['generated_byte_start'], 4)
        self.assertEqual(second['generated_byte_end'], 10)

    def test_token_crossing_multiple_word_boundaries_retains_byte_overlaps(self):
        result = self.report([b'foo flur', b'bex bar', b'.'])
        self.assertEqual([w['word'] for w in result['words']], ['foo', 'flurbex', 'bar'])
        word = result['words'][1]
        self.assertTrue(word['structural_candidate'])
        self.assertTrue(word['token_overlaps'][0]['token_crosses_word_start'])
        self.assertTrue(word['token_overlaps'][1]['token_crosses_word_end'])
        self.assertEqual([(x['token_byte_start'], x['token_byte_end']) for x in word['token_overlaps']], [(4, 8), (0, 3)])
        for row in result['words']:
            self.assertEqual(b''.join(bytes.fromhex(x['bytes_hex']) for x in row['token_overlaps']), row['word'].encode())

    def test_trailing_partial_word_waits_for_a_delimiter_event(self):
        vocabulary = {0: b' flur', 1: b'bex', 2: b',', 3: b'next'}
        partial = words.analyze_generated_words([], [0, 1], vocabulary)
        self.assertFalse(partial['words'][0]['complete'])
        self.assertEqual(partial['candidate_word_indices'], [])
        complete = words.analyze_generated_words([], [0, 1, 2, 3], vocabulary)
        self.assertTrue(complete['words'][0]['complete'])
        self.assertEqual(complete['candidate_word_indices'], [0])
        self.assertFalse(complete['words'][1]['complete'])

    def test_non_ascii_fragments_identifiers_and_compounds_are_conservatively_rejected(self):
        for pieces in [[b' ca', b'f', b'\xc3\xa9 '], [b'\xc3\xa9flur', b'bex '],
                       [b' flur', b'bex2 '], [b'3flur', b'bex '],
                       [b' flur', b'bex_ '], [b' flur', b"bex's "],
                       [b' flur', b'bex-like ']]:
            result = self.report(pieces)
            self.assertEqual(result['candidate_word_indices'], [], pieces)
        result = self.report([b' caf', b'\xc3', b'\xa9 flur', b'bex '])
        self.assertTrue(result['valid_utf8'])
        self.assertEqual([result['words'][i]['word'] for i in result['candidate_word_indices']], ['flurbex'])
        invalid = self.report([b' flur', b'bex', b'\xc3'])
        self.assertFalse(invalid['valid_utf8'])
        self.assertEqual(invalid['candidate_word_indices'], [])

    def test_special_token_text_or_touching_boundary_cannot_be_a_candidate(self):
        result = self.report([b' flur', b'bex', b'<|endoftext|>', b' '], special_token_ids=[2])
        self.assertEqual(result['candidate_word_indices'], [])
        self.assertTrue(result['tokens'][2]['special'])
        for word in result['words']:
            self.assertIn('word_overlaps_or_touches_special_token', word['rejection_reasons'])
        result = self.report([b'flur', b'bex', b' '], special_token_ids=[0])
        self.assertFalse(result['words'][0]['structural_candidate'])

    def test_empty_generated_stream_and_invalid_input_validation(self):
        result = words.analyze_generated_words([0], [], {0: b'hello'})
        self.assertEqual(result['candidate_word_indices'], [])
        for invalid in [{0: b''}, {True: b'a'}, {-1: b'a'}, {0: 'a'}, {}]:
            with self.assertRaises(ValueError):
                words.analyze_generated_words([], [], invalid)
        for bad_ids in [[True], [1], [-1], [0.0], ['0']]:
            with self.assertRaises(ValueError):
                words.analyze_generated_words([], bad_ids, {0: b'a'})
        with self.assertRaises(ValueError):
            self.report([b'foo'], lexicon=['foo'])
        with self.assertRaises(ValueError):
            self.report([b'foo'], lexicon=['caf\xe9'], lexicon_source='fixture')
        with self.assertRaises(ValueError):
            self.report([b'foo'], lexicon_source='without any lexicon')

    def native_fixture(self):
        vocabulary = {0: b'to be ', 1: b'flur', 2: b'bex', 3: b'.', 4: b'<|endoftext|>'}
        generated = [1, 2, 3]
        metadata = {'complete': True, 'prompt': 'to be ', 'initial_token_ids': [0],
                    'generated_token_ids': generated, 'all_token_ids': [0]+generated,
                    'steps': 3, 'special_token_ids': [4], 'vocab_size': len(vocabulary)}
        events, cursor = [], 0
        for index, token in enumerate(generated):
            events.append({'index': index, 'absolute_token_index': 1+index, 'token_id': token,
                           'piece_hex': vocabulary[token].hex(), 'generated_byte_start': cursor,
                           'generated_byte_end': cursor+len(vocabulary[token])})
            cursor += len(vocabulary[token])
        return metadata, events, vocabulary

    def test_native_events_are_verified_without_retokenizing(self):
        metadata, events, vocabulary = self.native_fixture()
        result = words.analyze_native_events(metadata, events, vocabulary, generated_bytes=b'flurbex.')
        self.assertTrue(result['native_event_bytes_verified'])
        self.assertFalse(result['sampling_or_logits_audited'])
        self.assertEqual([result['words'][i]['word'] for i in result['candidate_word_indices']], ['flurbex'])
        for field, value in [('index', 1), ('absolute_token_index', 0), ('token_id', 0),
                              ('generated_byte_start', 1), ('generated_byte_end', 5),
                              ('piece_hex', b'FLUR'.hex())]:
            changed = copy.deepcopy(events)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                words.analyze_native_events(metadata, changed, vocabulary)
        for field, value in [('complete', False), ('steps', True), ('steps', 2),
                              ('all_token_ids', [1, 2, 3]),
                              ('all_token_ids', [0.0, 1.0, 2.0, 3.0]),
                              ('all_token_ids', [False, 1, 2, 3]), ('prompt', 'to be')]:
            changed = dict(metadata, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                words.analyze_native_events(changed, events, vocabulary)
        with self.assertRaises(ValueError):
            words.analyze_native_events(metadata, events, vocabulary, generated_bytes=b'flurbex!')

    def test_cli_adapter_hashes_events_raw_bytes_ids_and_refuses_overwrite(self):
        metadata, events, vocabulary = self.native_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            native, tokenizer = base/'native', base/'tokenizer'
            native.mkdir()
            tokenizer.mkdir()
            (tokenizer/'tokenizer.json').write_text('{}')
            metadata['tokenizer_directory'] = str(tokenizer)
            metadata['files'] = {'events': {'file': 'events.jsonl'},
                                 'generated_bytes': {'file': 'generated.bin'},
                                 'tokens': {'file': 'tokens.i32', 'dtype': 'int32', 'shape': [4]}}
            (native/'metadata.json').write_text(json.dumps(metadata))
            (native/'events.jsonl').write_text('\n'.join(json.dumps(event) for event in events)+'\n')
            (native/'generated.bin').write_bytes(b'flurbex.')
            (native/'tokens.i32').write_bytes(struct.pack('<4i', *metadata['all_token_ids']))
            fake = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: object()))
            with mock.patch.dict('sys.modules', {'tokenizers': fake}), \
                    mock.patch.object(verify, 'gpt2_token_bytes', return_value=vocabulary):
                report = words.run(native, tokenizer, base/'report.json')
                self.assertTrue(report['all_checked_inputs_unchanged'])
                self.assertEqual(set(report['input_records']),
                    {'metadata', 'tokenizer', 'source', 'byte_decoder_source', 'events', 'generated_bytes', 'tokens'})
                with self.assertRaises(FileExistsError):
                    words.run(native, tokenizer, base/'report.json')
                (native/'tokens.i32').write_bytes(struct.pack('<4i', 0, 2, 1, 3))
                with self.assertRaisesRegex(ValueError, 'token artifact'):
                    words.run(native, tokenizer, base/'different.json')


if __name__ == '__main__':
    unittest.main()
