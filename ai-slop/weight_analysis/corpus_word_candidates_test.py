"""Synthetic word/byte/native-sequence lookup tests; no GPU or real corpus."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from . import corpus_word_candidates as cw
from . import generated_words


def fixture(corpus_pieces, generated_ids=None):
    pieces = [b' flur', b'bex', b'.']
    for piece in corpus_pieces:
        if piece not in pieces:
            pieces.append(piece)
    vocabulary = dict(enumerate(pieces))
    ids = [0, 1, 2] if generated_ids is None else generated_ids
    report = generated_words.analyze_generated_words([], ids, vocabulary)
    tokens = np.array([pieces.index(piece) for piece in corpus_pieces], dtype=np.int32)
    offsets = np.array([0, *np.cumsum([len(piece) for piece in corpus_pieces])], dtype=np.uint64)
    return report, b''.join(corpus_pieces), tokens, offsets, vocabulary


class CorpusWordTest(unittest.TestCase):
    def test_casefold_matches_preserve_actual_case_ids_and_word_piece_boundaries(self):
        f = fixture([b'flur', b'bex', b'. ', b'FLUR', b'BEX', b'.'])
        result = cw.analyze(*f, split_byte=9)
        word = result['candidates'][0]
        self.assertEqual(word['corpus_occurrence_count'], 2)
        self.assertEqual(word['corpus_occurrences_by_partition'],
                         dict(training_prefix=1, heldout_suffix=1, crosses_split=0))
        first, second = word['occurrences']
        self.assertEqual(first['spelling'], 'flurbex')
        self.assertEqual(second['spelling'], 'FLURBEX')
        self.assertTrue(first['exact_case_match'])
        self.assertFalse(second['exact_case_match'])
        self.assertFalse(first['native_token_ids_equal_generated_word_overlap_ids'])
        self.assertFalse(first['full_token_piece_bytes_equal_generated'])
        self.assertTrue(first['clipped_word_piece_bytes_equal_generated'])
        self.assertFalse(second['clipped_word_piece_bytes_equal_generated'])
        self.assertTrue(second['clipped_word_piece_ascii_casefold_equal_generated'])
        self.assertEqual((second['byte_start'], second['line'], second['byte_column']), (9, 1, 10))

    def test_independent_unicode_whole_word_boundaries_not_substrings(self):
        corpus = ("flurbex flurbexish preflurbex 3flurbex flurbex_ flurbex's "
                  "flurbex-like éflurbex flurbex\u0301 —flurbex—").encode()
        spans = cw.whole_word_spans(corpus, 'FLURBEX')
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0][:2], (0, 7))
        self.assertEqual(spans[-1][2:], ('—', '—'))
        self.assertEqual(cw.whole_word_spans(b'flurbex', 'flurbex'), [(0, 7, None, None)])
        for word in ('', 'flur-bex', 'flurbex2', 'é'):
            with self.assertRaises(ValueError):
                cw.whole_word_spans(corpus, word)

    def test_native_tokens_can_cross_both_word_boundaries(self):
        f = fixture([b'foo flur', b'bex bar', b'.'])
        word = cw.analyze(*f, split_byte=len(f[1]))['candidates'][0]
        occurrence = word['occurrences'][0]
        self.assertEqual(occurrence['native_span_byte_start'], 0)
        self.assertEqual(occurrence['native_span_byte_end'], 15)
        self.assertEqual((occurrence['byte_start'], occurrence['byte_end']), (4, 11))
        first, last = occurrence['native_token_overlaps']
        self.assertTrue(first['token_crosses_word_start'])
        self.assertTrue(last['token_crosses_word_end'])
        self.assertEqual(first['offset_within_token_start'], 4)
        self.assertEqual(last['offset_within_token_end'], 3)
        self.assertEqual(b''.join(bytes.fromhex(x['overlap_piece_hex'])
                                for x in occurrence['native_token_overlaps']), b'flurbex')

    def test_identical_word_text_can_have_different_native_piece_count(self):
        f = fixture([b'flurbex!'])  # A punctuation-containing vocabulary piece.
        word = cw.analyze(*f, split_byte=7)['candidates'][0]
        occurrence = word['occurrences'][0]
        self.assertEqual(len(occurrence['native_token_ids']), 1)
        self.assertEqual(len(word['generated_word_overlap_token_ids']), 2)
        self.assertFalse(occurrence['clipped_word_piece_bytes_equal_generated'])
        self.assertFalse(occurrence['native_token_ids_equal_generated_word_overlap_ids'])
        self.assertEqual(occurrence['word_partition'], 'training_prefix')
        self.assertEqual(occurrence['native_token_span_partition'], 'crosses_split')

    def test_exact_native_id_sequence_match_is_reported_without_reencoding(self):
        f = fixture([b' flur', b'bex', b'.'])
        word = cw.analyze(*f, split_byte=0)['candidates'][0]
        self.assertEqual(word['exact_native_token_id_sequence_occurrence_count'], 1)
        occurrence = word['occurrences'][0]
        self.assertTrue(occurrence['full_token_piece_bytes_equal_generated'])
        self.assertEqual(occurrence['word_partition'], 'heldout_suffix')
        self.assertTrue(occurrence['native_token_overlaps'][0]['token_crosses_word_start'])

    def test_custom_split_can_cross_word_or_token_and_default_is_explicit(self):
        f = fixture([b'flur', b'bex', b'. more text'])
        crossed = cw.analyze(*f, split_byte=3)
        self.assertEqual(crossed['candidates'][0]['occurrences'][0]['word_partition'], 'crosses_split')
        self.assertFalse(crossed['split']['is_native_token_boundary'])
        self.assertEqual(crossed['split']['native_tokens_crossing_split'], 1)
        boundary = cw.analyze(*f, split_byte=4)
        self.assertTrue(boundary['split']['is_native_token_boundary'])
        self.assertEqual(boundary['split']['native_tokens_crossing_split'], 0)
        default = cw.analyze(*f)
        self.assertEqual(default['split']['byte_boundary'], cw.cv.split_boundary(f[1]))
        self.assertEqual(default['split']['source'], 'current_default_policy')
        for value in (-1, len(f[1])+1, True, 3.0):
            with self.assertRaises(ValueError):
                cw.analyze(*f, split_byte=value)

    def test_line_and_character_columns_differ_for_utf8_prefix(self):
        f = fixture([b'first\n', 'é '.encode(), b'flur', b'bex', b'!'])
        occurrence = cw.analyze(*f, split_byte=0)['candidates'][0]['occurrences'][0]
        self.assertEqual((occurrence['line'], occurrence['byte_column'], occurrence['character_column']), (2, 4, 3))
        self.assertEqual(occurrence['byte_start'], 9)

    def test_repeated_candidates_and_zero_hits_are_retained_without_classification(self):
        f = fixture([b'nothing here.'], generated_ids=[0, 1, 2, 0, 1, 2])
        result = cw.analyze(*f, split_byte=0)
        self.assertEqual(result['candidate_count'], 2)
        self.assertEqual([x['word_index'] for x in result['candidates']], [0, 1])
        self.assertEqual([x['corpus_occurrence_count'] for x in result['candidates']], [0, 0])
        self.assertEqual([x['unfamiliarity'] for x in result['candidates']], ['not_assessed']*2)
        self.assertFalse(result['target_selected'])
        self.assertFalse(result['unfamiliarity_assessed'])
        partial = fixture([b'nothing here.'], generated_ids=[0, 1])
        self.assertEqual(cw.analyze(*partial, split_byte=0)['candidate_count'], 0)

    def test_invalid_utf8_and_split_codepoint_rejected(self):
        f = fixture([b'\xff'])
        with self.assertRaises(UnicodeDecodeError):
            cw.analyze(*f, split_byte=0)
        f = fixture(['é '.encode(), b'flur', b'bex', b'!'])
        with self.assertRaisesRegex(ValueError, 'UTF-8 codepoint'):
            cw.analyze(*f, split_byte=1)
        f = list(fixture([b'\xc3']))
        f[0] = generated_words.analyze_generated_words([], [0, 1, 2, 3], f[4])
        with self.assertRaises(ValueError):
            cw.validate_generated_report(f[0], f[4])

    def test_invalid_native_export_and_tampered_generated_records_rejected(self):
        f = fixture([b' flur', b'bex', b'.'])
        for tokens, offsets in ((np.array([-1, 1, 2]), f[3]),
                                (f[2], f[3].astype(float)),
                                (f[2], np.array([0, 4, 8, 9])),
                                (f[2], f[3][:-1])):
            with self.assertRaises(ValueError):
                cw.analyze(f[0], f[1], tokens, offsets, f[4], split_byte=0)
        for mutate in (lambda x: x['candidate_word_indices'].clear(),
                       lambda x: x['words'][0].update(word='flurbexx'),
                       lambda x: x['tokens'][0].update(special=1),
                       lambda x: x.update(schema_version=True)):
            bad = copy.deepcopy(f[0]); mutate(bad)
            with self.assertRaises(ValueError):
                cw.analyze(bad, *f[1:], split_byte=0)


class CliIntegrityTest(unittest.TestCase):
    def make_files(self, root):
        f = fixture([b'flur', b'bex', b'! other text'])
        report, corpus, tokens, offsets, vocabulary = f
        tokenizer = root/'tokenizer'; tokenizer.mkdir()
        (tokenizer/'tokenizer.json').write_text('{}')
        native = root/'native'; native.mkdir()
        ids = report['generated_token_ids']
        metadata = dict(complete=True, prompt='', initial_token_ids=[], generated_token_ids=ids,
                        all_token_ids=ids, steps=len(ids), special_token_ids=[],
                        vocab_size=len(vocabulary), tokenizer_directory=str(tokenizer))
        (native/'metadata.json').write_text(json.dumps(metadata))
        cursor = 0; events = []
        for index, token in enumerate(ids):
            piece = vocabulary[token]
            events.append(dict(index=index, absolute_token_index=index, token_id=token,
                               generated_byte_start=cursor, generated_byte_end=cursor+len(piece), piece_hex=piece.hex()))
            cursor += len(piece)
        (native/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
        (native/'generated.bin').write_bytes(b''.join(vocabulary[t] for t in ids))
        (native/'tokens.i32').write_bytes(np.array(ids,dtype='<i4').tobytes())
        report['native_event_bytes_verified'] = True
        report['all_checked_inputs_unchanged'] = True
        report['input_records'] = {key: cw.cv.file_record(path) for key,path in dict(
            metadata=native/'metadata.json', tokenizer=tokenizer/'tokenizer.json',
            source=Path(generated_words.__file__), byte_decoder_source=Path(cw.__file__).with_name('verify.py'),
            events=native/'events.jsonl', generated_bytes=native/'generated.bin', tokens=native/'tokens.i32').items()}
        (root/'generated_words.json').write_text(json.dumps(report))
        (root/'corpus').write_bytes(corpus)
        (root/'tokens').write_bytes(tokens.astype('<u4').tobytes())
        (root/'offsets').write_bytes(offsets.astype('<u8').tobytes())
        return vocabulary

    def call(self, root, vocabulary):
        with mock.patch('tokenizers.Tokenizer', SimpleNamespace(from_file=lambda _: object())), \
             mock.patch.object(cw, 'gpt2_token_bytes', return_value=vocabulary):
            return cw.run(root/'generated_words.json', root/'corpus', root/'tokens', root/'offsets',
                          root/'tokenizer', root/'result.json', split_byte=7)

    def test_end_to_end_hashes_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); vocabulary = self.make_files(root)
            result = self.call(root, vocabulary)
            self.assertTrue(result['complete'])
            self.assertTrue(result['all_checked_inputs_unchanged'])
            self.assertEqual(result['candidates_with_training_prefix_occurrences'], 1)
            for record in [*result['inputs'].values(), *result['sources'].values(),
                           *result['generated_origin_records'].values()]:
                self.assertEqual(cw.cv.file_record(record['path']), record)
            before = (root/'result.json').read_bytes()
            with self.assertRaises(FileExistsError):
                self.call(root, vocabulary)
            self.assertEqual(before, (root/'result.json').read_bytes())

    def test_changed_inputs_fail_before_completed_output(self):
        for when in ('before', 'during'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); vocabulary = self.make_files(root)
                if when == 'before':
                    (root/'native/events.jsonl').write_text('changed')
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        self.call(root, vocabulary)
                else:
                    original = cw.analyze
                    def mutate(*args, **kwargs):
                        result = original(*args, **kwargs)
                        (root/'corpus').write_bytes((root/'corpus').read_bytes()+b'changed')
                        return result
                    with mock.patch.object(cw, 'analyze', side_effect=mutate):
                        with self.assertRaisesRegex(ValueError, 'changed'):
                            self.call(root, vocabulary)
                self.assertFalse((root/'result.json').exists())

    def test_provenance_and_metadata_substitution_are_rejected(self):
        for field in ('vocab_size', 'tokenizer_directory', 'source'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); vocabulary = self.make_files(root)
                report = json.loads((root/'generated_words.json').read_text())
                if field == 'source':
                    alternate = root/'unrelated_source.py'
                    alternate.write_text('not the frozen decoder')
                    report['input_records']['source'] = cw.cv.file_record(alternate)
                else:
                    path = root/'native/metadata.json'
                    metadata = json.loads(path.read_text())
                    metadata[field] = True if field == 'vocab_size' else str(root/'other_tokenizer')
                    path.write_text(json.dumps(metadata))
                    report['input_records']['metadata'] = cw.cv.file_record(path)
                (root/'generated_words.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'provenance|metadata'):
                    self.call(root, vocabulary)
                self.assertFalse((root/'result.json').exists())

    def test_partial_native_export_element_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); vocabulary = self.make_files(root)
            (root/'offsets').write_bytes((root/'offsets').read_bytes()+b'x')
            with self.assertRaisesRegex(ValueError, 'partial'):
                self.call(root, vocabulary)
            self.assertFalse((root/'result.json').exists())


if __name__ == '__main__':
    unittest.main()
