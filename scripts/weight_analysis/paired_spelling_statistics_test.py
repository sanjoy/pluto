"""Independent small corpus oracles; never invoke a tokenizer or GPU."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_spelling_statistics as audit
from . import paired_training, paired_word_cases


def pattern(result, ids):
    return next(p for p in result['patterns'] if p['token_ids'] == list(ids))


class SequenceStatisticsTest(unittest.TestCase):
    def test_scalar_pair_triple_and_unigram_oracle(self):
        tokens = [198, 1475, 68, 2797, 13, 409, 68, 2797,
                  1086, 2797, 3109, 68, 2797, 68, 13, 68]
        starts = [1, 10]
        result = audit.sequence_statistics(tokens, starts)
        counts = Counter(tokens)
        for token in audit.PIECES:
            item = result['pieces'][str(token)]
            pairs = Counter(b for a, b in zip(tokens, tokens[1:]) if a == token)
            self.assertEqual(item['occurrences'], counts[token])
            self.assertEqual(item['next_token_target_count'], tokens[1:].count(token))
            self.assertEqual(item['successor_opportunities'], tokens[:-1].count(token))
            self.assertEqual({v['token_id']: v['count'] for v in item['successors']}, pairs)
        for ids in audit.PATTERNS:
            item = pattern(result, ids)
            found = [i for i in range(len(tokens)-len(ids)+1)
                     if tokens[i:i+len(ids)] == list(ids)]
            inside = sum(any(set(range(i, i+len(ids))) <= set(range(s, s+3))
                             for s in starts) for i in found)
            self.assertEqual(item['occurrences'], len(found))
            self.assertEqual(item['inside_one_replacement_span'], inside)

    def test_first_last_token_denominators(self):
        result = audit.sequence_statistics([68, 2797, 68], [])
        e = result['pieces']['68']
        self.assertEqual(e['occurrences'], 2)
        self.assertEqual(e['next_token_target_count'], 1)
        self.assertEqual(e['successor_opportunities'], 1)
        self.assertEqual(e['next_token_target_frequency'], .5)
        self.assertEqual(pattern(result, (68, 2797))['conditional_last_token_frequency'], 1)
        # A terminal Ex/e prefix cannot be in the denominator for predicting unt.
        terminal = audit.sequence_statistics([1475, 68], [])
        self.assertIsNone(pattern(terminal, (1475, 68, 2797))['conditional_last_token_frequency'])

    def test_split_boundary_never_supplies_a_transition(self):
        train, test = [198, 68], [2797, 198]
        for split in (train, test):
            self.assertEqual(pattern(audit.sequence_statistics(split, []),
                                     (68, 2797))['occurrences'], 0)
        self.assertEqual(pattern(audit.sequence_statistics(train+test, []),
                                 (68, 2797))['occurrences'], 1)

    def test_adjacent_spans_are_not_one_span(self):
        result = audit.sequence_statistics([68, 2797, 68, 2797, 1475, 68], [0, 3])
        pair = pattern(result, (68, 2797))
        self.assertEqual(pair['occurrences'], 2)
        self.assertEqual(pair['inside_one_replacement_span'], 1)
        self.assertEqual(pair['other_span_overlap'], 1)
        self.assertEqual(pair['wholly_outside_replacement_spans'], 0)

    def test_lowercase_survives_uppercase_replacement(self):
        original = [1475, 68, 2797, 198, 409, 68, 2797, 198, 1086, 2797]
        replacement = [21733, 303, 400, *original[3:]]
        before, after = (audit.sequence_statistics(t, [0]) for t in (original, replacement))
        self.assertEqual(pattern(before, (68, 2797))['occurrences'], 2)
        self.assertEqual(pattern(after, (68, 2797))['occurrences'], 1)
        self.assertEqual(pattern(after, (409, 68, 2797))['wholly_outside_replacement_spans'], 1)
        self.assertEqual(after['pieces']['2797']['outside_replacement_spans'], 2)
        self.assertEqual(pattern(after, (1086, 2797))['occurrences'], 1)

    def test_empty_and_single_token_have_no_conditional_estimate(self):
        for tokens in ([], [68]):
            result = audit.sequence_statistics(tokens, [])
            self.assertEqual(result['next_token_target_count'], 0)
            for p in result['patterns']:
                self.assertEqual(p['occurrences'], 0)
                self.assertIsNone(p['conditional_last_token_frequency'])

    def test_bad_ids_and_span_geometry(self):
        for tokens in ([True], [1.0], [-1], [50257], [[68]], ['68']):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                audit.sequence_statistics(tokens, [])
        for starts in ([True], [-1], [4], [1, 0], [0, 2], [0, 0], [1.0]):
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                audit.sequence_statistics([198]*6, starts)


class FrozenInputTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        spellings = {**audit.KNOWN_BYTES, 198: b'\n', 13: b'.',
                     21733: b' Nu', 303: b've', 400: b'th', 45: b'N', 45177: b'uve'}
        train = [198, 1475, 68, 2797, 198, 409, 68, 2797, 13, 1086, 2797, 198]
        test = [198, 3109, 68, 2797, 198, 68, 13]
        streams = {'original.training': train, 'original.test': test,
                   'replacement.training': [198, 21733, 303, 400, *train[4:]],
                   'replacement.test': [198, 45, 45177, 400, *test[4:]]}
        for domain in audit.DOMAINS:
            streams[domain+'.full'] = streams[domain+'.training']+streams[domain+'.test']
        self.manifest = dict(format='pluto-paired-corpus-training-v1',
                             replacement={'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True},
                             inputs={}, alignment={})
        arrays = {}
        for name, values in streams.items():
            text = b''.join(spellings[v] for v in values)
            ids = np.asarray(values, dtype='<u4')
            offsets = np.asarray([0, *np.cumsum([len(spellings[v]) for v in values])], dtype='<u8')
            arrays[name] = (text, ids, offsets)
            paths = {'text': self.root/(name+'.txt'), 'token_ids': self.root/(name+'.tokens.bin'),
                     'offsets': self.root/(name+'.offsets.bin')}
            for key, payload in zip(paths, (text, ids.tobytes(), offsets.tobytes())):
                paths[key].write_bytes(payload)
            item = {key: str(path) for key, path in paths.items()}
            item.update({key+'_sha256': paired_word_cases._record(path)['sha256']
                         for key, path in paths.items()})
            item['export'] = dict(token_dtype='<u4', offset_dtype='<u8', roundtrip_verified=True,
                                  token_count=len(ids), offset_count=len(offsets), corpus_bytes=len(text))
            self.manifest['inputs'][name] = item
        self.manifest['split_byte'] = len(arrays['original.training'][0])
        for split in ('full', *audit.SPLITS):
            a, b = (arrays[d+'.'+split] for d in audit.DOMAINS)
            self.manifest['alignment'][split] = paired_training.verify_alignment(
                a[0], b[0], a[1], b[1], a[2], b[2])
        (self.root/'word_cases').mkdir()
        self.publish_manifest()

    def publish_manifest(self):
        path = self.root/'manifest.json'
        path.write_text(json.dumps(self.manifest))
        (self.root/'word_cases/cases.json').write_text(json.dumps(dict(
            format='pluto-paired-word-cases-v1', manifest=paired_word_cases._record(path))))

    def test_native_report_preserves_shared_spelling_and_provenance(self):
        report = audit.analyze(self.root)
        self.assertFalse(report['inference_performed'])
        self.assertFalse(report['gpu_work_performed'])
        self.assertEqual(len(report['provenance']), 20)
        self.assertEqual(len(report['sources']), 4)
        self.assertTrue(report['full_and_split_exports_identical'])
        train = report['statistics']['training']
        self.assertEqual(train['original']['case_insensitive_exeunt_text_counts'],
                         {'Exeunt': 1, 'exeunt': 1})
        self.assertEqual(train['replacement']['case_insensitive_exeunt_text_counts'], {'exeunt': 1})
        example = train['replacement']['e_unt_outside_replacement_examples'][0]
        self.assertEqual(example['predecessor_token_id'], 409)
        self.assertEqual(report['statistics']['test']['original']['full_text_byte_base'],
                         self.manifest['split_byte'])
        self.assertIn('denominator_definitions', report)

    def test_tampered_input_hash_rejected(self):
        path = Path(self.manifest['inputs']['original.training']['token_ids'])
        path.write_bytes(b'\0'*path.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit.analyze(self.root)

    def test_wrong_split_and_alignment_rejected(self):
        self.manifest['split_byte'] += 1
        self.publish_manifest()
        with self.assertRaisesRegex(ValueError, 'split concatenation'):
            audit.analyze(self.root)
        self.manifest['split_byte'] -= 1
        self.manifest['alignment']['training']['replacements'] += 1
        self.publish_manifest()
        with self.assertRaisesRegex(ValueError, 'alignment differs'):
            audit.analyze(self.root)

    def test_changed_manifest_anchor_rejected(self):
        self.manifest['split_byte'] += 1
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'authenticate'):
            audit.analyze(self.root)

    def test_input_change_during_computation_rejected(self):
        original = audit.sequence_statistics
        altered = False

        def mutate(*args):
            nonlocal altered
            if not altered:
                altered = True
                path = Path(self.manifest['inputs']['original.training']['text'])
                path.write_bytes(path.read_bytes()+b'!')
            return original(*args)

        with mock.patch.object(audit, 'sequence_statistics', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                audit.analyze(self.root)

    def test_exclusive_output_and_serializable_report(self):
        path = self.root/'report.json'
        audit.main(['--root', str(self.root), '--output', str(path)])
        payload = path.read_bytes()
        self.assertEqual(json.loads(payload)['format'], 'pluto-paired-spelling-statistics-v1')
        with self.assertRaises(FileExistsError):
            audit.main(['--root', str(self.root), '--output', str(path)])
        self.assertEqual(path.read_bytes(), payload)


if __name__ == '__main__':
    unittest.main()
