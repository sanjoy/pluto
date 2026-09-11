"""Independent synthetic alignment tests; mocked exports never invoke a GPU."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_lowercase_inputs as amendment
from . import paired_training, paired_word_cases


# Synthetic native vocabulary for unit tests. Real-case token IDs are measured
# independently by the authenticated production CPU tokenizer, not assumed here.
SPELLINGS = {198: b'\n', 1475: b' Ex', 68: b'e', 2797: b'unt', 409: b' ex',
             21733: b' Nu', 303: b've', 400: b'th', 299: b' nu', 13: b'.'}


def native(ids):
    pieces = [SPELLINGS[i] for i in ids]
    return (b''.join(pieces), np.asarray(ids, dtype='<u4'),
            np.asarray([0, *np.cumsum([len(p) for p in pieces])], dtype='<u8'))


class AlignmentTest(unittest.TestCase):
    def test_both_case_preserving_replacements_and_internal_boundaries(self):
        a = native([198, 1475, 68, 2797, 198, 409, 68, 2797, 13])
        b = native([198, 21733, 303, 400, 198, 299, 303, 400, 13])
        result = amendment.verify_alignment(a[0], b[0], a[1], b[1], a[2], b[2])
        self.assertEqual(result['replacement_case_counts'], {'Exeunt': 1, 'exeunt': 1})
        self.assertEqual(result['changed_token_ids'], 6)
        self.assertEqual([(o['source_spelling'], o['target_spelling']) for o in result['occurrences']],
                         [('Exeunt', 'Nuveth'), ('exeunt', 'nuveth')])
        self.assertEqual([o['token_start'] for o in result['occurrences']], [1, 5])

    def test_unrelated_ids_and_boundaries_rejected(self):
        a = native([198, 1475, 68, 2797, 198, 13])
        b = native([198, 21733, 303, 400, 198, 13])
        ids = b[1].copy()
        ids[-1] = 198
        with self.assertRaisesRegex(ValueError, 'outside replacement spans'):
            amendment.verify_alignment(a[0], b[0], a[1], ids, a[2], b[2])
        # Shift an outer word boundary while retaining monotone offsets.
        offsets = b[2].copy()
        offsets[1] += 1
        with self.assertRaises(ValueError):
            amendment.verify_alignment(a[0], b[0], a[1], b[1], a[2], offsets)

    def test_wrong_case_or_other_text_change_rejected(self):
        a = native([409, 68, 2797])
        b = native([21733, 303, 400])
        with self.assertRaisesRegex(ValueError, 'unrelated text'):
            amendment.verify_alignment(a[0], b[0], a[1], b[1], a[2], b[2])

    def test_non_three_token_replacement_rejected(self):
        a = native([409, 68, 2797])
        b = native([299, 303, 400])
        with self.assertRaises(ValueError):
            amendment.verify_alignment(a[0], b[0], a[1], [299, 303, 400, 13], a[2], [0, 3, 4, 5, 7])

    def test_unmatched_case_variants_preserved(self):
        self.assertEqual(amendment.replace_words(b'Exeunt exeunt EXEUNT eXeunt'),
                         b'Nuveth nuveth EXEUNT eXeunt')


class PreparationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)/'old'
        self.root.mkdir()
        self.output = Path(tmp.name)/'amended'
        (self.root/'bin').mkdir()
        (self.root/'tokenizer').mkdir()
        (self.root/'word_cases').mkdir()
        (self.root/'bin/tokenize_corpus').write_bytes(b'authenticated synthetic CPU tokenizer fixture')
        (self.root/'tokenizer/tokenizer.json').write_text('{}')
        train = [198, 1475, 68, 2797, 198, 409, 68, 2797, 198]
        test = [198, 1475, 68, 2797, 13]
        streams = {'original.training': train, 'original.test': test,
                   'replacement.training': [198, 21733, 303, 400, *train[4:]],
                   'replacement.test': [198, 21733, 303, 400, 13]}
        for domain in ('original', 'replacement'):
            streams[domain+'.full'] = streams[domain+'.training']+streams[domain+'.test']
        self.manifest = dict(format='pluto-paired-corpus-training-v1',
            replacement={'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True},
            binaries={'tokenize_corpus': paired_word_cases._record(self.root/'bin/tokenize_corpus')},
            tokenizer_files={'tokenizer.json': paired_word_cases._record(self.root/'tokenizer/tokenizer.json')},
            inputs={}, alignment={})
        arrays = {}
        for name, ids in streams.items():
            arrays[name] = native(ids)
            text, tokens, offsets = arrays[name]
            path = self.root/(name+'.txt')
            path.write_bytes(text)
            item = self.export_item(path, self.root/(name+'.tokens.bin'), tokens, offsets)
            # An unsafe saved command must have no role in constructing argv.
            item['command'] = ['DO_NOT_EXECUTE_SAVED_COMMAND']
            self.manifest['inputs'][name] = item
        self.manifest['split_byte'] = len(arrays['original.training'][0])
        for split in ('full', 'training', 'test'):
            a, b = arrays['original.'+split], arrays['replacement.'+split]
            self.manifest['alignment'][split] = paired_training.verify_alignment(
                a[0], b[0], a[1], b[1], a[2], b[2])
        self.publish()

    def export_item(self, path, token_path, tokens, offsets):
        token_path.write_bytes(tokens.tobytes())
        offset_path = Path(str(token_path)+'.offsets.bin')
        offset_path.write_bytes(offsets.tobytes())
        export = dict(format='pluto-native-token-ids-v1', token_dtype='<u4', offset_dtype='<u8',
                      token_count=len(tokens), offset_count=len(offsets), corpus_bytes=len(path.read_bytes()),
                      roundtrip_verified=True)
        item = dict(text=str(path), token_ids=str(token_path), offsets=str(offset_path), export=export)
        for key in ('text', 'token_ids', 'offsets'):
            item[key+'_sha256'] = paired_word_cases._record(item[key])['sha256']
        return item

    def publish(self):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        (self.root/'word_cases/cases.json').write_text(json.dumps(dict(
            format='pluto-paired-word-cases-v1', manifest=paired_word_cases._record(self.root/'manifest.json'))))

    def fake_native_export(self, command, **kwargs):
        self.assertEqual(len(command), 4)
        self.assertEqual(command[0], str(self.output/'bin/tokenize_corpus'))
        self.assertEqual(command[1], str(self.output/'tokenizer'))
        self.assertNotIn('DO_NOT_EXECUTE_SAVED_COMMAND', command)
        text = Path(command[2]).read_bytes()
        ids, cursor = [], 0
        choices = sorted(SPELLINGS.items(), key=lambda kv: -len(kv[1]))
        while cursor < len(text):
            for token, piece in choices:
                if text.startswith(piece, cursor):
                    ids.append(token)
                    cursor += len(piece)
                    break
            else:
                self.fail('fixture text not representable by synthetic native vocabulary')
        _, tokens, offsets = native(ids)
        item = self.export_item(Path(command[2]), Path(command[3]), tokens, offsets)
        return subprocess.CompletedProcess(command, 0, json.dumps(item['export']), '')

    def test_amendment_preserves_originals_and_publishes_all_alignment(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with mock.patch.object(amendment.subprocess, 'run', side_effect=self.fake_native_export) as run:
            result = amendment.prepare(self.root/'manifest.json', self.output)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(result['alignment']['full']['replacement_case_counts'], {'Exeunt': 2, 'exeunt': 1})
        self.assertEqual(result['comparison_to_old_replacement']['training']['additional_changed_token_ids'], 3)
        self.assertEqual(result['comparison_to_old_replacement']['test']['additional_changed_token_ids'], 0)
        self.assertFalse(result['training_launched'])
        self.assertFalse(result['gpu_work_performed'])
        self.assertTrue(result['full_and_split_exports_identical'])
        self.assertEqual(json.loads((self.output/'amendments.json').read_text()), result)
        for path, payload in before.items():
            self.assertEqual(path.read_bytes(), payload)
        for item in result['provenance']+result['sources']:
            self.assertEqual(paired_word_cases._record(item['path']), item)

    def test_existing_output_and_symlink_rejected_before_export(self):
        self.output.mkdir()
        with mock.patch.object(amendment.subprocess, 'run') as run:
            with self.assertRaises(FileExistsError):
                amendment.prepare(self.root/'manifest.json', self.output)
            run.assert_not_called()
        alias = self.output.parent/'alias'
        alias.symlink_to(self.output)
        with self.assertRaises(FileExistsError):
            amendment.prepare(self.root/'manifest.json', alias)

    def test_changed_binary_and_manifest_anchor_rejected_before_output(self):
        path = self.root/'bin/tokenize_corpus'
        before = path.read_bytes()
        path.write_bytes(before+b'!')
        with self.assertRaisesRegex(ValueError, 'binary or asset changed'):
            amendment.prepare(self.root/'manifest.json', self.output)
        self.assertFalse(self.output.exists())
        path.write_bytes(before)
        (self.root/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'authenticate'):
            amendment.prepare(self.root/'manifest.json', self.output)

    def test_cpu_export_failure_does_not_publish_completion(self):
        with mock.patch.object(amendment.subprocess, 'run', return_value=subprocess.CompletedProcess(
                [], 1, '', 'synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'tokenizer failed'):
                amendment.prepare(self.root/'manifest.json', self.output)
        self.assertTrue(self.output.exists())
        self.assertFalse((self.output/'amendments.json').exists())

    def test_mid_run_original_mutation_rejected(self):
        def mutate(command, **kwargs):
            result = self.fake_native_export(command, **kwargs)
            path = self.root/'bin/tokenize_corpus'
            if not path.read_bytes().endswith(b'!'):
                path.write_bytes(path.read_bytes()+b'!')
            return result
        with mock.patch.object(amendment.subprocess, 'run', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                amendment.prepare(self.root/'manifest.json', self.output)
        self.assertFalse((self.output/'amendments.json').exists())


if __name__ == '__main__':
    unittest.main()
