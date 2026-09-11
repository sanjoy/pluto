"""Small native layouts checked against literal-byte and scalar token oracles."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_indentation_statistics as audit
from . import paired_training, paired_word_cases


SPELLINGS = {198: b'\n', 628: b'\n\n', 220: b' ', 1475: b' Ex', 3109: b'Ex',
             68: b'e', 2797: b'unt', 29739: b' Exit', 13: b'.',
             409: b' ex', 21733: b' Nu', 303: b've', 400: b'th'}


def native(values):
    pieces = [SPELLINGS[v] for v in values]
    return (np.asarray(values, dtype='<u4'),
            np.asarray([0, *np.cumsum([len(p) for p in pieces])], dtype='<u8'), b''.join(pieces))


class LayoutTest(unittest.TestCase):
    def test_first_piece_space_is_part_of_indentation(self):
        data = native([198, *([220]*57), 1475, 68, 2797, 198])
        result = audit.statistics(*data)
        self.assertEqual(result['indentation_only_histogram'], {'58': 1})
        self.assertEqual(result['maximum_raw_consecutive_spaces'], 58)
        self.assertEqual(result['maximum_native_consecutive_space220_tokens'], 57)
        observed = result['after_newline_spaces']['57']['strict_newline198']
        self.assertEqual(observed['conditioning_opportunities'], 1)
        self.assertEqual(observed['selected_token_frequencies']['1475'], 1)
        absent = result['after_newline_spaces']['63']['strict_newline198']
        self.assertEqual(absent['conditioning_opportunities'], 0)
        self.assertIsNone(absent['selected_token_frequencies']['1475'])

    def test_exact_63_token_context_requires_64_literal_spaces_for_spaced_ex(self):
        data = native([198, *([220]*63), 1475, 68, 2797])
        result = audit.statistics(*data)
        self.assertEqual(data[2].count(b'\n'+b' '*63), 1)
        self.assertEqual(result['indentation_only_histogram'], {'64': 1})
        self.assertEqual(result['after_newline_spaces']['63']['strict_newline198']
                         ['conditioning_opportunities'], 1)

    def test_merged_newline_is_not_two_prediction_opportunities(self):
        result = audit.statistics(*native([628, 220, 1475, 68, 2797]), space_counts=(0, 1))
        for count in ('0', '1'):
            tables = result['after_newline_spaces'][count]
            self.assertEqual(tables['any_native_token_ending_newline']['conditioning_opportunities'], 1)
            self.assertEqual(tables['strict_newline198']['conditioning_opportunities'], 0)
            self.assertEqual(tables['preceding_newline_token_counts'], {'628': 1})

    def test_scalar_next_token_oracle_includes_further_spaces(self):
        values = [198, 220, 220, 220, 1475, 68, 2797, 198, 220, 29739, 13, 628, 198]
        result = audit.statistics(*native(values), space_counts=(0, 1, 3, 63))
        for n in (0, 1, 3, 63):
            expected = Counter()
            for i, token in enumerate(values):
                if token == 198 and i+n+1 < len(values) and values[i+1:i+n+1] == [220]*n:
                    expected[values[i+n+1]] += 1
            table = result['after_newline_spaces'][str(n)]['strict_newline198']
            self.assertEqual(table['conditioning_opportunities'], sum(expected.values()))
            self.assertEqual({p['token_id']: p['count'] for p in table['successors']}, expected)

    def test_inline_column_is_not_indentation(self):
        result = audit.statistics(*native([13, 220, 1475, 68, 2797, 198, 3109, 68, 2797]))
        self.assertEqual(result['indentation_only_histogram'], {'0': 1})
        self.assertEqual(result['inline_word_byte_column_histogram'], {'3': 1})
        self.assertEqual(result['inline_count'], 1)
        self.assertEqual(result['exact_exeunt_count'], 2)

    def test_split_boundary_and_eof_supply_no_missing_target(self):
        train, test = native([198, 220]), native([220, 1475, 68, 2797])
        for data in (train, test):
            table = audit.statistics(*data, space_counts=(2,))['after_newline_spaces']['2']
            self.assertEqual(table['strict_newline198']['conditioning_opportunities'], 0)
        combined = native([198, 220, 220, 1475, 68, 2797])
        self.assertEqual(audit.statistics(*combined, space_counts=(2,))['after_newline_spaces']['2']
                         ['strict_newline198']['conditioning_opportunities'], 1)

    def test_literal_space_run_oracle(self):
        data = native([220, 1475, 68, 2797, 198, 220, 220, 29739, 13])
        result = audit.statistics(*data)
        lengths, count = [], 0
        for byte in data[2]+b'!':
            if byte == 32:
                count += 1
            elif count:
                lengths.append(count)
                count = 0
        self.assertEqual(result['maximum_raw_consecutive_spaces'], max(lengths))
        self.assertEqual(result['raw_space_run_length_histogram'],
                         {str(k): v for k, v in Counter(lengths).items()})

    def test_bad_native_offsets_spelling_and_request_rejected(self):
        tokens, offsets, text = native([198, 220, 1475, 68, 2797])
        for counts in ((), (-1,), (True,), (1, 1)):
            with self.assertRaises(ValueError):
                audit.statistics(tokens, offsets, text, space_counts=counts)
        bad = offsets.copy()
        bad[2] = bad[1]
        with self.assertRaises(ValueError):
            audit.statistics(tokens, bad, text)
        with self.assertRaisesRegex(ValueError, 'unexpected bytes'):
            audit.statistics(tokens, offsets, text.replace(b' ', b'x', 1))
        with self.assertRaises(ValueError):
            audit.word_layout(b'Exeunt', True)


class FrozenInputTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        train = [198, 220, 1475, 68, 2797, 198]
        test = [198, *([220]*57), 1475, 68, 2797, 198]
        values = {'original.training': train, 'original.test': test,
                  'replacement.training': [198, 220, 21733, 303, 400, 198],
                  'replacement.test': [198, *([220]*57), 21733, 303, 400, 198]}
        for domain in ('original', 'replacement'):
            values[domain+'.full'] = values[domain+'.training']+values[domain+'.test']
        self.manifest = dict(format='pluto-paired-corpus-training-v1',
                             replacement={'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True},
                             inputs={}, alignment={})
        arrays = {}
        for name, stream in values.items():
            ids, offsets, text = native(stream)
            arrays[name] = ids, offsets, text
            paths = {key: self.root/(name+'.'+key) for key in ('text', 'token_ids', 'offsets')}
            for key, payload in zip(paths, (text, ids.tobytes(), offsets.tobytes())):
                paths[key].write_bytes(payload)
            item = {key: str(path) for key, path in paths.items()}
            item.update({key+'_sha256': paired_word_cases._record(path)['sha256'] for key, path in paths.items()})
            item['export'] = dict(token_dtype='<u4', offset_dtype='<u8', roundtrip_verified=True,
                                  token_count=len(ids), offset_count=len(offsets), corpus_bytes=len(text))
            self.manifest['inputs'][name] = item
        self.manifest['split_byte'] = len(arrays['original.training'][2])
        for split in ('full', 'training', 'test'):
            a, b = arrays['original.'+split], arrays['replacement.'+split]
            self.manifest['alignment'][split] = paired_training.verify_alignment(
                a[2], b[2], a[0], b[0], a[1], b[1])
        (self.root/'word_cases').mkdir()
        self.publish()

    def publish(self):
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        self.cases = dict(format='pluto-paired-word-cases-v1',
            manifest=paired_word_cases._record(self.root/'manifest.json'),
            cases=[dict(kind='word', split=s, occurrence_index=0, target_source=dict(
                token_start=self.manifest['alignment'][s]['occurrences'][0]['token_start']))
                for s in ('training', 'test')])
        (self.root/'word_cases/cases.json').write_text(json.dumps(self.cases))

    def test_authenticated_report_and_selected_case_source_geometry(self):
        result = audit.analyze(self.root)
        self.assertEqual(len(result['provenance']), 20)
        self.assertEqual(len(result['sources']), 4)
        self.assertFalse(result['inference_performed'])
        self.assertEqual(result['statistics']['training']['indentation_only_histogram'], {'2': 1})
        self.assertEqual(result['statistics']['test']['indentation_only_histogram'], {'58': 1})
        selected = {v['split']: v for v in result['selected_word_source_layouts']}
        self.assertEqual(selected['test']['full_word_byte_start'], self.manifest['split_byte']+59)
        self.assertEqual(selected['test']['total_indentation_spaces'], 58)

    def test_modified_input_and_manifest_anchor_rejected(self):
        path = Path(self.manifest['inputs']['original.training']['text'])
        before = path.read_bytes()
        path.write_bytes(before+b'!')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit.analyze(self.root)
        path.write_bytes(before)
        (self.root/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'authenticate'):
            audit.analyze(self.root)

    def test_split_mismatch_and_case_coordinate_rejected(self):
        self.manifest['split_byte'] += 1
        self.publish()
        with self.assertRaisesRegex(ValueError, 'claimed split'):
            audit.analyze(self.root)
        self.manifest['split_byte'] -= 1
        self.publish()
        self.cases['cases'][0]['target_source']['token_start'] += 1
        (self.root/'word_cases/cases.json').write_text(json.dumps(self.cases))
        with self.assertRaisesRegex(ValueError, 'token coordinate'):
            audit.analyze(self.root)

    def test_changed_input_during_computation_rejected(self):
        original = audit.statistics
        changed = False

        def mutate(*args, **kwargs):
            nonlocal changed
            if not changed:
                changed = True
                path = Path(self.manifest['inputs']['replacement.training']['text'])
                path.write_bytes(path.read_bytes()+b'!')
            return original(*args, **kwargs)

        with mock.patch.object(audit, 'statistics', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                audit.analyze(self.root)

    def test_exclusive_json_output(self):
        path = self.root/'report.json'
        audit.main(['--root', str(self.root), '--output', str(path)])
        before = path.read_bytes()
        self.assertEqual(json.loads(before)['format'], 'pluto-paired-indentation-statistics-v1')
        with self.assertRaises(FileExistsError):
            audit.main(['--root', str(self.root), '--output', str(path)])
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
