"""Synthetic native-shaped fixtures; no tokenizer process, training, or GPU."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import paired_lowercase_cases as cases
from . import paired_lowercase_inputs as amendment
from . import paired_training, paired_word_cases as old_cases


BYTES = {64: b'a', 198: b'\n', 1475: b' Ex', 3109: b'Ex', 68: b'e', 2797: b'unt',
         21733: b' Nu', 45: b'N', 45177: b'uve', 303: b've', 400: b'th',
         409: b' ex', 14364: b' nu'}


def native(ids):
    pieces = [BYTES[i] for i in ids]
    return dict(tokens=np.asarray(ids, dtype='<u4'), text=b''.join(pieces),
                offsets=np.asarray([0, *np.cumsum([len(p) for p in pieces])], dtype='<u8'))


class LowercaseCasesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.old_root = self.root/'old'
        self.amended_root = self.root/'amended'
        self.old_root.mkdir()
        self.amended_root.mkdir()
        self.output = self.amended_root/'word_cases'
        self.old_streams = {}
        for split in ('training', 'test'):
            original = [64]*140
            original[-1] = 198
            for start, first in ((10, 1475), (45, 3109), (80, 1475)):
                original[start:start+3] = [first, 68, 2797]
            if split == 'training':
                for start in (4, 65):
                    original[start:start+3] = [409, 68, 2797]
            replaced = original.copy()
            for start, ids in ((10, [21733, 303, 400]), (45, [45, 45177, 400]),
                               (80, [21733, 303, 400])):
                replaced[start:start+3] = ids
            self.old_streams['original.'+split] = original
            self.old_streams['replacement.'+split] = replaced
        for domain in ('original', 'replacement'):
            self.old_streams[domain+'.full'] = self.old_streams[domain+'.training']+self.old_streams[domain+'.test']
        self.old_inputs = {name: native(ids) for name, ids in self.old_streams.items()}
        self.manifest = dict(format='pluto-paired-corpus-training-v1',
            replacement={'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True},
            inputs={name: self.export(self.old_root, name, data) for name, data in self.old_inputs.items()},
            alignment={}, split_byte=len(self.old_inputs['original.training']['text']))
        for split in ('full', 'training', 'test'):
            a, b = (self.old_inputs[f'{domain}.{split}'] for domain in ('original', 'replacement'))
            self.manifest['alignment'][split] = paired_training.verify_alignment(
                a['text'], b['text'], a['tokens'], b['tokens'], a['offsets'], b['offsets'])
        self.manifest_path = self.old_root/'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.old_report = old_cases.prepare(self.manifest_path, self.old_root/'word_cases',
            contexts_per_split=3, controls_per_split=0, prefix_tokens=4, context_length=16)
        self.old_case_path = self.old_root/'word_cases/cases.json'
        self.old_packed = np.fromfile(self.old_report['packed_batch']['path'], dtype='<i4').reshape(2, 24, 16)
        for split in ('training', 'test'):
            for start in (110, 120):
                case, x, y = self.control(split, start, len(self.old_report['cases']))
                self.old_report['cases'].append(case)
                self.old_packed = np.concatenate((self.old_packed, np.asarray([[x], [y]])), axis=1)
                self.old_report['selection']['selected'][split]['control_token_starts'].append(start)
        self.save_old_cases()
        self.updated = {}
        for name, original_ids in self.old_streams.items():
            ids = original_ids.copy()
            if name.startswith('replacement.'):
                for index in range(len(ids)-2):
                    if ids[index:index+3] == [409, 68, 2797]:
                        ids[index:index+3] = [14364, 303, 400]
            self.updated[name] = native(ids)
        self.amendment = dict(format='pluto-paired-lowercase-amendment-v1', complete=True,
            replacement_rules=[dict(source='Exeunt', target='Nuveth'), dict(source='exeunt', target='nuveth')],
            split_byte=self.manifest['split_byte'], alignment={},
            inputs={name: self.manifest['inputs'][name] if name.startswith('original.')
                    else self.export(self.amended_root, name, data) for name, data in self.updated.items()},
            sources=[cases._record(amendment.__file__)])
        for split in ('full', 'training', 'test'):
            a, b = (self.updated[f'{domain}.{split}'] for domain in ('original', 'replacement'))
            self.amendment['alignment'][split] = amendment.verify_alignment(
                a['text'], b['text'], a['tokens'], b['tokens'], a['offsets'], b['offsets'])
        self.amendment_path = self.amended_root/'amendments.json'
        self.refresh_amendment()

    def export(self, root, name, data):
        paths = dict(text=root/(name+'.txt'), token_ids=root/(name+'.tokens.bin'),
                     offsets=root/(name+'.offsets.bin'))
        paths['text'].write_bytes(data['text'])
        paths['token_ids'].write_bytes(data['tokens'].tobytes())
        paths['offsets'].write_bytes(data['offsets'].tobytes())
        result = dict(export=dict(token_count=len(data['tokens']), offset_count=len(data['offsets']),
            corpus_bytes=len(data['text']), token_dtype='<u4', offset_dtype='<u8', roundtrip_verified=True))
        for key, path in paths.items():
            result[key] = str(path)
            result[key+'_sha256'] = cases._record(path)['sha256']
        return result

    def control(self, split, start, index):
        data = self.old_inputs['original.'+split]
        begin = max(0, start-4)
        prefix = data['tokens'][begin:start]
        target = data['tokens'][start:start+3].tolist()
        x, y, rows = old_cases._case_sequence(prefix, target, 16)
        case = dict(case_index=index, kind='control', split=split, occurrence_index=None,
            prefix_domain='shared', target_source_domain='original', target='control_next_3',
            context_id=f'{split}:control:{start}', target_ids=target,
            target_source=old_cases._target_source(data, start), scored_rows=rows,
            prefix=dict(token_start=begin, token_end=start, length=len(prefix),
                byte_start=int(data['offsets'][begin]), byte_end=int(data['offsets'][start]),
                token_ids_sha256=hashlib.sha256(prefix.astype('<i4').tobytes()).hexdigest()))
        return case, x, y

    def save_old_cases(self):
        self.old_report['case_count'] = len(self.old_report['cases'])
        Path(self.old_report['packed_batch']['path']).write_bytes(self.old_packed.astype('<i4').tobytes())
        self.old_report['packed_batch'] = cases._record(self.old_report['packed_batch']['path'])
        self.old_case_path.write_text(json.dumps(self.old_report))

    def refresh_amendment(self):
        self.amendment['old_manifest'] = cases._record(self.manifest_path)
        self.amendment['old_word_cases'] = cases._record(self.old_case_path)
        records = [self.amendment['old_manifest'], self.amendment['old_word_cases']]
        for bundle in (self.manifest['inputs'], self.amendment['inputs']):
            for item in bundle.values():
                records += [cases._record(item[key]) for key in ('text', 'token_ids', 'offsets')]
        self.amendment['provenance'] = list({r['path']: r for r in records}.values())
        self.amendment_path.write_text(json.dumps(self.amendment))

    def prepare(self):
        return cases.prepare(self.amendment_path, self.old_case_path, self.output,
                             context_length=16, prefix_tokens=4)

    def test_preserves_old_order_and_remaps_shifted_occurrence_indices_by_start(self):
        result = self.prepare()
        self.assertEqual(result['case_count'], 36)
        self.assertEqual(result['selection']['inherited_case_count'], 28)
        self.assertEqual([c['source_case_index'] for c in result['cases'][:28]], list(range(28)))
        for old, new in zip(self.old_report['cases'], result['cases'][:28]):
            self.assertEqual(new['prefix']['token_end'], old['prefix']['token_end'])
            self.assertEqual(new['target_ids'], old['target_ids'])
            self.assertEqual(new['source_context_id'], old['context_id'])
        first = next(c for c in result['cases'] if c['split']=='training' and
                     c['prefix']['token_end']==10 and c['kind']=='word')
        self.assertEqual(first['source_occurrence_index'], 0)
        self.assertEqual(first['occurrence_index'], 1)
        last = next(c for c in result['cases'] if c['split']=='training' and
                    c['prefix']['token_end']==80 and c['kind']=='word')
        self.assertEqual(last['source_occurrence_index'], 2)
        self.assertEqual(last['occurrence_index'], 4)

    def test_lowercase_case_pairs_native_ids_and_union_are_exact(self):
        result = self.prepare()
        lower = result['cases'][28:]
        self.assertEqual(len(lower), 8)
        self.assertEqual({c['prefix']['token_end'] for c in lower}, {4, 65})
        self.assertEqual({c['spelling_variant'] for c in lower}, {'lowercase'})
        self.assertEqual({c['target'] for c in lower}, {'exeunt', 'nuveth'})
        for c in lower:
            self.assertEqual(c['candidate_pair'], {'original':'exeunt', 'replacement':'nuveth'})
            self.assertEqual(c['target_ids'], [409,68,2797] if c['target']=='exeunt' else [14364,303,400])
            self.assertIsNone(c['source_case_index'])
        self.assertEqual(result['word_piece_ids'], [45,68,303,400,409,1475,2797,3109,14364,21733,45177])
        self.assertEqual(result['selection']['lowercase_occurrences'], {'training':2, 'test':0})
        self.assertIn('no cases invented', result['selection']['selected']['test']['lowercase_case_coverage'])

    def test_rebuilds_amended_replacement_prefix_but_preserves_original_prefix(self):
        result = self.prepare()
        for domain in ('original', 'replacement'):
            c = next(c for c in result['cases'] if c['split']=='training' and
                     c['kind']=='word' and c['prefix']['token_end']==10 and c['prefix_domain']==domain)
            old = self.old_report['cases'][c['source_case_index']]
            equal = c['prefix']['token_ids_sha256'] == old['prefix']['token_ids_sha256']
            self.assertEqual(equal, domain=='original')

    def test_native_batch_geometry_teacher_forcing_and_controls_are_unchanged(self):
        result = self.prepare()
        packed = np.fromfile(result['packed_batch']['path'], dtype='<i4').reshape(2,36,16)
        self.assertEqual(result['packed_batch']['bytes'], 8*36*16)
        np.testing.assert_array_equal(packed[0,:,1:], packed[1,:,:-1])
        for c in result['cases']:
            i, length = c['case_index'], c['prefix']['length']
            self.assertEqual(c['scored_rows'], [length-1,length,length+1])
            self.assertEqual(packed[1,i,c['scored_rows']].tolist(), c['target_ids'])
            self.assertTrue(np.all(packed[1,i,length+2:] == old_cases.EOS))
            if c['kind']=='control':
                np.testing.assert_array_equal(packed[:,i], self.old_packed[:,c['source_case_index']])

    def test_old_control_overlapping_new_lowercase_span_fails_without_output(self):
        index = next(i for i,c in enumerate(self.old_report['cases']) if c['kind']=='control' and c['split']=='training')
        case, x, y = self.control('training', 4, index)
        self.old_report['cases'][index] = case
        self.old_packed[:,index] = np.asarray([x,y])
        self.old_report['selection']['selected']['training']['control_token_starts'] = [4,120]
        self.save_old_cases()
        self.refresh_amendment()
        with self.assertRaisesRegex(ValueError, 'control overlaps'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_wrong_capitalized_or_lowercase_label_is_rejected(self):
        self.old_report['cases'][0]['target'] = 'exeunt'
        self.save_old_cases()
        self.refresh_amendment()
        with self.assertRaisesRegex(ValueError, 'candidate label'):
            self.prepare()
        self.old_report['cases'][0]['target'] = 'Exeunt'
        self.save_old_cases()
        self.refresh_amendment()
        lower = next(o for o in self.amendment['alignment']['training']['occurrences'] if o['source_spelling']=='exeunt')
        lower.update(source_spelling='Exeunt', target_spelling='Nuveth')
        self.amendment_path.write_text(json.dumps(self.amendment))
        with self.assertRaisesRegex(ValueError, 'spelling/alignment'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_missing_old_crossing_and_changed_input_hash_are_rejected(self):
        # A same-sized duplicate cannot quietly replace a missing domain crossing.
        self.old_report['cases'][1] = dict(self.old_report['cases'][0], case_index=1)
        self.old_packed[:,1] = self.old_packed[:,0]
        self.save_old_cases()
        self.refresh_amendment()
        with self.assertRaisesRegex(ValueError, 'duplicate inherited word crossing'):
            self.prepare()
        path = Path(self.amendment['inputs']['replacement.training']['token_ids'])
        path.write_bytes(b'xxxx'+path.read_bytes()[4:])
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.prepare()

    def test_wrong_geometry_and_existing_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
            cases.prepare(self.amendment_path, self.old_case_path, self.output,
                          context_length=32, prefix_tokens=4)
        self.output.mkdir()
        marker = self.output/'preserve'
        marker.write_text('old output')
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(marker.read_text(), 'old output')

    def test_all_input_files_remain_unchanged_and_provenance_rehashes(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = self.prepare()
        for p, raw in before.items():
            self.assertEqual(p.read_bytes(), raw)
        for item in result['provenance']+result['sources']:
            self.assertEqual(cases._record(item['path']), item)
        self.assertEqual(json.loads((self.output/'cases.json').read_text()), result)


if __name__ == '__main__':
    unittest.main()
