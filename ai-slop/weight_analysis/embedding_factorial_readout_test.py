"""CPU-only planted logit/weight evidence; never executes CUDA or a trainer."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import embedding_factorial_readout as readout
from .checkpoint import GPT2Config, tensor_manifest
from .paired_weight_patch import create_patch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def native_score(row, target):
    """Scalar specification of the C++ scoring order, independent of NumPy sum."""
    winner = max(range(len(row)), key=lambda i: (float(row[i]), -i))
    maximum, total = float(row[winner]), 0.0
    for value in row:
        total += math.exp(float(value) - maximum)
    nll = math.log(total) + maximum - float(row[target])
    rank = 1 + sum(float(value) > float(row[target]) or
                   (float(value) == float(row[target]) and i < target)
                   for i, value in enumerate(row))
    return {'nll': nll, 'rank': rank, 'argmax': winner}


class Fixture:
    def __init__(self, root, *, boundary=False):
        self.root, self.boundary = root, boundary
        self.config = GPT2Config(vocab_size=64, padded_vocab_size=80, context_length=16,
                                 n_layers=1, d_model=4, n_heads=2, d_ff=8)
        self.rows = [11, 12, 13, 21, 22, 23]
        self.a, self.d, self.j = (root / name for name in ('a', 'donor', 'j'))
        for directory, shift in ((self.a, 0), (self.d, .125)):
            directory.mkdir()
            for spec in tensor_manifest(self.config):
                values = np.arange(math.prod(spec.shape), dtype='<f4').reshape(spec.shape) * .001
                values += spec.index + shift
                values.tofile(directory / spec.filename)
        self.patch = create_patch(self.a, self.d, self.j, embedding_rows=self.rows,
                                   config=self.config)
        self.manifest_path = root / 'manifest.json'
        write_json(self.manifest_path, {'format': 'pluto-paired-corpus-training-v1',
                   'replacement': {'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True}})
        self.binary = root / 'fixture_binary'
        self.binary.write_bytes(b'Synthetic CPU fixture, not an executable probe.\n')
        self.cases_path = root / 'cases.json'
        self.packed_path = root / 'packed.bin'
        self.native_batch_path = root / 'native_packed.bin'
        self.row_path = root / 'selected_rows.i32.bin'
        self.scores = root / 'scores'
        self.scores.mkdir()
        self.output = root / 'readout.json'
        self.cases, xs, ys = [], [], []

        def append(split, occurrence, domain, word, ids, pieces, kind, prefix):
            start = 10 + occurrence * 10
            if boundary and kind == 'word':
                # Same word/prefix, different following target: word aggregates
                # must not weight the duplicate word twice for two delimiters.
                following = 9 + (occurrence == 1)
                ids = ids + [following]
                pieces = pieces + [b'\n' if following == 9 else b'!']
                kind = 'word_next_native'
            target_source = {'bytes_hex': b''.join(pieces).hex(),
                             'native_piece_bytes_hex': [p.hex() for p in pieces]}
            case = {'case_index': len(self.cases), 'kind': kind, 'split': split,
                    'context_id': f'{split}:{kind}:{occurrence}', 'prefix_domain': domain,
                    'target': word, 'target_ids': ids, 'prefix': {
                        'length': len(prefix), 'token_start': start-len(prefix), 'token_end': start,
                        'token_ids_sha256': hashlib.sha256(np.asarray(prefix, dtype='<i4').tobytes()).hexdigest()},
                    'target_source': target_source,
                    'scored_rows': list(range(len(prefix)-1, len(prefix)-1+len(ids)))}
            if kind == 'word_next_native':
                case.update(word_token_count=3, next_native_token={
                    'id': ids[3], 'bytes_hex': pieces[3].hex()})
            if kind == 'shared_piece':
                case['piece_id'] = ids[0]
            self.cases.append(case)
            sequence = np.full(self.config.context_length+1, 63, dtype='<i4')
            sequence[:len(prefix)] = prefix
            sequence[len(prefix):len(prefix)+len(ids)] = ids
            xs.append(sequence[:-1])
            ys.append(sequence[1:])

        for split in ('training', 'test'):
            for occurrence in range(3):
                prefix = [1, 2, 3, 4] if occurrence < 2 else [1, 11, 3, 4]
                for domain in ('original', 'replacement'):
                    append(split, occurrence, domain, 'Exeunt', [11, 12, 13],
                           [b'Ex', b'e', b'unt'], 'word', prefix)
                    append(split, occurrence, domain, 'Nuveth', [21, 22, 23],
                           [b'N', b'uv', b'eth'], 'word', prefix)
            for occurrence in range(2):
                append(split, occurrence, 'shared', 'control_next_3',
                       [11, 6, 7] if boundary else [6, 7, 8], [b'a', b'b', b'c'],
                       'shared_piece' if boundary else 'control', [1, 2, 3, 4])
        self.packed = np.asarray([xs, ys], dtype='<i4')
        self.packed.tofile(self.packed_path)
        self.plan = {'format': 'pluto-paired-supplemental-cases-v1' if boundary else
                     'pluto-paired-word-cases-v1', 'case_count': len(self.cases),
                     'context_length': self.config.context_length, 'vocab_size': 64, 'eos_token_id': 63,
                     'candidate_words': {'original': 'Exeunt', 'replacement': 'Nuveth'},
                     'manifest': readout._record(self.manifest_path), 'source_inputs': [],
                     'packed_batch': readout._record(self.packed_path), 'cases': self.cases}
        write_json(self.cases_path, self.plan)
        self.kind = 'word_next_native' if boundary else None
        self.configure_selection(self.kind)

    def configure_selection(self, kind):
        self.kind = kind
        self.indices = [i for i, c in enumerate(self.cases) if kind is None or c['kind'] == kind]
        selected_cases = [self.cases[i] for i in self.indices]
        self.native_packed = np.asarray(self.packed[:, self.indices, :], dtype='<i4')
        self.native_packed.tofile(self.native_batch_path)
        selected_rows = np.asarray([c['scored_rows'] for c in selected_cases], dtype='<i4')
        selected_rows.tofile(self.row_path)
        count, length = selected_rows.shape
        self.arrays = {cell: np.empty((count, length, 64), dtype='<f4') for cell in readout.CELLS}
        for index, case in enumerate(selected_cases):
            for position, row in enumerate(case['scored_rows']):
                prefix = self.native_packed[0, index, :row+1]
                touched = set(prefix.tolist()) & set(self.rows)
                ids = np.arange(64)
                base = np.asarray(((ids * 3 + int(prefix.sum())) % 11) * .125, dtype='<f4')
                output_effect = np.asarray(np.isin(ids, self.rows) * ((ids % 7)-3) * .25, dtype='<f4')
                input_effect = np.asarray(len(touched) * ((ids * 7) % 17) * .0625, dtype='<f4')
                interaction = np.asarray(bool(touched) * np.isin(ids, self.rows) * .125, dtype='<f4')
                for cell, values in (('AA', base), ('AJ', base+output_effect),
                                      ('JA', base+input_effect),
                                      ('JJ', base+output_effect+input_effect+interaction)):
                    self.arrays[cell][index, position] = values
        self.metadata = {
            'format': 'pluto-embedding-factorial-v1', 'complete': True,
            'recipient': str(self.a), 'patched': str(self.j), 'binary': str(self.binary),
            'batch': str(self.native_batch_path), 'rows': str(self.row_path),
            'case_count': count, 'rows_per_case': length, 'batch_sequences': 1,
            'context_length': self.config.context_length, 'vocabulary': 64,
            'temperature': 1, 'logits_dtype': '<f4', 'logits_shape': [count, length, 64],
            'normalization': 'recipient final LayerNorm fixed in all four cells',
            'score_definition': 'CPU FP64 log-sum-exp over all logical native FP32 logits',
            'checks': {**{key: True for key in readout.CHECKS},
                       'native_diagonal_verification_scope': 'selected_rows_full_padded_vocabulary'},
            'changed_embedding_rows': self.rows, 'cells': {}}
        self.write_scores()

    def amend_capitalization(self):
        """Use distinct synthetic lower-case IDs and an eleven-row patch.

        One training context becomes lowercase while another title context has
        the SAME base prefix. This detects accidental four-candidate pooling.
        These IDs are a toy vocabulary, not claims about GPT-2 tokenization.
        """
        contract=readout.case_contract
        self.rows += [31,41,42,43,44]
        self.j=self.root/'eleven_row_patch'
        self.patch=create_patch(self.a,self.d,self.j,embedding_rows=self.rows,config=self.config)
        for index,case in enumerate(self.cases):
            if case['kind'] not in ('word','word_next_native'):
                case.update(spelling_variant=None,candidate_pair=None,target_source_domain='original')
                continue
            role='original' if case['target']=='Exeunt' else 'replacement'
            lower=case['split']=='training' and case['context_id'].endswith(':0')
            variant='lowercase' if lower else 'title'
            case.update(spelling_variant=variant,candidate_pair=contract.PAIRS[variant],target_source_domain=role)
            if lower:
                case['target']=contract.PAIRS[variant][role]
                case['target_ids'][0]=31 if role=='original' else 41
                pieces=[bytes.fromhex(p) for p in case['target_source']['native_piece_bytes_hex']]
                pieces[0]=pieces[0].lower()
                case['target_source'].update(bytes_hex=b''.join(pieces).hex(),
                                             native_piece_bytes_hex=[p.hex() for p in pieces])
            prefix_length=case['prefix']['length']
            prefix=self.packed[0,index,:prefix_length].copy()
            sequence=np.full(self.config.context_length+1,63,dtype='<i4')
            sequence[:prefix_length]=prefix
            sequence[prefix_length:prefix_length+len(case['target_ids'])]=case['target_ids']
            self.packed[:,index,:]=[sequence[:-1],sequence[1:]]
        self.packed.tofile(self.packed_path)
        amendment_path=self.root/'amendments.json'
        write_json(amendment_path,dict(format='pluto-paired-lowercase-amendment-v1',complete=True,
            old_manifest=self.plan['manifest'],replacement_rules=[{'source':'Exeunt','target':'Nuveth'},
                                                                 {'source':'exeunt','target':'nuveth'}]))
        self.plan.update(format=contract.AMENDED[1 if self.boundary else 0],complete=True,
            candidate_pairs=contract.PAIRS,amendment=readout._record(amendment_path),
            old_manifest=self.plan.pop('manifest'),provenance=[],sources=[],
            packed_batch=readout._record(self.packed_path))
        del self.plan['candidate_words']
        write_json(self.cases_path,self.plan)
        self.configure_selection(self.kind)

    def write_scores(self):
        for cell in readout.CELLS:
            filename = cell + '.logits.f32.bin'
            self.arrays[cell].tofile(self.scores / filename)
            cases = []
            for index, source_index in enumerate(self.indices):
                source = self.cases[source_index]
                scores = [dict(row=row, target=target, **native_score(
                    self.arrays[cell][index, position], target)) for position, (row, target) in enumerate(
                    zip(source['scored_rows'], source['target_ids']))]
                cases.append({'case_index': index, 'tokens': scores,
                               'selected_rows_nll_sum': sum(s['nll'] for s in scores)})
            self.metadata['cells'][cell] = {'logits_file': filename, 'cases': cases}
        write_json(self.scores / 'metadata.json', self.metadata)

    def execution(self):
        inputs = [self.cases_path, self.native_batch_path, self.row_path, self.binary,
                  self.j / 'patch.json']
        inputs += [directory / spec.filename for directory in (self.a, self.j)
                   for spec in tensor_manifest(self.config)]
        before = [readout._record(path) for path in inputs]
        output_files = [self.scores / 'metadata.json'] + [
            self.scores / (cell + '.logits.f32.bin') for cell in readout.CELLS]
        command = [str(self.binary), '--recipient', str(self.a), '--patched=' + str(self.j),
                   '--batch', str(self.native_batch_path), '--rows', str(self.row_path),
                   '--output_dir', str(self.scores),
                   '--rows_per_case=' + str(self.metadata['rows_per_case']), '--batch_sequences=1']
        self.execution_path = self.root / 'execution.json'
        value = {'format': 'pluto-paired-probe-execution-v1', 'pid': 1234, 'returncode': 0,
                 'started_utc': '2026-09-10T01:00:00+00:00',
                 'finished_utc': '2026-09-10T01:01:00Z', 'command': command,
                 'inputs_before': before, 'inputs_after': copy.deepcopy(before),
                 'outputs': [readout._record(path) for path in output_files]}
        write_json(self.execution_path, value)
        return value

    def analyze(self, **options):
        kwargs = dict(expected_rows=self.rows, case_kind=self.kind, config=self.config)
        kwargs.update(options)
        return readout.analyze(self.cases_path, self.scores, self.j / 'patch.json',
                                self.output, **kwargs)


class FactorialMathTest(unittest.TestCase):
    def test_all_50257_logits_and_stable_tie_rules(self):
        row = np.zeros(50257, dtype='<f4')
        row[-1] = 10
        measured = readout.score_logits(row, 0)
        self.assertAlmostEqual(measured['nll'], math.log(50256 + math.exp(10)), places=12)
        self.assertEqual(measured['argmax'], 50256)
        self.assertEqual(measured['rank'], 2)
        tied = readout.score_logits(np.array([0., -0., 0.], dtype='<f4'), 2)
        self.assertEqual(tied['rank'], 3)
        self.assertEqual(tied['argmax'], 0)
        self.assertAlmostEqual(tied['nll'], math.log(3))

    def test_extreme_finite_logits_and_invalid_inputs(self):
        largest = np.finfo(np.float32).max
        row = np.array([largest, -largest], dtype='<f4')
        self.assertEqual(readout.score_logits(row, 1)['nll'], 2 * float(largest))
        for bad in (np.array([], dtype='<f4'), np.array([np.inf], dtype='<f4'),
                    np.array([np.nan], dtype='<f4'), np.array([0.], dtype='<f8')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                readout.score_logits(bad, 0)
        with self.assertRaises(ValueError):
            readout.score_logits(np.zeros(3, dtype='<f4'), 3)

    def test_effect_signs_and_interaction_closure(self):
        effect = readout.factorial_effects({'AA': -10., 'AJ': -7., 'JA': -8., 'JJ': -4.})
        self.assertEqual(effect, {'input': 2., 'output': 3., 'interaction': 1., 'joint': 6.})
        self.assertEqual(effect['input'] + effect['output'] + effect['interaction'], effect['joint'])


class FactorialReadoutTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='factorial-readout-test-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fixture = Fixture(self.root)

    def test_validates_logits_patch_bytes_dedup_and_effects(self):
        result = self.fixture.analyze()
        self.assertFalse(result['execution_provenance']['verified'])
        self.assertTrue(result['checks']['all_logical_logits_recomputed'])
        self.assertTrue(result['checks']['unexposed_full_logits_byte_equal'])
        self.assertEqual(result['patch']['actual_changed_rows'], self.fixture.rows)
        self.assertEqual(len(result['per_case']), 28)
        group = next(g for g in result['groups'] if (g['kind'], g['split'], g['prefix_domain'], g['target'])
                     == ('word', 'training', 'deduplicated_all', 'Exeunt'))
        self.assertEqual(group['frozen_case_count'], 6)
        self.assertEqual(group['word_three']['unique_event_count'], 2)
        self.assertEqual(group['selected_sequence']['unique_event_count'], 2)
        self.assertEqual(group['tokens'][0]['unique_event_count'], 2)
        first = result['per_case'][0]
        self.assertEqual(first['token_log_probability_effects'][0]['input'], 0)
        self.assertEqual(first['token_log_probability_effects'][0]['interaction'], 0)
        effect = first['log_probability_effects']
        self.assertAlmostEqual(effect['joint'], effect['input'] + effect['output'] + effect['interaction'])
        for cell in readout.CELLS:
            score = group['word_three']['cells'][cell]
            self.assertAlmostEqual(score['geometric_mean_probability'], math.exp(-score['mean_nll']))
            self.assertGreaterEqual(score['arithmetic_mean_probability'], score['geometric_mean_probability'] - 1e-14)
        odds = next(g for g in result['word_odds'] if (g['split'], g['prefix_domain']) ==
                    ('training', 'deduplicated_all'))
        self.assertEqual(odds['unique_paired_context_count'], 2)
        for cell in readout.CELLS:
            self.assertAlmostEqual(sum(odds['mean_token_log_odds'][cell]), odds['mean_log_odds'][cell])
        self.assertEqual(json.loads(self.fixture.output.read_text()), result)
        self.assertTrue(any(r['path'].endswith('AA.logits.f32.bin') for r in result['files']))
        exposure = next(g for g in result['exposure_groups'] if
                        (g['kind'], g['split'], g['target_position']) == ('word', 'training', 0))
        self.assertEqual(exposure['unique_causal_prefix_count'], 2)
        self.assertEqual(exposure['unexposed_causal_prefix_count'], 1)

    def test_valid_execution_record_covers_inputs_outputs_and_command(self):
        self.fixture.execution()
        result = self.fixture.analyze(execution_record=self.fixture.execution_path)
        self.assertTrue(result['execution_provenance']['verified'])
        self.assertEqual(result['execution_provenance']['required_output_count'], 5)

    def test_amended_variants_never_pool_identical_prefixes_and_eleven_rows_work(self):
        self.fixture.amend_capitalization()
        result=self.fixture.analyze()
        self.assertEqual(len(result['patch']['actual_changed_rows']),11)
        lower=[g for g in result['word_odds'] if g['spelling_variant']=='lowercase']
        self.assertEqual(len(lower),3)
        self.assertTrue(all(g['split']=='training' for g in lower))
        self.assertTrue(all(g['candidate_pair']=={'original':'exeunt','replacement':'nuveth'} for g in lower))
        title=[g for g in result['word_odds'] if g['spelling_variant']=='title']
        self.assertEqual(len(title),6)
        for group in lower+title:
            for cell in readout.CELLS:
                self.assertAlmostEqual(sum(group['mean_token_log_odds'][cell]),group['mean_log_odds'][cell])
        lower_items=[i for i in result['per_case'] if i['spelling_variant']=='lowercase']
        self.assertEqual(len(lower_items),4)
        self.assertEqual({i['target_source_domain'] for i in lower_items},{'original','replacement'})
        self.assertTrue(any(r['path'].endswith('paired_case_contract.py') for r in result['files']))

    def test_amended_requires_correct_role_pair_and_native_spelling(self):
        self.fixture.amend_capitalization()
        for field,value in (('target_source_domain','replacement'),('candidate_pair',{'original':'Exeunt','replacement':'Nuveth'}),
                            ('spelling_variant','title')):
            case=self.fixture.cases[0]; original=copy.deepcopy(case)
            case[field]=value; write_json(self.fixture.cases_path,self.fixture.plan)
            with self.subTest(field=field),self.assertRaises(ValueError): self.fixture.analyze()
            case.clear(); case.update(original)

    def test_amended_supplemental_fourth_token_stays_separate(self):
        path=self.root/'supplemental_fixture'; path.mkdir()
        fixture=Fixture(path,boundary=True); fixture.amend_capitalization()
        result=fixture.analyze()
        self.assertTrue(any(g['spelling_variant']=='lowercase' for g in result['word_odds']))
        self.assertTrue(all('exact_next_native_token' in g for g in result['groups']))

    def test_rejects_missing_or_changed_execution_input_and_output_coverage(self):
        pristine = self.fixture.execution()
        for mutate in (
                lambda e: e['inputs_after'].pop(),
                lambda e: (e['inputs_before'].pop(), e['inputs_after'].pop()),
                lambda e: e['outputs'].pop(),
                lambda e: e.update(returncode=False),
                lambda e: e.update(returncode=1),
                lambda e: e['command'].__setitem__(2, str(self.fixture.d)),
                lambda e: e['command'].append('--batch_sequences=1'),
                lambda e: e.update(finished_utc='2026-09-09T01:00:00Z')):
            with self.subTest(mutate=mutate):
                execution = copy.deepcopy(pristine)
                mutate(execution)
                write_json(self.fixture.execution_path, execution)
                with self.assertRaises((ValueError, KeyError)):
                    self.fixture.analyze(execution_record=self.fixture.execution_path)
                self.assertFalse(self.fixture.output.exists())

    def test_rejects_native_checks_geometry_and_rows(self):
        pristine = copy.deepcopy(self.fixture.metadata)
        for mutate in (
                lambda m: m.update(complete=False),
                lambda m: m.update(temperature=.8),
                lambda m: m['checks'].update(native_diagonals_byte_equal=False),
                lambda m: m['checks'].update(native_diagonal_verification_scope='all_rows'),
                lambda m: m.update(changed_embedding_rows=[11]),
                lambda m: m.update(logits_shape=[28, 3, 63]),
                lambda m: m['cells']['AA'].update(logits_file='../AA.logits.f32.bin')):
            with self.subTest(mutate=mutate):
                metadata = copy.deepcopy(pristine)
                mutate(metadata)
                write_json(self.fixture.scores / 'metadata.json', metadata)
                with self.assertRaises(ValueError):
                    self.fixture.analyze()
        write_json(self.fixture.scores / 'metadata.json', pristine)
        rows = np.fromfile(self.fixture.row_path, dtype='<i4')
        rows[0] += 1
        rows.tofile(self.fixture.row_path)
        with self.assertRaisesRegex(ValueError, 'row file'):
            self.fixture.analyze()

    def test_scores_not_trusted_and_nonfinite_logits_rejected(self):
        pristine = copy.deepcopy(self.fixture.metadata)
        for field, value in (('nll', 100), ('rank', 1), ('argmax', 63), ('target', 63), ('row', 10)):
            metadata = copy.deepcopy(pristine)
            metadata['cells']['AA']['cases'][0]['tokens'][0][field] = value
            write_json(self.fixture.scores / 'metadata.json', metadata)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.fixture.analyze()
        write_json(self.fixture.scores / 'metadata.json', pristine)
        path = self.fixture.scores / 'AA.logits.f32.bin'
        values = np.fromfile(path, dtype='<f4')
        values[-1] = np.inf
        values.tofile(path)
        with self.assertRaisesRegex(ValueError, 'finite FP32'):
            self.fixture.analyze()

    def test_unexposed_input_invariance_checks_full_logits_not_only_scored_target(self):
        # Alter an unrelated output row consistently in all duplicate prefixes,
        # then recompute metadata. A target-only comparison would miss this.
        for index, source in enumerate(self.fixture.indices):
            case = self.fixture.cases[source]
            if case['prefix']['token_ids_sha256'] == self.fixture.cases[0]['prefix']['token_ids_sha256']:
                self.fixture.arrays['JA'][index, 0, 60] += .25
        self.fixture.write_scores()
        with self.assertRaisesRegex(ValueError, 'unexposed input path'):
            self.fixture.analyze()

    def test_duplicate_causal_prefixes_checked_even_if_future_target_differs(self):
        # Original/replacement candidate cases differ after row 3, but that
        # future word is irrelevant to row 3's logits in a causal decoder.
        for cell in readout.CELLS:
            self.fixture.arrays[cell][1, 0, 60] += .25
        self.fixture.write_scores()
        with self.assertRaisesRegex(ValueError, 'identical causal prefixes'):
            self.fixture.analyze()

    def test_output_dictionary_cannot_change_unselected_raw_logit_columns(self):
        # This passes the unexposed AA=JA / AJ=JJ checks, but falsely changes
        # the output classifier row for an unselected token in both AJ and JJ.
        for cell in ('AJ', 'JJ'):
            self.fixture.arrays[cell][:, :, 60] += .25
        self.fixture.write_scores()
        with self.assertRaisesRegex(ValueError, 'unselected raw-logit column'):
            self.fixture.analyze()

    def test_rejects_wrong_patch_selection_hashes_and_unselected_mutations(self):
        with self.assertRaisesRegex(ValueError, 'predeclared'):
            self.fixture.analyze(expected_rows=self.fixture.rows[:-1])
        path = self.fixture.j / 'weight_1.bin'
        values = np.fromfile(path, dtype='<f4')
        values[0] += 1
        values.tofile(path)
        with self.assertRaisesRegex(ValueError, 'weight hash mismatch'):
            self.fixture.analyze()
        patch_path = self.fixture.j / 'patch.json'
        patch = json.loads(patch_path.read_text())
        patch['output']['weights_sha256']['weight_1.bin'] = readout._record(path)['sha256']
        write_json(patch_path, patch)
        with self.assertRaisesRegex(ValueError, 'nonembedding patch bytes'):
            self.fixture.analyze()

    def test_refuses_existing_output_symlink_logits_duplicate_json_and_truncation(self):
        self.fixture.output.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.fixture.analyze()
        self.assertEqual(self.fixture.output.read_text(), 'preserve')
        self.fixture.output.unlink()
        path = self.fixture.scores / 'AA.logits.f32.bin'
        actual = self.root / 'logits.saved'
        path.rename(actual)
        path.symlink_to(actual)
        with self.assertRaisesRegex(ValueError, 'nonsymlink'):
            self.fixture.analyze()
        path.unlink()
        actual.rename(path)
        with path.open('r+b') as stream:
            stream.truncate(path.stat().st_size - 4)
        with self.assertRaisesRegex(ValueError, 'file size'):
            self.fixture.analyze()
        (self.fixture.scores / 'metadata.json').write_text('{"complete":true,"complete":false}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            self.fixture.analyze()

    def test_changes_in_frozen_packed_inputs_are_rejected(self):
        values = self.fixture.packed.copy()
        values[0, 0, 0] = 40
        values.tofile(self.fixture.packed_path)
        with self.assertRaisesRegex(ValueError, 'hash/size/path mismatch'):
            self.fixture.analyze()

    def test_input_changes_during_analysis_prevent_publication(self):
        original = readout._aggregates
        def change_after_computation(items):
            result = original(items)
            self.fixture.binary.write_bytes(b'changed after input snapshot')
            return result
        with mock.patch.object(readout, '_aggregates', side_effect=change_after_computation):
            with self.assertRaisesRegex(ValueError, 'input changed during readout'):
                self.fixture.analyze()
        self.assertFalse(self.fixture.output.exists())

    def test_output_cannot_add_files_to_a_source_checkpoint(self):
        self.fixture.output = self.fixture.j / 'report.json'
        with self.assertRaisesRegex(ValueError, 'inside a source checkpoint'):
            self.fixture.analyze()
        self.assertFalse(self.fixture.output.exists())

    def test_missing_candidate_and_full_logit_truncation_are_not_silent(self):
        # Relabeling all Nuveth cases as controls would evade paired odds if
        # labels were trusted without their frozen domain/target semantics.
        plan = copy.deepcopy(self.fixture.plan)
        plan['cases'][1]['target'] = 'Unknown'
        write_json(self.fixture.cases_path, plan)
        with self.assertRaisesRegex(ValueError, 'word identity'):
            self.fixture.analyze()


class BoundaryReadoutTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='factorial-boundary-test-')
        self.addCleanup(temp.cleanup)
        self.fixture = Fixture(Path(temp.name), boundary=True)

    def test_word_and_exact_following_token_are_separate_and_subset_is_checked(self):
        f = self.fixture
        f.execution()
        result = f.analyze(execution_record=f.execution_path)
        self.assertEqual(len(result['per_case']), 24)
        self.assertEqual(result['case_kind_selection'], 'word_next_native')
        group = next(g for g in result['groups'] if (g['split'], g['prefix_domain'], g['target']) ==
                     ('training', 'deduplicated_all', 'Exeunt'))
        self.assertEqual(group['word_three']['unique_event_count'], 2)
        self.assertEqual(group['selected_sequence']['unique_event_count'], 3)
        self.assertEqual(group['exact_next_native_token']['unique_event_count'], 3)
        self.assertIn('not any delimiter', group['exact_next_native_token']['interpretation'])
        for cell in readout.CELLS:
            self.assertGreater(group['selected_sequence']['cells'][cell]['mean_nll'],
                               group['word_three']['cells'][cell]['mean_nll'])
            first = result['per_case'][0]['cells'][cell]
            self.assertAlmostEqual(first['sequence_nll'],
                                   first['word_three_nll'] + first['exact_next_native_token_nll'])
            self.assertAlmostEqual(first['word_and_exact_next_native_token_probability'],
                                   first['word_three_probability'] *
                                   first['exact_next_native_token_conditional_probability'])

    def test_shared_piece_subset_and_frozen_source_indices_preserved(self):
        f = self.fixture
        f.configure_selection('shared_piece')
        result = f.analyze()
        self.assertEqual(len(result['per_case']), 4)
        self.assertEqual(result['word_odds'], [])
        self.assertEqual([item['source_case_index'] for item in result['per_case']], f.indices)
        self.assertTrue(all(item['kind'] == 'shared_piece' for item in result['per_case']))

    def test_missing_subset_selection_and_wrong_next_token_rejected(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, 'mixed row counts'):
            f.analyze(case_kind=None)
        plan = copy.deepcopy(f.plan)
        plan['cases'][0]['next_native_token']['id'] += 1
        write_json(f.cases_path, plan)
        with self.assertRaisesRegex(ValueError, 'exact next native token'):
            f.analyze()


if __name__ == '__main__':
    unittest.main()
