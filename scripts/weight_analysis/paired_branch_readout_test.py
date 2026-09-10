"""CPU-only synthetic weight/score records; these tests never claim GPU evidence."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import paired_branch_readout as readout
from .checkpoint import GPT2Config, tensor_manifest
from .paired_weight_patch import create_patch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


class Fixture:
    def __init__(self, root, branch='attention', scope='whole_branch', baseline_copies=True, spaced=False):
        self.root = root
        self.config = GPT2Config(vocab_size=64, padded_vocab_size=80, context_length=16,
                                 n_layers=1, d_model=4, n_heads=2, d_ff=8)
        self.specs = tensor_manifest(self.config)
        self.paths = {role: root / role / 'step_100' for role in readout.ROLES}
        for role in readout.ROLES:
            self.paths[role].parent.mkdir()
        for role, shift in (('recipient', 0), ('donor', .125)):
            self.paths[role].mkdir()
            for spec in self.specs:
                values = np.arange(math.prod(spec.shape), dtype='<f4') * .001 + spec.index + shift
                values.tofile(self.paths[role] / spec.filename)
        projection, norm = ('attn', 'ln1') if branch == 'attention' else ('mlp', 'ln2')
        selected = [s for s in self.specs if (
            s.name.startswith(f'blocks.0.{projection}.output.') if scope == 'output_write' else
            s.name.startswith(f'blocks.0.{projection}.') or s.name.startswith(f'blocks.0.{norm}.'))]
        self.patch = create_patch(self.paths['recipient'], self.paths['donor'], self.paths['patched'],
                                   tensors=[s.name for s in selected], config=self.config)
        self.intervention = {'name': 'step_100_replacement_to_original_block_0_attention_whole',
                             'kind': branch + '_' + scope, 'step': 100,
                             'recipient_arm': 'original', 'donor_arm': 'replacement',
                             'tensors': [s.name for s in selected], 'embedding_rows': [],
                             'selection_specs': [s.to_dict() for s in selected]}
        for role, source in (('recipient', 'original'), ('donor', 'replacement')):
            self.intervention[role + '_checkpoint'] = {'step': 100, 'path': str(self.paths[role]),
                'sha256': self.patch['sources'][source]['weights_sha256']}
        self.native_paths = dict(self.paths)
        if baseline_copies:
            for role in ('recipient', 'donor'):
                self.native_paths[role] = root / (role + '_copy') / 'step_100'
                self.native_paths[role].parent.mkdir()
                create_patch(self.paths[role], self.paths['donor' if role == 'recipient' else 'recipient'],
                             self.native_paths[role], config=self.config)
        self.binary = root / 'synthetic_binary'
        self.binary.write_bytes(b'CPU fixture, not a real executable')
        self.manifest_path = root / 'manifest.json'
        write_json(self.manifest_path, {'format': 'pluto-paired-corpus-training-v1',
                   'replacement': {'from': 'Exeunt', 'to': 'Nuveth', 'case_sensitive': True}})
        self.cases, self.plans, self.packed = {}, {}, {}
        for suite in readout.SUITES:
            directory = root / ('cases_' + suite)
            directory.mkdir()
            self.cases[suite] = directory / 'cases.json'
            rows, xs, ys = [], [], []

            def append(split, occurrence, domain, word, ids, pieces, kind):
                prefix = [1, 2, 3, 4] if word in readout.WORDS else [6, 2, 3, 4]
                if spaced and occurrence == 1 and word in readout.WORDS:
                    prefix = [1, 2, 3, 5]
                    ids, pieces = [ids[0]+3, *ids[1:]], [b' '+pieces[0], *pieces[1:]]
                if suite == 'supplemental' and kind == 'word':
                    ids, pieces, kind = ids + [9], pieces + [b'\n'], 'word_next_native'
                case = {'case_index': len(rows), 'kind': kind, 'split': split,
                        'context_id': f'{split}:{occurrence}', 'prefix_domain': domain,
                        'target': word, 'target_ids': ids,
                        'prefix': {'length': len(prefix), 'token_ids_sha256': hashlib.sha256(
                            np.asarray(prefix, dtype='<i4').tobytes()).hexdigest()},
                        'target_source': {'bytes_hex': b''.join(pieces).hex(),
                                          'native_piece_bytes_hex': [p.hex() for p in pieces]},
                        'scored_rows': list(range(3, 3 + len(ids)))}
                if kind == 'word_next_native':
                    case.update(word_token_count=3, next_native_token={'id': 9, 'bytes_hex': '0a'})
                if kind == 'shared_piece':
                    case['piece_id'] = ids[0]
                sequence = np.full(17, 63, dtype='<i4')
                sequence[:4] = prefix
                sequence[4:4+len(ids)] = ids
                rows.append(case)
                xs.append(sequence[:-1])
                ys.append(sequence[1:])

            for split in ('training', 'test'):
                for occurrence in range(2):
                    for domain in ('original', 'replacement'):
                        append(split, occurrence, domain, 'Exeunt', [11, 12, 13],
                               [b'Ex', b'e', b'unt'], 'word')
                        append(split, occurrence, domain, 'Nuveth', [21, 22, 23],
                               [b'N', b'uve', b'th'], 'word')
                append(split, 3, 'shared', 'control_next_3', [11, 5, 6], [b'Ex', b'i', b't'],
                       'control' if suite == 'main' else 'shared_piece')
            packed = np.asarray([xs, ys], dtype='<i4')
            packed.tofile(directory / 'packed.bin')
            self.packed[suite] = packed
            plan = {'format': 'pluto-paired-word-cases-v1' if suite == 'main' else
                     'pluto-paired-supplemental-cases-v1', 'case_count': len(rows),
                    'context_length': 16, 'vocab_size': 64, 'eos_token_id': 63,
                    'candidate_words': {'original': 'Exeunt', 'replacement': 'Nuveth'},
                    'manifest': readout._record(self.manifest_path), 'cases': rows,
                    'packed_batch': readout._record(directory / 'packed.bin'), 'source_inputs': []}
            if suite == 'supplemental':
                plan.update(source_word_cases=readout._record(self.cases['main']),
                            source_word_packed_batch=self.plans['main']['packed_batch'])
            self.plans[suite] = plan
            write_json(self.cases[suite], plan)
        self.scores = {role: {} for role in readout.ROLES}
        self.executions = {role: {} for role in readout.ROLES}
        self.output = root / 'report.json'
        for role in readout.ROLES:
            for suite in readout.SUITES:
                directory = root / (role + '_' + suite + '_scores')
                directory.mkdir()
                self.scores[role][suite] = directory
                self.executions[role][suite] = root / (role + '_' + suite + '_execution.json')
                plan = self.plans[suite]
                losses = np.full((plan['case_count'], 16), 2.0, dtype='<f4')
                argmax = np.ones(losses.shape, dtype='<i4')
                for index, case in enumerate(plan['cases']):
                    original = case['target'] == 'Exeunt'
                    if case['target'] in readout.WORDS:
                        nll = {'recipient': [1., .25, .5] if original else [5., 1., 1.],
                               'donor': [4., 1., 1.] if original else [1., .25, .5],
                               'patched': [2., .5, .5] if original else [5., 1.5, 1.5]}[role]
                    else:
                        nll = [2., 2., 2.] if role != 'patched' else [2.25, 2., 2.]
                    if len(case['target_ids']) == 4:
                        nll += [1.25]
                    losses[index, case['scored_rows']] = nll
                losses.tofile(directory / 'losses.f32.bin')
                argmax.tofile(directory / 'argmax.i32.bin')
                metadata = {'schema_version': 1, 'kind': 'paired_loss_probe', 'complete': True,
                            'checkpoint_directory': str(self.native_paths[role]), 'checkpoint_step': 100,
                            'batch_file': plan['packed_batch']['path'], 'binary_file': str(self.binary),
                            'output_directory': str(directory), 'batch_sequences': 1,
                            'batch_bytes': plan['packed_batch']['bytes'], 'case_count': plan['case_count'],
                            'passage_count': plan['case_count'], 'context_length': 16,
                            'output_shape': [plan['case_count'], 16], 'vocab_size': 64,
                            'padded_vocab_size': 80, 'layers': 1, 'width': 4, 'heads': 2,
                            'head_dimension': 2, 'feed_forward_width': 8,
                            'checkpoint_unique_weight_count': len(self.specs), 'temperature': 1,
                            'loss_dtype': '<f4', 'argmax_dtype': '<i4', 'loss_file': 'losses.f32.bin',
                            'argmax_file': 'argmax.i32.bin', 'byte_order': 'little',
                            'finite_losses': True, 'argmax_valid': True, 'optimizer_steps': 0,
                            'backward_calls': 0, 'checkpoint_writes': 0}
                write_json(directory / 'metadata.json', metadata)
                self.refresh_execution(role, suite)

    def refresh_execution(self, role, suite):
        directory, checkpoint = self.scores[role][suite], self.native_paths[role]
        inputs = [self.binary, self.cases[suite], Path(self.plans[suite]['packed_batch']['path'])]
        inputs += [checkpoint / spec.filename for spec in self.specs]
        if (checkpoint / 'patch.json').exists():
            inputs += [checkpoint / 'patch.json']
        before = [readout._record(path) for path in inputs]
        value = {'format': 'pluto-paired-probe-execution-v1', 'pid': 12345, 'returncode': 0,
                 'started_utc': '2026-09-10T12:00:00+00:00', 'finished_utc': '2026-09-10T12:01:00+00:00',
                 'inputs_before': before, 'inputs_after': before,
                 'outputs': [readout._record(directory / name) for name in
                             ('metadata.json', 'losses.f32.bin', 'argmax.i32.bin')],
                 'command': [str(self.binary), f'--checkpoint={checkpoint}',
                             f'--batch={self.plans[suite]["packed_batch"]["path"]}',
                             f'--output_dir={directory}', '--batch_sequences=1']}
        write_json(self.executions[role][suite], value)

    def amend_capitalization(self):
        """Synthetic lower/title candidates share a prefix but not token IDs."""
        contract=readout.case_contract
        amendment_path=self.root/'amendments.json'
        write_json(amendment_path,dict(format='pluto-paired-lowercase-amendment-v1',complete=True,
            old_manifest=self.plans['main']['manifest'],replacement_rules=[{'source':'Exeunt','target':'Nuveth'},
                                                                          {'source':'exeunt','target':'nuveth'}]))
        for suite in readout.SUITES:
            plan=self.plans[suite]
            for index,case in enumerate(plan['cases']):
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
                    case['target_source'].update(bytes_hex=b''.join(pieces).hex(),native_piece_bytes_hex=[p.hex() for p in pieces])
                prefix_length=case['prefix']['length']
                sequence=np.full(17,63,dtype='<i4'); sequence[:prefix_length]=self.packed[suite][0,index,:prefix_length]
                sequence[prefix_length:prefix_length+len(case['target_ids'])]=case['target_ids']
                self.packed[suite][:,index,:]=[sequence[:-1],sequence[1:]]
            self.packed[suite].tofile(plan['packed_batch']['path'])
            plan.update(format=contract.AMENDED[0 if suite=='main' else 1],complete=True,candidate_pairs=contract.PAIRS,
                        amendment=readout._record(amendment_path),old_manifest=plan.pop('manifest'),
                        provenance=[],sources=[],packed_batch=readout._record(plan['packed_batch']['path']))
            del plan['candidate_words']
            if suite=='supplemental':
                plan.update(source_word_cases=readout._record(self.cases['main']),
                            source_word_packed_batch=self.plans['main']['packed_batch'])
            write_json(self.cases[suite],plan)
        for role in readout.ROLES:
            for suite in readout.SUITES: self.refresh_execution(role,suite)

    def analyze(self):
        return readout.analyze(self.intervention, self.paths['patched'] / 'patch.json',
                               self.cases['main'], self.cases['supplemental'], self.scores,
                               self.output, executions=self.executions, config=self.config)

    def mutate_losses(self, role, suite, callback):
        path = self.scores[role][suite] / 'losses.f32.bin'
        losses = np.fromfile(path, dtype='<f4').reshape(-1, 16)
        callback(losses)
        losses.tofile(path)
        self.refresh_execution(role, suite)


class BranchReadoutTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def test_all_six_native_records_and_byte_exact_copy_baselines(self):
        result = self.fixture.analyze()
        self.assertTrue(result['complete'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(result['selection_semantics'], {'block': 0, 'branch': 'attention',
                         'scope': 'whole_branch', 'includes_pre_layernorm': True, 'includes_output_bias': True})
        self.assertTrue(self.fixture.output.is_file())

    def test_amended_case_pairs_do_not_pool_even_when_prefixes_match(self):
        self.fixture.amend_capitalization()
        result=self.fixture.analyze()
        self.assertEqual(len(result['odds']['per_context']),3)
        groups=result['odds']['groups']
        self.assertEqual({(g['split'],g['spelling_variant']) for g in groups},
                         {('training','title'),('training','lowercase'),('test','title')})
        lower=next(g for g in groups if g['spelling_variant']=='lowercase')
        self.assertEqual(lower['candidate_pair'],{'original':'exeunt','replacement':'nuveth'})
        self.assertNotIn('mean_log_odds_Nuveth_over_Exeunt',lower)
        self.assertEqual(lower['effect_nats']['patched_minus_recipient'],.25)
        self.assertEqual(lower['both_words_degraded_case_count'],1)
        row=next(r for r in result['odds']['per_context'] if r['spelling_variant']=='lowercase')
        self.assertEqual(set(row['word_log_probabilities']),{'exeunt','nuveth'})
        self.assertNotIn('log_odds_Nuveth_over_Exeunt',row)
        self.assertTrue(result['checks']['main_supplemental_word_scores_byte_equivalent'])
        self.assertTrue(any(r['path'].endswith('paired_case_contract.py') for r in result['provenance']))

    def test_amended_spelling_mismatch_rejected_before_scores(self):
        self.fixture.amend_capitalization()
        case=self.fixture.plans['main']['cases'][0]
        case['target_source_domain']='replacement'
        write_json(self.fixture.cases['main'],self.fixture.plans['main'])
        with self.assertRaisesRegex(ValueError,'source corpus role'): self.fixture.analyze()

    def test_amended_policy_must_include_lowercase(self):
        self.fixture.amend_capitalization()
        path=self.fixture.root/'amendments.json'; value=readout._json(path)
        value['replacement_rules'].pop(); write_json(path,value)
        for suite in readout.SUITES:
            self.fixture.plans[suite]['amendment']=readout._record(path)
            write_json(self.fixture.cases[suite],self.fixture.plans[suite])
        with self.assertRaisesRegex(ValueError,'replacement policy'): self.fixture.analyze()

    def test_dedup_domains_occurrences_and_separate_splits(self):
        result = self.fixture.analyze()
        groups = [g for g in result['groups'] if g['kind'] == 'word'
                  and g['prefix_domain'] == 'deduplicated_all']
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(g['case_count'] == 1 and g['source_case_count'] == 4 for g in groups))
        self.assertEqual(len(result['odds']['per_context']), 2)
        self.assertEqual([g['case_count'] for g in result['odds']['groups']], [1, 1])

    def test_preference_improves_even_though_both_words_degrade(self):
        result = self.fixture.analyze()
        row = result['odds']['per_context'][0]
        self.assertTrue(row['both_words_less_likely_after_patch'])
        self.assertEqual(row['effect_nats']['patched_minus_recipient'], .25)
        self.assertGreater(row['effect_nats']['transfer_fraction'], 0)
        word = next(g for g in result['groups'] if g['kind'] == 'word' and g['target'] == 'Exeunt')
        self.assertAlmostEqual(word['metrics']['sequence']['models']['patched']['geometric_mean_probability'],
                               math.exp(-3.))
        self.assertEqual(word['metrics']['tokens'][0]['effect_nats']['patched_minus_recipient'], -1.)

    def test_exact_next_token_and_shared_piece_collateral_are_separate(self):
        result = self.fixture.analyze()
        next_token = next(g for g in result['groups'] if g['kind'] == 'word_next_native')
        self.assertEqual(next_token['metrics']['following_native_token']['effect_nats']['patched_minus_recipient'], 0)
        shared = next(g for g in result['groups'] if g['kind'] == 'shared_piece')
        self.assertEqual(shared['metrics']['tokens'][0]['effect_nats']['patched_minus_recipient'], -.25)
        self.assertEqual({g['piece_id'] for g in result['shared_piece_groups']}, {11})
        self.assertEqual({g['split'] for g in result['shared_piece_groups']}, {'training', 'test'})

    def test_native_tokenization_variants_have_separate_word_and_odds_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp), spaced=True)
            result = fixture.analyze()
        variants = [g for g in result['variant_groups'] if g['prefix_domain'] == 'deduplicated_all']
        self.assertEqual({g['leading_space'] for g in variants}, {False, True})
        self.assertEqual({g['kind'] for g in variants}, {'word', 'word_next_native'})
        self.assertEqual(len(variants), 16)
        self.assertTrue(all(g['case_count'] == 1 and g['source_case_count'] == 2 for g in variants))
        self.assertEqual(len(result['odds']['variant_groups']), 4)
        self.assertEqual({g['case_count'] for g in result['odds']['variant_groups']}, {1})
        self.assertEqual({g['case_count'] for g in result['odds']['groups']}, {2})

    def test_all_branch_scope_variants(self):
        for branch in ('attention', 'mlp'):
            for scope in ('whole_branch', 'output_write'):
                with self.subTest(branch=branch, scope=scope), tempfile.TemporaryDirectory() as temp:
                    fixture = Fixture(Path(temp), branch, scope, baseline_copies=False)
                    result = fixture.analyze()
                    self.assertEqual(result['selection_semantics']['branch'], branch)
                    self.assertEqual(result['selection_semantics']['scope'], scope)

    def test_transfer_fraction_requires_nontrivial_gap_and_is_not_clipped(self):
        for gap in (0., 1e-8, -1e-8):
            self.assertIsNone(readout.effect({'recipient': -1., 'donor': -1.+gap, 'patched': -2.})['transfer_fraction'])
        self.assertEqual(readout.effect({'recipient': -3., 'donor': -2., 'patched': -1.})['transfer_fraction'], 2.)
        self.assertEqual(readout.effect({'recipient': -3., 'donor': -2., 'patched': -4.})['transfer_fraction'], -1.)

    def test_refuses_missing_or_mutated_execution_provenance(self):
        path = self.fixture.executions['patched']['main']
        run = readout._json(path)
        run['inputs_before'] = run['inputs_before'][1:]
        run['inputs_after'] = run['inputs_before']
        write_json(path, run)
        with self.assertRaisesRegex(ValueError, 'omits required'):
            self.fixture.analyze()

    def test_refuses_failed_execution(self):
        path = self.fixture.executions['recipient']['supplemental']
        run = readout._json(path)
        run['returncode'] = 1
        write_json(path, run)
        with self.assertRaisesRegex(ValueError, 'failed native execution'):
            self.fixture.analyze()

    def test_refuses_bad_native_shape_even_when_execution_hashes_are_updated(self):
        path = self.fixture.scores['donor']['main'] / 'metadata.json'
        metadata = readout._json(path)
        metadata['output_shape'] = [16, self.fixture.plans['main']['case_count']]
        write_json(path, metadata)
        self.fixture.refresh_execution('donor', 'main')
        with self.assertRaisesRegex(ValueError, 'incompatible/incomplete'):
            self.fixture.analyze()

    def test_refuses_checkpoint_step_metadata_disagreement(self):
        path = self.fixture.scores['donor']['main'] / 'metadata.json'
        metadata = readout._json(path)
        metadata['checkpoint_step'] = 99
        write_json(path, metadata)
        self.fixture.refresh_execution('donor', 'main')
        with self.assertRaisesRegex(ValueError, 'directory/step mismatch'):
            self.fixture.analyze()

    def test_refuses_argmax_depending_on_future_word_target(self):
        path = self.fixture.scores['patched']['main'] / 'argmax.i32.bin'
        values = np.fromfile(path, dtype='<i4').reshape(-1, 16)
        values[1, 3] = 2
        values.tofile(path)
        self.fixture.refresh_execution('patched', 'main')
        with self.assertRaisesRegex(ValueError, 'inconsistent argmax'):
            self.fixture.analyze()

    def test_metrics_deduplicate_only_irrelevant_following_token_variants(self):
        def item(last, logs):
            return {'split': 'training', 'prefix_ids_sha256': 'a', 'prefix_length': 4,
                    'target_ids': [11, 12, 13, last],
                    'models': {role: {'token_log_probabilities': logs} for role in readout.ROLES}}
        rows = [item(9, [-1., -.25, -.5, -1.]), item(10, [-1., -.25, -.5, -2.])]
        first_three = readout._metric(rows, range(3))
        full = readout._metric(rows, range(4))
        self.assertEqual(first_three['case_count'], 1)
        self.assertEqual(full['case_count'], 2)
        self.assertEqual(first_three['models']['recipient']['mean_log_probability'], -1.75)
        self.assertEqual(full['models']['recipient']['mean_log_probability'], -3.25)

    def test_execution_command_must_match_actual_frozen_checkpoint(self):
        path = self.fixture.executions['patched']['main']
        run = readout._json(path)
        run['command'][1] = '--checkpoint=/different/step_100'
        write_json(path, run)
        with self.assertRaisesRegex(ValueError, 'command path mismatch'):
            self.fixture.analyze()

    def test_refuses_nonfinite_losses(self):
        self.fixture.mutate_losses('patched', 'main', lambda a: a.__setitem__((0, 3), np.nan))
        with self.assertRaisesRegex(ValueError, 'invalid native loss'):
            self.fixture.analyze()

    def test_refuses_duplicate_or_cross_suite_causal_disagreement(self):
        self.fixture.mutate_losses('patched', 'supplemental', lambda a: a.__setitem__((0, 3), 3.))
        with self.assertRaisesRegex(ValueError, 'inconsistent losses'):
            self.fixture.analyze()

    def test_refuses_wrong_patch_scope(self):
        self.fixture.intervention['kind'] = 'attention_output_write'
        with self.assertRaisesRegex(ValueError, 'incorrect whole-branch'):
            self.fixture.analyze()

    def test_refuses_mismatched_step_or_donor(self):
        self.fixture.intervention['donor_checkpoint']['step'] = 99
        with self.assertRaisesRegex(ValueError, 'source identity'):
            self.fixture.analyze()

    def test_refuses_weight_tampering(self):
        path = self.fixture.paths['patched'] / 'weight_0.bin'
        with path.open('r+b') as stream:
            stream.write(np.asarray([123.], dtype='<f4').tobytes())
        with self.assertRaisesRegex(ValueError, 'checkpoint weight hash'):
            self.fixture.analyze()

    def test_refuses_selected_bytes_from_wrong_source_even_with_honest_hash(self):
        spec = next(s for s in self.fixture.specs if s.name in self.fixture.intervention['tensors'])
        destination = self.fixture.paths['patched'] / spec.filename
        destination.write_bytes((self.fixture.paths['recipient'] / spec.filename).read_bytes())
        path = self.fixture.paths['patched'] / 'patch.json'
        patch = readout._json(path)
        patch['output']['weights_sha256'][spec.filename] = readout._record(destination)['sha256']
        write_json(path, patch)
        with self.assertRaisesRegex(ValueError, 'selected/unselected bytes'):
            self.fixture.analyze()

    def test_refuses_overwrite_or_output_inside_checkpoint(self):
        self.fixture.output.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.fixture.analyze()
        self.assertEqual(self.fixture.output.read_text(), 'preserve')
        self.fixture.output = self.fixture.paths['recipient'] / 'report.json'
        with self.assertRaisesRegex(ValueError, 'outside input directories'):
            self.fixture.analyze()

    def test_refuses_reordered_target_rows(self):
        plan = self.fixture.plans['main']
        plan['cases'][0]['scored_rows'] = [4, 3, 5]
        write_json(self.fixture.cases['main'], plan)
        with self.assertRaisesRegex(ValueError, 'target order/row mismatch'):
            self.fixture.analyze()


if __name__ == '__main__':
    unittest.main()
