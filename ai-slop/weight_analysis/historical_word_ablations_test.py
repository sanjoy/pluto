"""Synthetic CPU tests; their numbers are not historical experimental evidence."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from .checkpoint import GPT2Config, tensor_manifest
from . import historical_word_ablations as readout


def write_json(path, value):
    path.write_text(json.dumps(value))


class Fixture:
    """A tiny complete native archive, including growing and sliding contexts."""

    def __init__(self, base):
        self.base = Path(base)
        self.root = self.base / 'run'
        self.generation = self.base / 'generation'
        self.checkpoint = self.base / 'step_17'
        self.repository = self.base / 'repo'
        for path in (self.root, self.generation, self.checkpoint, self.repository):
            path.mkdir()
        self.config = GPT2Config(vocab_size=5, padded_vocab_size=8, context_length=4,
                                 n_layers=1, d_model=2, n_heads=2, d_ff=4)
        self.source = self.repository / readout.PRODUCER_SOURCE
        self.source.parent.mkdir(parents=True)
        self.source_bytes = b'// Synthetic producer snapshot, not GPU evidence.\n'
        self.source.write_bytes(self.source_bytes)
        self.source_record = readout.record(self.source)
        self.manifest_path = self.base / 'evidence.json'
        for spec in tensor_manifest(self.config):
            (self.checkpoint / spec.filename).write_bytes(
                np.arange(spec.nbytes // 4, dtype='<f4').tobytes())
        initial, generated = [0, 1], [2, 3, 4, 0, 1]
        all_ids = initial + generated
        byte_map = {0: b'q', 1: b'r', 2: b' x', 3: b'y', 4: b'p'}
        self.generation_metadata = dict(
            complete=True, checkpoint_directory=str(self.checkpoint),
            context_length=4, vocab_size=5, padded_vocab_size=8,
            initial_token_ids=initial, generated_token_ids=generated,
            all_token_ids=all_ids, steps=len(generated),
            events_file='events.jsonl', logits_file='logits.f32')
        write_json(self.generation / 'metadata.json', self.generation_metadata)
        events, runs, originals = [], [], []
        for step, target in enumerate(generated):
            absolute = len(initial) + step
            start = max(0, absolute - self.config.context_length)
            ids = all_ids[start:absolute]
            event = dict(index=step, step=step, absolute_token_index=absolute,
                         context_start=start, context_length=len(ids),
                         output_row=len(ids)-1, token_id=target,
                         logits_byte_offset=step*5*4, piece_hex=byte_map[target].hex())
            events.append(event)
            runs.append(dict(step=step, interventions=True,
                             context_token_ids=ids, event=dict(event)))
            native = self.root / f'trace_step_{step}'
            native.mkdir()
            # Extremely large finite padding must have NO probability mass.
            clean = np.array([-1., 0., 1., 2., 3., 1000., 2000., 3000.], dtype='<f4')
            clean[target] += np.float32(0.2 * step)
            originals.append(clean[:5].tobytes())
            for filename in ('logits.f32', 'replay.f32', 'padding.f32'):
                (native / filename).write_bytes(clean.tobytes())
            arms = []
            for index, (name, identity) in enumerate(readout.expected_arms(self.config).items()):
                values = clean.copy()
                values[target] -= np.float32((index+1) * (0.1 + step * 0.03))
                filename = name + '.f32'
                (native / filename).write_bytes(values.tobytes())
                arms.append(dict(name=name, **identity, scale=0, logits_file=filename,
                                 shape=[1, 8], role='selected_row', restoration_verified_bytes=True))
            checks = {k: True for k in (
                'clean_replay_logits_byte_equal', 'alternate_padding_logits_byte_equal',
                'final_lens_logits_byte_equal', 'attention_residual_replay_byte_equal',
                'mlp_residual_replay_byte_equal')}
            metadata = dict(complete=True, probe_kind='token_trace',
                checkpoint_directory=str(self.checkpoint), vocab_size=5,
                padded_vocab_size=8, context_length=4, token_ids=ids,
                target_id=target, prompt_rows=len(ids), selected_row=len(ids)-1,
                checkpoint_unique_weight_count=len(tensor_manifest(self.config)),
                byte_order='little', retokenized=False, optimizer_steps=0,
                backward_calls=0, checkpoint_writes=0, checks=checks,
                files={'logits': dict(file='logits.f32', dtype='float32', shape=[1, 8])},
                parity_files={'clean_replay': 'replay.f32', 'alternate_padding': 'padding.f32'},
                interventions=arms)
            write_json(native / 'metadata.json', metadata)
        (self.generation / 'events.jsonl').write_text(
            '\n'.join(json.dumps(event) for event in events) + '\n')
        (self.generation / 'logits.f32').write_bytes(b''.join(originals))
        self.plan = dict(output_directory=str(self.root), vocab_size=5,
            padded_vocab_size=8, n_layers=1, n_heads=2,
            checkpoint_directory=str(self.checkpoint),
            generation_directory=str(self.generation),
            generation_logits=str(self.generation / 'logits.f32'),
            selections=[dict(word='xy', first=0, end=2), dict(word='pqr', first=2, end=5)],
            runs=runs)
        self.refresh()

    def refresh(self):
        """Re-enroll intentional fixture changes to test semantics after hashes."""
        self.plan['inputs'] = [self.source_record] + [readout.record(p)
            for directory in (self.checkpoint, self.generation)
            for p in sorted(directory.iterdir()) if p.is_file()]
        write_json(self.root / 'plan.json', self.plan)
        files = []
        for path in sorted(self.root.rglob('*')):
            if path.is_file():
                item = readout.record(path)
                files.append(dict(original_path=item.pop('path'), **item))
        write_json(self.manifest_path, dict(complete=True, source_commit_sha='a'*40, files=files))

    def metadata(self, step=0):
        return json.loads((self.root / f'trace_step_{step}/metadata.json').read_text())

    def replace_metadata(self, value, step=0):
        write_json(self.root / f'trace_step_{step}/metadata.json', value)
        self.refresh()

    def analyze(self):
        completed = subprocess.CompletedProcess([], 0, stdout=self.source_bytes, stderr=b'')
        with mock.patch.object(readout.subprocess, 'run', return_value=completed):
            return readout.analyze(self.manifest_path, self.root, self.repository, config=self.config)


class NumericalTest(unittest.TestCase):
    def test_logical_vocabulary_excludes_padding(self):
        info = readout.token_readout([0., 0., 0., 10000.], 1, 3)
        self.assertAlmostEqual(info['probability'], 1/3)
        self.assertAlmostEqual(info['nll'], math.log(3))
        self.assertEqual(info['target_rank'], 2)

    def test_scalar_probability_and_shift_invariance(self):
        a = readout.token_readout([0., 1., 2.], 2, 3)
        b = readout.token_readout([100., 101., 102.], 2, 3)
        self.assertEqual(a['nll'], b['nll'])
        self.assertAlmostEqual(a['probability'], math.exp(2)/(1+math.e+math.exp(2)))

    def test_extreme_logits_preserve_finite_nll_after_probability_underflow(self):
        info = readout.token_readout([10000., -10000.], 1, 2)
        self.assertEqual(info['probability'], 0)
        self.assertEqual(info['nll'], 20000.)

    def test_sequence_uses_one_arm_and_separates_suffix(self):
        result = readout.sequence_readout([2., 3., 4.], [1., 1., 1.])
        self.assertEqual(result['first_piece']['nll'], 2.)
        self.assertEqual(result['conditional_suffix']['nll'], 7.)
        self.assertEqual(result['joint']['nll'], 9.)
        self.assertEqual(result['joint']['delta_nll'], 6.)
        self.assertAlmostEqual(result['joint']['probability'], math.exp(-9))
        self.assertAlmostEqual(result['joint']['probability_ratio'], math.exp(-6))
        self.assertEqual(result['joint']['token_count'], 3)

    def test_empty_suffix_and_ratio_overflow_are_explicit(self):
        result = readout.sequence_readout([0.], [10000.])
        self.assertEqual(result['conditional_suffix']['probability'], 1.)
        self.assertEqual(result['conditional_suffix']['token_count'], 0)
        self.assertIsNone(result['joint']['probability_ratio'])
        self.assertEqual(result['joint']['log_probability_ratio'], 10000.)
        json.dumps(result, allow_nan=False)

    def test_bad_numerical_inputs(self):
        for args in (([1., 2.], True, 2), ([1., 2.], 2, 2),
                     ([1., float('nan')], 0, 2), ([1., float('inf')], 0, 2),
                     ([[1., 2.]], 0, 2), ([1., 2.], 0, 3), ([1.], 0, 1)):
            with self.assertRaises(ValueError):
                readout.token_readout(*args)
        for args in (([],), ([-1.],), ([float('nan')],), ([1.], [1., 2.]),
                     ([1.], [-1.]), ([1.], [float('inf')])):
            with self.assertRaises(ValueError):
                readout.sequence_readout(*args)


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(self.temporary.name)

    def test_complete_aggregation_and_all_native_arm_joins(self):
        result = self.fixture.analyze()
        self.assertEqual(result['counts'], dict(words=2, token_predictions=5,
                         interventions_per_word=4, native_piece_ablations=20))
        self.assertFalse(result['new_deterministic_experiment_result'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(result['temperature'], 1)
        self.assertEqual(result['words'][0]['token_sequence_bytes_hex'], b' xy'.hex())
        self.assertEqual([p['context_start'] for p in result['words'][1]['pieces']], [0, 1, 2])
        for word in result['words']:
            for arm in word['interventions']:
                self.assertEqual([p['generation_step'] for p in arm['per_piece']], word['generation_steps'])
                self.assertAlmostEqual(arm['joint']['probability'],
                                       math.prod(p['probability'] for p in arm['per_piece']))
                self.assertAlmostEqual(arm['joint']['delta_nll'],
                                       math.fsum(p['delta_nll'] for p in arm['per_piece']))
        self.assertEqual(len(result['cross_word_comparisons']), 4)
        for row in result['cross_word_comparisons']:
            self.assertEqual(set(row['words']), {'xy', 'pqr'})
        for item in result['provenance'] + result['analysis_sources']:
            self.assertEqual(readout.record(item['path']), item)
        json.dumps(result, allow_nan=False)

    def test_dirty_current_producer_is_not_used(self):
        self.fixture.source.write_text('// Different current source.\n')
        result = self.fixture.analyze()
        self.assertEqual(result['historical_producer_source']['sha256'],
                         hashlib.sha256(self.fixture.source_bytes).hexdigest())
        self.assertFalse(result['historical_producer_source']['current_worktree_source_equality_claimed'])

    def test_hash_tampering_is_rejected(self):
        path = self.fixture.root / 'trace_step_0/ablation.block0.head0.f32'
        path.write_bytes(b'x' + path.read_bytes()[1:])
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.fixture.analyze()

    def test_checkpoint_wrong_size_and_nonfinite_are_rejected(self):
        path = self.fixture.checkpoint / 'weight_0.bin'
        original = path.read_bytes()
        for raw in (original[:-4], np.float32(np.nan).tobytes() + original[4:]):
            with self.subTest(raw_bytes=len(raw)):
                path.write_bytes(raw)
                self.fixture.refresh()
                with self.assertRaisesRegex(ValueError, 'wrong-sized or nonfinite'):
                    self.fixture.analyze()

    def test_extra_weight_inventory_is_rejected(self):
        (self.fixture.checkpoint / 'weight_999.bin').write_bytes(b'1234')
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'weight inventory'):
            self.fixture.analyze()

    def test_original_generation_and_padded_parity_are_checked(self):
        for filename in ('replay.f32', 'padding.f32'):
            path = self.fixture.root / 'trace_step_0' / filename
            original = path.read_bytes()
            path.write_bytes(original[:-4] + np.float32(99).tobytes())
            self.fixture.refresh()
            with self.assertRaisesRegex(ValueError, 'full-padded'):
                self.fixture.analyze()
            path.write_bytes(original)
        path = self.fixture.generation / 'logits.f32'
        path.write_bytes(np.float32(99).tobytes() + path.read_bytes()[4:])
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'original generation'):
            self.fixture.analyze()

    def test_context_identity_cannot_be_replaced_by_matching_logits(self):
        metadata = self.fixture.metadata(3)
        metadata['token_ids'][0] = (metadata['token_ids'][0] + 1) % 5
        self.fixture.replace_metadata(metadata, 3)
        with self.assertRaisesRegex(ValueError, 'generation context'):
            self.fixture.analyze()

    def test_step_coordinates_and_original_event_are_checked(self):
        self.fixture.plan['runs'][3]['event']['context_start'] = 0
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'original generation event'):
            self.fixture.analyze()

    def test_missing_duplicate_wrong_identity_or_unrestored_arm_is_rejected(self):
        original = self.fixture.metadata()
        mutations = [
            lambda a: a.pop(),
            lambda a: a.__setitem__(1, dict(a[0])),
            lambda a: a[0].update(restoration_verified_bytes=False),
            lambda a: a[2].update(head_or_neuron=1),
            lambda a: a[0].update(scale=0.5),
            lambda a: a[0].update(logits_file='../outside.f32'),
            lambda a: a[1].update(logits_file=a[0]['logits_file']),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = json.loads(json.dumps(original))
                mutation(value['interventions'])
                self.fixture.replace_metadata(value)
                with self.assertRaises(ValueError):
                    self.fixture.analyze()

    def test_incomplete_trace_bad_dtype_or_nonfinite_logits_is_rejected(self):
        original = self.fixture.metadata()
        for update in ({'complete': False}, {'retokenized': True}, {'backward_calls': 1}):
            value = dict(original, **update)
            self.fixture.replace_metadata(value)
            with self.assertRaisesRegex(ValueError, 'incomplete native trace'):
                self.fixture.analyze()
        value = json.loads(json.dumps(original))
        value['files']['logits']['dtype'] = 'bf16'
        self.fixture.replace_metadata(value)
        with self.assertRaisesRegex(ValueError, 'schema mismatch'):
            self.fixture.analyze()
        self.fixture.replace_metadata(original)
        path = self.fixture.root / 'trace_step_0/ablation.block0.head0.f32'
        path.write_bytes(np.float32(np.inf).tobytes() + path.read_bytes()[4:])
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.fixture.analyze()

    def test_word_label_and_selection_integrity(self):
        self.fixture.plan['selections'][0]['word'] = 'notxy'
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'spell'):
            self.fixture.analyze()
        self.fixture.plan['selections'][0]['word'] = 'xy'
        self.fixture.plan['selections'].append(dict(word='overlap', first=1, end=3))
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.fixture.analyze()

    def test_end_rehash_detects_file_change_after_use(self):
        original = readout.token_readout
        changed = False

        def change_after_read(*args, **kwargs):
            nonlocal changed
            if not changed:
                path = self.fixture.root / 'plan.json'
                path.write_bytes(path.read_bytes() + b'\n')
                changed = True
            return original(*args, **kwargs)

        with mock.patch.object(readout, 'token_readout', side_effect=change_after_read):
            with self.assertRaisesRegex(ValueError, 'changed during analysis'):
                self.fixture.analyze()

    def test_symlink_and_duplicate_manifest_identity_are_rejected(self):
        path = self.fixture.root / 'trace_step_0/logits.f32'
        saved = path.with_name('saved.f32')
        path.rename(saved)
        path.symlink_to(saved.name)
        with self.assertRaisesRegex(ValueError, 'nonsymlink'):
            self.fixture.analyze()
        path.unlink()
        saved.rename(path)
        manifest = json.loads(self.fixture.manifest_path.read_text())
        manifest['files'].append(manifest['files'][0])
        write_json(self.fixture.manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.fixture.analyze()

    def test_cli_refuses_existing_output_before_analysis(self):
        output = self.fixture.base / 'result.json'
        output.write_text('preserve me')
        with mock.patch.object(readout, 'analyze') as analyze:
            with self.assertRaises(FileExistsError):
                readout.main(['--evidence-manifest', str(self.fixture.manifest_path),
                              '--run-directory', str(self.fixture.root), '--output', str(output)])
        analyze.assert_not_called()
        self.assertEqual(output.read_text(), 'preserve me')


class GitRecoveryTest(unittest.TestCase):
    def test_git_recovers_committed_source_despite_dirty_worktree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / readout.PRODUCER_SOURCE
            source.parent.mkdir(parents=True)
            source.write_text('// The saved fixture producer.\n')
            expected = readout.record(source)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
            git('init', '-q')
            git('add', readout.PRODUCER_SOURCE)
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                'commit', '-qm', 'fixture')
            revision = git('rev-parse', 'HEAD').decode().strip()
            source.write_text('// A new, different current producer.\n')
            info = readout.recover_producer_source(root, revision, expected)
            self.assertEqual(info['sha256'], expected['sha256'])
            self.assertTrue(info['matched_historical_plan'])
            for bad_revision in ('HEAD', '-n1', 'a'*39, 'A'*40):
                with self.assertRaisesRegex(ValueError, 'full Git commit'):
                    readout.recover_producer_source(root, bad_revision, expected)
            with self.assertRaisesRegex(ValueError, 'differs'):
                readout.recover_producer_source(root, revision, dict(expected, sha256='0'*64))
            with self.assertRaisesRegex(ValueError, 'could not recover'):
                readout.recover_producer_source(root, 'f'*40, expected)


if __name__ == '__main__':
    unittest.main()
