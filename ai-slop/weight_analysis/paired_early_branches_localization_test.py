"""CPU-only checks for early attention/MLP/rest coordinate interventions."""

from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from . import paired_complement_localization as localization
from .checkpoint import GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest


LAYOUT = 'early_branches'
CELLS = {'E': (), 'A': ('A',), 'M': ('M',), 'R': ('R',),
         'AM': ('A', 'M'), 'AR': ('A', 'R'), 'MR': ('M', 'R'),
         'EC': ('A', 'M', 'R')}
CONFIG = GPT2Config(vocab_size=16, padded_vocab_size=20, context_length=8,
                    n_layers=2, d_model=4, n_heads=2, d_ff=8)


def polynomial(subset):
    a, m, r = (float(axis in subset) for axis in ('A', 'M', 'R'))
    return 2*a + 3*m + 5*r + 7*a*m + 11*a*r + 13*m*r + 17*a*m*r


class EarlyLayoutTest(unittest.TestCase):
    def test_production_tensor_partition_has_exact_file_indices_and_parameter_counts(self):
        groups = localization.groups(layout=LAYOUT)
        specs = {spec.name: spec for spec in tensor_manifest()}
        expected = {'A': {index for block in range(4) for index in range(2+12*block, 8+12*block)},
                    'M': {index for block in range(4) for index in range(8+12*block, 14+12*block)},
                    'R': {1, *range(50, 100)}}
        self.assertEqual(set(groups), set(expected))
        for axis, indices in expected.items():
            self.assertEqual({specs[name].index for name in groups[axis]}, indices)
        self.assertEqual({axis: len(names) for axis, names in groups.items()},
                         {'A': 24, 'M': 24, 'R': 51})
        self.assertEqual({axis: sum(specs[name].nbytes // 4 for name in names)
                          for axis, names in groups.items()},
                         {'A': 4206592, 'M': 8402944, 'R': 13134848})
        flattened = [name for names in groups.values() for name in names]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(specs) - {'token_embedding.weight'})

    def test_each_early_branch_contains_its_own_norm_and_all_projection_biases(self):
        groups = localization.groups(layout=LAYOUT)
        suffixes = {'A': ('ln1.scale', 'ln1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                          'attn.output.weight', 'attn.output.bias'),
                    'M': ('ln2.scale', 'ln2.bias', 'mlp.input.weight', 'mlp.input.bias',
                          'mlp.output.weight', 'mlp.output.bias')}
        for axis, names in suffixes.items():
            self.assertEqual(set(groups[axis]),
                             {f'blocks.{block}.{name}' for block in range(4) for name in names})
        self.assertTrue({'position_embedding.weight', 'final_norm.scale', 'final_norm.bias'}
                        <= set(groups['R']))

    def test_tiny_even_layout_uses_early_half_and_rest_without_production_index_assumptions(self):
        groups = localization.groups(CONFIG, layout=LAYOUT)
        self.assertEqual({axis: len(names) for axis, names in groups.items()},
                         {'A': 6, 'M': 6, 'R': 15})
        self.assertTrue(all(name.startswith('blocks.0.') for axis in ('A', 'M') for name in groups[axis]))
        self.assertEqual({name for name in groups['R'] if name.startswith('blocks.')},
                         {spec.name for spec in tensor_manifest(CONFIG) if spec.name.startswith('blocks.1.')})
        for layers in (1, 3, 5):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                localization.groups(replace(CONFIG, n_layers=layers), layout=LAYOUT)

    def test_layout_and_cube_names_are_explicit_and_legacy_defaults_are_unchanged(self):
        self.assertEqual(localization.cube_subsets(layout=LAYOUT), CELLS)
        self.assertEqual(set(localization.groups()), {'Q', 'H', 'L'})
        self.assertEqual(localization.groups(), localization.groups(layout='complement'))
        self.assertEqual(localization.cube_subsets(), {
            'E': (), 'Q': ('Q',), 'H': ('H',), 'L': ('L',),
            'QH': ('Q', 'H'), 'QL': ('Q', 'L'), 'HL': ('H', 'L'), 'EC': ('Q', 'H', 'L')})
        for layout in ('early', '', None):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                localization.groups(layout=layout)
            with self.assertRaises(ValueError):
                localization.cube_subsets(layout=layout)

    def test_polynomial_effects_match_both_complement_backgrounds(self):
        scores = {cell: -100.0 + polynomial(subset) for cell, subset in CELLS.items()}
        actual = localization.score_cube(scores, layout=LAYOUT)
        self.assertEqual(actual['additions'], {'A': 2.0, 'M': 3.0, 'R': 5.0})
        self.assertEqual(actual['removals'], {'A': 37.0, 'M': 40.0, 'R': 46.0})
        self.assertEqual(actual['pair_interactions'], {
            'AM': {'absent': 7.0, 'present': 24.0},
            'AR': {'absent': 11.0, 'present': 28.0},
            'MR': {'absent': 13.0, 'present': 30.0}})
        self.assertEqual(actual['three_way'], 17.0)

    def test_null_and_negative_effects_preserve_the_early_axis_labels(self):
        for changed in (None, 'A', 'M', 'R'):
            values = {cell: -10.0 - (2.0 if changed in subset else 0.0)
                      for cell, subset in CELLS.items()}
            actual = localization.score_cube(values, layout=LAYOUT)
            expected = {axis: -2.0 if axis == changed else 0.0 for axis in ('A', 'M', 'R')}
            self.assertEqual(actual['additions'], expected)
            self.assertEqual(actual['removals'], expected)
            self.assertEqual(actual['three_way'], 0.0)
            self.assertTrue(all(value == {'absent': 0.0, 'present': 0.0}
                                for value in actual['pair_interactions'].values()))

    def test_missing_old_layout_extra_and_nonfinite_cube_cells_are_rejected(self):
        values = dict.fromkeys(CELLS, -1.0)
        invalid = [{key: value for key, value in values.items() if key != 'AM'},
                   dict(values, Q=-1.0), dict.fromkeys(localization.cube_subsets(), -1.0)]
        invalid.extend(dict(values, A=value) for value in
                       (float('nan'), float('inf'), -float('inf'), True, '-1.0', None))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                localization.score_cube(value, layout=LAYOUT)
        with self.assertRaises(ValueError):
            localization.score_cube(values)

    def test_selection_rejects_legacy_groups_duplicates_and_missing_embedding_rows(self):
        for subset in (('Q',), ('H',), ('L',), ('E',), ('EC',), ('A', 'A'), 'AM'):
            with self.subTest(subset=subset), self.assertRaises(ValueError):
                localization._selection(subset, [1, 2], CONFIG, layout=LAYOUT)
        for rows in ([], [True], [1.0], [1, 1], [-1], [16]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                localization._selection(('A',), rows, CONFIG, layout=LAYOUT)
        tensors, rows = localization._selection(('M', 'A'), [2, 1], CONFIG, layout=LAYOUT)
        self.assertEqual(rows, [1, 2])
        self.assertEqual(set(tensors), {spec.name for spec in tensor_manifest(CONFIG)
                                        if spec.name.startswith('blocks.0.')})


class PatchFixture:
    def __init__(self, test):
        temporary = tempfile.TemporaryDirectory(prefix='early-branches-cpu-test-')
        test.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.a = self.root / 'recipient'
        self.d = self.root / 'donor'
        self.rows = list(range(11))
        self.specs = tensor_manifest(CONFIG)
        for path, shift in ((self.a, 0), (self.d, 10000)):
            path.mkdir()
            for spec in self.specs:
                values = (np.arange(np.prod(spec.shape), dtype='<f4')
                          + np.float32(100 * spec.index + shift)).reshape(spec.shape)
                if spec.index == 0 and not shift:
                    values[-1, -1] = np.float32(-0.0)
                values.tofile(path / spec.filename)
        self.initial = {str(path): self.hashes(path) for path in (self.a, self.d)}

    def hashes(self, path):
        return {spec.filename: sha256_file(path / spec.filename) for spec in self.specs}

    def build(self, cell, *, layout=LAYOUT):
        directory = self.root / (layout + '_' + cell)
        directory.mkdir(exist_ok=True)
        return localization.build_model(self.a, self.d, directory / 'step_331',
            localization.cube_subsets(layout=layout)[cell], self.rows,
            config=CONFIG, layout=layout)

    def validate(self, model, subset, *, layout=LAYOUT):
        return localization.validate_model(model['patch']['path'], self.a, self.d,
            subset, self.rows, config=CONFIG, layout=layout)


class EarlyPatchTest(unittest.TestCase):
    def setUp(self):
        self.f = PatchFixture(self)

    def test_all_eight_real_models_have_exact_selected_bytes_and_fixed_padded_embedding(self):
        f = self.f
        recipient, donor = GPT2Checkpoint(f.a, CONFIG), GPT2Checkpoint(f.d, CONFIG)
        groups = localization.groups(CONFIG, layout=LAYOUT)
        embedding_bytes = None
        for cell, subset in CELLS.items():
            model = f.build(cell)
            self.assertEqual(f.validate(model, subset), model)
            path = Path(model['paths']['patched'])
            actual = GPT2Checkpoint(path, CONFIG, check_finite=True)
            selected = {name for axis in subset for name in groups[axis]}
            for spec in f.specs:
                expected = np.array(donor[spec.name] if spec.name in selected else recipient[spec.name])
                if spec.index == 0:
                    expected[f.rows] = donor[spec.name][f.rows]
                self.assertEqual(actual[spec.name].tobytes(), expected.tobytes(), (cell, spec.name))
                destination = (path / spec.filename).stat()
                self.assertEqual(destination.st_nlink, 1)
                for source in (f.a, f.d):
                    source_info = (source / spec.filename).stat()
                    self.assertNotEqual((destination.st_dev, destination.st_ino),
                                        (source_info.st_dev, source_info.st_ino))
            current_embedding = actual['token_embedding.weight'].tobytes()
            if embedding_bytes is None:
                embedding_bytes = current_embedding
            self.assertEqual(current_embedding, embedding_bytes)
            self.assertEqual(actual['token_embedding.weight'][CONFIG.vocab_size:].tobytes(),
                             recipient['token_embedding.weight'][CONFIG.vocab_size:].tobytes())
        for source in (f.a, f.d):
            self.assertEqual(f.hashes(source), f.initial[str(source)])

    def test_old_e_h_ql_ec_anchors_are_weight_identical_to_new_e_am_r_ec(self):
        for old, new in (('E', 'E'), ('H', 'AM'), ('QL', 'R'), ('EC', 'EC')):
            with self.subTest(old=old, new=new):
                legacy = self.f.build(old, layout='complement')
                early = self.f.build(new)
                self.assertEqual(legacy['hashes']['patched'], early['hashes']['patched'])

    def test_wrong_layout_or_branch_norm_selection_is_rejected(self):
        model = self.f.build('A')
        with self.assertRaises(ValueError):
            self.f.validate(model, ('H',), layout='complement')
        with self.assertRaises(ValueError):
            self.f.validate(model, ('M',))
        path = Path(model['patch']['path'])
        patch = json.loads(path.read_text())
        patch['selection']['tensors'] = sorted(localization.groups(CONFIG, layout=LAYOUT)['M'])
        path.write_text(json.dumps(patch))
        with self.assertRaises(ValueError):
            self.f.validate(model, ('M',))

    def test_padding_corruption_is_rejected_even_with_a_forged_output_hash(self):
        model = self.f.build('EC')
        path = Path(model['paths']['patched']) / 'weight_0.bin'
        values = np.fromfile(path, dtype='<f4')
        values[-1] = 1.0
        values.tofile(path)
        patch_path = Path(model['patch']['path'])
        patch = json.loads(patch_path.read_text())
        patch['output']['weights_sha256']['weight_0.bin'] = sha256_file(path)
        patch_path.write_text(json.dumps(patch))
        with self.assertRaisesRegex(ValueError, 'padded|embedding'):
            self.f.validate(model, ('A', 'M', 'R'))

    def test_an_unchanged_r_tensor_must_still_be_an_independent_copy(self):
        model = self.f.build('A')
        path = Path(model['paths']['patched']) / 'weight_1.bin'
        path.unlink()
        os.link(self.f.a / path.name, path)
        with self.assertRaises(ValueError):
            self.f.validate(model, ('A',))

    def test_invalid_layout_is_rejected_before_creating_a_model(self):
        output = self.f.root / 'not_created'
        with self.assertRaises(ValueError):
            localization.build_model(self.f.a, self.f.d, output, ('A',), self.f.rows,
                                     config=CONFIG, layout='unknown')
        self.assertFalse(output.exists())


class EarlyAggregationTest(unittest.TestCase):
    def test_each_case_requires_the_exact_early_cube_cell_set(self):
        item = dict(suite='main', kind='control', split='test', prefix_domain='shared',
                    target='control_next_3', prefix_sha256='same', prefix_length=2,
                    target_ids=[1, 2, 3],
                    cells={cell: dict(token_log_probability=[-1., -2., -3.]) for cell in CELLS})
        for change in ('missing', 'extra', 'legacy'):
            changed = copy.deepcopy(item)
            if change == 'missing':
                del changed['cells']['AR']
            elif change == 'extra':
                changed['cells']['Q'] = copy.deepcopy(changed['cells']['A'])
            else:
                changed['cells'] = {cell: dict(token_log_probability=[-1., -2., -3.])
                                    for cell in localization.cube_subsets()}
            with self.subTest(change=change), self.assertRaises(ValueError):
                localization._aggregates([changed], layout=LAYOUT)

    def test_aliases_and_following_tokens_keep_metric_dedup_and_early_effects(self):
        items = []
        for domain in ('original', 'replacement'):
            for fourth, logp in ((12, -2.0), (13, -4.0)):
                items.append(dict(suite='supplemental', kind='word_next_native',
                    split='training', spelling_variant='title', prefix_domain=domain,
                    word_has_leading_space=False, target='Exeunt', prefix_sha256='same',
                    prefix_length=2, target_ids=[1, 2, 3, fourth],
                    cells={cell: dict(token_log_probability=[-100.0 + polynomial(subset), -1.0, -2.0, logp])
                           for cell, subset in CELLS.items()}))
        groups = localization._aggregates(items, layout=LAYOUT)
        aggregate = next(group for group in groups if group['prefix_domain'] == 'deduplicated_all')
        word = aggregate['metrics']['word_three']
        self.assertEqual(word['unique_event_count'], 1)
        self.assertEqual(word['alias_case_count'], 4)
        self.assertEqual(word['effects']['additions'], {'A': 2.0, 'M': 3.0, 'R': 5.0})
        self.assertEqual(word['effects']['three_way'], 17.0)
        fourth = aggregate['metrics']['exact_next_native_token']
        self.assertEqual(fourth['unique_event_count'], 2)
        self.assertEqual(fourth['cells']['E']['mean_log_probability'], -3.0)
        self.assertEqual(fourth['effects']['three_way'], 0.0)

    def test_alias_conflicts_in_a_new_cell_cannot_be_silently_deduplicated(self):
        item = dict(suite='main', kind='word', split='test', prefix_domain='original',
                    target='Exeunt', prefix_sha256='same', prefix_length=2, target_ids=[1, 2, 3],
                    cells={cell: dict(token_log_probability=[-1., -2., -3.]) for cell in CELLS})
        alias = copy.deepcopy(item)
        alias['prefix_domain'] = 'replacement'
        alias['cells']['AR']['token_log_probability'][0] -= .25
        with self.assertRaises(ValueError):
            localization._aggregates([item, alias], layout=LAYOUT)


class ArchiveDelegationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='early-archive-delegation-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / 'scores'
        self.directory.mkdir()
        for filename in ('metadata.json', 'losses.f32.bin', 'argmax.i32.bin'):
            (self.directory / filename).write_bytes(b'CPU mocked scorer fixture')
        roles = ('recipient', 'donor', 'patched')
        self.model = dict(records=[], paths={role: str(self.root / role) for role in roles},
                          hashes={role: {} for role in roles}, weight_records={role: [] for role in roles})
        self.archives = {}
        for kind in ('case', 'execution'):
            path = self.root / (kind + '_archive.json')
            path.write_text(json.dumps({'fixture_kind': kind}))
            self.archives[kind] = SimpleNamespace(manifest_record=localization.native._record(path))
        self.cases = dict(plan={'cases': [dict(case_index=0, scored_rows=[0, 1, 2], target_ids=[1, 2, 3])]},
                          packed=np.asarray([[[5, 1, 2, 3]], [[1, 2, 3, 15]]], dtype='<i4'))
        self.scored = dict(losses=np.ones((1, 4), dtype='<f4'), argmax=np.ones((1, 4), dtype='<i4'))
        self.case_reader = self.enterContext(mock.patch.object(localization.branch, '_cases',
                                                              return_value=self.cases))
        self.score_reader = self.enterContext(mock.patch.object(localization.branch, '_scores',
                                                               return_value=self.scored))

    def load(self, **kwargs):
        return localization.load_native(self.model, self.root / 'cases.json', 'main',
            self.directory, self.root / 'execution.json', config=CONFIG, **kwargs)

    def test_case_and_execution_archives_are_forwarded_separately_and_only_when_supplied(self):
        for use_case, use_execution in ((False, False), (True, False), (False, True), (True, True)):
            options = {}
            if use_case:
                options['case_archive'] = self.archives['case']
            if use_execution:
                options['execution_archive'] = self.archives['execution']
            with self.subTest(use_case=use_case, use_execution=use_execution):
                result = self.load(**options)
                self.assertEqual(self.case_reader.call_args.kwargs,
                                 {'source_archive': self.archives['case']} if use_case else {})
                self.assertEqual(self.score_reader.call_args.kwargs,
                                 {'source_archive': self.archives['execution']} if use_execution else {})
                self.assertEqual({record['path'] for record in result['records']},
                                 {archive.manifest_record['path'] for archive in options.values()})

    def test_both_archive_manifests_require_their_declared_current_hash(self):
        for kind in ('case', 'execution'):
            archive = self.archives[kind]
            path = Path(archive.manifest_record['path'])
            before = path.read_bytes()
            path.write_bytes(b'changed manifest')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.load(**{kind + '_archive': archive})
            path.write_bytes(before)
        self.case_reader.assert_not_called()
        self.score_reader.assert_not_called()

    def test_archive_manifest_mutation_during_score_validation_is_rejected(self):
        archive = self.archives['execution']

        def change_manifest(*args, **kwargs):
            Path(archive.manifest_record['path']).write_bytes(b'changed during validation')
            return self.scored

        self.score_reader.side_effect = change_manifest
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.load(execution_archive=archive)


class EarlyFullReadoutTest(unittest.TestCase):
    def setUp(self):
        self.f = PatchFixture(self)
        self.models = {cell: self.f.build(cell) for cell in CELLS}
        self.output = self.f.root / 'readout.json'
        self.case_paths, self.case_data = {}, {}
        for suite in ('main', 'supplemental'):
            directory = self.f.root / ('cases_' + suite)
            directory.mkdir()
            cases, inputs, targets = [], [], []
            for role in ('original', 'replacement'):
                ids = [1, 2, 3] if role == 'original' else [4, 5, 6]
                if suite == 'supplemental':
                    ids.append(7)
                prefix = np.asarray([8, 9], dtype='<i4')
                sequence = np.full(CONFIG.context_length + 1, 15, dtype='<i4')
                sequence[:2] = prefix
                sequence[2:2+len(ids)] = ids
                cases.append(dict(case_index=len(cases), kind='word' if suite == 'main' else 'word_next_native',
                    split='test', context_id='shared-prefix', prefix_domain='original',
                    target='Exeunt' if role == 'original' else 'Nuveth', target_ids=ids,
                    spelling_variant='title', candidate_pair={'original': 'Exeunt', 'replacement': 'Nuveth'},
                    target_source_domain=role, scored_rows=list(range(1, 1+len(ids))),
                    prefix=dict(length=2, token_ids_sha256=hashlib.sha256(prefix.tobytes()).hexdigest()),
                    target_source=dict(native_piece_bytes_hex=['45' if role == 'original' else '4e'])))
                inputs.append(sequence[:-1])
                targets.append(sequence[1:])
            packed = np.asarray([inputs, targets], dtype='<i4')
            batch_path = directory / 'packed.bin'
            packed.tofile(batch_path)
            batch_record = localization.native._record(batch_path)
            plan = dict(format='pluto-paired-lowercase-word-cases-v1' if suite == 'main'
                        else 'pluto-paired-lowercase-supplemental-cases-v1',
                        cases=cases, case_count=len(cases), packed_batch=batch_record)
            if suite == 'supplemental':
                plan.update(source_word_cases=self.case_data['main']['record'],
                            source_word_packed_batch=self.case_data['main']['batch'])
            path = directory / 'cases.json'
            path.write_text(json.dumps(plan))
            self.case_paths[suite] = path
            self.case_data[suite] = dict(plan=plan, record=localization.native._record(path),
                                         batch=batch_record, packed=packed)
        self.scores = {cell: {suite: self.f.root / (cell + '_' + suite + '_scores')
                             for suite in self.case_paths} for cell in CELLS}
        self.executions = {cell: {suite: self.f.root / (cell + '_' + suite + '_execution.json')
                                 for suite in self.case_paths} for cell in CELLS}
        self.by_path = {model['paths']['patched']: cell for cell, model in self.models.items()}

        def load(model, cases_path, suite, directory, execution, config=CONFIG, *, case_archive=None):
            cell = self.by_path[model['paths']['patched']]
            data = self.case_data[suite]
            losses = np.full((len(data['plan']['cases']), CONFIG.context_length), 5.0, dtype='<f4')
            for case in data['plan']['cases']:
                rows = case['scored_rows']
                losses[case['case_index'], rows[:3]] = 100.0 - polynomial(CELLS[cell])
                if len(rows) == 4:
                    losses[case['case_index'], rows[3]] = 2.0
            records = [*model['records'], data['record'], data['batch']]
            if case_archive is not None:
                records.append(case_archive.manifest_record)
            return dict(cases=data, losses=losses, argmax=np.ones(losses.shape, dtype='<i4'),
                        records=records)

        self.native = self.enterContext(mock.patch.object(localization, 'load_native', side_effect=load))

    def analyze(self, **kwargs):
        return localization.analyze_cube(self.models, self.case_paths, self.scores,
            self.executions, self.output, rows=self.f.rows, config=CONFIG, layout=LAYOUT, **kwargs)

    def test_complete_readout_uses_real_patch_validation_and_new_format_with_all_eight_cells(self):
        result = self.analyze()
        self.assertEqual(result['format'], localization.EARLY_BRANCHES_FORMAT)
        self.assertEqual(result['format'], 'pluto-paired-early-branches-localization-v1')
        self.assertEqual(result['layout'], LAYOUT)
        self.assertEqual(result['cells'], {cell: list(subset) for cell, subset in CELLS.items()})
        self.assertEqual(set(result['groups']), {'A', 'M', 'R'})
        self.assertEqual(self.native.call_count, 16)
        self.assertEqual(len(result['per_case']), 4)
        self.assertFalse(result['model_forward_performed'])
        self.assertFalse(result['baseline_controls_certified'])
        self.assertFalse(result['goal_completion_claimed'])
        self.assertNotIn('case_source_archive', result)
        self.assertTrue(all(call.kwargs == {'config': CONFIG} for call in self.native.call_args_list))
        for case in result['per_case']:
            self.assertEqual(set(case['cells']), set(CELLS))
            effects = case['metrics']['word_three']['effects']
            self.assertEqual(effects['additions'], {'A': 6.0, 'M': 9.0, 'R': 15.0})
            self.assertEqual(effects['three_way'], 51.0)
        self.assertTrue(result['groups_readout'])
        localization.training.verify_records(result['files'])
        self.assertEqual(json.loads(self.output.read_text()), result)

    def test_readout_records_and_forwards_the_explicit_case_archive_without_execution_relabeling(self):
        manifest = self.f.root / 'case_archive.json'
        manifest.write_text(json.dumps({'fixture': 'case archive'}))
        archive = SimpleNamespace(manifest_record=localization.native._record(manifest))
        result = self.analyze(case_archive=archive)
        self.assertEqual(result['case_source_archive'], archive.manifest_record)
        self.assertIn(archive.manifest_record, result['files'])
        self.assertEqual(self.native.call_count, 16)
        self.assertTrue(all(call.kwargs == {'config': CONFIG, 'case_archive': archive}
                            for call in self.native.call_args_list))

    def test_missing_early_cube_cell_is_rejected_before_loading_native_results(self):
        del self.models['MR']
        with self.assertRaises(ValueError):
            self.analyze()
        self.native.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_changed_real_model_weights_prevent_successful_readout(self):
        path = Path(self.models['E']['paths']['patched']) / 'weight_1.bin'
        values = np.fromfile(path, dtype='<f4')
        values[0] += 1.0
        values.tofile(path)
        with self.assertRaises(ValueError):
            self.analyze()
        self.native.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
