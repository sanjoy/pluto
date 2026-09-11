"""CPU-only exact-byte checks for LN2 versus early MLP parameter patches.

Only temporary, tiny checkpoints are constructed. No native executable,
training process, real checkpoint, or GPU is used by these tests.
"""

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import paired_complement_localization as core
from . import paired_mlp_affine as affine
from .checkpoint import GPT2Checkpoint, GPT2Config, sha256_file, tensor_manifest


CONFIG = GPT2Config(vocab_size=16, padded_vocab_size=20, context_length=8,
                    n_layers=2, d_model=4, n_heads=2, d_ff=8)
PRODUCTION_ROWS = [45, 68, 303, 400, 409, 1475, 2797, 3109, 14364, 21733, 45177]
SUBSETS = {'E': [], 'N': ['N'], 'F': ['F'], 'M': ['N', 'F']}


class SelectionTest(unittest.TestCase):
    def test_production_groups_have_exact_tensor_indices_and_parameter_counts(self):
        selections = affine.selections()
        specs = {spec.name: spec for spec in tensor_manifest()}
        indices = {
            'E': set(),
            'N': {offset + 12*block for block in range(4) for offset in (8, 9)},
            'F': {offset + 12*block for block in range(4) for offset in (10, 11, 12, 13)},
        }
        indices['M'] = indices['N'] | indices['F']
        self.assertEqual(set(selections), set(SUBSETS))
        for cell, expected in indices.items():
            self.assertEqual(selections[cell], sorted(selections[cell]))
            self.assertEqual(len(selections[cell]), len(set(selections[cell])))
            self.assertEqual({specs[name].index for name in selections[cell]}, expected)
        self.assertEqual({cell: len(names) for cell, names in selections.items()},
                         {'E': 0, 'N': 8, 'F': 16, 'M': 24})
        self.assertEqual({cell: sum(specs[name].nbytes // 4 for name in names)
                          for cell, names in selections.items()},
                         {'E': 0, 'N': 4096, 'F': 8398848, 'M': 8402944})
        self.assertFalse(set(selections['N']) & set(selections['F']))
        self.assertEqual(selections['M'], sorted(core.groups(layout='early_branches')['M']))

    def test_tiny_even_layout_uses_early_half_not_fixed_production_indices(self):
        for layers in (2, 4):
            config = replace(CONFIG, n_layers=layers)
            selections = affine.selections(config)
            expected_n = {f'blocks.{block}.ln2.{field}'
                          for block in range(layers // 2) for field in ('scale', 'bias')}
            expected_f = {f'blocks.{block}.mlp.{projection}.{field}'
                          for block in range(layers // 2)
                          for projection in ('input', 'output') for field in ('weight', 'bias')}
            self.assertEqual(selections, {'E': [], 'N': sorted(expected_n),
                'F': sorted(expected_f), 'M': sorted(expected_n | expected_f)})
        for layers in (1, 3, 5):
            with self.subTest(layers=layers), self.assertRaises(ValueError):
                affine.selections(replace(CONFIG, n_layers=layers))

    def test_production_row_contract_accepts_only_the_fixed_canonical_membership(self):
        tensors, rows = affine._selection('N', list(reversed(PRODUCTION_ROWS)), GPT2Config())
        self.assertEqual(tensors, affine.selections()['N'])
        self.assertEqual(rows, PRODUCTION_ROWS)

    def test_manifest_matrix_shape_dtype_or_index_forgery_is_rejected(self):
        original = tensor_manifest(CONFIG)
        for changes in ({'shape': (CONFIG.d_ff, CONFIG.d_model)},
                        {'dtype': '>f4'}, {'index': 11}):
            forged = tuple(replace(spec, **changes) if spec.name == 'blocks.0.mlp.input.weight'
                           else spec for spec in original)
            with self.subTest(changes=changes), mock.patch.object(affine, 'tensor_manifest', return_value=forged):
                with self.assertRaises(ValueError):
                    affine.selections(CONFIG)


class PatchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mlp-affine-cpu-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.a, self.d = self.root/'recipient', self.root/'donor'
        self.output = self.root/'patched'
        self.rows = [1, 5, 15]
        self.specs = tensor_manifest(CONFIG)
        self.by_name = {spec.name: spec for spec in self.specs}
        for path, shift in ((self.a, 0), (self.d, 10000)):
            path.mkdir()
            for spec in self.specs:
                values = (np.arange(np.prod(spec.shape), dtype='<f4') +
                          np.float32(100*spec.index + shift)).reshape(spec.shape)
                if spec.index == 0:
                    values[-1, -1] = np.float32(-0.0 if path == self.a else 0.0)
                    if path == self.d:
                        values[self.rows[0], 0] = np.float32(-0.0)
                values.tofile(path/spec.filename)
        self.original_hashes = {str(path): self.hashes(path) for path in (self.a, self.d)}

    def hashes(self, path):
        return {spec.filename: sha256_file(path/spec.filename) for spec in self.specs}

    def build(self, cell='N', *, rows=None, output=None, a=None, d=None, config=CONFIG):
        return affine.build_model(self.a if a is None else a, self.d if d is None else d,
            self.output if output is None else output, cell,
            self.rows if rows is None else rows, config=config)

    def validate(self, cell='N', *, rows=None, output=None, a=None, d=None, patch_path=None):
        output = self.output if output is None else output
        return affine.validate_model(output/'patch.json' if patch_path is None else patch_path,
            self.a if a is None else a, self.d if d is None else d, cell,
            self.rows if rows is None else rows, config=CONFIG)

    def metadata(self):
        return json.loads((self.output/'patch.json').read_text())

    def publish_metadata(self, value):
        (self.output/'patch.json').write_text(json.dumps(value))

    def change_weight_and_forge_hash(self, name, change, *, source=None):
        directory = self.output if source is None else source
        path = directory/self.by_name[name].filename
        values = np.fromfile(path, dtype='<f4').reshape(self.by_name[name].shape)
        change(values)
        values.tofile(path)
        metadata = self.metadata()
        if source is None:
            hashes = metadata['output']['weights_sha256']
        else:
            hashes = metadata['sources']['original' if source == self.a else 'replacement']['weights_sha256']
        hashes[path.name] = sha256_file(path)
        self.publish_metadata(metadata)

    def test_all_four_cells_have_exact_selected_unselected_and_padded_bytes(self):
        recipient, donor = GPT2Checkpoint(self.a, CONFIG), GPT2Checkpoint(self.d, CONFIG)
        embeddings = []
        for cell in SUBSETS:
            with self.subTest(cell=cell):
                output = self.root/cell
                model = self.build(cell, output=output)
                self.assertEqual(model, self.validate(cell, output=output))
                self.assertEqual(model['subset'], SUBSETS[cell])
                self.assertEqual(model['selection'], affine.selections(CONFIG)[cell])
                self.assertEqual(set(model), {'paths', 'hashes', 'weight_records', 'patch',
                                              'subset', 'selection', 'records'})
                actual = GPT2Checkpoint(output, CONFIG, check_finite=True)
                for spec in self.specs:
                    expected = np.array(donor[spec.name] if spec.name in model['selection']
                                        else recipient[spec.name])
                    if spec.index == 0:
                        expected[self.rows] = donor[spec.name][self.rows]
                    self.assertEqual(actual[spec.name].tobytes(), expected.tobytes(), spec.name)
                    destination = (output/spec.filename).stat()
                    self.assertEqual(destination.st_nlink, 1)
                    for source in (self.a, self.d):
                        origin = (source/spec.filename).stat()
                        self.assertNotEqual((destination.st_dev, destination.st_ino),
                                            (origin.st_dev, origin.st_ino))
                embedding = actual['token_embedding.weight']
                self.assertTrue(np.signbit(embedding[-1, -1]))
                self.assertTrue(np.signbit(embedding[self.rows[0], 0]))
                self.assertEqual(embedding[CONFIG.vocab_size:].tobytes(),
                                 recipient['token_embedding.weight'][CONFIG.vocab_size:].tobytes())
                embeddings.append(embedding.tobytes())
        self.assertTrue(all(value == embeddings[0] for value in embeddings))
        for source in (self.a, self.d):
            self.assertEqual(self.hashes(source), self.original_hashes[str(source)])

    def test_e_and_m_are_byte_identical_to_existing_early_cube_anchors(self):
        for cell, subset in (('E', ()), ('M', ('M',))):
            with self.subTest(cell=cell):
                model = self.build(cell, output=self.root/('new_'+cell))
                previous = core.build_model(self.a, self.d, self.root/('early_'+cell),
                    subset, self.rows, config=CONFIG, layout='early_branches')
                self.assertEqual(model['hashes'], previous['hashes'])
                for spec in self.specs:
                    self.assertEqual((Path(model['paths']['patched'])/spec.filename).read_bytes(),
                                     (Path(previous['paths']['patched'])/spec.filename).read_bytes())

    def test_reordered_tiny_rows_are_canonicalized_without_changing_membership(self):
        model = self.build('F', rows=list(reversed(self.rows)))
        self.assertEqual(self.metadata()['selection']['embedding_rows'], self.rows)
        self.assertEqual(model, self.validate('F', rows=list(reversed(self.rows))))

    def test_invalid_cells_and_rows_fail_before_creating_output(self):
        for cell in ('A', 'R', 'NF', '', None, ['N'], True):
            with self.subTest(cell=cell), self.assertRaises(ValueError):
                self.build(cell)
            self.assertFalse(self.output.exists())
        for rows in ([], [True], [1.0], [1, 1], [-1], [CONFIG.vocab_size], [20], '1'):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.build('N', rows=rows)
            self.assertFalse(self.output.exists())

    def test_production_configuration_rejects_any_other_logical_row_set(self):
        for rows in (self.rows, PRODUCTION_ROWS[:-1], PRODUCTION_ROWS + [2],
                     [*PRODUCTION_ROWS[:-1], 45178]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, 'eleven frozen E rows'):
                self.build('N', rows=rows, config=GPT2Config())
            self.assertFalse(self.output.exists())

    def test_wrong_cell_or_forged_tensor_selection_cannot_relabel_existing_bytes(self):
        self.build('N')
        with self.assertRaises(ValueError):
            self.validate('F')
        metadata = self.metadata()
        metadata['selection']['tensors'] = affine.selections(CONFIG)['F']
        self.publish_metadata(metadata)
        with self.assertRaises(ValueError):
            self.validate('F')

    def test_forged_metadata_identity_geometry_selection_and_validation_fail(self):
        self.build('M')
        original = self.metadata()
        changes = {
            'format': lambda m: m.update(format='different-patch'),
            'complete': lambda m: m.update(complete=False),
            'geometry': lambda m: m['config'].update(n_heads=1),
            'source': lambda m: m['sources']['original'].update(path=str(self.d)),
            'output': lambda m: m['output'].update(path=str(self.root/'other')),
            'rows': lambda m: m['selection'].update(embedding_rows=[1, 5]),
            'ranges': lambda m: m['selection']['embedding_row_byte_ranges'][0].__setitem__(0, 0),
            'tied': lambda m: m['selection'].update(embedding_is_tied_to_lm_head=False),
            'validation': lambda m: m['validation'].update(selected_bytes_equal_replacement=False),
            'count': lambda m: m['output']['weights_sha256'].update(weight_999_bin='0'*64),
        }
        for label, change in changes.items():
            metadata = copy.deepcopy(original)
            change(metadata)
            self.publish_metadata(metadata)
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.validate('M')

    def test_current_patcher_path_hash_and_loader_hash_are_all_required(self):
        self.build('F')
        original = self.metadata()
        source = Path(original['implementation']['path'])
        clone = self.root/'same_bytes_different_producer.py'
        clone.write_bytes(source.read_bytes())
        for field, value in (('path', str(clone)), ('sha256', '0'*64),
                             ('checkpoint_loader_sha256', '0'*64)):
            metadata = copy.deepcopy(original)
            metadata['implementation'][field] = value
            self.publish_metadata(metadata)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate('F')

    def test_selected_matrix_corruption_fails_even_with_matching_forged_hash(self):
        self.build('F')
        self.change_weight_and_forge_hash('blocks.0.mlp.input.weight',
            lambda values: values.__setitem__((0, 1), values[0, 1] + np.float32(1)))
        with self.assertRaises(ValueError):
            self.validate('F')

    def test_unselected_late_matrix_corruption_fails_with_matching_forged_hash(self):
        self.build('M')
        self.change_weight_and_forge_hash('blocks.1.mlp.output.weight',
            lambda values: values.__setitem__((0, 0), values[0, 0] + np.float32(1)))
        with self.assertRaises(ValueError):
            self.validate('M')

    def test_padding_signed_zero_change_fails_even_with_matching_forged_hash(self):
        self.build('E')
        self.change_weight_and_forge_hash('token_embedding.weight',
            lambda values: values.__setitem__((-1, -1), np.float32(0.0)))
        with self.assertRaises(ValueError):
            self.validate('E')

    def test_selected_embedding_signed_zero_change_fails_with_forged_hash(self):
        self.build('N')
        self.change_weight_and_forge_hash('token_embedding.weight',
            lambda values: values.__setitem__((self.rows[0], 0), np.float32(0.0)))
        with self.assertRaises(ValueError):
            self.validate('N')

    def test_nonfinite_values_fail_even_with_matching_forged_hash(self):
        self.build('F')
        self.change_weight_and_forge_hash('blocks.0.mlp.output.bias',
            lambda values: values.__setitem__(0, np.float32(np.nan)))
        with self.assertRaises(ValueError):
            self.validate('F')

    def test_changed_source_bytes_cannot_be_hidden_by_forged_source_hash(self):
        self.build('F')
        self.change_weight_and_forge_hash('blocks.0.mlp.input.weight',
            lambda values: values.__setitem__((0, 0), values[0, 0] + np.float32(1)), source=self.d)
        with self.assertRaises(ValueError):
            self.validate('F')

    def test_missing_or_extra_weight_files_and_wrong_sizes_fail(self):
        self.build('N')
        weight = self.output/'weight_1.bin'
        original = weight.read_bytes()
        weight.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.validate('N')
        weight.write_bytes(original[:-4])
        with self.assertRaises(ValueError):
            self.validate('N')
        weight.write_bytes(original)
        (self.output/'weight_999.bin').write_bytes(b'\0'*4)
        with self.assertRaises(ValueError):
            self.validate('N')

    def test_source_directory_and_ancestor_symlinks_are_rejected(self):
        alias = self.root/'recipient_link'
        alias.symlink_to(self.a, target_is_directory=True)
        ancestor = self.root/'ancestor_link'
        ancestor.symlink_to(self.root, target_is_directory=True)
        for source in (alias, ancestor/'recipient'):
            with self.subTest(source=str(source)), self.assertRaises(ValueError):
                self.build(a=source)
            self.assertFalse(self.output.exists())

    def test_source_weight_symlink_is_rejected(self):
        weight = self.a/'weight_1.bin'
        copy_path = self.root/'original_weight_1.bin'
        copy_path.write_bytes(weight.read_bytes())
        weight.unlink()
        weight.symlink_to(copy_path)
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def test_output_ancestor_symlink_or_source_descendant_is_rejected(self):
        alias = self.root/'output_parent_link'
        alias.symlink_to(self.root, target_is_directory=True)
        for output in (alias/'new_output', self.a/'new_output', self.d/'new_output'):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                self.build(output=output)
            self.assertFalse(output.exists())

    def test_existing_output_is_not_overwritten(self):
        self.build()
        before = self.hashes(self.output)
        metadata = (self.output/'patch.json').read_bytes()
        with self.assertRaises((ValueError, FileExistsError)):
            self.build('F')
        self.assertEqual(self.hashes(self.output), before)
        self.assertEqual((self.output/'patch.json').read_bytes(), metadata)

    def test_validation_rejects_symlinked_patch_or_weight(self):
        self.build('E')
        alias = self.root/'patch_link.json'
        alias.symlink_to(self.output/'patch.json')
        with self.assertRaises(ValueError):
            self.validate('E', patch_path=alias)
        weight = self.output/'weight_1.bin'
        weight.unlink()
        weight.symlink_to(self.a/weight.name)
        with self.assertRaises(ValueError):
            self.validate('E')

    def test_same_bytes_hardlinked_output_is_not_an_independent_copy(self):
        self.build('E')
        weight = self.output/'weight_1.bin'
        weight.unlink()
        os.link(self.a/weight.name, weight)
        with self.assertRaises(ValueError):
            self.validate('E')


if __name__ == '__main__':
    unittest.main()
