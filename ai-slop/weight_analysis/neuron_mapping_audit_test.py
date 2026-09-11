"""Synthetic end-to-end checks of the completed-run numerical audit."""

import json
from pathlib import Path
import unittest

import numpy as np

from . import neuron_mapping_independent as independent
from . import neuron_mapping_report_test as fixture


class IndependentAuditTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.ReportFilesTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        batch_path = Path(self.fixture.run['batch_tokens_file'])
        batch_path.parent.mkdir()
        batch_path.write_bytes(self.fixture.batch.tobytes())
        checkpoint = self.fixture.base['checkpoint']
        directory = Path(checkpoint['checkpoint_directory'])
        directory.mkdir()
        checkpoint['weight_sha256'] = {}
        for name in ('weight_84.bin', 'weight_96.bin'):
            path = directory / name
            path.write_bytes(np.zeros((2048, 512), dtype='<f4').tobytes())
            checkpoint['weight_sha256'][name] = independent.identity(path)['sha256']
        self.report = self.fixture.report()

    def test_scalar_means_dense_projector_and_all_selections_agree(self):
        result = independent.audit(self.fixture.output)
        self.assertTrue(result['all_checks_passed'])
        self.assertEqual(result['max_raw_mean_delta_accuracy_absolute_error'], 0)
        self.assertLess(result['max_dense_eigh_projector_absolute_error'], 1e-12)
        self.assertEqual(result['checked_all_group_comparators'], 64)
        self.assertEqual(len(result['selected_checks']), 16)
        json.dumps(result, allow_nan=False)

    def test_changed_raw_measurement_rejected(self):
        path = Path(next(iter(self.report['raw_files'].values()))['path'])
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, 'measurement changed'):
            independent.audit(self.fixture.output)

    def test_changed_rank_rejected(self):
        entry = next(row for row in self.report['mapping']['selected']
                     if row['selected_group_index'] is not None)
        entry['confirmation_rank_all64']['rank'] += 1
        self.fixture.output.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'rank mismatch'):
            independent.audit(self.fixture.output)

    def test_changed_reported_mean_rejected(self):
        self.report['per_passage_mean_nll']['confirmation']['clean_before'][0] += .25
        self.fixture.output.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, 'numerical report mismatch'):
            independent.audit(self.fixture.output)


if __name__ == '__main__':
    unittest.main()
