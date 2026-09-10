"""Small synthetic oracles; no model/GPU execution or measured research claim."""

import unittest

import numpy as np

from . import head_position_math as h


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.original = np.full((8, 6), 0x3f80, dtype='<u2')
        self.options = dict(context_length=4, sequence=1, query=2, head=1, head_dim=2)

    def test_positions_partition_one_sequence(self):
        self.assertEqual(h.selected_rows(8, 4, 1, 2, 'query_only').tolist(), [6])
        self.assertEqual(h.selected_rows(8, 4, 1, 2, 'other_queries').tolist(), [4, 5, 7])
        self.assertEqual(h.selected_rows(8, 4, 1, 2, 'all_queries').tolist(), [4, 5, 6, 7])

    def test_exact_head_scope_and_all_supported_doses(self):
        for scope in h.SCOPES:
            for dose, bits in ((0, 0), (.5, 0x3f00), (1, 0x3f80)):
                changed = self.original.copy()
                rows = h.selected_rows(8, 4, 1, 2, scope)
                changed[rows, 2:4] = bits
                report = h.audit_context(self.original, changed, scope=scope, scale=dose, **self.options)
                self.assertEqual(report['selected_element_count'], len(rows) * 2)
                self.assertEqual(report['changed_element_count'], 0 if dose == 1 else len(rows) * 2)

    def test_other_sequence_head_query_or_incorrect_dose_rejected(self):
        for position in ((2, 2), (6, 0), (5, 2), (6, 2)):
            changed = self.original.copy()
            changed[6, 2:4] = 0x3f00
            changed[position] ^= 1
            with self.assertRaisesRegex(ValueError, 'exact position/head dose'):
                h.audit_context(self.original, changed, scope='query_only', scale=.5, **self.options)

    def test_identity_preserves_negative_zero_and_zero_erases_it(self):
        self.original[6, 2:4] = [0x8000, 0x8003]
        h.audit_context(self.original, self.original.copy(), scope='query_only', scale=1, **self.options)
        changed = self.original.copy(); changed[6, 2:4] = [0x8000, 0x8002]
        h.audit_context(self.original, changed, scope='query_only', scale=.5, **self.options)
        changed[6, 2:4] = 0
        h.audit_context(self.original, changed, scope='query_only', scale=0, **self.options)

    def test_context_one_has_empty_other_scope(self):
        report = h.audit_context(self.original, self.original.copy(), context_length=1,
            sequence=3, query=0, head=0, head_dim=2, scope='other_queries', scale=0)
        self.assertEqual(report['selected_element_count'], 0)

    def test_bad_selection_and_nonfinite_outside_site_rejected(self):
        for args in ((8, 3, 0, 0, 'all_queries'), (8, 4, 2, 0, 'query_only'),
                     (8, 4, 0, -1, 'query_only'), (8, 4, 0, True, 'query_only'),
                     (0, 4, 0, 0, 'all_queries'), (8, 4, 0, 0, 'unknown')):
            with self.assertRaises(ValueError): h.selected_rows(*args)
        bad = self.original.copy(); bad[0, 0] = 0x7f80
        with self.assertRaises(ValueError):
            h.audit_context(self.original, bad, scope='query_only', scale=1, **self.options)
        for update in ({'head_dim': 0}, {'head': True}, {'head_dim': 4}):
            with self.assertRaises(ValueError):
                h.audit_context(self.original, self.original, scope='query_only', scale=1,
                                **{**self.options, **update})


class CausalLogitsTest(unittest.TestCase):
    def test_query_only_preserves_other_sequence_and_prior_rows(self):
        clean = np.zeros((8, 5), dtype='<f4'); changed = clean.copy(); changed[6:] = 1
        report = h.audit_causal_logits(clean, changed, context_length=4, sequence=1,
                                      query=2, scope='query_only', scale=0)
        self.assertEqual(report['unchanged_row_count'], 6)
        for row in range(6):
            bad = changed.copy(); bad[row, 4] = 1
            with self.assertRaises(ValueError):
                h.audit_causal_logits(clean, bad, context_length=4, sequence=1,
                                      query=2, scope='query_only', scale=0)

    def test_other_queries_cannot_affect_query_zero(self):
        clean = np.zeros((4, 2), dtype='<f4'); changed = clean.copy(); changed[1:] = 1
        h.audit_causal_logits(clean, changed, context_length=4, sequence=0,
                              query=0, scope='other_queries', scale=.5)
        changed[0, 0] = 1
        with self.assertRaises(ValueError):
            h.audit_causal_logits(clean, changed, context_length=4, sequence=0,
                                  query=0, scope='other_queries', scale=.5)

    def test_identity_compares_all_physical_columns_and_signed_zero_bits(self):
        clean = np.zeros((4, 5), dtype='<f4'); changed = clean.copy()
        changed[3, 4] = -0.0
        with self.assertRaises(ValueError):
            h.audit_causal_logits(clean, changed, context_length=4, sequence=0,
                                  query=3, scope='all_queries', scale=1)

    def test_nonfinite_wrong_shape_dtype_or_dose_rejected(self):
        clean = np.zeros((4, 3), dtype='<f4')
        for bad in (np.full_like(clean, np.nan), np.zeros((4, 2), dtype='<f4'), clean.astype('<f8')):
            with self.assertRaises(ValueError):
                h.audit_causal_logits(clean, bad, context_length=4, sequence=0,
                                      query=3, scope='all_queries', scale=1)
        with self.assertRaises(ValueError):
            h.audit_causal_logits(clean, clean, context_length=4, sequence=0,
                                  query=3, scope='all_queries', scale=True)


class EffectsTest(unittest.TestCase):
    def test_fixed_rival_and_opposing_local_whole_effects(self):
        report = h.partition_effects(dict(clean=[1, 2, 0], query_only=[3, 2, 0],
            other_queries=[-2, 2, 0], all_queries=[-3, 2, 10]), 0)
        self.assertEqual(report['fixed_rival_id'], 1)
        self.assertEqual(report['scores']['all_queries']['argmax'], 2)
        self.assertEqual(report['interaction']['delta_fixed_rival_margin'], -3)
        self.assertTrue(report['query_and_all_margin_effects_have_opposite_signs'])
        for effect in report['effects'].values():
            self.assertEqual(effect['delta_nll'], -effect['delta_log_probability'])

    def test_additive_logits_do_not_imply_additive_log_probabilities(self):
        report = h.partition_effects(dict(clean=[0, 0, 0], query_only=[1, 0, 0],
            other_queries=[2, 0, 0], all_queries=[3, 0, 0]), 0)
        self.assertEqual(report['interaction']['delta_fixed_rival_margin'], 0)
        self.assertNotAlmostEqual(report['interaction']['delta_log_probability'], 0)

    def test_full_identity_has_zero_effects(self):
        report = h.partition_effects({key: [1000, 999, 998] for key in ('clean', *h.SCOPES)}, 0)
        self.assertFalse(report['query_and_all_margin_effects_have_opposite_signs'])
        self.assertTrue(all(value == 0 for effect in report['effects'].values() for value in effect.values()))

    def test_missing_extra_mismatched_or_nonfinite_rows_rejected(self):
        rows = {key: [0, 0] for key in ('clean', *h.SCOPES)}
        for changed in ({key: value for key, value in rows.items() if key != 'other_queries'},
                        {**rows, 'extra': [0, 0]}, {**rows, 'query_only': [0, 0, 0]},
                        {**rows, 'other_queries': [np.nan, 0]}):
            with self.assertRaises(ValueError): h.partition_effects(changed, 0)


if __name__ == '__main__':
    unittest.main()
