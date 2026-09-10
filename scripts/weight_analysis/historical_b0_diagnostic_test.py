"""Numerical helpers only; synthetic tests do not substitute for raw evidence."""

import math
import unittest
import numpy as np

from .historical_b0_diagnostic import distribution, compare_logits, residual_metrics


class HistoricalB0DiagnosticTest(unittest.TestCase):
    def test_uniform_and_shift_invariance(self):
        p,logp,entropy=distribution([7.,7.,7.,7.])
        np.testing.assert_array_equal(p,[.25]*4)
        np.testing.assert_array_equal(logp,[-math.log(4)]*4)
        self.assertEqual(entropy,math.log(4))
        np.testing.assert_array_equal(distribution([0.,1.,2.])[0],distribution([100.,101.,102.])[0])

    def test_target_loss_rank_and_degradation(self):
        result=compare_logits([0.,2.,2.],[0.,0.,0.],2)
        self.assertEqual(result['clean']['target_rank'],2)
        self.assertEqual(result['ablated']['target_rank'],3)
        self.assertLess(result['delta_log_probability'],0)
        self.assertGreater(result['ablated']['entropy_nats'],result['clean']['entropy_nats'])
        self.assertAlmostEqual(result['clean']['target_probability'],math.exp(2)/(1+2*math.exp(2)))

    def test_extreme_logits_are_stable(self):
        p,logp,entropy=distribution([10000.,-10000.])
        np.testing.assert_array_equal(p,[1.,0.])
        np.testing.assert_array_equal(logp,[0.,-20000.])
        self.assertEqual(entropy,0)

    def test_residual_norms_rounding_and_undefined_ratios(self):
        result=residual_metrics([1.,-1.],[10.,-10.],[11.25,-11.])
        self.assertEqual(result['write_to_before_l2_ratio'],10)
        self.assertEqual(result['centered_write_to_before_ratio'],10)
        self.assertEqual(result['residual_add_rounding_l2'],.25)
        self.assertIsNone(residual_metrics([0.,0.],[1.,2.],[1.,2.])['write_to_before_l2_ratio'])
        self.assertIsNone(residual_metrics([1.,1.],[1.,2.],[2.,3.])['centered_write_to_before_ratio'])

    def test_bad_inputs(self):
        for values in ([1.], [1.,np.nan], [1.,np.inf], [[1.,2.]]):
            with self.assertRaises(ValueError): distribution(values)
        for args in (([1.,2.],[1.,2.],True),([1.,2.],[1.,2.],2),([1.,2.],[1.,2.,3.],1)):
            with self.assertRaises(ValueError): compare_logits(*args)
        for args in (([],[],[]),([1.,2.],[1.],[1.,2.]),([1.,np.nan],[1.,2.],[1.,2.])):
            with self.assertRaises(ValueError): residual_metrics(*args)


if __name__=='__main__': unittest.main()
