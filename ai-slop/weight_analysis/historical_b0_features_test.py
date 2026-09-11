"""Small arithmetic/control fixtures, not a claim of new GPU measurements."""
import unittest
import numpy as np
from .historical_b0_features import activation_ranking,support_summary


class HistoricalB0FeaturesTest(unittest.TestCase):
    def test_signed_mass_cancellation_and_coverage(self):
        s=support_summary([4.,2.,1.,1.,-7.,0.])
        self.assertEqual((s['positive_count'],s['negative_count'],s['zero_count']),(4,1,1))
        self.assertEqual((s['positive_mass'],s['negative_mass'],s['net']),(8.,-7.,1.))
        self.assertEqual(s['positive_mass_fractions']['1'],.5)
        self.assertEqual(s['positive_neurons_to_reach'],{'0.5':1,'0.9':4})
        self.assertEqual(s['top_positive'][0]['neuron'],0)

    def test_no_positive_mass(self):
        s=support_summary([-1.,-2.,0.])
        self.assertIsNone(s['positive_mass_fractions']['1'])
        self.assertIsNone(s['positive_neurons_to_reach']['0.9'])
        self.assertEqual(s['top_positive'],[])

    def test_activation_ranks_ties_and_missing_same_token_controls(self):
        g=np.array([[3.,1.],[1.,2.],[2.,2.],[2.,3.]])
        a=activation_ranking(g,[10,11,12,13],3,0)
        self.assertEqual(a['rank_by_strictly_greater'],2)
        self.assertEqual(a['same_current_token_other_positions'],[])
        self.assertEqual([r['position'] for r in a['largest_activations']],[0,2,3,1])
        a=activation_ranking(g,[13,11,12,13],3,0)
        self.assertEqual(a['same_current_token_other_positions'],[0])

    def test_invalid_inputs(self):
        for terms in ([],[[1.]], [1.,np.nan]):
            with self.assertRaises(ValueError):support_summary(terms)
        for args in (([[1.]],[1],1,0),([[1.]],[1],0,1),([[np.inf]],[1],0,0),([[1.]],[1,2],0,0)):
            with self.assertRaises(ValueError):activation_ranking(*args)


if __name__=='__main__':unittest.main()
