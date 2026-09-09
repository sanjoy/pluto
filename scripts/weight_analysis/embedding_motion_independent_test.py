"""Dense vocabulary-space oracles independent of the small-Gram formulas."""

import unittest
import numpy as np

from .embedding_motion import centered_motion


class DenseMotionOracleTest(unittest.TestCase):
    def test_dense_all_three_motion_definitions(self):
        rng=np.random.default_rng(927)
        before=rng.normal(size=(19,4))
        after=1.13*before+rng.normal(size=(19,4))*.3
        measured=centered_motion(before,after)
        a=before-before.mean(axis=0); b=after-after.mean(axis=0)
        # Reversed covariance gives the transpose of the alignment rotation.
        u,_,vt=np.linalg.svd(a.T@b)
        aligned=b@(u@vt).T
        scale=np.sum(a*aligned)/np.sum(a*a)
        for mode,delta,q in [('raw',b-a,1.),('procrustes',aligned-a,1.),
                             ('rotation_and_fitted_scale',aligned-scale*a,scale)]:
            actual=measured[mode]
            score=a@delta.T
            gram=(q*a+delta)@(q*a+delta).T-(q*a)@(q*a).T
            matrices={'score_frobenius':score,'score_symmetric_frobenius':(score+score.T)/2,
                      'score_antisymmetric_frobenius':(score-score.T)/2,'gram_change_frobenius':gram}
            for name,matrix in matrices.items():
                expected=float(np.sum(matrix*matrix))
                observed=actual[name]['computed_signed_squared_norm']
                self.assertLessEqual(abs(expected-observed),actual[name]['squared_roundoff_allowance']+1e-12)
                self.assertAlmostEqual(actual[name]['squared_norm'],expected,places=9)
            self.assertAlmostEqual(actual['delta_frobenius_norm'],np.linalg.norm(delta),places=12)

    def test_rotation_can_be_directed_without_gram_change(self):
        rng=np.random.default_rng(13); a=rng.normal(size=(9,2))
        t=.7; rotation=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        result=centered_motion(a,a@rotation)
        self.assertGreater(result['raw']['score_antisymmetric_frobenius']['norm'],1)
        self.assertFalse(result['raw']['gram_change_frobenius']['squared_norm_above_roundoff'])
        self.assertLess(result['procrustes']['delta_frobenius_norm'],1e-12)
        self.assertAlmostEqual(result['fitted_scale'],1.,places=12)

    def test_procrustes_does_not_remove_all_token_directionality(self):
        rng=np.random.default_rng(4); values=rng.normal(size=(11,4))
        values-=values.mean(axis=0)
        orthogonal,_=np.linalg.qr(values)
        a=orthogonal[:,:2]; b=a+.25*orthogonal[:,2:4]
        result=centered_motion(a,b)['procrustes']
        self.assertLess(result['coordinate_cross_skew_norm'],1e-12)
        self.assertGreater(result['score_antisymmetric_frobenius']['norm'],.1)
        self.assertAlmostEqual(result['score_antisymmetric_frobenius']['squared_norm'],.0625,places=12)

    def test_common_rotation_and_independent_translations_preserve_metrics(self):
        rng=np.random.default_rng(8)
        a=rng.normal(size=(21,3)); b=a+.2*rng.normal(size=(21,3))
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        first=centered_motion(a,b)
        changed=centered_motion(a@rotation+[7.,-3.,2.],b@rotation+[-4.,5.,1.])
        for mode in ('raw','procrustes','rotation_and_fitted_scale'):
            for metric in ('score_frobenius','score_symmetric_frobenius','score_antisymmetric_frobenius','gram_change_frobenius'):
                np.testing.assert_allclose(first[mode][metric]['squared_norm'],changed[mode][metric]['squared_norm'],rtol=1e-11,atol=1e-12)


if __name__=='__main__':
    unittest.main()
