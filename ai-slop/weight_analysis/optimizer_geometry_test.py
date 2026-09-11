"""Independent small numerical controls, never tests against the real corpus."""

import unittest
import numpy as np

from .optimizer_geometry import (invert_known_first_update, recover_from_first_moments,
                                 run_experiment, simulate_adamw, span_metrics)


class OptimizerGeometryTest(unittest.TestCase):
    def test_rank_two_gradient_becomes_full_rank(self):
        g=np.arange(6)[:,None]-np.arange(6)[None,:]+.25
        observed=g/(np.abs(g)+1e-8)
        metrics=span_metrics(g, observed)
        self.assertEqual(metrics['reference_rank'],2)
        self.assertEqual(metrics['observed_rank'],6)
        self.assertGreater(metrics['relative_energy_outside_reference_span'],.1)

    def test_rank_one_does_not_preserve_particular_span(self):
        x=np.array([1.,2.,4.,8.]); g=x[:,None]*np.array([-1.,2.,3.])
        result=span_metrics(g,np.sign(g))
        self.assertEqual(result['reference_rank'],1)
        self.assertEqual(result['observed_rank'],1)
        self.assertGreater(result['relative_energy_outside_reference_span'],.1)
        self.assertLess(span_metrics(g,g)['relative_energy_outside_reference_span'],1e-28)

    def test_zero_and_orthogonal_subspaces(self):
        x=np.eye(3)[:,:1]; y=np.eye(3)[:,1:]
        self.assertAlmostEqual(span_metrics(x,y)['relative_energy_outside_reference_span'],1.)
        self.assertTrue(span_metrics(x,np.zeros((3,2)))['observed_zero'])
        self.assertEqual(span_metrics(np.zeros((3,2)),y)['reference_rank'],0)

    def test_adam_scalar_steps_against_independent_loop(self):
        gs=np.array([.3,-.7,.2]); eta=.02; beta1=.7; beta2=.8; eps=.03; decay=.2
        result=simulate_adamw([[1.5]],gs[:,None,None],eta,beta1,beta2,eps,decay)
        m=v=0.; w=1.5
        for t,g in enumerate(gs,1):
            m=beta1*m+(1-beta1)*g; v=beta2*v+(1-beta2)*g*g
            u=(m/(1-beta1**t))/((v/(1-beta2**t))**.5+eps)
            w=w-eta*(u+decay*w)
            self.assertAlmostEqual(result['first'][t,0,0],m)
            self.assertAlmostEqual(result['second'][t,0,0],v)
            self.assertAlmostEqual(result['weights'][t,0,0],w)

    def test_recover_gradients_only_given_moments(self):
        rng=np.random.default_rng(19); gs=rng.normal(size=(4,3,2))
        result=simulate_adamw(np.zeros((3,2)),gs)
        for i in range(4):
            np.testing.assert_allclose(recover_from_first_moments(result['first'][i],result['first'][i+1],.9),gs[i],atol=1e-15)

    def test_conditional_first_step_inverse_and_ill_conditioning(self):
        g=np.array([[-2.,-.1,0.,.7,4.]])
        eps=.05
        np.testing.assert_allclose(invert_known_first_update(g/(abs(g)+eps),eps),g,rtol=1e-14,atol=1e-15)
        for u in ([[1.]], [[-1.]], [[1.1]], [[float('nan')]]):
            with self.assertRaises(ValueError): invert_known_first_update(u,eps)

    def test_full_control_experiment(self):
        r=run_experiment(); t=r['ten_steps_fixed_input_span']
        self.assertEqual(t['sgd_sum']['observed_rank'],2)
        self.assertEqual(t['known_final_first_moment']['observed_rank'],2)
        self.assertEqual(t['adam_endpoint']['observed_rank'],6)
        self.assertLess(t['endpoint_weighted_update_max_error'],1e-10)
        self.assertLess(t['recover_each_gradient_with_saved_moments_max_error'],1e-13)
        self.assertLess(t['pure_decay_corrected_residual_max_abs'],1e-10)
        self.assertTrue(r['fp32_scalar_collision']['same_bits'])
        self.assertEqual(r['fp32_scalar_collision']['weight_bits_hex'],['0x3f7fea60']*2)
        self.assertGreater(r['known_first_step_ideal_inverse']['maximum_inverse_absolute_derivative'],1e8)

    def test_invalid_inputs(self):
        for matrix in ([],[1,2],[[np.inf]]):
            with self.assertRaises(ValueError): span_metrics(matrix,[[1.]])
        with self.assertRaises(ValueError): span_metrics([[1.]],[[1.],[2.]])
        with self.assertRaises(ValueError): simulate_adamw([[1.]],np.zeros((0,1,1)))
        with self.assertRaises(ValueError): simulate_adamw([[1.]],np.zeros((1,2,1)))
        for option,value in [('beta1',1),('beta2',-1),('epsilon',0),('learning_rate',0),('weight_decay',-1)]:
            with self.assertRaises(ValueError): simulate_adamw([[1.]],[[[0.]]],**{option:value})
        with self.assertRaises(ValueError): recover_from_first_moments([[1.]],[[2.,3.]],.9)


if __name__=='__main__':
    unittest.main()
