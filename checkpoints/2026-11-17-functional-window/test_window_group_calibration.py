"""Reference checks for finite-sample group correction and selective risk."""

import itertools
import math
import unittest

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

from window_group_calibration import (
    evaluate_decisions,
    evaluate_interval,
    evaluate_upper,
    fit_group_interval,
    fit_group_upper,
    fit_selective_group,
    take_actions,
)


class GroupCalibrationTests(unittest.TestCase):
    def test_rank_by_enumerated_holdout(self):
        # Under an exchangeable permutation of these groups, a single group's
        # held position is uniform. Every target in that mother must be covered.
        margins = np.array([[s, s-2, s-5] for s in range(10)],dtype=float)
        hits = []
        for index in range(10):
            calibration = margins[np.arange(10)!=index]
            fit = fit_group_upper(calibration,np.zeros_like(calibration),alpha=.1)
            self.assertEqual(fit.rank,9)
            self.assertEqual(fit.n_groups,9)
            hits.append(np.all(margins[index]<=fit.apply(np.zeros((1,3)))[0]))
        self.assertEqual(sum(hits),9)

    def test_rows_repeat_without_increasing_group_count(self):
        residual = np.arange(18,dtype=float).reshape(9,2)
        source = fit_group_upper(residual,np.zeros_like(residual))
        expanded = np.repeat(residual,500,axis=1)
        repeated = fit_group_upper(expanded,np.zeros_like(expanded))
        self.assertEqual(source,repeated)
        self.assertEqual(source.radius,17)

    def test_two_sided_scaled_band_and_explicit_mask(self):
        # Endpoints in different units must receive proportional corrections.
        y = np.tile([3.,-6.,np.nan],(9,1))
        nominal = np.tile([1.,2.,np.nan],(9,1))
        valid = np.tile([True,True,False],(9,1))
        fit = fit_group_interval(y,-nominal,nominal,scale=[1.,2.,np.nan],valid=valid)
        self.assertAlmostEqual(fit.radius,2.)
        low, high = fit.apply_interval(-nominal[:,:2],nominal[:,:2],scale=[1.,2.])
        np.testing.assert_array_equal(low[0],[-3.,-6.])
        np.testing.assert_array_equal(high[0],[3.,6.])
        report = evaluate_interval(y[:,:2],low,high)
        self.assertEqual(report['simultaneous_mother_coverage'],1.)
        self.assertEqual(report['mean_width'],9.)

    def test_vacuous_rank_reports_no_useful_acceptance(self):
        values = np.zeros((8,2))
        upper_fit = fit_group_upper(values,values,alpha=.1)
        self.assertEqual(upper_fit.metadata(),{'radius':None,'vacuous':True,'rank':9,'n_groups':8,'alpha':.1})
        upper = upper_fit.apply(values)
        self.assertTrue(np.isposinf(upper).all())
        report = evaluate_upper(values,upper,median=values)
        self.assertEqual(report['simultaneous_mother_coverage'],1.)
        self.assertEqual(report['finite_width_fraction'],0.)
        self.assertIsNone(report['mean_width'])
        decisions = evaluate_decisions(values,upper<=0)
        self.assertEqual(decisions['accepted_request_fraction'],0.)
        self.assertIsNone(decisions['unsafe_fraction_among_active_mothers'])

    def test_choice_gather_preserves_mother_case_window_axes(self):
        y = np.arange(48).reshape(2,2,3,4)
        selection = np.array([[[0,1,3],[3,2,1]],[[1,2,0],[0,1,3]]])
        expected = [[[y[i,j,k,selection[i,j,k]] for k in range(3)]
                     for j in range(2)] for i in range(2)]
        np.testing.assert_array_equal(take_actions(y,selection),expected)
        with self.assertRaises(ValueError):
            take_actions(y,selection.astype(float))

    def test_bonferroni_bound_inverts_reference_binomial(self):
        problems = [x for x in itertools.product([1,5,48,80],[0,1,4,40]) if x[1]<=x[0]]
        for n,failures in problems:
            with self.subTest(n=n,failures=failures):
                truth = -np.ones((n,2))
                truth[:failures,0] = 1.
                prob = np.zeros_like(truth)
                fit = fit_selective_group(truth,prob,cutoffs=[.1,.2,.4],max_risk=.2,delta=.05)
                row = fit['candidates'][0]
                expected = (1. if n==failures else
                            brentq(lambda p:binom.cdf(failures,n,p)-.05/3,0.,1.))
                self.assertEqual(row['active_mothers'],n)
                self.assertEqual(row['unsafe_mothers'],failures)
                self.assertAlmostEqual(row['upper_risk'],expected,places=10)

    def test_selectivity_cannot_certify_empty_or_overconfident_family(self):
        n = 48
        truth = np.column_stack((-np.ones(n),np.r_[np.ones(12),-np.ones(n-12)]))
        prob = np.tile([.1,.5],(n,1))
        selection = fit_selective_group(truth,prob,cutoffs=[.05,.1,.2,.4,.5],max_risk=.1)
        self.assertEqual(selection['cutoff'],.4)
        self.assertAlmostEqual(selection['candidates'][1]['upper_risk'],1-math.pow(.01,1/n),places=12)
        no_cert = fit_selective_group(truth,prob,cutoffs=[.01,.04],max_risk=.2)
        self.assertIsNone(no_cert['cutoff'])
        self.assertFalse(no_cert['certified'])
        self.assertEqual(no_cert['candidates'][0]['upper_risk'],1.)

    def test_decision_accounting_and_mother_bootstrap(self):
        selected_margins = np.array([[-1.,.1,1.],[-1.,-1.,-1.],[1.,1.,1.]])
        accepted = np.array([[True,True,True],[True,False,False],[False,False,False]])
        probs = np.array([[0.,1.,.5],[0.,.01,.99],[1.,.5,1.]])
        result = evaluate_decisions(selected_margins,accepted,probability=probs,bootstrap=200,seed=309)
        self.assertEqual(result['mothers'],3)
        self.assertEqual(result['requests'],9)
        self.assertEqual(result['unsafe_mothers'],1)
        self.assertEqual(result['unsafe_accepted_requests'],2)
        self.assertAlmostEqual(result['accepted_request_fraction'],4/9)
        self.assertAlmostEqual(result['successful_request_fraction'],2/9)
        self.assertAlmostEqual(result['unsafe_fraction_among_active_mothers'],.5)
        self.assertAlmostEqual(result['any_unsafe_mother_fraction'],1/3)
        self.assertAlmostEqual(result['chosen_failure_brier'],np.mean((probs-(selected_margins>0))**2))
        self.assertEqual(sum(r['requests'] for r in result['probability_reliability']),9)
        raw = result['mother_bootstrap']['ratios']['accepted_request_fraction']
        self.assertEqual(raw['usable_draws'],200)
        self.assertTrue(raw['interval'][0]<=4/9<=raw['interval'][1])


if __name__ == '__main__':
    unittest.main()
