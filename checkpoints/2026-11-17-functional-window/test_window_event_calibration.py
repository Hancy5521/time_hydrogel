"""Finite-rank and exact-binomial checks independent of a materials simulator."""

import itertools
import math
import unittest

import numpy as np

from window_event_calibration import (
    apply_marginal_event,
    certify_policy,
    certify_policy_sequence,
    critical_unsafe_score,
    fit_marginal_event,
    fit_policy_sequence,
    single_policy_power,
)


def mass(k, n, p):
    return math.comb(n,k)*p**k*(1-p)**(n-k)


class EventCalibrationTests(unittest.TestCase):
    def test_group_critical_reduces_only_over_unsafe_recommendations(self):
        y=np.array([[-2.,.1,.2],[0.,-1.,-2.],[1.,2.,3.]])
        s=np.array([[-30.,.3,.2],[-.1,.1,.2],[.1,.2,.3]])
        valid=np.array([[True,True,True],[True,True,True],[False,True,True]])
        np.testing.assert_array_equal(critical_unsafe_score(y,s,valid=valid),[.2,np.inf,.2])

    def test_exchangeable_rank_including_ties_and_no_failures(self):
        for critical,expected in (([0.,1.,2.,3.,np.inf],2),([0.,0.,0.,3.,np.inf],0)):
            y=np.array([1.,1.,1.,1.,-1.])
            s=np.array(critical)
            s[-1]=0.
            unsafe=0
            for i in range(len(y)):
                mask=np.arange(len(y))!=i
                fit=fit_marginal_event(y[mask],s[mask],alpha=.4)
                self.assertEqual(fit['rank'],2)
                unsafe+=(y[i]>0 and apply_marginal_event(s[i:i+1],fit)[0])
            self.assertEqual(unsafe,expected)

    def test_small_data_and_no_error_conventions(self):
        never=fit_marginal_event([1.,-1.],[.1,.2],alpha=.1)
        self.assertFalse(apply_marginal_event([-.2,.2],never).any())
        all_eligible=fit_marginal_event([-1.]*12,np.linspace(-1,1,12),alpha=.1)
        self.assertTrue(apply_marginal_event([-.2,.2],all_eligible).all())

    def test_iid_tolerance_rank_equals_exact_binomial_enumeration(self):
        n,alpha,delta=48,.1,.05
        rank=max(k for k in range(n+1) if sum(mass(j,n,alpha) for j in range(k))<=delta)
        fit=fit_marginal_event(np.ones(n),np.arange(n),alpha=alpha,delta=delta)
        self.assertEqual(fit['rank'],rank)
        self.assertEqual(fit['threshold'],rank-1)

    def test_exact_separate_risks_activity_and_empty_policy(self):
        n=50
        y=np.tile([1.,1.,-1.],(n,1))
        y[2:,0:2]=-1
        use=np.zeros(y.shape,dtype=bool)
        use[:20]=True
        fit=certify_policy(y,use,max_risk=.25,min_activity=.20)
        self.assertEqual((fit['active_mothers'],fit['unsafe_active_mothers']), (20,2))
        self.assertAlmostEqual(fit['risk_p'],sum(mass(j,20,.25) for j in range(3)))
        self.assertAlmostEqual(fit['activity_p'],sum(mass(j,n,.20) for j in range(20,n+1)))
        self.assertEqual(fit['joint_p'],max(fit['activity_p'],fit['risk_p']))
        self.assertFalse(certify_policy(y,np.zeros(y.shape,bool))['certified'])

    def test_sequence_stops_before_untested_high_utility_policy(self):
        n=64
        y=np.full((n,2),-1.)
        y[:,1]=1.
        good=np.tile([True,False],(n,1))
        risky=np.ones(y.shape,dtype=bool)
        seq=certify_policy_sequence([('a',y,good),('b',y,risky),('c',y,good)])
        self.assertEqual(seq['selected'],'a')
        self.assertEqual(seq['untested'],['c'])
        scores=np.tile([-.1,.1],(n,1))
        seq=fit_policy_sequence(y,scores,cutoffs=[.1,.2,.3])
        self.assertEqual(seq['cutoff'],.1)
        self.assertEqual(seq['untested'],[.3])

    def test_power_against_explicit_categorical_workloads(self):
        n,p,q=5,.8,.2
        power=0.
        for realization in itertools.product(range(3),repeat=n):
            m=sum(r!=0 for r in realization)
            bad=sum(r==2 for r in realization)
            passes_risk=sum(mass(j,m,.6) for j in range(bad+1))<=.2
            passes_use=sum(mass(j,n,.2) for j in range(m,n+1))<=.2
            if passes_risk and passes_use:
                power+=math.prod((1-p,p*(1-q),p*q)[i] for i in realization)
        self.assertAlmostEqual(power,single_policy_power(n,true_activity=p,true_risk=q,max_risk=.6,
                                                         min_activity=.2,delta=.2))


if __name__=='__main__':
    unittest.main()
