# Calibrating a usable whole-window decision at the mother level

The benchmark forecasts a signed margin \(Y_{g,j,a}\) for each candidate action \(a\), request \(j\) and mother \(g\). The margin is a maximum over the registered window clauses and time grid, so a positive margin is a failure. The eight requests per mother share underlying simulation conditions and must stay together in calibration, testing and resampling.

The previous simultaneous upper band protects every catalogue action and allows arbitrary reselection among certified actions. In the fresh spatial calibration study, the nominal depth-probe model activated 2/288 requests under the catalogue band and 43/288 under its frozen-choice band. The latter had two failed requests in one mother. The difference illustrates the cost of covering every candidate, including margins far from the decision. Quantile coverage alone leaves conditional safety among accepted mothers unresolved. [Independent spatial calibration audit](runs/patch_window_spatial_calibration_20261116T142630930437Z/AUDIT.json)

## Exact joint risk and activity targets

Using independent development mothers, freeze a mapping from observed inputs to action \(a_{gj}\) and its acceptance score \(s_{gj}\), with lower scores safer. The mapping can inspect the entire posterior catalogue. Freeze one global cutoff \(t\) and accept if \(s_{gj}<t\). Validate the chosen actions under the complete operating policy.

\[
A_g=\mathbf 1[\text{any accepted request}],\qquad
U_g=\mathbf 1[\text{any accepted request has }Y_{g,j,a_{gj}}>0].
\]

The proposed targets are a ceiling \(R_{\max}\) for \(\Pr(U=1\mid A=1)\) and a floor \(a_{\min}\) for \(\Pr(A=1)\). At the illustrative benchmark thresholds \(R_{\max}=.25\) and \(a_{\min}=.10\), satisfying both ensures that at least 7.5% of mothers receive a recommendation and have no unsafe accepted request. Report request throughput, positive margin regret, outcomes conditional on acceptance and paired utility alongside activity. An activity count can hide a low number of useful recommendations per mother.

Let \(M=\sum A_g\), \(K=\sum U_g\), and \(n\) be the number of IID calibration mothers. Conditional on \(M\), \(K\) is binomial with \(M\) trials and risk \(\Pr(U=1\mid A=1)\). Test

\[
p_R=F_{\mathrm{Bin}(M,R_{\max})}(K),\qquad
p_A=\Pr\{\mathrm{Bin}(n,a_{\min})\ge M\}.
\]

Joint certification requires \(\max(p_R,p_A)\le\delta\). If either target fails in the population, that component must pass its size-\(\delta\) test before a false joint certificate can issue. The individually shown Clopper–Pearson bounds do not constitute simultaneous or post-selection confidence bounds. With \(M=0\), \(p_R=1\) and the rule cannot certify.

A fixed sequence of complete policies can be ordered on development and tested at level \(\delta\), stopping at its first failed joint test. Returning any of the passed policies controls false certification because certification of the first invalid policy would require its valid p-value to pass. This ordering can give more power than Bonferroni over all cutoffs. It can also block good later policies behind a failing first choice. One development-locked primary rule is the cleanest current experiment. Choosing a certificate across algorithms requires a common prespecified sequence or multiplicity correction. Same-information baselines can be scored descriptively on the same held cohort. Sections 2.3.1, B and D of the author version were consulted. [Learn then Test](https://doi.org/10.1214/24-AOAS1998) · [Author preprint v5](https://arxiv.org/abs/2110.01052)

## Decision-aligned marginal alternative

When the target is unconditional incidence of unsafe mothers, define the minimum score that would first admit an unsafe recommendation,

\[
B_g=\min_{j:Y_{g,j,a_{gj}}>0}s_{gj},
\]

using \(+\infty\) when every proposed choice is safe. At threshold \(t\), \(U_g=\mathbf 1[B_g<t]\). For \(n\) exchangeable calibration mothers and a fresh exchangeable mother, the \(k=\lfloor(n+1)\alpha\rfloor\)th smallest \(B_g\) controls \(\Pr(U_{\rm new}=1)\le\alpha\) averaged across calibration and new draws. Strict inequality protects ties. If \(k=0\), use no recommendations. For a high-probability variant under IID calibration, use the largest \(k\) with \(F_{\mathrm{Bin}(n,\alpha)}(k-1)\le\delta\). Exceeding population rate \(\alpha\) would require the sample order statistic to exceed the lower \(\alpha\) quantile, an event bounded by this binomial tail.

A low unconditional failure rate can coexist with all accepted mothers failing when activity is rare. The marginal threshold is therefore a separate comparison, with no conditional certificate. This rank construction applies when actions are fixed functions of observations, or when threshold changes add nested recommendations at their original scores. If changing the cutoff replaces an accepted action, a violation can disappear as the cutoff rises. Directly certify that entire policy using the locked fixed-policy or sequence test. Retain the catalogue band when arbitrary future reselection must remain covered.

With a verified common posterior draw identity across requests, an action slate can be ranked by the weighted fraction of draws where any proposed margin exceeds zero. When cases have separate posterior weights, use a sum of within-case union scores as a ranking feature. A cross-case joint posterior would require conditioning shared latent draws on both cases. A score from available posterior failure probabilities and signed quantiles avoids assuming that coupling. The choice of ranking affects utility and feasibility of a nonempty certificate. Validity relies on independent mother labels and a locked policy.

## Changed mechanisms and sample sizing

An unflagged 50:50 mixture is a valid target if calibration and test mothers draw their mechanisms and measurements independently. A quota of exactly 50:50 by role lacks IID mixture sampling. Reusing a witness-noise generator address across mechanisms also violates the design. An actual instance and the repaired preflight are documented in [the random-stream correction](WINDOW_RANDOM_STREAM_CORRECTION_ZH.md). In two populations with activation \(a_z\), selective error \(r_z\) and mixture weight \(\pi\), the mixture error is

\[
\frac{\pi a_1r_1+(1-\pi)a_2r_2}{\pi a_1+(1-\pi)a_2}.
\]

A mixture certificate does not imply a certificate for either mechanism or for changed proportions. Within-mechanism protection can use a common unchanged policy and independent calibration from each declared population. Per-mother thresholds need an observable predeclared stratum rather than the hidden simulator flag. Reweighting under measured covariate shift requires a shared conditional outcome law, which an ion-exchange mechanism change can violate. [Tibshirani et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)

Exact power for a locked rule is the binomial average over \(M\). For counts passing the activity test, use the probability under the assumed true \(\mathrm{Bin}(M,r)\) of passing the selective-risk test. At one-sided \(\delta=.05\) and \(a_{\min}=.10\), plausible planning scenarios give the following powers. Rates are hypothetical.

| True activity | True selective risk | Risk target | Power at 48 | Power at 96 | Power at 224 | First n with ≥80% power |
|---:|---:|---:|---:|---:|---:|---:|
| .20 | .08 | .25 | .129 | .420 | .892 | 180 |
| .30 | .08 | .25 | .318 | .714 | .981 | 115 |
| .40 | .08 | .25 | .474 | .845 | .997 | 86 |
| .40 | .12 | .25 | .270 | .567 | .922 | 159 |
| .40 | .03 | .10 | .001 | .330 | .806 | 222 |
| .40 | .05 | .10 | .001 | .154 | .449 | 480 |

With zero observed unsafe active mothers, the level-.05 risk test requires at least 11 active mothers at a .25 target, or 29 at .10. The activity-floor requirement can raise the total size needed. Use a distinct development cohort for score selection and a fresh certification cohort of the stated mixture, then an untouched test cohort for paired observed use and safety evaluation. Preserve single-score and catalogue-envelope controls. If the rule cannot certify or useful throughput is weak, report the counts before deciding whether another local measurement is needed.

The [implementation](window_event_calibration.py) and [tests](test_window_event_calibration.py) include finite-rank enumeration, exact tail checks against a direct binomial sum, sequence stopping and independent categorical enumeration of planning power.
