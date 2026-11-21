"""Prospective paired noninferiority scores for independent simulated mothers."""

from operator import index

import numpy as np
from scipy.optimize import brentq
from scipy.stats import t


SCHEMA = "aam-paired-simultaneous-relative-noninferiority-v1"


def _position(method, names):
    if isinstance(method, str):
        return names.index(method)
    position = index(method)
    if position not in range(len(names)):
        raise ValueError("Method index is outside the locked methods.")
    return position


def _column(values, critical):
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    return mean, sd, mean + critical * sd / np.sqrt(len(values))


def _signed_margin_bound(difference, reference, critical):
    """Invert the simultaneous upper paired bound, when its root is unique."""
    scale = np.sqrt(len(reference))
    ref_mean = reference.mean()
    ref_sd = reference.std(ddof=1)
    slope = -ref_mean + critical * ref_sd / scale
    if slope >= 0.:
        # Without a positive lower bound for the denominator, this inversion
        # is not certified. The direct test at every locked margin still runs.
        return None

    mean = difference.mean()
    centered = difference - mean
    ref_centered = reference - ref_mean
    divisor = np.sqrt(len(reference) - 1) * scale

    def upper(margin):
        return float(mean - margin * ref_mean +
                     critical * np.linalg.norm(centered - margin * ref_centered) / divisor)

    zero = upper(0.)
    if zero == 0.:
        return 0.
    outside = 2. * zero / -slope
    low, high = sorted((0., float(outside)))
    return float(brentq(upper, low, high, xtol=1e-13))


def score_noninferiority(losses, methods, metrics, candidate, references, *,
                         primary_margin, sensitivity_margins=(), alpha=.05):
    """Score the locked candidate on every baseline and metric (smaller is better).

    Each row represents an independent mother; repeated paired estimates must
    already be averaged within that mother. The primary null for a comparison
    is E[C-(1+m)R] >= 0. A negative simultaneous upper bound rejects it.
    """
    x = np.asarray(losses, dtype=float)
    names, endpoints = tuple(methods), tuple(metrics)
    margins = (float(primary_margin), *map(float, sensitivity_margins))
    if (x.ndim != 3 or x.shape[1:] != (len(names), len(endpoints)) or len(x) < 2
            or not np.isfinite(x).all() or len(set(names)) != len(names)
            or not endpoints or len(set(endpoints)) != len(endpoints)
            or not all(isinstance(name, str) and name for name in (*names, *endpoints))
            or not np.isfinite(margins).all() or min(margins) < 0.
            or len(set(margins)) != len(margins) or not 0. < alpha < 1.):
        raise ValueError("Require finite independent mother rows and unique locked methods, metrics, and margins.")
    c = _position(candidate, names)
    refs = tuple(_position(method, names) for method in references)
    if not refs or len(set(refs)) != len(refs) or c in refs:
        raise ValueError("Comparison references must be distinct from the candidate and each other.")

    tests = len(refs) * len(endpoints)
    critical = float(t.isf(alpha / tests, len(x) - 1))
    pointwise = float(t.isf(alpha / 2., len(x) - 1))
    details = {}
    cases = [{"margin": m, "comparisons": {names[r]: {} for r in refs}} for m in margins]
    certified = []
    for r in refs:
        details[names[r]] = {}
        for j, endpoint in enumerate(endpoints):
            target, reference = x[:, c, j], x[:, r, j]
            delta = target - reference
            mean, sd, bound = _column(delta, critical)
            half = pointwise * sd / np.sqrt(len(delta))
            minimum = _signed_margin_bound(delta, reference, critical)
            certified.append(minimum)
            details[names[r]][endpoint] = {
                "candidate_mean": float(target.mean()), "reference_mean": float(reference.mean()),
                "raw_candidate_minus_reference": mean, "raw_paired_sd": sd,
                "raw_pointwise_two_sided_interval": [mean - half, mean + half],
                "raw_simultaneous_upper": bound, "smallest_certifiable_relative_excess": minimum,
                "identical_scores": bool(np.array_equal(target, reference))}
            for case in cases:
                margin = case["margin"]
                mu, shifted_sd, upper = _column(delta - margin * reference, critical)
                case["comparisons"][names[r]][endpoint] = {
                    "shifted_mean": mu, "shifted_paired_sd": shifted_sd, "simultaneous_upper": upper,
                    "passed": bool(upper < 0.)}
    for case in cases:
        case["passed"] = all(result["passed"] for group in case["comparisons"].values()
                             for result in group.values())
    return {
        "schema": SCHEMA, "candidate": names[c], "references": [names[r] for r in refs],
        "metrics": list(endpoints), "mothers": len(x), "alpha": alpha, "family_comparisons": tests,
        "critical_one_sided_bonferroni_t": critical, "comparisons": details,
        "primary": cases[0], "sensitivity": cases[1:],
        "largest_endpoint_certifiable_relative_excess": (
            max(certified) if all(v is not None for v in certified) else None)}


def fixtures():
    """Test the simultaneous correction, sign, inversion and exact equal scores."""
    names = ("learned", "strong_a", "strong_b")
    metrics = ("path", "change")
    x = np.empty((32, 3, 2))
    x[:, 1, :] = (1., 2.)
    x[:, 2, :] = (1., 2.2)
    x[:, 0, :] = (1.01, 2.02)
    kwargs = {"primary_margin": .02, "sensitivity_margins": (.005,)}
    result = score_noninferiority(x, names, metrics, "learned", ("strong_a", "strong_b"), **kwargs)
    assert result == score_noninferiority(x, names, metrics, 0, (1, 2), **kwargs)
    assert result["primary"]["passed"] and not result["sensitivity"][0]["passed"]
    np.testing.assert_allclose(result["critical_one_sided_bonferroni_t"], t.isf(.05/4,31))
    np.testing.assert_allclose(result["largest_endpoint_certifiable_relative_excess"], .01, atol=1e-12)

    x[:, 1, 0] = 1. + np.linspace(-.2,.2,len(x))
    x[:, 0, 0] = 1.007 * x[:, 1, 0]
    x[:, 0, 1] = x[:, 1, 1]
    check = score_noninferiority(x, names, metrics, 0, (1,), primary_margin=.008)
    np.testing.assert_allclose(check["comparisons"]["strong_a"]["path"]["smallest_certifiable_relative_excess"],
                               .007, atol=1e-12)
    assert check["comparisons"]["strong_a"]["change"]["identical_scores"]
    assert check["primary"]["passed"]

    x[:,1,0] = np.r_[np.zeros(len(x)-1),1.]
    broad = score_noninferiority(x, names, metrics, 0, (1,), primary_margin=.008)
    assert broad["largest_endpoint_certifiable_relative_excess"] is None
    assert not broad["primary"]["passed"]
    return {"passed": True, "native_calls": 0, "fitted_models": 0}


if __name__ == "__main__":
    print(fixtures())
