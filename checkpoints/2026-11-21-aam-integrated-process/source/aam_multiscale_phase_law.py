"""Monotone mother-ESS temperatures with one exact shared phase per path."""

from operator import index

import numpy as np
from scipy.special import logsumexp


SCHEMA = "aam-factorized-parent-ess-power-with-exact-shared-phase-v1"
PHASES = 16
TARGET_ESS = (64, 256, 1024, 4096)
COARSE = tuple(f"ess{n}" for n in TARGET_ESS)
COMPONENTS = (*COARSE, "phase_rbf")
_SPLIT_STEPS = 53


def camera_log_likelihood(clean, query, noise):
    """Joint original likelihood for all three frames at each single phase."""
    x, y, sigma = (np.asarray(a, dtype=np.float64) for a in (clean, query, noise))
    if (x.ndim != 4 or x.shape[1:] != (PHASES, 3, 4) or not len(x)
            or y.shape != (3, 4) or sigma.shape != (4,) or np.any(sigma <= 0.)
            or any(not np.isfinite(a).all() for a in (x, y, sigma))):
        raise ValueError("Use the original three snapshot rows, 16 shared phases, and positive camera noise.")
    d = (x - y) / sigma
    return -.5 * np.einsum("nsij,nsij->ns", d, d, optimize=False)


def _ess(centered, exponent):
    z = np.exp(exponent * centered)
    return float(z.sum()**2 / np.sum(z * z))


def _scale(centered, target, exact):
    """Largest exponent in [0,1] whose mother ESS is at least target."""
    if target <= exact:
        return 1.
    if target == len(centered):
        return 0.
    less, greater = 0., 1.
    for _ in range(_SPLIT_STEPS):
        mid = .5 * (less + greater)
        if _ess(centered, mid) >= target:
            less = mid
        else:
            greater = mid
    return less


def temper_log_likelihood(log_likelihood, targets=TARGET_ESS):
    """Temper the marginalized mother evidence, preserving exact phase conditionals.

    If E_i = log(mean_s exp(L_is)), then P_beta(i) is proportional to
    exp(beta E_i), and P_beta(s|i) is exactly softmax_s(L_is). Increasing
    beta cannot increase the mother ESS because

        d log ESS / d beta = 2 (E_{P_beta} E_i - E_{P_{2 beta}} E_i) <= 0,

    and the derivative of the latter expectation versus its own parameter
    is Var(E_i). A joint phase-wise power can fail this monotonicity.
    The exact arm always has beta=1. A requested lower ESS never sharpens
    that arm. Every positive theoretical mass is retained in log form.
    """
    values = np.asarray(log_likelihood, dtype=np.float64)
    ordered = tuple(index(t) for t in targets)
    if (values.ndim != 2 or values.shape[1] != PHASES or not len(values) or not np.isfinite(values).all()
            or not ordered or ordered != tuple(sorted(set(ordered))) or not 1 <= ordered[0] <= ordered[-1] <= len(values)):
        raise ValueError("Use a finite mother-by-16 joint likelihood and increasing attainable ESS targets.")
    maxima = values.max(axis=1)
    offset = values - maxima[:, None]
    log_sum_phase = logsumexp(offset, axis=1)
    phase_conditional = offset - log_sum_phase[:, None]
    evidence = maxima + log_sum_phase - np.log(PHASES)
    centered = evidence - evidence.max()
    exact_ess = _ess(centered, 1.)
    beta = {f"ess{t}": _scale(centered, t, exact_ess) for t in ordered}
    beta["exact"] = 1.
    nominal = {f"ess{t}": t for t in ordered}
    nominal["exact"] = None
    probabilities, logs, rows = {}, {}, {}
    for name, exponent in beta.items():
        a = exponent * centered
        log_parent = a - logsumexp(a)
        log_atom = log_parent[:, None] + phase_conditional
        mass = np.exp(log_atom)
        mass /= mass.sum()
        p = mass.sum(axis=1)
        underflow = int(np.count_nonzero(mass == 0.))
        probabilities[name] = mass.ravel()
        logs[name] = log_atom.ravel()
        rows[name] = {"target_mother_ess": nominal[name], "parent_exponent": exponent,
                      "log_mother_ess": float(-logsumexp(2. * log_parent)),
                      "mother_ess": float(1. / np.sum(p * p)),
                      "atom_ess": float(1. / np.sum(mass * mass)),
                      "largest_mother_mass": float(p.max()), "largest_atom_mass": float(mass.max()),
                      "numerically_zero_atoms": underflow,
                      "positive_mothers": int(np.count_nonzero(p)),
                      "capped_at_original_likelihood": exponent == 1. and nominal[name] is not None}
    return {"schema": SCHEMA, "weights": probabilities, "log_weights": logs, "diagnostics": rows,
            "mothers": len(values), "phase_count": PHASES}


def multiscale_laws(clean, query, noise, targets=TARGET_ESS):
    """Return the same-atom laws for one original noisy query, with no fitting."""
    return temper_log_likelihood(camera_log_likelihood(clean, query, noise), targets)


def mix_laws(components, coefficients):
    """Combine probability laws in a fixed caller-declared component order."""
    x, g = np.asarray(components, dtype=np.float64), np.asarray(coefficients, dtype=np.float64)
    if (x.ndim != 2 or not x.size or g.shape != (len(x),)
            or any(not np.isfinite(a).all() or np.any(a < 0.) for a in (x, g))
            or not np.allclose(x.sum(axis=1), 1., rtol=0., atol=2e-13)
            or not np.isclose(g.sum(), 1., rtol=0., atol=2e-13)):
        raise ValueError("Each complete law and the component mixture must be normalized and nonnegative.")
    result = np.einsum("k,ki->i", g, x, optimize=False)
    return result / result.sum()


def fixtures():
    """Check endpoint identities, the ESS bracket, and no finite-mass flooring."""
    rng = np.random.default_rng(319963)
    logl = -rng.random((96, PHASES)) * 8. - np.arange(96)[:, None]
    out = temper_log_likelihood(logl, (4, 16, 64, 96))
    wanted = np.exp(logl - logsumexp(logl))
    exact = out["weights"]["exact"].reshape(logl.shape)
    np.testing.assert_allclose(exact, wanted, rtol=2e-14, atol=3e-16)
    last = 1.
    original_phase = logl - logsumexp(logl, axis=1, keepdims=True)
    for target in (4, 16, 64, 96):
        name = f"ess{target}"
        h = out["diagnostics"][name]
        assert h["parent_exponent"] <= last and abs(h["mother_ess"] - target) < 2e-10
        logged = out["log_weights"][name].reshape(logl.shape)
        np.testing.assert_allclose(logged - logsumexp(logged, axis=1, keepdims=True),
                                   original_phase, rtol=0., atol=3e-14)
        last = h["parent_exponent"]
    fake = np.array([[0., -12.], [-1., -1.]])
    raw_group_ess = []
    for temp in (.2, .7):
        parent = logsumexp(temp * fake, axis=1)
        raw_group_ess.append(_ess(parent - parent.max(), 1.))
    assert raw_group_ess[1] > raw_group_ess[0]
    difficult = np.tile(np.linspace(-1800., 0., PHASES), (4, 1))
    stress = temper_log_likelihood(difficult, (2, 4))
    assert stress["diagnostics"]["ess2"]["parent_exponent"] == 1.
    assert stress["diagnostics"]["ess2"]["numerically_zero_atoms"] > 0
    assert np.isfinite(stress["log_weights"]["ess2"]).all()
    first, second = out["weights"]["ess64"], out["weights"]["ess96"]
    np.testing.assert_allclose(mix_laws((first, second), [.25, .75]),
                               .25*first+.75*second, rtol=0., atol=1e-16)
    return {"schema": SCHEMA, "native_calls": 0, "exact_likelihood_endpoints": True,
            "mother_ess_bisection": True, "counterexample_to_joint_phase_power_monotonicity": True}


if __name__ == "__main__":
    print(fixtures())
