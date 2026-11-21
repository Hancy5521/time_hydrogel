"""Full-support weights and paired energy-score estimates for a phase atlas."""

from itertools import combinations
from operator import index

import numpy as np
from scipy.special import logsumexp


SCHEMA = "aam-large-common-phase-atlas-whole-support-energy-v1"
PHASES = 16
PAIR_DRAWS = 32768
METHODS = ("phase_gaussian", "phase_rbf", "raw_gaussian", "raw_rbf", "phase_uniform")
METRICS = ("early18", "hidden_delta32_6", "hidden_t3_6", "hidden_t2_6")
LOCAL_SHAPE = (3, 2, 3)


def _normalized_log(log_weights):
    values = np.asarray(log_weights, dtype=np.float64)
    normalizer = logsumexp(values)
    if not np.isfinite(normalizer):
        raise ValueError("Every atlas weighting method needs positive finite total log mass.")
    w = np.exp(values - normalizer)
    return w / w.sum()


def _rbf(logs, mixture):
    kernel = _normalized_log(logs)
    return mixture * kernel + (1. - mixture) / kernel.size


def _embed_raw(w):
    scaffold = np.zeros((len(w), PHASES), dtype=np.float64)
    scaffold[:, 0] = w
    return scaffold


def _stats(w):
    mothers = w.sum(axis=1)
    phases = w.sum(axis=0)
    return {"atom_ess": float(1. / np.sum(w * w)),
            "mother_ess": float(1. / np.sum(mothers * mothers)),
            "largest_atom_mass": float(w.max()), "largest_mother_mass": float(mothers.max()),
            "positive_atoms": int(np.count_nonzero(w)), "positive_mothers": int(np.count_nonzero(mothers)),
            "phase_marginal": phases}


def probabilities(clean, measured_base, query, noise, camera_scale, kernel_sq_scale, setting):
    """Compute exact conditional likelihoods on a fixed finite legal bank.

    Each mother has 16 shared shifts of all three frames. Its archived base
    measurement residual is reused unchanged for all deterministic copies in
    the matched RBF. We evaluate all registered support points. After
    self-normalization, the finite-bank Gaussian law only approximates the
    original continuous-prior posterior. It preserves the original sensor
    likelihood exactly on the finite bank.
    """
    clean, base, y, noise, scale = (np.asarray(a, dtype=np.float64)
                                    for a in (clean, measured_base, query, noise, camera_scale))
    if (clean.ndim != 4 or clean.shape[1:] != (PHASES, 3, 4) or not len(clean)
            or base.shape != (len(clean), 3, 4) or y.shape != (3, 4)
            or noise.shape != (4,) or scale.shape not in ((3, 4), (4,))
            or any(not np.isfinite(x).all() for x in (clean, base, y, noise, scale))
            or np.any(noise <= 0.) or np.any(scale <= 0.)):
        raise ValueError("Use all 16 shared phases, original three camera rows, and frozen positive scales.")
    bandwidth, mixture = (float(x) for x in setting)
    sq_scale = float(kernel_sq_scale)
    if (not np.isfinite(bandwidth) or bandwidth <= 0. or not np.isfinite(mixture) or not 0. <= mixture <= 1.
            or not np.isfinite(sq_scale) or sq_scale <= 0.):
        raise ValueError("Use the original registered width, squared scale, and RBF mixture.")
    width = bandwidth * np.sqrt(sq_scale)
    if not np.isfinite(width) or width <= 0.:
        raise ValueError("The combined frozen width must remain finite and positive.")
    difference = (clean - y) / noise
    exact = -.5 * np.einsum("nsij,nsij->ns", difference, difference, optimize=False)
    noisy_difference = (clean + (base - clean[:, 0])[:, None] - y) / (scale * width)
    noisy_difference[:, 0] = (base - y) / (scale * width)
    rbf_score = -.5 * np.einsum("nsij,nsij->ns", noisy_difference, noisy_difference, optimize=False)
    laws = {
        "phase_gaussian": _normalized_log(exact),
        "phase_rbf": _rbf(rbf_score, mixture),
        "raw_gaussian": _embed_raw(_normalized_log(exact[:, 0])),
        "raw_rbf": _embed_raw(_rbf(rbf_score[:, 0], mixture)),
        "phase_uniform": np.full_like(exact, 1. / exact.size),
    }
    diagnostics = {name: _stats(p) for name, p in laws.items()}
    diagnostics.update(phase_rbf_unmixed=_stats(_normalized_log(rbf_score)),
                       raw_rbf_unmixed=_stats(_embed_raw(_normalized_log(rbf_score[:, 0]))))
    return {"schema": SCHEMA, "weights": {name: p.ravel() for name, p in laws.items()},
            "diagnostics": diagnostics,
            "mothers": len(clean), "phases": PHASES}


def pair_uniforms(rng, draws=PAIR_DRAWS):
    """Independent U,V within every draw; reuse one array across methods."""
    size = index(draws)
    if size < 2:
        raise ValueError("At least two draws are required to estimate pair error.")
    return rng.random((size, 2))


def _flat_histories(value):
    z = np.asarray(value, dtype=np.float64)
    if z.ndim not in (4, 5) or z.shape[-3:] != LOCAL_SHAPE or not len(z) or not np.isfinite(z).all():
        raise ValueError("Supply complete original three-time, two-port, three-feature histories.")
    return z.reshape(-1, *LOCAL_SHAPE)


def _weights(value, n):
    p = np.asarray(value, dtype=np.float64)
    if (p.shape != (n,) or not np.isfinite(p).all() or np.any(p < 0.)
            or not np.isclose(p.sum(), 1., rtol=0., atol=2e-13)):
        raise ValueError("Every method must provide normalized mass on the same full atom order.")
    return p


def _uniforms(value):
    u = np.asarray(value, dtype=np.float64)
    if (u.ndim != 2 or len(u) < 2 or u.shape[1] != 2 or not np.isfinite(u).all()
            or np.any(u < 0.) or np.any(u >= 1.)):
        raise ValueError("Use more than one pair of uniforms in the half-open unit interval.")
    return u


def _rms(value, axis):
    return np.sqrt(np.mean(np.square(value), axis=axis))


def _distances(x, y):
    dz = x - y
    return np.stack((_rms(dz, (-3, -2, -1)), _rms(dz[..., 2, :, :] - dz[..., 1, :, :], (-2, -1)),
                     _rms(dz[..., 2, :, :], (-2, -1)), _rms(dz[..., 1, :, :], (-2, -1))), axis=-1)


def _inverse_cdf(p, main, repeat):
    positive = np.flatnonzero(p > 0.)
    cdf = np.minimum(np.cumsum(p[positive], dtype=np.float64), 1.)
    cdf[-1] = 1.
    return tuple(positive[np.searchsorted(cdf, u, side="right")] for u in (main, repeat))


def _sample_distances(x, ids):
    return _distances(x[ids[:, 0]], x[ids[:, 1]])


def _standard_error(pair_values):
    return .5 * pair_values.std(axis=0, ddof=1) / np.sqrt(len(pair_values))


def _terms(linear, main, repeat):
    pair = main.mean(axis=0)
    pair_repeat = repeat.mean(axis=0)
    score = linear - .5 * pair
    score_repeat = linear - .5 * pair_repeat
    difference = np.abs(score - score_repeat)
    return {"linear_exact": linear, "pair_main": pair, "pair_repeat": pair_repeat,
            "energy_main": score, "energy_repeat": score_repeat,
            "energy_monte_carlo_se": _standard_error(main),
            "energy_repeat_monte_carlo_se": _standard_error(repeat),
            "absolute_energy_repeat_difference": difference,
            "l1_energy_repeat_difference": float(difference.sum())}


def energy_scores(local, weights, truth, uniforms, repeat_uniforms):
    """Full linear term and unbiased independent-pair estimate of the spread.

    Each pair is drawn iid from the full weighted finite atlas with replacement.
    Main/repeat directions must come from independent addressed streams. Within
    each stream the same two uniforms per pair are used by every method, so
    paired differences have a direct numerical Monte Carlo SE. Samples never
    determine the predictive support or its probabilities.
    """
    x = _flat_histories(local)
    y = np.asarray(truth, dtype=np.float64)
    u, v = _uniforms(uniforms), _uniforms(repeat_uniforms)
    if y.shape != LOCAL_SHAPE or not np.isfinite(y).all() or len(u) != len(v) or not weights:
        raise ValueError("Require one finite original target, nonempty weights, and matched draw counts.")
    if np.array_equal(u, v):
        raise ValueError("The checking replicate must use a separate pair stream.")
    first = _distances(x, y)
    scores, pair_values = {}, {}
    for name, weight in weights.items():
        p = _weights(weight, len(x))
        # Explicitly include every atom in this exact linear expectation.
        linear = np.einsum("n,nm->m", p, first, optimize=False)
        index_main, index_repeat = _inverse_cdf(p, u, v)
        main, repeat = _sample_distances(x, index_main), _sample_distances(x, index_repeat)
        scores[name] = _terms(linear, main, repeat)
        pair_values[name] = (main, repeat)
    contrasts = {}
    for left, right in combinations(scores, 2):
        a, b = scores[left], scores[right]
        main, repeat = (m - n for m, n in zip(pair_values[left], pair_values[right], strict=True))
        diff = a["linear_exact"] - b["linear_exact"]
        contrasts[f"{left}_minus_{right}"] = _terms(diff, main, repeat)
    return {"schema": SCHEMA, "metrics": METRICS, "pair_draws": len(u), "atoms": len(x),
            "methods": scores, "contrasts": contrasts}


def fixtures():
    """Dummy shared phase likelihood and exact finite pair enumeration."""
    rng = np.random.default_rng(180909)
    clean = rng.normal(size=(3, PHASES, 3, 4))
    noisy = clean[:, 0] + rng.normal(size=(3, 3, 4)) * .1
    truth_y = rng.normal(size=(3, 4))
    noise = np.arange(1, 5) * .5
    camera = np.linspace(.5, 1.5, 12).reshape(3, 4)
    result = probabilities(clean, noisy, truth_y, noise, camera, 1.2, (.5, .8))
    for name, w in result["weights"].items():
        np.testing.assert_allclose(w.sum(), 1., rtol=0., atol=2e-15)
        if name.startswith("raw"):
            assert np.count_nonzero(w.reshape(3, PHASES)[:, 1:]) == 0
    log = np.array([[-.5 * (((template-truth_y)/noise)**2).sum() for template in row] for row in clean])
    compare = _normalized_log(log)
    np.testing.assert_allclose(result["weights"]["phase_gaussian"].reshape(3, PHASES), compare,
                               rtol=0., atol=2e-15)
    manual = np.exp(-.5 * (((noisy-truth_y)/camera)**2).sum(axis=(1, 2))/(.5**2*1.2))
    np.testing.assert_allclose(result["weights"]["raw_rbf"].reshape(3, PHASES)[:, 0],
                               .8 * manual / manual.sum() + .2 / 3, rtol=0., atol=2e-15)
    local = np.zeros((2, *LOCAL_SHAPE))
    local[1, 2] = 1.
    p = np.array([.25, .75])
    q = np.array([.5, .5])
    marginal = np.array([.125, .375, .625, .875])
    uniform = np.column_stack((np.repeat(marginal, len(marginal)), np.tile(marginal, len(marginal))))
    exact = energy_scores(local, {"p": p, "q": q}, local[0], uniform, uniform[::-1])
    difference = _distances(local[1], local[0])
    np.testing.assert_allclose(exact["methods"]["p"]["energy_main"], p[1]**2 * difference,
                               rtol=0., atol=2e-15)
    np.testing.assert_allclose(exact["contrasts"]["p_minus_q"]["energy_repeat"],
                               (p[1]**2-q[1]**2) * difference, rtol=0., atol=2e-15)
    return {"schema": SCHEMA, "native_calls": 0, "analytic_shared_phase_likelihood": True,
            "matched_raw_rbf": True, "exact_finite_pair_enumeration": True}


if __name__ == "__main__":
    print(fixtures())
