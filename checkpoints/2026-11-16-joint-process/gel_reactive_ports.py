"""Conservative two-port finite-volume model for irreversible crosslinking."""

import numpy as np
from scipy.fft import dct, idct


def schedules(settings):
    """Same total input and same input at each port integrated over time."""
    output = [{"early_fraction": .5, "early_split": 0., "late_split": 0.}]
    for fraction in settings["early_fractions"]:
        for split in settings["early_splits"]:
            output.append({"early_fraction": fraction, "early_split": split,
                           "late_split": -split * fraction / (1 - fraction)})
    return output


def local_binding(free, bound, rate, capacity, duration):
    """Exact local update for c + sites -> bound; all arrays are cellwise."""
    sites = capacity * (1 - bound)
    small = np.minimum(free, sites)
    difference = np.abs(free - sites)
    kernel = duration * rate / capacity
    exponent = -kernel * difference
    integral = np.divide(-np.expm1(exponent), difference,
                         out=np.broadcast_to(kernel, difference.shape).copy(), where=difference > 0)
    remaining = small * np.exp(exponent) / (1 + small * integral)
    transfer = small - remaining
    return free - transfer, bound + transfer / capacity


def transport_coefficients(cells, diffusivity, step):
    spatial = 2. * cells**2 * (np.cos(np.pi * np.arange(cells) / cells) - 1.)
    rates = diffusivity[..., None] * spatial
    exact = np.exp(rates * step)
    gains = np.divide(np.expm1(rates * step), rates, out=np.full_like(rates, step), where=rates != 0)
    return exact, gains


def prepare_step(cells, diffusivity, step, feed, split):
    exact, gains = transport_coefficients(cells, diffusivity, step)
    sources = np.zeros((*np.shape(split), cells))
    sources[..., 0] = cells * feed * (1 + split) / 2
    sources[..., -1] = cells * feed * (1 - split) / 2
    return exact, gains * dct(sources, type=2, norm="ortho", axis=-1)


def advance(free, bound, rate, capacity, operators, dt):
    free, bound = local_binding(free, bound, rate, capacity, dt / 2)
    exact, add = operators
    modes = exact * dct(free, type=2, norm="ortho", axis=-1) + add
    free = idct(modes, type=2, norm="ortho", axis=-1)
    if free.min() < -1e-9:
        raise ArithmeticError("Negative free reagent in diffusion step.")
    return local_binding(np.maximum(free, 0), bound, rate, capacity, dt / 2)


def material_fields(parameters, coordinates):
    """Parameters are D, global rate, capacity, signed gradient, curvature."""
    d, global_rate, capacity, gradient, curvature = np.asarray(parameters).T
    centered = 2 * coordinates - 1
    rate = global_rate[:, None] * np.exp(gradient[:, None] * centered + curvature[:, None] * (centered**2 - 1 / 3))
    return d, rate, capacity[:, None]


def bank(parameters, plans, cfg, dose_rate, *, cells=None, dt=None):
    cells = cfg["cells"] if cells is None else cells
    dt = cfg["step"] if dt is None else dt
    times = np.arange(round(cfg["final_time"] / dt) + 1) * dt
    prefix_count, fed_count = (round(cfg[key] / dt) for key in ("prefix_duration", "feed_duration"))
    coordinates = (np.arange(cells) + .5) / cells
    n, a = len(parameters), len(plans)
    d, rate, capacity = material_fields(parameters, coordinates)
    prefix = prepare_step(cells, d, dt, cfg["prefix_feed"], np.zeros(n))
    early = prepare_step(cells, d[:, None], dt, dose_rate, np.broadcast_to([s["early_split"] for s in plans], (n, a)))
    late = prepare_step(cells, d[:, None], dt, dose_rate, np.broadcast_to([s["late_split"] for s in plans], (n, a)))
    sealed = prepare_step(cells, d[:, None], dt, 0., np.zeros((n, a)))
    early_count = np.array([round(fed_count * s["early_fraction"]) for s in plans])
    if any(abs(fed_count * s["early_fraction"] - q) > 1e-10 for s, q in zip(plans, early_count, strict=True)):
        raise ValueError("The control clock must align with split changes.")
    bounds = [((coordinates >= interval[0]) & (coordinates <= interval[1])) for interval in cfg["regions"]]
    if any(not region.any() for region in bounds):
        raise ValueError("A target region has no cells.")
    free = np.zeros((n, cells))
    bound = np.zeros_like(free)
    for _ in range(prefix_count):
        free, bound = advance(free, bound, rate, capacity, prefix, dt)
    observed = np.stack([np.array([np.interp(x, coordinates, row) for row in field])
                         for field in (free, bound) for x in cfg["probe_locations"]], axis=1)
    free = np.broadcast_to(free[:, None, :], (n, a, cells)).copy()
    bound = np.broadcast_to(bound[:, None, :], (n, a, cells)).copy()
    offset = 0
    max_balance = 0.
    trajectory = np.empty((n, a, len(times) - prefix_count, 2))
    trajectory[:, :, 0] = np.stack([bound[..., mask].mean(axis=-1) for mask in bounds], axis=-1)
    for i in range(prefix_count, len(times) - 1):
        if offset < fed_count:
            use_early = offset < early_count
            gains = np.where(use_early[None, :, None], early[1], late[1])
            operator = early[0], gains
        else:
            operator = sealed
        free, bound = advance(free, bound, rate[:, None, :], capacity[:, None], operator, dt)
        offset += 1
        total_input = cfg["prefix_feed"] * cfg["prefix_duration"] + min(offset, fed_count) * dt * dose_rate
        mass = (free + capacity[:, None] * bound).mean(axis=-1)
        max_balance = max(max_balance, float(np.max(np.abs(mass - total_input))))
        trajectory[:, :, offset] = np.stack([bound[..., mask].mean(axis=-1) for mask in bounds], axis=-1)
    return times[prefix_count:], observed, trajectory, max_balance


def first_crossings(times, paths, level):
    hits = paths >= level
    exists = hits.any(axis=-2)
    hi = np.clip(hits.argmax(axis=-2), 1, len(times) - 1)
    before = np.take_along_axis(paths, (hi - 1)[..., None, :], axis=-2).squeeze(axis=-2)
    after = np.take_along_axis(paths, hi[..., None, :], axis=-2).squeeze(axis=-2)
    portion = np.divide(level - before, after - before, out=np.zeros_like(before), where=after > before)
    return np.where(exists, np.where(hits[..., 0, :], times[0], times[hi - 1] + portion * (times[hi] - times[hi - 1])), np.inf)
