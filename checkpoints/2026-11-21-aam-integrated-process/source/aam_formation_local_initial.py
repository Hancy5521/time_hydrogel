"""Explicit mu5 initial variables and prior-correct microscopic initial moves.

Import has no random draws, initialization, propagation, fitting or I/O.  This
module deliberately does not call the finite-seed ``setup``/``initialize``
functions.  The new mu5 law is shared by truth, training and all V5 native arms.
The eight initial MBAA retain their nominal source labels and unwrapped normal
preimages; their wrapped x coordinates alone are not the latent variables.
"""
from __future__ import annotations

import hashlib
import json
from math import lgamma, log, sqrt

import numpy as np

from aam_edge_fork import State
from aam_mbaa_kinetics import Recipe
from aam_snapshot_process import PORTS
from aam_spatial_motion import Motion


SCHEMA = "aam-mbaa-explicit-initial-mu5-v1"
LOCAL_KINDS = ("initiators", "aam_position", "mbaa_position", "side_label", "global")
LABEL_ALPHA = .65
NORMAL_SCALE = .62
POSITION_STEP = .10
PCN_SCALE = .20
FIELDS = ("schema", "I", "left", "normal_x", "points")


def _configuration(config):
    """Validate the fixed V5 recipe without calling physical or RNG code."""
    recipe = Recipe(**config["recipe"])
    motion = Motion(**config["motion"])
    if (recipe.aam, recipe.mbaa, recipe.initial_mbaa) != (64, 14, 8):
        raise ValueError("mu5 is registered for64AAm,14MBAA,8initialMBAA.")
    if motion.box != 6.:
        raise ValueError("mu5 source ports and position moves use the fixed box6.")
    return recipe, motion


def _derived_x(left, normal_x, box):
    return (np.where(left, PORTS[0], PORTS[1])+NORMAL_SCALE*normal_x) % box


def validate_initial(config, initial):
    """Check the numeric-tree identity and derived coordinates; never mutate."""
    recipe, motion = _configuration(config)
    if not isinstance(initial, dict) or set(initial) != set(FIELDS):
        raise TypeError("Use exactly the registered numeric initial-variable tree.")
    if initial["schema"] != SCHEMA:
        raise ValueError("Unknown explicit initial-variable schema.")
    initiators = initial["I"]
    if (isinstance(initiators, (bool, np.bool_))
            or not isinstance(initiators, (int, np.integer))
            or not 6 <= initiators <= 12):
        raise ValueError("I must be an integer in6..12.")
    left, normal_x, points = (initial[name] for name in ("left", "normal_x", "points"))
    if (not isinstance(left, np.ndarray) or left.dtype != np.dtype(np.bool_)
            or left.shape != (recipe.initial_mbaa,)):
        raise TypeError("left must be a bool[8] array; True denotes PORTS[0].")
    if (not isinstance(normal_x, np.ndarray) or normal_x.dtype != np.dtype(np.float64)
            or normal_x.shape != (recipe.initial_mbaa,) or not np.isfinite(normal_x).all()):
        raise TypeError("normal_x must be finite float64[8] normal preimages.")
    if (not isinstance(points, np.ndarray) or points.dtype != np.dtype(np.float64)
            or points.shape != (recipe.aam+recipe.mbaa, 3) or not np.isfinite(points).all()
            or np.any(points < 0.) or np.any(points >= motion.box)):
        raise ValueError("points must be finite float64[78,3] in[0,6).")
    target = _derived_x(left, normal_x, motion.box)
    if not np.array_equal(points[recipe.aam:recipe.aam+recipe.initial_mbaa, 0], target):
        raise ValueError("Initial MBAA x must exactly derive from its retained label and normal preimage.")


def clone_initial(initial):
    """Copy every numeric field; no arrays are shared with the source tree."""
    if not isinstance(initial, dict) or set(initial) != set(FIELDS) or initial["schema"] != SCHEMA:
        raise TypeError("Expected the registered explicit initial-variable tree.")
    return {"schema": SCHEMA, "I": int(initial["I"]),
            "left": np.array(initial["left"], dtype=np.bool_, copy=True),
            "normal_x": np.array(initial["normal_x"], dtype=np.float64, copy=True),
            "points": np.array(initial["points"], dtype=np.float64, copy=True)}


def draw_initial(config, rng):
    """Draw mu5, retaining labels but integrating away their auxiliary Beta P.

    The passed generator is caller-owned.  No hidden seed/substream is created,
    and no unused initial-MBAA x coordinate is drawn or retained.
    """
    recipe, motion = _configuration(config)
    initiators = int(rng.integers(6, 13))
    probability = float(rng.beta(LABEL_ALPHA, LABEL_ALPHA))
    left = np.asarray(rng.random(recipe.initial_mbaa) < probability, dtype=np.bool_)
    normal_x = np.asarray(rng.standard_normal(recipe.initial_mbaa), dtype=np.float64)
    points = np.empty((recipe.aam+recipe.mbaa, 3), dtype=np.float64)
    points[:recipe.aam] = rng.uniform(0., motion.box, (recipe.aam, 3))
    start, stop = recipe.aam, recipe.aam+recipe.initial_mbaa
    points[start:stop, 0] = _derived_x(left, normal_x, motion.box)
    points[start:stop, 1:] = rng.uniform(0., motion.box, (recipe.initial_mbaa, 2))
    points[stop:] = rng.uniform(0., motion.box, (recipe.mbaa-recipe.initial_mbaa, 3))
    initial = {"schema": SCHEMA, "I": initiators, "left": left,
               "normal_x": normal_x, "points": points}
    validate_initial(config, initial)
    return initial


def build_packet(config, initial, factors=None):
    """Construct the existing complete native packet directly, without setup RNG.

    This is initial State construction only, not a reaction or Brownian step.
    Every one of the16State arrays and four parameter arrays is newly owned.
    """
    validate_initial(config, initial)
    recipe, motion = _configuration(config)
    rates = np.asarray(config["rates"], dtype=np.float64)
    factors = np.asarray(config["factors"] if factors is None else factors, dtype=np.float64)
    corridor = float(config["corridor"])
    if (rates.shape != (8, 2) or factors.shape != (3, 5)
            or not np.isfinite(rates).all() or not np.isfinite(factors).all()
            or np.any(rates < 0.) or np.any(factors <= 0.)
            or not np.isfinite(corridor) or corridor <= 0.):
        raise ValueError("Invalid native rate/factor/corridor parameters.")
    n = recipe.aam+recipe.mbaa
    initiators = int(initial["I"])
    slots = recipe.aam+2*recipe.mbaa+initiators//2+recipe.mbaa+1
    points = initial["points"].copy()
    present = np.arange(n, dtype=np.int64) < recipe.aam+recipe.initial_mbaa
    available = np.zeros(2*n, dtype=np.bool_)
    available[:2*recipe.aam:2] = True
    available[2*recipe.aam:2*(recipe.aam+recipe.initial_mbaa)] = True
    state = State(
        points, points.copy(), present, np.zeros(2*n, dtype=np.bool_), available,
        np.full(2*n, -1, dtype=np.int64), np.arange(n, dtype=np.int64),
        np.zeros((slots, 2), dtype=np.int64), np.zeros((slots, 3), dtype=np.int64),
        np.zeros((slots, 13), dtype=np.float64), np.arange(n, dtype=np.int64),
        np.zeros((n, 3), dtype=np.int64), np.zeros(n, dtype=np.int64),
        np.zeros(3, dtype=np.bool_),
        np.asarray([0, 0, 0, initiators, 0, 0, 0], dtype=np.int64),
        np.zeros(4, dtype=np.float64))
    material = np.asarray((recipe.aam, recipe.mbaa, recipe.initial_mbaa, recipe.feed_time),
                          dtype=np.float64)
    geometry = np.asarray((motion.step, motion.diffusion, motion.box, motion.spring, motion.rest,
                           motion.repulsion, motion.core, motion.capture, motion.inner, corridor),
                          dtype=np.float64)
    return state, material, geometry, rates.copy(), factors.copy()


def label_log_mass(left_count, count=8):
    """Log mass of a *specific* collapsed label vector; no binomial factor."""
    if (isinstance(left_count, (bool, np.bool_))
            or not isinstance(left_count, (int, np.integer))
            or not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_))
            or count < 0 or not 0 <= left_count <= count):
        raise ValueError("Invalid collapsed label counts.")
    alpha = LABEL_ALPHA
    return (lgamma(alpha+left_count)+lgamma(alpha+count-left_count)-lgamma(2*alpha+count)
            -2*lgamma(alpha)+lgamma(2*alpha))


def side_flip_log_ratio(left_count, old_left, count=8):
    """Prior/proposal log ratio for a symmetric single labelled-source flip."""
    label_log_mass(left_count, count)
    if not isinstance(old_left, (bool, np.bool_)):
        raise TypeError("old_left must be a Boolean source label.")
    remaining_left = int(left_count)-int(old_left)
    if not 0 <= remaining_left <= count-1:
        raise ValueError("The flipped labelled entry is inconsistent with the supplied count.")
    log_right_to_left = log(LABEL_ALPHA+remaining_left)-log(LABEL_ALPHA+count-1-remaining_left)
    return -log_right_to_left if old_left else log_right_to_left


def _metadata(kind, old, new, selected, *, step=0):
    return {"schema": "aam-mu5-initial-proposal-v1", "kind": kind,
            "selected_indices": np.asarray(selected, dtype=np.int64).copy(),
            "old_I": int(old["I"]), "new_I": int(new["I"]),
            "old_nL": int(np.count_nonzero(old["left"])),
            "new_nL": int(np.count_nonzero(new["left"])), "initiator_step": int(step),
            "self_loop": initial_fingerprint(old) == initial_fingerprint(new)}


def propose_initial(config, initial, rng, *, mode="local", kind=None):
    """Return candidate, log[mu(new)q(old|new)/mu(old)q(new|old)], metadata.

    The return is a proposal, not an accepted update.  In particular a label
    flip still requires its nonzero collapsed prior ratio at beta=0.  All
    non-self proposals must rebuild and replay the entire prefix externally.
    """
    validate_initial(config, initial)
    recipe, motion = _configuration(config)
    if mode not in ("local", "global"):
        raise ValueError("Initial proposal mode must be local or global.")
    if mode == "global":
        if kind not in (None, "global"):
            raise ValueError("The global-only arm cannot request a local move.")
        kind = "global"
    elif kind is None:
        kind = LOCAL_KINDS[int(rng.integers(len(LOCAL_KINDS)))]
    if kind not in LOCAL_KINDS:
        raise ValueError("Unknown registered initial move kind.")
    if kind == "global":
        candidate = draw_initial(config, rng)
        return candidate, 0., _metadata(kind, initial, candidate, [])
    candidate = clone_initial(initial)
    ratio, step = 0., 0
    if kind == "initiators":
        step = -1 if int(rng.integers(2)) == 0 else 1
        proposed = int(initial["I"])+step
        if 6 <= proposed <= 12:
            candidate["I"] = proposed
        selected = []
    elif kind == "aam_position":
        selected = np.asarray(rng.choice(recipe.aam, size=4, replace=False), dtype=np.int64)
        candidate["points"][selected] = (candidate["points"][selected]
                                          +rng.uniform(-POSITION_STEP, POSITION_STEP, (4, 3))) % motion.box
    elif kind == "mbaa_position":
        j = int(rng.integers(recipe.initial_mbaa))
        selected = [recipe.aam+j]
        candidate["normal_x"][j] = (sqrt(1.-PCN_SCALE**2)*candidate["normal_x"][j]
                                     +PCN_SCALE*float(rng.standard_normal()))
        candidate["points"][recipe.aam+j, 0] = _derived_x(
            candidate["left"][j:j+1], candidate["normal_x"][j:j+1], motion.box)[0]
        candidate["points"][recipe.aam+j, 1:] = (candidate["points"][recipe.aam+j, 1:]
                                                +rng.uniform(-POSITION_STEP, POSITION_STEP, 2)) % motion.box
    else:  # side_label, chosen independently of the current label values
        j = int(rng.integers(recipe.initial_mbaa))
        selected = [recipe.aam+j]
        ratio = side_flip_log_ratio(int(np.count_nonzero(initial["left"])), initial["left"][j])
        candidate["left"][j] = not candidate["left"][j]
        candidate["points"][recipe.aam+j, 0] = _derived_x(
            candidate["left"][j:j+1], candidate["normal_x"][j:j+1], motion.box)[0]
    validate_initial(config, candidate)
    return candidate, float(ratio), _metadata(kind, initial, candidate, selected, step=step)


def refresh_stock_initial(config, initial, rng):
    """Exact prior refresh of only the six absent MBAA coordinates.

    This returns variables only.  The path layer must separately copy/sync the
    absent coordinates in all actual packets/frames and verify readout identity.
    """
    validate_initial(config, initial)
    recipe, motion = _configuration(config)
    refreshed = clone_initial(initial)
    start = recipe.aam+recipe.initial_mbaa
    indices = np.arange(start, recipe.aam+recipe.mbaa, dtype=np.int64)
    refreshed["points"][indices] = rng.uniform(0., motion.box, (len(indices), 3))
    validate_initial(config, refreshed)
    metadata = _metadata("stock_refresh", initial, refreshed, indices)
    metadata.update({"schema": "aam-mu5-absent-stock-refresh-v1", "native_calls": 0,
                     "informative_fingerprint_before": initial_fingerprint(initial, informative_only=True),
                     "informative_fingerprint_after": initial_fingerprint(refreshed, informative_only=True),
                     "scope": "Absent initial coordinates only; not evidence of early structure recovery."})
    return refreshed, metadata


def initial_fingerprint(initial, *, informative_only=False):
    """Typed identity, not a physical-mode count or identifiability certificate."""
    if not isinstance(initial, dict) or set(initial) != set(FIELDS) or initial["schema"] != SCHEMA:
        raise TypeError("Expected the registered explicit initial-variable tree.")
    h = hashlib.sha256()
    h.update((SCHEMA+"|"+("informative" if informative_only else "complete")).encode("ascii"))
    arrays = (("I", np.asarray(initial["I"], dtype=np.int64)),
              ("left", np.asarray(initial["left"], dtype=np.bool_)),
              ("normal_x", np.asarray(initial["normal_x"], dtype=np.float64)),
              ("points", np.asarray(initial["points"], dtype=np.float64)[:72]
               if informative_only else np.asarray(initial["points"], dtype=np.float64)))
    for name, array in arrays:
        h.update(name.encode("ascii"))
        h.update(array.dtype.str.encode("ascii"))
        h.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()
