"""Enumerate an independent finite conditional-particle-kernel fixture exactly."""

from argparse import ArgumentParser
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
from math import prod


BITS = (0, 1)
WINDOW_VALUES = (1, -2, 4)
HISTORIES = tuple(product(BITS, repeat=4))
MOTHERS = tuple(product(BITS, repeat=2))
HALF = Fraction(1, 2)
MODES = ("conditional_pg", "full_suffix_pgas", "one_step_ancestor")


def state(path):
    """The sign and amplitude of future increments depend on prior tape bits."""
    initial, *innovations = path
    position = initial
    count = initial
    for bit in innovations:
        position += (2*bit-1)*(1+count)
        count += bit
    return position


@lru_cache(maxsize=None)
def likelihood(prefix):
    """Use positive rational measurement factors with nonconstant weights."""
    stage = len(prefix)-2
    return Fraction(1, 2+abs(state(prefix)-WINDOW_VALUES[stage]))


def normalized(weights):
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def parents(ensemble, reference, mode):
    if mode == "conditional_pg":
        return (Fraction(0), Fraction(1))
    cut = len(ensemble[0])
    end = cut+1 if mode == "one_step_ancestor" else len(reference)
    factors = []
    for prefix in ensemble:
        path = prefix + reference[cut:]
        factors.append(prod(likelihood(path[:j]) for j in range(cut+1, end+1)))
    return normalized([likelihood(p)*f for p, f in zip(ensemble, factors, strict=True)])


def update(ensemble, mass, reference, mode):
    cut = len(ensemble[0])
    free = normalized([likelihood(prefix) for prefix in ensemble])
    conditional = parents(ensemble, reference, mode)
    for i, j, bit in product(BITS, repeat=3):
        conditional_mass = mass * free[i] * conditional[j] * HALF
        if conditional_mass:
            yield (ensemble[i]+(bit,), ensemble[j]+(reference[cut],)), conditional_mass


def row(reference, mode):
    """List probabilities of every possible Markov successor of reference."""
    result = dict.fromkeys(HISTORIES, Fraction(0))
    for first in MOTHERS:
        ensemble = (first, reference[:2])
        for second, step_mass in update(ensemble, Fraction(1, 4), reference, mode):
            for third, branch_mass in update(second, step_mass, reference, mode):
                final = normalized([likelihood(path) for path in third])
                for path, mass in zip(third, final, strict=True):
                    result[path] += branch_mass * mass
    return tuple(result.values())


def independent_target():
    """Enumeration under four independent Bernoulli tapes and three factors."""
    return normalized([Fraction(1,16)*prod(likelihood(path[:j]) for j in (2,3,4))
                       for path in HISTORIES])


def run_fixture():
    assert state((0,0)) == state((1,0)) == -1
    assert state((0,0,1)) != state((1,0,1))
    target = independent_target()
    assert sum(target) == 1 and all(p > 0 for p in target)
    assert len(set(target)) > 1
    witness = {}
    for mode in MODES:
        kernel = [row(path, mode) for path in HISTORIES]
        # Positive rows also rule out a degenerate identity-only invariance.
        assert all(sum(r) == 1 and all(e > 0 for e in r) for r in kernel)
        diff = [sum(p*k for p, k in zip(target, column, strict=True)) - original
                for column, original in zip(zip(*kernel, strict=True), target, strict=True)]
        tv = sum(abs(value) for value in diff) / 2
        witness[mode] = {"preserves_exact_target": tv == 0, "stationarity_tv": float(tv)}
        if mode == "one_step_ancestor":
            assert tv > 0
            index = max(range(len(diff)), key=lambda i: abs(diff[i]))
            witness[mode].update(witness_tape=HISTORIES[index], exact_witness_delta=str(diff[index]))
        else:
            assert tv == 0
    return {"states":len(HISTORIES), "particles":2, "stages":3, "native_calls":0,
            "uses_material_observations":False, "visible_state_is_not_markov":True,
            "arithmetic":"exact fractions",
            "passed":True, "results":witness}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run_fixture(), indent=2))
