"""Independently check the shared-atlas forecast, controls and paired decision."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from aam_formation_actual_rollout import packet_fingerprint
from aam_formation_cycle_driver import np_load, read, sha, write
from aam_formation_path_cycle import load_tree, stamp
from audit_aam_amortized_continuous_inverse import (
    GOALS, TIMES, bound, close, exact, inspect_choice, need, risk_one, source_integrity, time, verify_prefixes,
)
from audit_aam_continuous_inverse import goal_loss
from audit_aam_endpoint_tape_move import rng
from audit_aam_large_phase_atlas import exact_neighbor_score, hash_weights, independent_energy, native_weights
from audit_aam_path_noninferiority import (
    METHODS as PATH_METHODS, METRICS as PATH_METRICS, PAIR_DRAWS, PAIR_SEED,
    digest_valid, independent_ess_law, independently_bound,
)
from audit_aam_stationary_distillation import baseline_laws, lens


UNITS = list(range(2026163000, 2026163512))
ANCHORS = (0, 73, 146, 219, 292, 365, 438, 511)
ALPHA = .025
INVERSE_METHODS = ("ess1024_cophase", "ess1024_uniform_phase", "old_population", "fixed")
PATH_MARGINS = (.01, 0., .005, .02)
INVERSE_MARGINS = (.05, 0., .01, .02, .03)
BANK = np.arange(2026145000, 2026153192)
NAMESPACE = "aam-original-recovery-inverse-same-atlas-ess1024-20261121"


def verify_family(losses, report, methods, metrics, margins):
    n = len(losses)
    count = (len(methods) - 1) * len(metrics)
    critical = t.isf(ALPHA / count, n-1)
    need(report["mothers"] == n and report["family_comparisons"] == count
         and report["candidate"] == methods[0] and tuple(report["references"]) == methods[1:]
         and tuple(report["metrics"]) == metrics and report["alpha"] == ALPHA,
         "Bonferroni comparison family differs from registration.")
    close(report["critical_one_sided_bonferroni_t"], critical)
    cases = [report["primary"], *report["sensitivity"]]
    exact([row["margin"] for row in cases], margins)
    comparison = losses[:, :1] - losses[:, 1:]
    raw_mean, raw_sd, raw_upper = independently_bound(comparison, critical)
    pointwise = t.isf(ALPHA/2, n-1)
    for j, ref in enumerate(methods[1:]):
        for k, metric in enumerate(metrics):
            reported = report["comparisons"][ref][metric]
            close(reported["candidate_mean"], losses[:, 0, k].mean())
            close(reported["reference_mean"], losses[:, j+1, k].mean())
            close(reported["raw_candidate_minus_reference"], raw_mean[j,k])
            close(reported["raw_paired_sd"], raw_sd[j,k])
            close(reported["raw_simultaneous_upper"], raw_upper[j,k])
            half = pointwise * raw_sd[j,k] / np.sqrt(n)
            close(reported["raw_pointwise_two_sided_interval"], raw_mean[j,k] + [-half, half])
    for margin, case in zip(margins, cases, strict=True):
        mu, sd, upper = independently_bound(comparison - margin*losses[:, 1:], critical)
        for j, ref in enumerate(methods[1:]):
            for k, metric in enumerate(metrics):
                reported = case["comparisons"][ref][metric]
                close(reported["shifted_mean"], mu[j,k])
                close(reported["shifted_paired_sd"], sd[j,k])
                close(reported["simultaneous_upper"], upper[j,k])
                need(reported["passed"] is bool(upper[j,k] < 0.), "Individual noninferiority gate differs.")
        need(case["passed"] is bool(np.all(upper < 0.)), "Conjunctive family gate differs.")
    return report["primary"]["passed"]


def audit_references(out, reg, reference, choices, prefix, freeze_hash):
    rows = reference["rows"]
    exact([row["unit"] for row in rows], UNITS)
    losses, blocks = [], []
    calls = 0
    goal_indices = np.arange(len(GOALS))
    for row, choice, packet in zip(rows, choices, prefix, strict=True):
        folder = out/"reference"/str(row["unit"])
        record = read(bound(row["reference"], folder/"COMPLETE.json"))
        need(record["unit"] == row["unit"] and record["freeze_sha256"] == freeze_hash
             and record["new_c3_identity"] == packet_fingerprint(packet),
             "The reference does not descend from the registered frozen mother.")
        data = np_load(bound(record["reference"], folder/"REFERENCE_VALUES.npz"))
        expected = np.asarray(sorted(set(map(tuple,choice.reshape(-1,2)))))
        exact(data["choices"], choice)
        exact(data["controls"], expected)
        exact(expected[data["mapping"]], choice)
        need(data["values"].shape == (24, len(expected), 3, 6) and len(record["branches"]) == 24,
             "Every distinct frozen control requires 24 common-random-stream futures.")
        count = 24 * len(expected) * 3
        need(record["native_calls"] == row["native_calls"] == count and time(record) <= time(reference),
             "Native budget differs from deducting identical controls.")
        for k, branch in enumerate(record["branches"]):
            path = folder/f"R{k}_RAW.npz"
            need(Path(branch["path"]).resolve() == path.resolve() and path.stat().st_size == branch["bytes"],
                 "A reference continuation ledger is missing or detached.")
        calls += count
        loss, components = goal_loss(data["values"], reg["normalization"]["mechanical_scale"])
        losses.append(loss[:,data["mapping"],goal_indices].mean(axis=0))
        blocks.append(components[:,data["mapping"],goal_indices,:].mean(axis=0))
    return np.asarray(losses), np.asarray(blocks), calls


def validate_cache(out, cache, reg):
    """Bind the precise selected mother and phase of each packed loss coefficient."""
    bank = reg["bank"]
    exact(reg["bank_ids"], BANK)
    ordered = np.sort(np.random.default_rng(np.random.SeedSequence([20261121,43541,8192,384]))
                      .choice(8192,size=384,replace=False))
    exact(bank["selected_rows"], ordered)
    expected = BANK[ordered]
    exact(reg["route_ids"], expected)
    exact([r["unit"] for r in bank["selected_records"]], expected)
    rows = cache["rows"]
    exact([r["unit"] for r in rows], expected)
    exact([r["row"] for r in rows], ordered)
    compact = []
    for source, row in zip(bank["selected_records"], rows, strict=True):
        need(source["raw_block"] == row["original_raw_block"] and row["native_calls"] == 126,
             "Cached risk is not attached to the original parent.")
        p = out/"cache"/f'{row["unit"]}_PHASE_RISKS.npz'
        data = np_load(bound(row["compact"], p))
        exact(data["unit"], row["unit"])
        exact(data["row"], row["row"])
        raw = np_load(bound(row["raw"], out/"cache"/f'{row["unit"]}_RAW.npz'))
        exact(raw["origin"].item(), row["c3_sha256"])
        existing, _ = goal_loss(raw["values"].reshape(6,7,3,6), reg["normalization"]["mechanical_scale"])
        close(existing, data["losses"][0], atol=2e-11)
        compact.append(data["losses"])
    compact = np.asarray(compact)
    declared = np_load(bound(cache["arrays"], out/"CACHE_LOSSES.npz"))
    exact(declared["units"], expected)
    exact(declared["losses"], compact)
    return ordered,compact


def anchor_check(query, unit, generating_raw, descriptors, full_joint, bank, old, settings,
                 route_positions, phase_losses, choices, path_replicas, physics):
    """An independent query law must feed both original readback and cached planning."""
    oracle, temperature = independent_ess_law(bank, old, query)
    close(full_joint.ravel(), oracle, atol=2e-10)
    need(hash_weights(full_joint.ravel()) == descriptors["ess1024"],
         "Shared anchor mass does not have the original recovery digest.")
    weights = native_weights(bank, query, old["noise"], old["camera_scale"],
                             old["kernel_sq_scale"], settings["kernel"])
    weights.update({key: val[0] for key,val in baseline_laws(old, {"atlas_settings":settings},query[None]).items()})
    weights["ess1024"] = oracle
    truth = lens(generating_raw["packets"], physics, {})[1][0]
    pairs = [rng(PAIR_SEED,unit,k).random((PAIR_DRAWS,2)) for k in range(2)]
    for j, label in enumerate(PATH_METHODS):
        if j < 3:
            # Independent weights are close in floating point; frozen ESS mass
            # keeps the exact draws used by the original query score.
            w = full_joint.ravel() if j == 0 else weights[label]
            estimate = np.asarray([independent_energy(bank["local"], w, truth[None],u)[0] for u in pairs])
        else:
            value = exact_neighbor_score(old["bank"]["local"], weights[label], truth[None])
            estimate = np.broadcast_to(value, (2,2))
        close(estimate, path_replicas[:, j], atol=2.e-10)
    restricted = oracle.reshape(8192,16)[route_positions]
    total = restricted.sum()
    restricted /= total
    uniform = np.broadcast_to(restricted.sum(axis=1,keepdims=True)/16, restricted.shape)
    for j,w in enumerate((restricted, uniform)):
        risks = np.tensordot(w, phase_losses, axes=([0,1],[0,1]))
        # Read each action risk via independently convolved Bernoulli cells,
        # and compare with polynomial stationary points over all six times.
        chosen = choices[j]
        inspect_choice(risks, chosen, [risk_one(risks,a,g) for g,a in enumerate(chosen)])
    return {"ess_exponent": temperature, "retained_joint_mass": float(total),
            "restricted_mother_ess": float(1./np.sum(np.square(restricted.sum(axis=1))))}


def audit(out):
    names = ("REGISTERED","RUN","CACHE_STARTED","CACHE_CLOSED","ACQUIRED","FORECAST_DESCRIPTORS",
             "PREDICTIONS_FROZEN","REVEAL_STARTED","REFERENCES_CLOSED","SCORED")
    files = {k:out/f"{k}.json" for k in names}
    reg,start,cache_start,cache,acq,desc,freeze,reveal,refs,score = map(read,files.values())
    moments = list(map(time,(reg,start,cache_start,cache,acq,desc,freeze,reveal,refs,score)))
    need(all(a <= b for a,b in zip(moments,moments[1:]))
         and moments[6] < moments[8], "Prediction freeze must predate reference generation.")
    need(reg["namespace"] == NAMESPACE and reg["units"] == UNITS
         and tuple(reg["path_methods"]) == PATH_METHODS and tuple(reg["path_metrics"]) == PATH_METRICS
         and tuple(reg["inverse_methods"]) == INVERSE_METHODS and tuple(reg["goals"]) == GOALS
         and reg["alpha"] == ALPHA and reg["primary_path_margin"] == PATH_MARGINS[0]
         and reg["primary_inverse_margin"] == INVERSE_MARGINS[0] and tuple(reg["anchor_positions"]) == ANCHORS
         and reg["reference_repeats"] == 24 and reg["new_fits"] == reg["wet_measurements"] == 0
         and reg["legacy_outcome_payloads_opened"] == 0 and start["before_native"]
         and not freeze["held_truth_opened"] and freeze["native_calls"] == score["native_planning"] == 0,
         "Original task, interfaces or analytical family differs from the prospective registration.")
    old_reg = reg["old_plan"]
    need(old_reg["physics"] == reg["physics"] and old_reg["bank_units"] == reg["bank_ids"]
         and not set(UNITS).intersection(old_reg["units"]+old_reg["old_atlas_units"]+reg["bank_ids"]),
         "The shared bank or fresh test identities overlap the registered prior tasks.")
    exact(reg["times"], TIMES)
    h = {k:sha(f) for k,f in files.items()}
    exact([r["registration_sha256"] for r in (start,cache_start,acq,freeze,score)], [h["REGISTERED"]]*5)
    need(cache["started_sha256"] == h["CACHE_STARTED"] and acq["cache_sha256"] == h["CACHE_CLOSED"]
         and freeze["cache_sha256"] == h["CACHE_CLOSED"] and freeze["acquisition_sha256"] == h["ACQUIRED"]
         and all(r["freeze_sha256"] == h["PREDICTIONS_FROZEN"] for r in (reveal,refs,score))
         and score["reference_sha256"] == h["REFERENCES_CLOSED"],
         "A frozen path or future reference is not bound to the registered complete acquisition.")
    bound(freeze["descriptors"], files["FORECAST_DESCRIPTORS"])
    source_integrity(reg,out/"SOURCE_SNAPSHOT")
    route_positions, phase_losses = validate_cache(out,cache,reg)
    full = np_load(bound(freeze["anchors"],out/"ANCHORS.npz"))
    exact(full["units"], [UNITS[j] for j in ANCHORS])
    need(full["full_joint"].shape == (8,8192,16)
         and np.isfinite(full["full_joint"]).all() and (full["full_joint"] >= 0.).all(),
         "Every physical anchor must retain all 8,192 mothers and 16 phase weights.")
    plan = np_load(bound(freeze["choices"],out/"CHOICES.npz"))
    packed = np_load(bound(score["arrays"],out/"SCORED_ARRAYS.npz"))
    exact(plan["units"], UNITS)
    exact(packed["units"], UNITS)
    exact(packed["choices"], plan["choices"])
    m = len(UNITS)
    choices = plan["choices"]
    replicas, path_losses = packed["replica"],packed["path_losses"]
    need(choices.shape == (m,4,2,2) and np.isfinite(choices).all()
         and np.isin(choices[...,0],TIMES).all()
         and ((choices[...,1] >= 0.) & (choices[...,1] <= 1.)).all()
         and plan["query"].shape == (m,3,4) and np.isfinite(plan["query"]).all()
         and replicas.shape == (m,2,7,2) and np.isfinite(replicas).all()
         and (replicas >= -1.e-13).all(), "A mother, legal control or original score is missing.")
    exact(choices[:,2],np.broadcast_to(reg["population_controls"],(m,2,2)))
    exact(choices[:,3],np.broadcast_to(reg["fixed_controls"],(m,2,2)))
    exact(replicas[:,0,3:],replicas[:,1,3:])
    close(replicas.mean(axis=1),path_losses)
    rows = desc["rows"]
    exact([r["unit"] for r in rows],UNITS)
    for acquired,r in zip(acq["held"],rows,strict=True):
        need(r["obs_sha256"] == acquired["query"]["sha256"] and set(r["digests"]) == set(PATH_METHODS)
             and all(digest_valid(v) for v in r["digests"].values()),
             "One of the seven recovery laws lost its query-only prediction receipt.")
    prefix = verify_prefixes(out/"held",acq["held"],UNITS,plan["query"],reg["physics"])
    losses, blocks, native = audit_references(out,reg,refs,choices,prefix,h["PREDICTIONS_FROZEN"])
    close(losses,packed["losses"])
    close(blocks,packed["blocks"])
    need(native == refs["native_calls"] == score["native_reference"] <= reg["simulation_reference_bound"]
         and len(UNITS) == score["mothers"] and acq["native_calls"] == score["native_acquisition"] == 3*m
         and score["cache_native"] == cache["native_calls"] == 384*126 and score["new_wet_measurements"] == 0,
         "Physical execution cost or fresh confirmation unit was miscounted.")
    bank_file = bound(reg["bank"]["bank_arrays"], Path(old_reg["bank_arrays"]))
    bank = np_load(bank_file)
    exact(bank["unit"], BANK)
    old_path = old_reg["old_model"]
    old = load_tree(old_path,reg["inputs"][old_path])
    details = {}
    for k,j in enumerate(ANCHORS):
        raw = load_tree(bound(acq["held"][j]["raw"],out/"held"/f"{UNITS[j]}_PREFIX.npz"))
        details[str(UNITS[j])] = anchor_check(plan["query"][j],UNITS[j],raw,rows[j]["digests"],
                full["full_joint"][k],bank,old,old_reg["old_settings"],route_positions,
                phase_losses,choices[j],replicas[j],reg["physics"])
    path_gate = verify_family(path_losses,score["path_ni"],PATH_METHODS,PATH_METRICS,PATH_MARGINS)
    inverse_gate = verify_family(losses,score["inverse_ni"],INVERSE_METHODS,GOALS,INVERSE_MARGINS)
    need(score["primary_gate"] is (path_gate and inverse_gate), "The joint primary gate differs.")
    return {**stamp(),"passed":True,"registration_sha256":h["REGISTERED"],"scored_sha256":h["SCORED"],
            "freeze_sha256":h["PREDICTIONS_FROZEN"],"reference_sha256":h["REFERENCES_CLOSED"],
            "mothers":m,"family_alpha":ALPHA,"path_gate":path_gate,"inverse_gate":inverse_gate,
            "primary_gate":score["primary_gate"],"independent_control_and_recovery_anchor_receipts":details,
            "all_384_cache_readouts_bound":True,"all_512_inverse_losses_recomputed":True,
            "all_18_primary_and_sensitivity_bounds_recomputed":True,
            "native_calls":0,"new_fits":0,"new_wet_measurements":0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    with threadpool_limits(limits=1):
        out = parser.parse_args().out.resolve(strict=True)
        write(out/"INDEPENDENT_AUDIT.json",audit(out))
        print("PASS independent shared-atlas readback", out)
