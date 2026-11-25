"""Work-fixed V3 microscopic history summaries, scales and empirical supports.

Import is inert.  No initialization, physical advance, fit, RNG or file write
occurs.  The eight readouts are fixed before V3: they describe existing States,
not SEC masses, unique full-graph recovery or measured gel points.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np


SCHEMA = "aam-mbaa-work-fixed-eight-micro-history-v3"
EARLY_TIMES = (.65, 1.35, 2.05)
FUTURE_TIMES = (3.25, 4.85, 7.20)
NAMES = ("AAm_lineage_Ln_over64", "AAm_lineage_Lw_over64",
         "largest_reacted_component_nodes_over78", "multigraph_cycle_rank_over14",
         "cumulative_tag5_over14", "cumulative_tag6_over14", "cumulative_tag9_over14",
         "arrived_MBAA_one_used_vinyl_over14")
SCALE_FLOOR = .05


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def extract_state(state):
    """Return (micro8,winding3) from an independent copy of the whole State.

    Ln/Lw use positive AAm-only lineage lengths L: Ln=sum(L)/len(L),
    Lw=sum(L**2)/sum(L). Termination never merges the stored owners.
    Largest reacted component counts reacted molecules, not mass; unreacted
    isolates are excluded. Beta=E-V+C includes every actual parallel edge;
    adding unreacted isolates to V and C cancels. Tags retain event-time meaning.
    Pendant reserve counts present MBAA with exactly one owned vinyl site.
    """
    from aam_snapshot_process import clone
    from aam_formation_cycle_data import check_actual_state
    copied = clone(state)
    for array in copied:
        array.flags.writeable = False
    check = check_actual_state(copied)
    if len(copied.points) != 78 or np.asarray(copied.winding).shape != (3,):
        raise ValueError("Expected the frozen64AAm/14MBAA native State.")
    owner = np.asarray(copied.owner).reshape(78, 2)
    aam_owner = owner[:64, 0]
    _, lengths = np.unique(aam_owner[aam_owner >= 0], return_counts=True)
    if len(lengths):
        ln = float(lengths.mean())
        lw = float(np.sum(lengths.astype(float)**2)/lengths.sum())
    else:
        ln = lw = 0.
    groups = np.asarray(check["groups"])
    present = np.asarray(copied.present)
    reacted = (owner >= 0).any(axis=1) & present
    largest = int(np.bincount(groups[reacted]).max()) if reacted.any() else 0
    components = len(np.unique(groups[present]))
    beta = int(copied.counts[1])-int(present.sum())+components
    if beta < 0:
        raise ValueError("Invalid multigraph cycle rank.")
    tags = copied.events[:int(copied.counts[0]), 1].astype(np.int64)
    count = np.bincount(tags, minlength=10)
    pendant = int(np.count_nonzero(present[64:] & ((owner[64:] >= 0).sum(axis=1) == 1)))
    micro = np.asarray([ln/64., lw/64., largest/78., beta/14.,
                        count[5]/14., count[6]/14., count[9]/14., pendant/14.], dtype=float)
    winding = np.asarray(copied.winding, dtype=bool).copy()
    if not np.isfinite(micro).all() or np.any(micro < 0.):
        raise ValueError("Invalid microscopic history readout;never replace it with fabricated zeros.")
    return micro, winding


def extract_frames(frames):
    """Three actual early frames, always ordered .65/1.35/2.05."""
    rows = [extract_state(frames[time]) for time in EARLY_TIMES]
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def extract_future(result):
    """Three saved actual future snapshots;does not continue the State."""
    if not result["valid"] or tuple(float(s["time"]) for s in result["snapshots"]) != FUTURE_TIMES:
        raise ValueError("Use the complete actual three-age future,not partial/reconstructed values.")
    rows = [extract_state(s["packet"][0]) for s in result["snapshots"]]
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def preflight_training_archives(parent):
    """Hash+metadata-read every64training prefix and768future archives.

    Does not decode physical States or compute features/scales.  Existing raw
    identities are bound to the original mother receipt and registration.
    """
    parent = Path(parent).resolve()
    root_path = parent/"REGISTERED.json"
    plan, root_hash = _read(root_path), _sha(root_path)
    if (plan["namespace"] != "aam-mbaa-property-inverse-cycle-v1-20261118"
            or plan["training_mothers"] != 64 or plan["training_suffixes"] != 2):
        raise ValueError("Use the frozen complete V1 64mother/two-suffix training set.")
    mothers, inputs = [], {str(root_path): root_hash}
    for mother in range(64):
        receipt_path = parent/"train"/f"parent_{mother:03d}"/"MOTHER_RECEIPT.json"
        receipt = _read(receipt_path)
        binding = receipt["binding"]
        if (receipt["status"] != "complete" or binding["role"] != "train"
                or binding["mother_id"] != mother or binding["root_registration_sha256"] != root_hash
                or Path(binding["root_registration"]).resolve() != root_path
                or binding["namespace"] != plan["namespace"]):
            raise ValueError("Training mother has another identity or incomplete physical output.")
        inputs[str(receipt_path)] = _sha(receipt_path)
        files = {}
        names = ["prefix_raw"]+[f"FUTURE_a{a:02d}_r{r:02d}" for r in range(2) for a in range(6)]
        for name in names:
            info = receipt["files"][name]
            path = Path(info["path"]).resolve()
            if path.parent != receipt_path.parent or path.stat().st_size != info["bytes"]:
                raise ValueError("Training raw file path/size differs from mother receipt.")
            actual = _sha(path)
            if actual != info["sha256"]:
                raise ValueError("Original training raw bytes changed: "+str(path))
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata_json"].item()))
            if (meta["schema"] != "aam-formation-cycle-lossless-numeric-tree-v1"
                    or meta["binding"] != binding or info["binding"] != binding):
                raise ValueError("Raw archive metadata is from another mother or schema.")
            files[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
            inputs[str(path)] = actual
        mothers.append({"mother": mother, "binding": binding, "files": files})
    training_path = parent/"TRAINING_BANK.npz"
    training_receipt = _read(parent/"TRAINING_BANK.json")
    if _sha(training_path) != training_receipt["sha256"]:
        raise ValueError("Frozen X training bank changed.")
    inputs[str(training_path)] = training_receipt["sha256"]
    fit_path = parent/"BASELINES/FIT_ARRAYS.npz"
    frozen = _read(parent/"BASELINES/FROZEN.json")
    if _sha(fit_path) != frozen["arrays_sha256"]:
        raise ValueError("Frozen KNN distance arrays changed.")
    inputs[str(fit_path)] = frozen["arrays_sha256"]
    return {"utc": datetime.now(timezone.utc).isoformat(), "schema": SCHEMA,
            "parent": str(parent), "training_mothers": 64, "prefix_archives": 64,
            "future_archives": 768, "raw_hashes_and_metadata_checked": 832,
            "physical_State_arrays_decoded": 0, "features_extracted": 0, "scales_computed": False,
            "mothers": mothers, "input_files": inputs, "native_calls": 0, "fits": 0,
            "scope": "Existing raw identity/readability preflight;not a new scientific experiment."}


def training_scales(early, future):
    """Work-fixed weighted population SD,computed only from64training mothers.

    Early block weight=.5/(64*3),future=.5/(64*2*6*3).
    Extractors already apply the fixed physical normalizations in NAMES.
    """
    early, future = np.asarray(early, float), np.asarray(future, float)
    if early.shape != (64, 3, 8) or future.shape != (64, 2, 6, 3, 8):
        raise ValueError("Frozen training layout must be64early and64x2x6future histories.")
    if not np.isfinite(early).all() or not np.isfinite(future).all():
        raise ValueError("All training histories must be retained and finite.")
    mean = .5*early.mean(axis=(0, 1))+.5*future.mean(axis=(0, 1, 2, 3))
    variance = .5*np.mean((early-mean)**2, axis=(0, 1))+.5*np.mean((future-mean)**2, axis=(0, 1, 2, 3))
    sd = np.sqrt(variance)
    return {"scale": np.maximum(sd, SCALE_FLOOR), "training_mean": mean,
            "training_SD": sd, "scale_floor": SCALE_FLOOR,
            "weight_rule": "Eachmother1/64;earlyhalf,futurehalf;withinblocktime/action/suffixequal."}


def load_training_histories(parent, *, preflight=None):
    """Existing64actual mothers only;no initialization,reaction or model fitting."""
    from aam_formation_cycle_data import _tree_load
    from aam_formation_actual_rollout import packet_fingerprint
    parent = Path(parent).resolve()
    if preflight is None:
        preflight = preflight_training_archives(parent)
    elif isinstance(preflight, (str, Path)):
        preflight = _read(preflight)
    if preflight["schema"] != SCHEMA or Path(preflight["parent"]).resolve() != parent:
        raise ValueError("Training preflight belongs to another schema/parent.")
    early, future = np.empty((64, 3, 8)), np.empty((64, 2, 6, 3, 8))
    early_w, future_w = np.empty((64, 3, 3), bool), np.empty((64, 2, 6, 3, 3), bool)
    for record in preflight["mothers"]:
        mother, binding = record["mother"], record["binding"]
        info = record["files"]["prefix_raw"]
        prefix = _tree_load(info["path"], binding, info["sha256"])
        if not prefix["valid"]:
            raise ValueError("No invalid or missing training prefix may be replaced.")
        early[mother], early_w[mother] = extract_frames(prefix["frames"])
        original_hash = packet_fingerprint(prefix["packet"])
        for suffix in range(2):
            for action in range(6):
                info = record["files"][f"FUTURE_a{action:02d}_r{suffix:02d}"]
                result = _tree_load(info["path"], binding, info["sha256"])
                if result["mother_packet_sha256"] != original_hash:
                    raise ValueError("Training future does not start at its original actual prefix.")
                future[mother, suffix, action], future_w[mother, suffix, action] = extract_future(result)
    if len(preflight["mothers"]) != 64 or sorted(r["mother"] for r in preflight["mothers"]) != list(range(64)):
        raise ValueError("Exactly all64distinct training mothers must be represented.")
    for path in (parent/"TRAINING_BANK.npz", parent/"BASELINES/FIT_ARRAYS.npz"):
        if _sha(path) != preflight["input_files"][str(path)]:
            raise ValueError("Frozen training array input changed after preflight.")
    with np.load(parent/"TRAINING_BANK.npz", allow_pickle=False) as archive:
        x = archive["X"].copy()
    with np.load(parent/"BASELINES/FIT_ARRAYS.npz", allow_pickle=False) as archive:
        if not np.array_equal(x, archive["X12"]):
            raise ValueError("KNN inputs and original training query bank differ.")
        xmean, xscale = archive["knn_Xmean"].copy(), archive["knn_Xscale"].copy()
    return {"schema": SCHEMA, "names": NAMES, "X": x, "early": early, "future": future,
            "early_winding": early_w, "future_winding": future_w,
            "knn_Xmean": xmean, "knn_Xscale": xscale,
            **training_scales(early, future), "input_files": preflight["input_files"],
            "new_training_mothers": 0, "new_native_calls": 0,
            "scope": "Actual existing wholehistory support;not independently generated molecules or wetmeasurements."}


def micro_baseline_predictions(bank, X_query):
    """Batch empirical128 and KNN8x2 whole-history supports,without fitting.

    Outputs have leading(query,draw) axes.  One mother and suffix index is used
    for every frame of that draw. Actions are later scored separately.
    """
    x = np.asarray(X_query, float)
    if x.ndim != 2 or x.shape[1] != 12 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Micro baselines receive only finite query x12.")
    if bank["schema"] != SCHEMA or np.asarray(bank["X"]).shape != (64, 12):
        raise ValueError("Use the frozen64mother micro history bank.")
    mean, scale = np.asarray(bank["knn_Xmean"]), np.asarray(bank["knn_Xscale"])
    if mean.shape != (12,) or scale.shape != (12,) or np.any(scale <= 0.):
        raise ValueError("Use the original frozen KNN distance standardization.")
    trainx = (bank["X"]-mean)/scale
    queryx = (x-mean)/scale
    distance = np.sum((queryx[:, None]-trainx[None])**2, axis=2)
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :8]
    all_mothers = np.broadcast_to(np.arange(64)[None], (len(x), 64))
    output = {}
    for name, mothers in (("empirical", all_mothers), ("knn8", nearest)):
        mi = np.repeat(mothers, 2, axis=1)
        si = np.broadcast_to(np.tile(np.arange(2), mothers.shape[1])[None], mi.shape)
        output[name] = {"early": bank["early"][mi], "future": bank["future"][mi, si],
                        "early_winding": bank["early_winding"][mi],
                        "future_winding": bank["future_winding"][mi, si],
                        "weights": np.full(mi.shape, 1./mi.shape[1]),
                        "mother_indices": mi.copy(), "suffix_indices": si.copy()}
    output["knn8"]["neighbor_parent_indices"] = nearest.copy()
    return output


def _energy(draws, weights, reference):
    from scipy.spatial.distance import cdist
    draws, reference = np.asarray(draws, float), np.asarray(reference, float)
    if draws.ndim != 2 or reference.ndim != 2 or draws.shape[1] != reference.shape[1] or not len(reference):
        raise ValueError("Energy score requires paired finite vector dimensions.")
    distance_xy = cdist(draws, reference)/np.sqrt(draws.shape[1])
    distance_xx = cdist(draws, draws)/np.sqrt(draws.shape[1])
    return float(np.mean(weights@distance_xy)-.5*(weights@distance_xx@weights))


def score_micro_history(early, future, early_winding, future_winding, weights, *,
                        reference_early, reference_future, reference_early_winding,
                        reference_future_winding, scale):
    """Early24,each-action full48/future24 weighted ES;winding Brier separate."""
    early, future = np.asarray(early, float), np.asarray(future, float)
    ew, fw = np.asarray(early_winding, float), np.asarray(future_winding, float)
    re, rf = np.asarray(reference_early, float), np.asarray(reference_future, float)
    rew, rfw = np.asarray(reference_early_winding, float), np.asarray(reference_future_winding, float)
    w, scale = np.asarray(weights, float), np.asarray(scale, float)
    d = len(early)
    if (early.shape != (d, 3, 8) or future.shape != (d, 6, 3, 8)
            or ew.shape != (d, 3, 3) or fw.shape != (d, 6, 3, 3)
            or re.shape != (3, 8) or rf.ndim != 4 or rf.shape[1:] != (6, 3, 8)
            or rew.shape != (3, 3) or rfw.shape != (len(rf), 6, 3, 3)
            or w.shape != (d,) or not d or not len(rf) or scale.shape != (8,)):
        raise ValueError("Wrong full microhistory/weight/reference layout.")
    if (any(not a.size or not np.isfinite(a).all() for a in (early, future, ew, fw, re, rf, rew, rfw, w, scale))
            or np.any(w < 0.) or abs(float(w.sum())-1.) > 2.e-12 or np.any(scale < SCALE_FLOOR)
            or any(np.any((a != 0.) & (a != 1.)) for a in (ew, fw, rew, rfw))):
        raise ValueError("All supports and original winding labels must be retained and valid.")
    ed, er = (early/scale).reshape(d, 24), (re/scale).reshape(1, 24)
    early_es = _energy(ed, w, er)
    full_es, future_es = [], []
    for action in range(6):
        fd = (future[:, action]/scale).reshape(d, 24)
        fr = (rf[:, action]/scale).reshape(len(rf), 24)
        full_es.append(_energy(np.concatenate((ed, fd), axis=1), w,
                               np.concatenate((np.repeat(er, len(rf), axis=0), fr), axis=1)))
        future_es.append(_energy(fd, w, fr))
    early_probability = np.einsum("d,dtk->tk", w, ew)
    future_probability = np.einsum("d,datk->atk", w, fw)
    early_brier = float(np.mean((early_probability-rew)**2))
    future_brier = np.mean((future_probability[None]-rfw)**2, axis=(0, 2, 3))
    full_brier = .5*early_brier+.5*future_brier
    return {"schema": SCHEMA, "early24_energy": early_es,
            "per_action_full48_energy": np.asarray(full_es), "full48_energy": float(np.mean(full_es)),
            "per_action_future24_energy": np.asarray(future_es), "future24_energy": float(np.mean(future_es)),
            "early_winding_brier": early_brier,
            "per_action_future_winding_brier": future_brier, "future_winding_brier": float(future_brier.mean()),
            "per_action_full_winding_brier": full_brier, "full_winding_brier": float(full_brier.mean()),
            "counterfactual_actions_scored_jointly": False,
            "scope": "Distributions of fixed8Dstructural/event summaries and axis-wise winding,not unique fullgraph recovery."}
