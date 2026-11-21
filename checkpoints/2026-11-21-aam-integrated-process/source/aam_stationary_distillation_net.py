"""A recurrent, low-dimensional tilt of a frozen legal trajectory probability law."""

import hashlib
from operator import index

import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch import nn

from aam_mlst_recurrent_process import _deterministic_cpu


SCHEMA = "aam-stationary-distillation-energy-gru24-v1"
GEOMETRY_SCHEMA = "aam-equal-early18-and-delta32-negative-type-spectral-v1"
METRICS = ("early18", "hidden_delta32_6")
LOCAL_SHAPE = (3, 2, 3)
CAMERA_SHAPE = (3, 4)
RANK = 16
HIDDEN_SIZE = 24
BATCH_SIZE = 32
EPOCHS = (8, 24, 64)
LEARNING_RATE = .003
WEIGHT_DECAY = .0001
TILT_PENALTY = .001


def _histories(value):
    x = np.asarray(value, dtype=np.float64)
    if x.shape == LOCAL_SHAPE:
        x = x[None, ...]
    if (x.ndim != 4 or x.shape[1:] != LOCAL_SHAPE or len(x) == 0
            or not np.isfinite(x).all()):
        raise ValueError("Require complete finite three-camera, two-aperture paths.")
    return x


def _metric_coordinates(local):
    delta = local[:, 2] - local[:, 1]
    return local.reshape(len(local), 18) / np.sqrt(18.), delta.reshape(len(local), 6) / np.sqrt(6.)


def _geometry_fingerprint(distance, codes):
    digest = hashlib.sha256()
    for array in (distance, codes):
        a = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(a.shape).encode("ascii"))
        digest.update(a.tobytes())
    return digest.hexdigest()


def build_geometry(atlas_local, rank=RANK):
    """Classical negative-type embedding of the fixed 50/50 path distance.

    Codes are leading coordinates of the exact double-centered Gram matrix.
    The score and training objective still use the full atlas distance matrix.
    """
    local, r = _histories(atlas_local), index(rank)
    if not 1 <= r < len(local):
        raise ValueError("Spectral rank must be positive and smaller than the legal atlas.")
    early, delta = _metric_coordinates(local)
    distance = .5 * (cdist(early, early) + cdist(delta, delta))
    row_mean = distance.mean(axis=1)
    centered = -.5 * (distance - row_mean[:, None] - row_mean[None, :] + row_mean.mean())
    with _deterministic_cpu(), torch.inference_mode():
        values, vectors = torch.linalg.eigh(torch.from_numpy(centered))
    eigenvalues, eigenvectors = values.numpy(), vectors.numpy()
    tolerance = np.finfo(np.float64).eps * len(local) * max(1., float(eigenvalues[-1]))
    if eigenvalues[0] < -tolerance or eigenvalues[-r] <= tolerance:
        raise ValueError("Require a negative-type geometry with all requested spectral directions identified.")
    directions = eigenvectors[:, -r:][:, ::-1].copy()
    pivots = np.argmax(np.abs(directions), axis=0)
    directions *= np.sign(directions[pivots, np.arange(r)])
    spectrum = eigenvalues[-r:][::-1].copy()
    codes = directions * np.sqrt(spectrum)
    return {"schema": GEOMETRY_SCHEMA, "metrics": METRICS, "rank": r, "distance": distance, "codes": codes,
            "spectral_solver": "deterministic PyTorch CPU float64 eigh",
            "eigenvalues": spectrum, "smallest_full_eigenvalue": float(eigenvalues[0]),
            "retained_gram_trace_fraction": float(spectrum.sum()/np.maximum(eigenvalues, 0.).sum()),
            "sha256": _geometry_fingerprint(distance, codes)}


def _geometry(value):
    distance = np.asarray(value["distance"], dtype=np.float64)
    codes = np.asarray(value["codes"], dtype=np.float64)
    if (value.get("schema") != GEOMETRY_SCHEMA or codes.ndim != 2
            or distance.shape != (len(codes), len(codes)) or codes.shape[1] != value["rank"]
            or not np.isfinite(codes).all() or not np.isfinite(distance).all()
            or np.any(distance < 0.) or not np.array_equal(distance, distance.T)
            or np.any(distance.diagonal() != 0.)
            or _geometry_fingerprint(distance, codes) != value["sha256"]):
        raise ValueError("Require the unchanged complete distance and spectral codes of the same legal atlas.")
    return distance, codes


def _probabilities(value, shape):
    p = np.asarray(value, dtype=np.float64)
    if (p.shape != shape or not np.isfinite(p).all() or np.any(p < 0.)
            or not np.allclose(p.sum(axis=-1), 1., atol=2e-13, rtol=0.)):
        raise ValueError("Require finite normalized probabilities on the original legal support.")
    return p


def cross_distances(atlas_local, target_local, weights=None):
    """Expected distance for the linear part of the joint held-aperture score.

    The target may be one authentic generating path or a weighted teacher law.
    All target paths are complete and their probabilities share one normalization.
    """
    atlas, target = _histories(atlas_local), _histories(target_local)
    mass = np.full(len(target), 1. / len(target)) if weights is None else _probabilities(weights, (len(target),))
    left, right = _metric_coordinates(atlas), _metric_coordinates(target)
    return .5 * sum(np.einsum("ij,j->i", cdist(x, y), mass, optimize=False)
                    for x, y in zip(left, right, strict=True))


def energy(weights, cross, distance):
    """Expected full-atlas energy score, one value per camera query."""
    cross = np.asarray(cross, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    if (cross.ndim != 2 or distance.shape != (cross.shape[1], cross.shape[1])
            or not np.isfinite(cross).all() or not np.isfinite(distance).all()):
        raise ValueError("Use matching query-to-target distances and a complete atlas pair matrix.")
    p = _probabilities(weights, cross.shape)
    pair = np.einsum("ni,ij->nj", p, distance, optimize=False)
    return np.sum(p * (cross - .5 * pair), axis=1)


def _network_cameras(cameras, camera_scale):
    x = np.asarray(cameras, dtype=np.float64)
    if x.shape == CAMERA_SHAPE:
        x = x[None, ...]
    scale = np.asarray(camera_scale, dtype=np.float64)
    if (x.ndim != 3 or x.shape[1:] != CAMERA_SHAPE or len(x) == 0 or not np.isfinite(x).all()
            or scale.shape not in ((4,), CAMERA_SHAPE) or not np.isfinite(scale).all() or np.any(scale <= 0.)):
        raise ValueError("Use the same three four-channel snapshots and the frozen positive camera scales.")
    scaled = x / scale
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled original cameras must remain finite.")
    return np.ascontiguousarray(scaled)


def _base_log(base):
    log = np.full_like(base, -np.inf)
    np.log(base, where=base > 0., out=log)
    return log


class _TiltNet(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.gru = nn.GRU(4, HIDDEN_SIZE, batch_first=True, device="cpu", dtype=torch.float64)
        self.output = nn.Linear(HIDDEN_SIZE, rank, device="cpu", dtype=torch.float64)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, normalized_cameras):
        history, _ = self.gru(normalized_cameras)
        return self.output(history[:, -1])


def _probability_tensor(tilt, base_log, codes):
    return torch.softmax(base_log + tilt @ codes.T, dim=1)


def _loss(net, inputs, base_log, codes, cross, distance):
    tilt = net(inputs)
    p = _probability_tensor(tilt, base_log, codes)
    score = torch.sum(p * (cross - .5 * (p @ distance)), dim=1)
    penalty = TILT_PENALTY * torch.sum(tilt.square(), dim=1)
    return score, penalty


def fit(cameras, base_weights, cross, geometry, camera_scale, *, seed, epochs=EPOCHS):
    """Fit one predeclared arm from training cameras and expected linear scores.

    Run the single-path and stationary-teacher arms with the same seed, inputs,
    epochs, and all other settings. Only the training cross term should differ.
    Snapshot indices are training iteration counts. DEV/held selection is external.
    """
    distance, codes = _geometry(geometry)
    normalized = _network_cameras(cameras, camera_scale)
    p = _probabilities(base_weights, (len(normalized), len(codes)))
    cross = np.asarray(cross, dtype=np.float64)
    seed = index(seed)
    checkpoints = tuple(index(k) for k in epochs)
    if (cross.shape != p.shape or not np.isfinite(cross).all() or np.any(cross < 0.)
            or seed < 0 or not checkpoints or checkpoints[0] < 1
            or tuple(sorted(set(checkpoints))) != checkpoints):
        raise ValueError("Use one complete nonnegative cross row per training mother and increasing fixed epochs.")
    shuffle_seed, model_seed = np.random.SeedSequence(seed).spawn(2)
    shuffle = np.random.default_rng(shuffle_seed)
    model_seed = int(model_seed.generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)
    tensors = [torch.from_numpy(np.array(a, copy=True, order="C"))
               for a in (normalized, _base_log(p), codes, cross, distance)]
    x, log, z, target, pair = tensors
    state, trace = {}, np.empty((checkpoints[-1], 2))
    with _deterministic_cpu():
        torch.manual_seed(model_seed)
        net = _TiltNet(codes.shape[1])
        optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for epoch in range(1, checkpoints[-1] + 1):
            order = shuffle.permutation(len(normalized))
            totals = np.zeros(2)
            for start in range(0, len(order), BATCH_SIZE):
                ids = order[start:start + BATCH_SIZE]
                value, penalty = _loss(net, x[ids], log[ids], z, target[ids], pair)
                loss = (value + penalty).mean()
                if not torch.isfinite(loss):
                    raise ArithmeticError("The full-atlas objective must remain finite.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                totals += (value.detach().sum().item(), penalty.detach().sum().item())
            trace[epoch - 1] = totals / len(order)
            if epoch in checkpoints:
                saved = {name: tensor.detach().numpy().copy() for name, tensor in net.state_dict().items()}
                if any(not np.isfinite(a).all() for a in saved.values()):
                    raise ArithmeticError("A trained CPU checkpoint must be finite.")
                state[str(epoch)] = saved
    return {
        "schema": SCHEMA, "states": state, "geometry_sha256": geometry["sha256"],
        "rank": codes.shape[1], "camera_scale": np.array(camera_scale, dtype=np.float64, copy=True),
        "seed": seed, "model_seed": model_seed, "epochs": checkpoints, "initial_residual": "zero_head",
        "training_mothers": len(normalized), "training_trace": trace, "trace_names": ("score", "tilt_penalty"),
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "tilt_penalty": TILT_PENALTY, "hidden_size": HIDDEN_SIZE, "torch_version": str(torch.__version__),
    }


def predict(model, geometry, cameras, base_weights, *, epoch, shrink=1.):
    """Return portable normalized atlas weights, optionally mixed with the frozen RBF."""
    _, codes = _geometry(geometry)
    if (model.get("schema") != SCHEMA or model.get("geometry_sha256") != geometry["sha256"]
            or model.get("hidden_size") != HIDDEN_SIZE or model.get("rank") != codes.shape[1]):
        raise ValueError("Use the same trained correction and legal atlas order.")
    i, alpha = index(epoch), float(shrink)
    if not 0. <= alpha <= 1. or str(i) not in model["states"]:
        raise ValueError("Use a frozen training checkpoint and a valid baseline mixture proportion.")
    x = _network_cameras(cameras, model["camera_scale"])
    p = _probabilities(base_weights, (len(x), len(codes)))
    if alpha == 0.:
        return p.copy()
    with _deterministic_cpu(), torch.inference_mode():
        net = _TiltNet(codes.shape[1])
        net.load_state_dict({name: torch.from_numpy(np.array(v, dtype=np.float64, copy=True))
                             for name, v in model["states"][str(i)].items()})
        tilt = net(torch.from_numpy(x))
        corrected = _probability_tensor(tilt, torch.from_numpy(_base_log(p)),
                                        torch.from_numpy(np.array(codes, copy=True))).numpy()
    final = (1. - alpha) * p + alpha * corrected
    return _probabilities(final, p.shape)


def fixtures():
    """Pure geometry, score, zero-residual, and gradient checks on dummy arrays."""
    gen = np.random.default_rng(77109)
    atlas = gen.random((20, *LOCAL_SHAPE))
    local = gen.random((3, *LOCAL_SHAPE))
    mass = np.array([.2, .3, .5])
    geometry = build_geometry(atlas)
    distance, codes = _geometry(geometry)
    assert codes.shape == (20, 16)
    np.testing.assert_allclose(codes.mean(axis=0), 0., rtol=0., atol=1.e-14)
    cross = cross_distances(atlas, local, mass)
    direct = np.stack([.5 * (
        np.sqrt(np.mean((atlas - target)**2, axis=(1, 2, 3)))
        + np.sqrt(np.mean(((atlas[:, 2]-atlas[:, 1]) - (target[2]-target[1]))**2, axis=(1, 2))))
        for target in local])
    np.testing.assert_allclose(cross, np.sum(mass[:, None] * direct, axis=0), rtol=0., atol=3.e-16)
    x = gen.random((2, *CAMERA_SHAPE))
    p = gen.random((2, len(atlas)))
    p[0, 0] = 0.
    p /= p.sum(axis=1, keepdims=True)
    y = np.broadcast_to(cross, p.shape).copy()
    with _deterministic_cpu():
        net = _TiltNet(codes.shape[1])
        q = _probability_tensor(net(torch.from_numpy(x)), torch.from_numpy(_base_log(p)), torch.from_numpy(codes))
        np.testing.assert_allclose(q.detach().numpy(), p, rtol=0., atol=2.e-16)
        netval, penalty = _loss(net, torch.from_numpy(x), torch.from_numpy(_base_log(p)), torch.from_numpy(codes),
                                torch.from_numpy(y), torch.from_numpy(distance))
        np.testing.assert_allclose(netval.detach().numpy(), energy(p, y, distance), rtol=0., atol=3.e-16)
        assert torch.all(penalty == 0.)
        netval.mean().backward()
        assert torch.isfinite(net.output.bias.grad).all() and torch.any(net.output.bias.grad != 0.)
        fixture_state = {name: tensor.detach().numpy().copy() for name, tensor in net.state_dict().items()}
    model = {"schema": SCHEMA, "states": {"8": fixture_state}, "geometry_sha256": geometry["sha256"],
             "rank": codes.shape[1], "hidden_size": HIDDEN_SIZE, "camera_scale": np.ones(CAMERA_SHAPE)}
    np.testing.assert_array_equal(predict(model, geometry, x, p, epoch=8, shrink=0.), p)
    np.testing.assert_allclose(predict(model, geometry, x, p, epoch=8, shrink=.5), p, rtol=0., atol=2.e-16)
    return {"schema": SCHEMA, "physics_calls": 0, "optimizer_steps": 0, "exact_geometry_and_energy": True,
            "portable_positive_residual_and_gradients": True}


if __name__ == "__main__":
    print(fixtures())
