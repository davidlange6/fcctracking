"""Stage 2: seed triplets per shard, normalisation ranges and target windows."""

import json

import numpy as np
import torch

from fcctrackfit.data.dataset import gather_tracks, usable_tracks
from fcctrackfit.data.shards import COL, layer_key, read_shard, write_seed_indices
from fcctrackfit.geometry.perigee import QOP, THETA
from fcctrackfit.model.fitter import RESIDUAL_FEATURES, make_targets
from fcctrackfit.seed.conformal import estimate_seed, select_seed_hits
from fcctrackfit.seed.residuals import seed_residuals


def add_seed_indices(path, seed_cfg, bz):
    shard = read_shard(path)
    feat = shard["hit_feat"]
    seedable = np.isin(feat[:, COL["system"]].astype(int), seed_cfg["systems"])
    timing = {}
    if seed_cfg.get("max_turn"):
        timing = dict(pos=shard["hit_pos"], time=feat[:, COL["t"]], bz=bz, max_turn=seed_cfg["max_turn"])
    idx = select_seed_hits(shard["offsets"], layer_key(feat), seedable, feat[:, COL["r"]], seed_cfg["rule"], **timing)
    write_seed_indices(path, idx, shard["offsets"])
    return (idx >= 0).all(1).mean()


def fit_resolution(delta, inv_p, sin_theta=None, n_bins=20, width_quantile=0.683, n_iter=3):
    """Fit the seed error per parameter as hypot(a, b * inv_p) * sin_theta^-c.

    delta: [N, 5] unscaled targets (make_targets with unit windows); inv_p, sin_theta: [N]
    from the seed. The width in each of n_bins equal-population bins is the width_quantile
    of |delta|. a^2 + b^2 inv_p^2 is fitted to the squared widths in inv_p bins with relative
    weights (linear least squares), and c to the log widths in sin_theta bins of the
    momentum-scaled delta; the two fits alternate n_iter times. Returns [5, 3] (a, b, c),
    c = 0 without sin_theta.
    """

    def binned(x):
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
        centre = np.array([np.median(x[which == k]) for k in range(n_bins)])
        return which, centre

    def widths(d, which):
        return np.array([np.quantile(np.abs(d[which == k]), width_quantile) for k in range(n_bins)])

    p_bin, x = binned(inv_p)
    if sin_theta is not None:
        t_bin, s = binned(sin_theta)
    out = []
    for j in range(delta.shape[1]):
        c = 0.0
        for _ in range(n_iter if sin_theta is not None else 1):
            d = delta[:, j] * (sin_theta**c if sin_theta is not None else 1.0)
            w = widths(d, p_bin)
            A = np.stack([1 / w**2, x**2 / w**2], 1)
            (a2, b2), *_ = np.linalg.lstsq(A, np.ones(n_bins), rcond=None)
            a2 = max(a2, 1e-3 * np.min(w) ** 2)  # keep the scale positive at high momentum
            a, b = np.sqrt(a2), np.sqrt(max(b2, 0.0))
            if sin_theta is None:
                break
            wt = widths(delta[:, j] / np.hypot(a, b * inv_p), t_bin)
            c = -np.polyfit(np.log(s), np.log(wt), 1)[0]
        out.append([float(a), float(b), float(c)])
    return np.array(out)


def compute_norm(paths, hit_columns, bz, qop_eps, residual_scale, max_tracks=200_000, window_quantile=0.999,
                 momentum_scaling=True):  # fmt: skip
    """Min-max ranges of every model input feature, the momentum dependence of the seed
    error (fit_resolution) and per-parameter target windows of the scaled targets."""
    feats, res, deltas, inv_p, sin_theta = [], [], [], [], []
    n = 0
    for path in paths:
        shard = read_shard(path)
        idx = usable_tracks(shard)[: max_tracks - n]
        b = gather_tracks(shard, idx)
        seed = estimate_seed(*(b["pos"][b["seed_idx"][:, k]] for k in range(3)), bz)
        du, dv, s = seed_residuals(b["pos"], b["cu_seqlens"], seed, bz)
        feats.append(b["feat"].numpy())
        res.append(np.stack([np.arcsinh(du.numpy() / residual_scale), np.arcsinh(dv.numpy() / residual_scale),
                             s.numpy()], 1))  # fmt: skip
        deltas.append(make_targets(b["truth"], seed, torch.ones(5, dtype=torch.float64), qop_eps).numpy())
        inv_p.append(seed[:, QOP].abs().numpy())
        sin_theta.append(torch.sin(seed[:, THETA]).abs().clamp(min=1e-3).numpy())
        n += len(idx)
        if n >= max_tracks:
            break
    feats = np.concatenate(feats)
    res = np.concatenate(res)
    deltas = np.concatenate(deltas)
    inv_p = np.concatenate(inv_p)
    sin_theta = np.concatenate(sin_theta)
    ranges = {c: _range(feats[:, COL[c]]) for c in hit_columns}
    ranges.update({c: _range(res[:, i]) for i, c in enumerate(RESIDUAL_FEATURES)})
    norm = {"features": ranges, "n_tracks": int(n), "qop_eps": qop_eps, "residual_scale": residual_scale}
    if momentum_scaling:
        resolution = fit_resolution(deltas, inv_p, sin_theta)
        deltas = deltas / (np.hypot(resolution[:, 0], resolution[:, 1] * inv_p[:, None])
                           * sin_theta[:, None] ** -resolution[:, 2])  # fmt: skip
        norm["resolution"] = resolution.tolist()
    norm["windows"] = np.quantile(np.abs(deltas), window_quantile, axis=0).tolist()
    return norm


def _range(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return [lo, hi if hi > lo else lo + 1.0]


def save_norm(norm, path):
    with open(path, "w") as f:
        json.dump(norm, f, indent=1)


def load_norm(path):
    with open(path) as f:
        return json.load(f)
