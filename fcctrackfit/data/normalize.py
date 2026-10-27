"""Stage 2: seed triplets per shard, normalisation ranges and target windows."""

import json

import numpy as np
import torch

from fcctrackfit.data.dataset import gather_tracks, usable_tracks
from fcctrackfit.data.shards import COL, add_dataset, layer_key, read_shard
from fcctrackfit.model.fitter import RESIDUAL_FEATURES, make_targets
from fcctrackfit.seed.conformal import estimate_seed, select_seed_hits
from fcctrackfit.seed.residuals import seed_residuals


def add_seed_indices(path, seed_cfg):
    shard = read_shard(path)
    feat = shard["hit_feat"]
    seedable = np.isin(feat[:, COL["system"]].astype(int), seed_cfg["systems"])
    idx = select_seed_hits(shard["offsets"], layer_key(feat), seedable, feat[:, COL["r"]], seed_cfg["rule"])
    add_dataset(path, "seed_idx", idx)
    return (idx >= 0).all(1).mean()


def compute_norm(paths, hit_columns, bz, qop_eps, residual_scale, max_tracks=200_000, window_quantile=0.999):
    """Min-max ranges of every model input feature and per-parameter target windows."""
    feats, res, deltas = [], [], []
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
        n += len(idx)
        if n >= max_tracks:
            break
    feats = np.concatenate(feats)
    res = np.concatenate(res)
    deltas = np.concatenate(deltas)
    ranges = {c: _range(feats[:, COL[c]]) for c in hit_columns}
    ranges.update({c: _range(res[:, i]) for i, c in enumerate(RESIDUAL_FEATURES)})
    windows = np.quantile(np.abs(deltas), window_quantile, axis=0)
    return {"features": ranges, "windows": windows.tolist(), "n_tracks": int(n),
            "qop_eps": qop_eps, "residual_scale": residual_scale}  # fmt: skip


def _range(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return [lo, hi if hi > lo else lo + 1.0]


def save_norm(norm, path):
    with open(path, "w") as f:
        json.dump(norm, f, indent=1)


def load_norm(path):
    with open(path) as f:
        return json.load(f)
