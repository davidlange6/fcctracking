"""Ragged HDF5 shard format shared by the edm4hep converter and the toy simulator.

Per shard (one HDF5 file):
    hit_pos    [N, 3] float64   hit positions (mm), time-ordered within a track
    hit_feat   [N, F] float32   per-hit features, column names in attr "hit_columns"
    offsets    [T+1]  int64     track t owns hits offsets[t]:offsets[t+1]
    truth      [T, 5] float64   truth perigee (d0, z0, phi, theta, q/p)
    kf         [T, 5] float64   baseline KF perigee (NaN where kf_ok is False)
    kf_ok      [T]    bool
    kf_cov     [T, 21] float32  edm4hep TrackState covMatrix (lower triangle, 6x6 incl. time)
    event      [T]    int64
    seed_idx   [T, 3] int64     (written by prepare) global hit indices of the seed triplet
attrs: detector, bz, hit_columns
"""

import h5py
import numpy as np

HIT_COLUMNS = (
    "x", "y", "z", "t", "r", "phi", "theta",
    "system", "layer", "side", "kind",
    "u_theta", "u_phi", "v_theta", "v_phi", "du", "dv",
    "wire_azim", "wire_stereo", "drift_dist", "drift_err",
)  # fmt: skip
COL = {name: i for i, name in enumerate(HIT_COLUMNS)}

KIND_SI_PLANE, KIND_SI_3D, KIND_DRIFT = 0, 1, 2


def layer_key(feat):
    """Unique integer per (system, side, layer) from a hit_feat array."""
    return (
        feat[:, COL["system"]].astype(np.int64) * 10000
        + (feat[:, COL["side"]].astype(np.int64) + 10) * 1000
        + feat[:, COL["layer"]].astype(np.int64)
    )


def derived_columns(pos):
    """r, phi, theta of hit positions."""
    r = np.hypot(pos[:, 0], pos[:, 1])
    return r, np.arctan2(pos[:, 1], pos[:, 0]), np.arctan2(r, pos[:, 2])


def write_shard(path, *, hit_pos, hit_feat, offsets, truth, kf, kf_ok, event, detector, bz, kf_cov=None):
    with h5py.File(path, "w") as f:
        f.attrs["detector"] = detector
        f.attrs["bz"] = bz
        f.attrs["hit_columns"] = list(HIT_COLUMNS)
        f.create_dataset("hit_pos", data=np.asarray(hit_pos, np.float64), compression="lzf")
        f.create_dataset("hit_feat", data=np.asarray(hit_feat, np.float32), compression="lzf")
        f.create_dataset("offsets", data=np.asarray(offsets, np.int64))
        f.create_dataset("truth", data=np.asarray(truth, np.float64))
        f.create_dataset("kf", data=np.asarray(kf, np.float64))
        f.create_dataset("kf_ok", data=np.asarray(kf_ok, bool))
        f.create_dataset("event", data=np.asarray(event, np.int64))
        if kf_cov is not None:
            f.create_dataset("kf_cov", data=np.asarray(kf_cov, np.float32))


def read_shard(path):
    """Load a whole shard into a dict of numpy arrays (+ attrs)."""
    with h5py.File(path, "r") as f:
        out = {k: f[k][()] for k in f.keys()}
        out["detector"] = f.attrs["detector"]
        out["bz"] = float(f.attrs["bz"])
        out["hit_columns"] = [str(c) for c in f.attrs["hit_columns"]]
    if tuple(out["hit_columns"]) != HIT_COLUMNS:
        raise ValueError(f"{path}: hit column layout differs from this code version")
    return out


def add_dataset(path, name, data):
    with h5py.File(path, "a") as f:
        if name in f:
            del f[name]
        f.create_dataset(name, data=data)
