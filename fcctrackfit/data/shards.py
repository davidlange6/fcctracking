"""Ragged HDF5 shard format shared by the edm4hep converter and the toy simulator.

On disk (format 2), per shard (one HDF5 file), N hits and T tracks:
    hit_pos     [N, 3] float32  hit positions (mm), time-ordered within a track
    hit_t       [N]    float32  hit time (ns)
    hit_id      [N, 4] int16    system, side, layer, kind
    hit_meas    [N, 6] float32  u_theta, u_phi, v_theta, v_phi, du, dv (silicon measurement frame)
    hit_drift   [N, 4] float32  wire_azim, wire_stereo, drift_dist, drift_err (only if drift hits exist)
    offsets     [T+1]  int64    track t owns hits offsets[t]:offsets[t+1]
    truth       [T, 5] float64  truth perigee (d0, z0, phi, theta, q/p)
    event       [T]    int64
    kf          [T, 5] float64  baseline KF perigee (only if any track has one; NaN where kf_ok is False)
    kf_ok       [T]    bool     (only with kf)
    kf_cov      [T, 21] float32 edm4hep TrackState covMatrix (only with kf)
    seed_local  [T, 3] int16    (written by prepare) seed triplet as indices within the track, -1 if none
attrs: format, detector, bz

float32 positions quantise at 0.12 um for r = 2.1 m, far below the sensor resolution;
the seed is still computed in float64 from them.

In memory, read_shard returns the uncompressed layout used by the rest of the code:
hit_pos [N, 3] float64, hit_feat [N, F] float32 with columns HIT_COLUMNS (r, phi,
theta recomputed), kf / kf_ok always present, and seed_idx [T, 3] as global hit indices.
"""

import h5py
import numpy as np

FORMAT = 2

HIT_COLUMNS = (
    "x", "y", "z", "t", "r", "phi", "theta",
    "system", "layer", "side", "kind",
    "u_theta", "u_phi", "v_theta", "v_phi", "du", "dv",
    "wire_azim", "wire_stereo", "drift_dist", "drift_err",
)  # fmt: skip
COL = {name: i for i, name in enumerate(HIT_COLUMNS)}

ID_COLUMNS = ("system", "side", "layer", "kind")
MEAS_COLUMNS = ("u_theta", "u_phi", "v_theta", "v_phi", "du", "dv")
DRIFT_COLUMNS = ("wire_azim", "wire_stereo", "drift_dist", "drift_err")

KIND_SI_PLANE, KIND_SI_3D, KIND_DRIFT = 0, 1, 2

COMPRESSION = dict(compression="gzip", compression_opts=4, shuffle=True)


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


def _cols(feat, names):
    return feat[:, [COL[c] for c in names]]


def write_shard(path, *, hit_pos, hit_feat, offsets, truth, event, detector, bz, kf=None, kf_ok=None, kf_cov=None):
    """Write a shard from the in-memory layout (hit_feat with HIT_COLUMNS)."""
    hit_feat = np.asarray(hit_feat, np.float32)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["detector"] = detector
        f.attrs["bz"] = bz
        f.create_dataset("hit_pos", data=np.asarray(hit_pos, np.float32), **COMPRESSION)
        f.create_dataset("hit_t", data=hit_feat[:, COL["t"]], **COMPRESSION)
        f.create_dataset("hit_id", data=_cols(hit_feat, ID_COLUMNS).astype(np.int16), **COMPRESSION)
        f.create_dataset("hit_meas", data=_cols(hit_feat, MEAS_COLUMNS), **COMPRESSION)
        drift = _cols(hit_feat, DRIFT_COLUMNS)
        if drift.any():
            f.create_dataset("hit_drift", data=drift, **COMPRESSION)
        f.create_dataset("offsets", data=np.asarray(offsets, np.int64), **COMPRESSION)
        f.create_dataset("truth", data=np.asarray(truth, np.float64), **COMPRESSION)
        f.create_dataset("event", data=np.asarray(event, np.int64), **COMPRESSION)
        if kf_ok is not None and np.any(kf_ok):
            f.create_dataset("kf", data=np.asarray(kf, np.float64), **COMPRESSION)
            f.create_dataset("kf_ok", data=np.asarray(kf_ok, bool), **COMPRESSION)
            if kf_cov is not None:
                f.create_dataset("kf_cov", data=np.asarray(kf_cov, np.float32), **COMPRESSION)


def read_shard(path):
    """Load a whole shard into the in-memory layout (see module docstring)."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{path}: shard format {f.attrs.get('format')} != {FORMAT}; reconvert it")
        pos = f["hit_pos"][()].astype(np.float64)
        n = len(pos)
        feat = np.zeros((n, len(HIT_COLUMNS)), np.float32)
        feat[:, COL["x"]], feat[:, COL["y"]], feat[:, COL["z"]] = pos.T
        feat[:, COL["t"]] = f["hit_t"][()]
        feat[:, COL["r"]], feat[:, COL["phi"]], feat[:, COL["theta"]] = derived_columns(pos)
        for name, cols in (("hit_id", ID_COLUMNS), ("hit_meas", MEAS_COLUMNS), ("hit_drift", DRIFT_COLUMNS)):
            if name in f:
                feat[:, [COL[c] for c in cols]] = f[name][()]
        out = {"hit_pos": pos, "hit_feat": feat, "detector": f.attrs["detector"], "bz": float(f.attrs["bz"])}
        for k in ("offsets", "truth", "event"):
            out[k] = f[k][()]
        n_tracks = len(out["truth"])
        if "kf" in f:
            out["kf"], out["kf_ok"] = f["kf"][()], f["kf_ok"][()]
            if "kf_cov" in f:
                out["kf_cov"] = f["kf_cov"][()]
        else:
            out["kf"] = np.full((n_tracks, 5), np.nan)
            out["kf_ok"] = np.zeros(n_tracks, bool)
        if "seed_local" in f:
            local = f["seed_local"][()].astype(np.int64)
            out["seed_idx"] = np.where(local >= 0, local + out["offsets"][:-1, None], -1)
    return out


def write_seed_indices(path, seed_idx, offsets):
    """Store global seed hit indices [T, 3] (-1 for none) as within-track int16 indices."""
    local = np.where(seed_idx >= 0, seed_idx - offsets[:-1, None], -1).astype(np.int16)
    with h5py.File(path, "a") as f:
        if "seed_local" in f:
            del f["seed_local"]
        f.create_dataset("seed_local", data=local, **COMPRESSION)
