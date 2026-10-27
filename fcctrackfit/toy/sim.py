"""Toy barrel-only tracker simulation, for CPU development without edm4hep samples.

Analytic helix propagation between concentric cylinders in a uniform field, a
Highland multiple-scattering kick at every layer and Gaussian smearing in r-phi
and z. Layout loosely follows CLD (VXD, IT, OT barrels, Bz = 2 T). Writes the
same shard format as the edm4hep converter (with kf_ok = False).
"""

import numpy as np
import torch

from fcctrackfit.data.shards import COL, HIT_COLUMNS, KIND_SI_PLANE, derived_columns
from fcctrackfit.geometry.perigee import K_BFIELD, perigee_from_state

C_MM_NS = 299.792458
MUON_MASS = 0.1056584

# (system, layer, radius mm, half length mm, sigma_rphi mm, sigma_z mm, x/X0)
CLD_LIKE_BARREL = [
    (1, 0, 17.5, 125.0, 0.003, 0.003, 0.003),
    (1, 1, 18.5, 125.0, 0.003, 0.003, 0.003),
    (1, 2, 37.0, 125.0, 0.003, 0.003, 0.003),
    (1, 3, 38.0, 125.0, 0.003, 0.003, 0.003),
    (1, 4, 57.0, 125.0, 0.003, 0.003, 0.003),
    (1, 5, 58.0, 125.0, 0.003, 0.003, 0.003),
    (3, 0, 127.0, 482.0, 0.007, 0.090, 0.01),
    (3, 1, 400.0, 692.0, 0.007, 0.090, 0.01),
    (3, 2, 670.0, 692.0, 0.007, 0.090, 0.01),
    (5, 0, 1000.0, 1264.0, 0.007, 0.090, 0.01),
    (5, 1, 1568.0, 1264.0, 0.007, 0.090, 0.01),
    (5, 2, 2136.0, 1264.0, 0.007, 0.090, 0.01),
]


def _wrap(a):
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi


def simulate(n, rng, bz=2.0, pt_range=(0.5, 100.0), log_pt=False, max_cos_theta=0.7,
             layers=CLD_LIKE_BARREL, vertex_sigma=(0.005, 0.005, 0.3), multiple_scattering=True):
    """Simulate n single muons from near the origin. Returns a shard dict."""
    lo, hi = pt_range
    pt = np.exp(rng.uniform(np.log(lo), np.log(hi), n)) if log_pt else rng.uniform(lo, hi, n)
    cos_t = rng.uniform(-max_cos_theta, max_cos_theta, n)
    theta = np.arccos(cos_t)
    phi = rng.uniform(-np.pi, np.pi, n)
    q = rng.choice([-1.0, 1.0], n)
    vtx = rng.normal(0.0, vertex_sigma, (n, 3))
    p = pt / np.sin(theta)
    mom = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt / np.tan(theta)], axis=1)

    truth = perigee_from_state(torch.from_numpy(vtx), torch.from_numpy(mom), torch.from_numpy(q), bz).numpy()

    pos = vtx.copy()
    direction = mom / p[:, None]
    time = np.zeros(n)
    beta = p / np.sqrt(p * p + MUON_MASS**2)
    alive = np.ones(n, dtype=bool)
    hits = [[] for _ in range(n)]

    for system, layer, radius, half_len, s_rphi, s_z, x0 in layers:
        dphi = np.arctan2(direction[:, 1], direction[:, 0])
        sin_t = np.hypot(direction[:, 0], direction[:, 1])
        ptc = p * sin_t
        rc = ptc / (K_BFIELD * abs(bz))
        h = -np.sign(q * bz)
        cx = pos[:, 0] - h * rc * np.sin(dphi)
        cy = pos[:, 1] + h * rc * np.cos(dphi)
        d = np.hypot(cx, cy)
        a = (radius**2 - rc**2 + d**2) / (2 * d)
        hgt2 = radius**2 - a**2
        alive &= hgt2 >= 0
        hgt = np.sqrt(np.clip(hgt2, 0, None))
        ux, uy = cx / d, cy / d
        psi_pos = np.arctan2(pos[:, 1] - cy, pos[:, 0] - cx)
        best = np.full(n, np.inf)
        best_xy = np.zeros((n, 2))
        for sgn in (1.0, -1.0):
            px = a * ux - sgn * hgt * uy
            py = a * uy + sgn * hgt * ux
            psi_p = np.arctan2(py - cy, px - cx)
            fwd = np.remainder(h * (psi_p - psi_pos), 2 * np.pi)
            fwd = np.where(fwd < 1e-12, 2 * np.pi, fwd)
            better = fwd < best
            best = np.where(better, fwd, best)
            best_xy[better] = np.stack([px, py], axis=1)[better]
        s_xy = rc * best
        new_z = pos[:, 2] + s_xy * direction[:, 2] / sin_t
        alive &= (np.abs(new_z) <= half_len) & (best < np.pi)  # no loopers in the toy
        path = s_xy / sin_t
        time = time + path / (beta * C_MM_NS)
        pos = np.stack([best_xy[:, 0], best_xy[:, 1], new_z], axis=1)
        dphi = dphi + h * best
        direction = np.stack([sin_t * np.cos(dphi), sin_t * np.sin(dphi), direction[:, 2]], axis=1)

        hit_phi = np.arctan2(pos[:, 1], pos[:, 0]) + rng.normal(0, s_rphi, n) / radius
        hit = np.stack([radius * np.cos(hit_phi), radius * np.sin(hit_phi), pos[:, 2] + rng.normal(0, s_z, n)], 1)
        for i in np.nonzero(alive)[0]:
            hits[i].append((hit[i], time[i], system, layer, s_rphi, s_z))

        if multiple_scattering:
            cos_inc = np.abs(direction[:, 0] * pos[:, 0] + direction[:, 1] * pos[:, 1]) / radius
            x = x0 / np.clip(cos_inc, 0.05, None)
            theta0 = 0.0136 / (beta * p) * np.sqrt(x) * (1 + 0.038 * np.log(x))
            e1 = np.cross(direction, np.array([0.0, 0.0, 1.0]))
            e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
            e2 = np.cross(direction, e1)
            k1 = rng.normal(0, 1, n) * theta0
            k2 = rng.normal(0, 1, n) * theta0
            direction = direction + k1[:, None] * e1 + k2[:, None] * e2
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    return _pack(hits, truth, bz)


def _pack(hits, truth, bz, min_hits=6):
    keep = [i for i, hs in enumerate(hits) if len(hs) >= min_hits]
    counts = np.array([len(hits[i]) for i in keep], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_hits = offsets[-1]
    pos = np.zeros((n_hits, 3))
    feat = np.zeros((n_hits, len(HIT_COLUMNS)), np.float32)
    j = 0
    for i in keep:
        for xyz, t, system, layer, s_rphi, s_z in hits[i]:
            pos[j] = xyz
            feat[j, COL["t"]] = t
            feat[j, COL["system"]] = system
            feat[j, COL["layer"]] = layer
            feat[j, COL["kind"]] = KIND_SI_PLANE
            feat[j, COL["du"]] = s_rphi
            feat[j, COL["dv"]] = s_z
            j += 1
    feat[:, COL["x"]], feat[:, COL["y"]], feat[:, COL["z"]] = pos.T
    feat[:, COL["r"]], feat[:, COL["phi"]], feat[:, COL["theta"]] = derived_columns(pos)
    n = len(keep)
    return dict(
        hit_pos=pos, hit_feat=feat, offsets=offsets, truth=truth[keep],
        kf=np.full((n, 5), np.nan), kf_ok=np.zeros(n, bool), event=np.arange(n), detector="toy", bz=bz,
    )  # fmt: skip
