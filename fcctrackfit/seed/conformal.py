"""Analytic three-hit helix seed.

Vectorised torch port of the ACTS ``estimateTrackParamsFromSeed`` conformal-map
estimate: in a frame with origin at the bottom hit and x-axis towards the middle
hit, the map (x, y) -> (x, y) / (x^2 + y^2) turns the circle through the three
hits into a straight line v = A u + B, from which curvature and direction at the
bottom hit follow in closed form. Unlike ACTS, cot(theta) uses the exact helical
transverse arc length between bottom and top hit. The state at the bottom hit is
then transported analytically to the perigee. Run in float64.
"""

import math

import torch

from fcctrackfit.geometry.perigee import K_BFIELD, perigee_from_state

C_MM_PER_NS = 299.792458


def estimate_seed(x1, x2, x3, bz):
    """Seed perigee [..., 5] from three space points [..., 3] ordered inner to outer."""
    rel = x2[..., :2] - x1[..., :2]
    ex = rel / torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
    ey = torch.stack([-ex[..., 1], ex[..., 0]], dim=-1)

    d2 = x2[..., :2] - x1[..., :2]
    d3 = x3[..., :2] - x1[..., :2]
    l2x = (d2 * ex).sum(-1)
    l3x = (d3 * ex).sum(-1)
    l3y = (d3 * ey).sum(-1)

    u1 = 1.0 / l2x  # middle hit sits on the local x-axis: v1 = 0
    rn3 = l3x * l3x + l3y * l3y
    u3 = l3x / rn3
    v3 = l3y / rn3
    a = v3 / (u3 - u1)
    b = v3 - a * u3

    # signed curvature: rho > 0 for clockwise rotation (as seen from +z)
    hyp = torch.hypot(torch.ones_like(a), a)
    rho = -2.0 * b / hyp
    q_over_pt = rho / (K_BFIELD * bz)

    dir_xy = ex + a.unsqueeze(-1) * ey
    dir_xy = dir_xy / torch.linalg.vector_norm(dir_xy, dim=-1, keepdim=True)

    chord = torch.sqrt(rn3)
    half = (0.5 * chord * rho.abs()).clamp(max=1.0)
    # 2 R asin(c / 2R), with the straight-line limit handled by the ratio
    s_xy = torch.where(half > 1e-12, chord * torch.asin(half) / half.clamp(min=1e-12), chord)
    cot_theta = (x3[..., 2] - x1[..., 2]) / s_xy

    pt = 1.0 / q_over_pt.abs()
    mom = torch.stack([pt * dir_xy[..., 0], pt * dir_xy[..., 1], pt * cot_theta], dim=-1)
    return perigee_from_state(x1, mom, torch.sign(q_over_pt), bz)


def select_seed_hits(offsets, layer_key, is_seedable, r=None, rule="spread", pos=None, time=None, bz=None,
                     max_turn=0.4 * math.pi):  # fmt: skip
    """Indices (global, into the flat hit array) of the seed triplet per track.

    rule "first3": the first three seedable hits (stored, i.e. time, order) on
    distinct layers. With closely spaced inner layers this gives a short lever
    arm and a poor curvature at high pT.
    rule "spread": the first seedable hit, the outermost seedable hit on the
    outgoing arm and the hit closest to mid-radius in between, all on distinct
    layers. Requires r. The arm ends where r first decreases. With pos, time and
    bz it also ends where the transverse turning angle since the first hit,
    estimated as c dt sin(theta) / R (beta = 1, an upper bound) from a "first3"
    helix, exceeds max_turn: a looper can complete whole turns between two hits,
    or return to a layer, without r decreasing.
    Tracks without a valid triplet get -1 entries.

    offsets: [T+1] int; layer_key: [N] int (unique per sensitive layer);
    is_seedable: [N] bool; r: [N] float; pos: [N, 3] mm; time: [N] ns. numpy arrays.
    """
    import numpy as np

    n_tracks = len(offsets) - 1
    out = np.full((n_tracks, 3), -1, dtype=np.int64)
    turn_rate = None
    if rule == "spread" and time is not None:
        # turning angle per ns of flight from a rough inner helix
        inner = select_seed_hits(offsets, layer_key, is_seedable, rule="first3")
        ok = (inner >= 0).all(1)
        turn_rate = np.zeros(n_tracks)
        if ok.any():
            x = torch.as_tensor(pos, dtype=torch.float64)
            rough = estimate_seed(*(x[inner[ok, k]] for k in range(3)), bz).numpy()
            radius = np.sin(rough[:, 3]) / (np.abs(rough[:, 4]) * K_BFIELD * abs(bz))
            turn_rate[ok] = C_MM_PER_NS * np.sin(rough[:, 3]) / radius
    for t in range(n_tracks):
        cand = [i for i in range(offsets[t], offsets[t + 1]) if is_seedable[i]]
        if rule == "first3":
            found, seen = [], set()
            for i in cand:
                if layer_key[i] not in seen:
                    found.append(i)
                    seen.add(layer_key[i])
                    if len(found) == 3:
                        out[t] = found
                        break
            continue
        if rule != "spread":
            raise ValueError(f"unknown seed rule {rule!r}")
        arm = []
        for i in cand:
            if arm and r[i] < r[arm[-1]]:
                break
            if arm and turn_rate is not None and turn_rate[t] * (time[i] - time[arm[0]]) > max_turn:
                break
            arm.append(i)
        if len(arm) < 3:
            continue
        first, last = arm[0], arm[-1]
        if layer_key[first] == layer_key[last]:
            continue
        r_mid = 0.5 * (r[first] + r[last])
        middle = [i for i in arm[1:-1] if layer_key[i] not in (layer_key[first], layer_key[last])]
        if middle:
            out[t] = (first, min(middle, key=lambda i: abs(r[i] - r_mid)), last)
    return out
