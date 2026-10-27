"""Per-hit residual features with respect to the seed helix.

For every hit, the reference point on the seed helix is the point at the same
turning angle as seen from the circle centre (radial projection in xy); z
follows from the helix pitch. The residual is decomposed in the curvilinear
frame at that point (U transverse in the xy-plane, V = T x U), giving
(delta_u, delta_v), plus the 3D path length s from the PCA. Turning angles are
unwrapped along the time-ordered hits so loopers keep a monotonic s.

Drift-chamber hits (hit_kind == 2) are not handled yet; see the plan, milestone 6.
"""

import torch

from fcctrackfit.geometry.perigee import helix_circle, wrap_angle


def segment_cumsum(x, cu_seqlens):
    """Cumulative sum of x [N, ...] restarting at every segment boundary."""
    cs = torch.cumsum(x, dim=0)
    starts = cu_seqlens[:-1]
    lengths = cu_seqlens[1:] - starts
    before = torch.where(starts > 0, cs[(starts - 1).clamp(min=0)], torch.zeros_like(cs[0]))
    return cs - torch.repeat_interleave(before, lengths, dim=0)


def seed_residuals(pos, cu_seqlens, seed, bz):
    """Returns (delta_u, delta_v, s) [N] each, float64.

    pos: [N, 3] float64 packed hit positions; cu_seqlens: [T+1] int64;
    seed: [T, 5] float64 seed perigee.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    cx, cy, radius, h, psi0, cot = (torch.repeat_interleave(v, lengths) for v in helix_circle(seed, bz))
    _, z0, _, theta, _ = torch.repeat_interleave(seed, lengths, dim=0).unbind(-1)

    psi = torch.atan2(pos[:, 1] - cy, pos[:, 0] - cx)
    first = torch.zeros_like(psi, dtype=torch.bool)
    first[cu_seqlens[:-1]] = True
    prev = torch.roll(psi, 1)
    # turning angle of the first hit relative to the PCA, then hit-to-hit increments
    step = torch.where(first, wrap_angle(psi - psi0), wrap_angle(psi - prev))
    dpsi = segment_cumsum(step, cu_seqlens)

    s_xy = h * radius * dpsi
    ref_x = cx + radius * torch.cos(psi)
    ref_y = cy + radius * torch.sin(psi)
    ref_z = z0 + s_xy * cot

    # tangent direction at the reference point
    tx = -h * torch.sin(psi) * torch.sin(theta)
    ty = h * torch.cos(psi) * torch.sin(theta)
    tz = torch.cos(theta)
    # U = z x T normalised, V = T x U
    ux, uy = -ty, tx
    un = torch.hypot(ux, uy)
    ux, uy = ux / un, uy / un
    vx, vy, vz = -tz * uy, tz * ux, tx * uy - ty * ux

    rx, ry, rz = pos[:, 0] - ref_x, pos[:, 1] - ref_y, pos[:, 2] - ref_z
    du = rx * ux + ry * uy
    dv = rx * vx + ry * vy + rz * vz
    s = s_xy / torch.sin(theta)
    return du, dv, s

