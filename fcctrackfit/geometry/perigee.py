"""Perigee parametrisation and helix geometry.

Units throughout the package: mm, GeV, ns, Tesla.

Internal ("paper") convention, reference point at the origin, PCA to the z-axis:
    p = (d0, z0, phi, theta, q/p)            index order given by PARAMS
    PCA = (-d0 sin(phi), d0 cos(phi), z0)    (ACTS and LCIO share this d0 sign)

LCIO / edm4hep TrackState convention:
    (D0, phi, omega, Z0, tanLambda) with omega = +1/R for clockwise rotation
    seen from +z (i.e. a positive particle for Bz > 0).

All functions are written for torch tensors (use float64) and broadcast over
leading dimensions.
"""

import math

import torch

PARAMS = ("d0", "z0", "phi", "theta", "qop")
D0, Z0, PHI, THETA, QOP = range(5)

# pT [GeV] = K_BFIELD * B [T] * R [mm]
K_BFIELD = 0.299792458e-3


def wrap_angle(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def perigee_from_state(pos, mom, q, bz):
    """Perigee w.r.t. the z-axis of the helix through ``pos`` with momentum ``mom``.

    pos, mom: [..., 3]; q: [...] (charge sign is what matters); bz: float.
    The PCA is taken on the branch of the circle nearest to ``pos`` in turning
    angle, so z0 is propagated backwards for displaced production points.
    """
    x, y, z = pos.unbind(-1)
    px, py, pz = mom.unbind(-1)
    pt = torch.hypot(px, py)
    radius = pt / (K_BFIELD * abs(bz))
    h = -torch.sign(q * bz)  # +1: counter-clockwise seen from +z
    phiv = torch.atan2(py, px)
    cx = x - h * radius * torch.sin(phiv)
    cy = y + h * radius * torch.cos(phiv)
    dc = torch.hypot(cx, cy)
    # point on the circle closest to the origin, radial vector from the centre
    ex = -radius * cx / dc
    ey = -radius * cy / dc
    pca_x = cx + ex
    pca_y = cy + ey
    phi0 = torch.atan2(h * ex, -h * ey)
    d0 = -pca_x * torch.sin(phi0) + pca_y * torch.cos(phi0)
    dpsi = wrap_angle(torch.atan2(ey, ex) - torch.atan2(y - cy, x - cx))
    s_xy = h * radius * dpsi
    z0 = z + s_xy * pz / pt
    theta = torch.atan2(pt, pz)
    p = torch.sqrt(pt * pt + pz * pz)
    return torch.stack([d0, z0, phi0, theta, torch.sign(q) / p], dim=-1)


def perigee_to_state(par, bz):
    """Inverse of :func:`perigee_from_state`: returns (pca [...,3], mom [...,3], q [...])."""
    d0, z0, phi, theta, qop = par.unbind(-1)
    pca = torch.stack([-d0 * torch.sin(phi), d0 * torch.cos(phi), z0], dim=-1)
    p = 1.0 / qop.abs()
    mom = torch.stack(
        [p * torch.sin(theta) * torch.cos(phi), p * torch.sin(theta) * torch.sin(phi), p * torch.cos(theta)],
        dim=-1,
    )
    return pca, mom, torch.sign(qop)


def helix_circle(par, bz):
    """Transverse circle of a perigee: (cx, cy, radius, h, psi0, cot_theta).

    psi0 is the angle of the PCA as seen from the centre; a point at transverse
    arc length s from the PCA sits at angle psi0 + h * s / radius.
    """
    d0, z0, phi, theta, qop = par.unbind(-1)
    pt = torch.sin(theta) / qop.abs()
    radius = pt / (K_BFIELD * abs(bz))
    h = -torch.sign(qop * bz)
    pca_x = -d0 * torch.sin(phi)
    pca_y = d0 * torch.cos(phi)
    cx = pca_x - h * radius * torch.sin(phi)
    cy = pca_y + h * radius * torch.cos(phi)
    psi0 = torch.atan2(pca_y - cy, pca_x - cx)
    return cx, cy, radius, h, psi0, 1.0 / torch.tan(theta)


def to_lcio(par, bz):
    """(d0, z0, phi, theta, q/p) -> LCIO (D0, phi, omega, Z0, tanLambda)."""
    d0, z0, phi, theta, qop = par.unbind(-1)
    omega = torch.sign(qop * bz) * K_BFIELD * abs(bz) * qop.abs() / torch.sin(theta)
    return torch.stack([d0, phi, omega, z0, 1.0 / torch.tan(theta)], dim=-1)


def from_lcio(lcio, bz):
    """LCIO (D0, phi, omega, Z0, tanLambda) -> (d0, z0, phi, theta, q/p)."""
    d0, phi, omega, z0, tanl = lcio.unbind(-1)
    theta = math.pi / 2 - torch.atan(tanl)
    pt = K_BFIELD * abs(bz) / omega.abs()
    q = torch.sign(omega) * math.copysign(1.0, bz)
    return torch.stack([d0, z0, phi, theta, q * torch.sin(theta) / pt], dim=-1)
