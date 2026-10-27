import math

import numpy as np
import pytest
import torch

from fcctrackfit.data.shards import COL, layer_key
from fcctrackfit.geometry.perigee import (
    PHI, QOP, from_lcio, helix_circle, perigee_from_state, perigee_to_state, to_lcio, wrap_angle,
)  # fmt: skip
from fcctrackfit.seed.conformal import estimate_seed, select_seed_hits
from fcctrackfit.seed.residuals import seed_residuals
from fcctrackfit.toy.sim import simulate


@pytest.fixture(autouse=True)
def float64_default():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def random_perigees(n, gen, pt_lo=0.5, pt_hi=100.0):
    d0 = torch.empty(n).uniform_(-2, 2, generator=gen)
    z0 = torch.empty(n).uniform_(-20, 20, generator=gen)
    phi = torch.empty(n).uniform_(-math.pi, math.pi, generator=gen)
    theta = torch.empty(n).uniform_(0.6, math.pi - 0.6, generator=gen)
    pt = torch.empty(n).uniform_(pt_lo, pt_hi, generator=gen)
    q = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0)
    return torch.stack([d0, z0, phi, theta, q * torch.sin(theta) / pt], -1)


def helix_point(par, s_xy, bz):
    cx, cy, radius, h, psi0, cot = helix_circle(par, bz)
    psi = psi0 + h * s_xy / radius
    return torch.stack([cx + radius * torch.cos(psi), cy + radius * torch.sin(psi), par[:, 1] + s_xy * cot], -1)


def assert_perigee_close(a, b, tol):
    diff = a - b
    diff[:, PHI] = wrap_angle(diff[:, PHI])
    rel_qop = diff[:, QOP] / b[:, QOP]
    assert diff[:, :4].abs().max() < tol, diff[:, :4].abs().max(0)
    assert rel_qop.abs().max() < tol, rel_qop.abs().max()


def test_state_roundtrip():
    gen = torch.Generator().manual_seed(1)
    for bz in (2.0, -2.0):
        par = random_perigees(1000, gen)
        pca, mom, q = perigee_to_state(par, bz)
        assert_perigee_close(perigee_from_state(pca, mom, q, bz), par, 1e-9)


def test_perigee_invariant_along_helix():
    gen = torch.Generator().manual_seed(2)
    bz = 2.0
    par = random_perigees(1000, gen)
    _, mom0, q = perigee_to_state(par, bz)
    s = torch.full((1000,), 300.0)
    pos = helix_point(par, s, bz)
    # momentum direction at the displaced point
    cx, cy, radius, h, psi0, _ = helix_circle(par, bz)
    psi = psi0 + h * s / radius
    pt = torch.hypot(mom0[:, 0], mom0[:, 1])
    mom = torch.stack([-h * pt * torch.sin(psi), h * pt * torch.cos(psi), mom0[:, 2]], -1)
    assert_perigee_close(perigee_from_state(pos, mom, q, bz), par, 1e-8)


def test_lcio_roundtrip_and_sign():
    gen = torch.Generator().manual_seed(3)
    par = random_perigees(500, gen)
    for bz in (2.0, -3.0):
        assert_perigee_close(from_lcio(to_lcio(par, bz), bz), par, 1e-12)
    # LCIO: positive particle in Bz > 0 has omega > 0
    lc = to_lcio(par, 2.0)
    assert torch.all(torch.sign(lc[:, 2]) == torch.sign(par[:, QOP]))


def test_seed_exact_on_ideal_helix():
    gen = torch.Generator().manual_seed(4)
    bz = 2.0
    par = random_perigees(2000, gen, pt_lo=1.0)
    pts = [helix_point(par, torch.full((2000,), s), bz) for s in (20.0, 40.0, 60.0)]
    assert_perigee_close(estimate_seed(*pts, bz), par, 1e-7)


def test_seed_hit_selection():
    offsets = np.array([0, 4, 6])
    key = np.array([1, 1, 2, 3, 5, 6])
    ok = np.ones(6, bool)
    out = select_seed_hits(offsets, key, ok, rule="first3")
    assert out[0].tolist() == [0, 2, 3]
    assert out[1].tolist() == [-1, -1, -1]
    # spread: outermost hit on the outgoing arm, middle closest to mid-radius
    offsets = np.array([0, 6])
    key = np.array([1, 2, 3, 4, 5, 6])
    r = np.array([10.0, 20.0, 50.0, 90.0, 100.0, 80.0])
    assert select_seed_hits(offsets, key, np.ones(6, bool), r).tolist() == [[0, 2, 4]]


def test_residuals_vanish_on_seed_helix():
    gen = torch.Generator().manual_seed(5)
    bz = 2.0
    par = random_perigees(50, gen, pt_lo=1.0)
    s_vals = torch.tensor([10.0, 30.0, 80.0, 200.0, 400.0])
    pos = torch.stack([helix_point(par, torch.full((50,), s), bz) for s in s_vals], 1).reshape(-1, 3)
    cu = torch.arange(0, 251, 5)
    du, dv, s = seed_residuals(pos, cu, par, bz)
    assert du.abs().max() < 1e-8 and dv.abs().max() < 1e-8
    s_expect = (s_vals[None, :] / torch.sin(par[:, 3:4])).reshape(-1)
    assert torch.allclose(s, s_expect, atol=1e-6)


def test_toy_seed_resolution_reasonable():
    shard = simulate(2000, np.random.default_rng(0), pt_range=(5.0, 50.0))
    feat = shard["hit_feat"]
    seed_idx = select_seed_hits(shard["offsets"], layer_key(feat), np.ones(len(feat), bool), feat[:, COL["r"]])
    good = (seed_idx >= 0).all(1)
    pos = torch.from_numpy(shard["hit_pos"])
    i = torch.from_numpy(seed_idx[good])
    seed = estimate_seed(pos[i[:, 0]], pos[i[:, 1]], pos[i[:, 2]], shard["bz"])
    truth = torch.from_numpy(shard["truth"][good])
    rel_qop = ((seed[:, QOP] - truth[:, QOP]) / truth[:, QOP]).abs().median()
    dphi = wrap_angle(seed[:, PHI] - truth[:, PHI]).abs().median()
    # coarse: the seed only has to land in the basin the model corrects
    assert rel_qop < 0.05 and dphi < 0.005
    assert (torch.sign(seed[:, QOP]) == torch.sign(truth[:, QOP])).double().mean() > 0.99
