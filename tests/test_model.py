import numpy as np
import torch

from fcctrackfit.data.dataset import gather_tracks
from fcctrackfit.data.normalize import compute_norm, fit_resolution
from fcctrackfit.data.shards import COL, layer_key
from fcctrackfit.model.fitter import QUANTILES, TrackFitter, apply_corrections, make_targets
from fcctrackfit.model.mingru import BiMinGRULayer, PackedIndex
from fcctrackfit.seed.conformal import select_seed_hits
from fcctrackfit.toy.sim import simulate
from fcctrackfit.train.losses import pinball_loss

COLUMNS = ["x", "y", "z", "t", "r", "phi", "theta", "system", "layer", "side", "du", "dv"]


def test_bimingru_matches_per_track_loop():
    torch.manual_seed(0)
    layer = BiMinGRULayer(8, 5)
    lengths = torch.tensor([3, 7, 1, 5])
    cu = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, 0)])
    x = torch.randn(int(cu[-1]), 8)
    out, (fwd, bwd) = layer(x, PackedIndex(cu))
    for b in range(len(lengths)):
        xs = x[cu[b] : cu[b + 1]]
        kf, nf, kb, nb = layer.proj(xs).chunk(4, -1)
        h = torch.zeros(5)
        hf = []
        for t in range(len(xs)):
            h = h + torch.sigmoid(kf[t]) * (nf[t] - h)
            hf.append(h)
        h = torch.zeros(5)
        hb = [None] * len(xs)
        for t in reversed(range(len(xs))):
            h = h + torch.sigmoid(kb[t]) * (nb[t] - h)
            hb[t] = h
        expect = torch.cat([torch.stack(hf), torch.stack(hb)], -1)
        assert torch.allclose(out[cu[b] : cu[b + 1]], expect, atol=1e-6)
        assert torch.allclose(fwd[b], hf[-1], atol=1e-6) and torch.allclose(bwd[b], hb[0], atol=1e-6)


def test_targets_roundtrip():
    gen = torch.Generator().manual_seed(0)
    seed = torch.randn(100, 5, generator=gen, dtype=torch.float64)
    truth = seed + 0.01 * torch.randn(100, 5, generator=gen, dtype=torch.float64)
    windows = torch.tensor([0.1, 0.5, 0.01, 0.01, 0.05], dtype=torch.float64)
    t = make_targets(truth.clone(), seed, windows, 0.02)
    back = apply_corrections(seed, t, windows, 0.02)
    assert torch.allclose(back, truth, atol=1e-12)
    resolution = torch.tensor([[0.003, 0.05, 1.5], [0.004, 0.1, 2.5], [1e-4, 0.004, 1], [1e-4, 0.004, 0],
                               [0.002, 0.01, 1]], dtype=torch.float64)  # fmt: skip
    t = make_targets(truth.clone(), seed, windows, 0.02, resolution)
    assert torch.allclose(apply_corrections(seed, t, windows, 0.02, resolution), truth, atol=1e-12)
    per_q = apply_corrections(seed, t.unsqueeze(1).expand(-1, 3, -1), windows, 0.02, resolution)
    assert torch.allclose(per_q, truth.unsqueeze(1).expand(-1, 3, -1), atol=1e-12)


def test_fit_resolution_recovers_model():
    rng = np.random.default_rng(0)
    n = 400_000
    inv_p = 1 / np.exp(rng.uniform(np.log(0.2), np.log(50), n))
    sin_theta = np.sqrt(1 - rng.uniform(-0.99, 0.99, n) ** 2)
    a, b, c = np.array([0.003, 1e-4]), np.array([0.05, 0.004]), np.array([1.5, 0.0])
    delta = rng.normal(size=(n, 2)) * np.hypot(a, b * inv_p[:, None]) * sin_theta[:, None] ** -c
    res = fit_resolution(delta, inv_p, sin_theta)
    assert np.allclose(res[:, 0], a, rtol=0.05) and np.allclose(res[:, 1], b, rtol=0.05)
    assert np.allclose(res[:, 2], c, atol=0.05)


def test_pinball_minimised_at_quantile():
    y = torch.randn(20000, 1)
    grid = torch.linspace(-3, 3, 601)
    losses = [pinball_loss(torch.full((20000, 1, 1), float(g)), y, quantiles=(0.8413,)).item() for g in grid]
    assert abs(grid[int(np.argmin(losses))].item() - 1.0) < 0.05


def _toy_batch(n=300):
    shard = simulate(n, np.random.default_rng(3), pt_range=(1.0, 50.0))
    feat = shard["hit_feat"]
    shard["seed_idx"] = select_seed_hits(shard["offsets"], layer_key(feat), np.ones(len(feat), bool), feat[:, COL["r"]])
    return shard


def test_fitter_forward_and_batch_independence(tmp_path):
    from fcctrackfit.data.shards import write_seed_indices, write_shard

    shard = _toy_batch()
    path = tmp_path / "s.h5"
    write_shard(path, **{k: v for k, v in shard.items() if k != "seed_idx"})
    write_seed_indices(path, shard["seed_idx"], shard["offsets"])
    norm = compute_norm([path], COLUMNS, 2.0, 0.02, 0.01)
    model = TrackFitter(COLUMNS, norm, 2.0, d_model=32, hidden=16, head_hidden=16).eval()
    idx = np.nonzero((shard["seed_idx"] >= 0).all(1))[0]
    full = gather_tracks(shard, idx[:40])
    q, seed = model(full)
    assert q.shape == (40, 5, len(QUANTILES)) and seed.dtype == torch.float64
    # a track's output must not depend on the other tracks in the batch
    q_sub, _ = model(gather_tracks(shard, idx[[5, 17]]))
    assert torch.allclose(q_sub, q[[5, 17]], atol=1e-5)
    pred, per_q = model.predict(full)
    assert pred.shape == (40, 5) and per_q.shape == (40, len(QUANTILES), 5)
