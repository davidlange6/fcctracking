"""Seed-guided track-fitting model: seed -> features -> encoder -> quantile corrections."""

import math

import torch
from torch import nn

from fcctrackfit.data.shards import COL
from fcctrackfit.geometry.perigee import PHI, QOP, THETA, wrap_angle
from fcctrackfit.model.mingru import BiMinGRUEncoder
from fcctrackfit.seed.conformal import estimate_seed
from fcctrackfit.seed.residuals import seed_residuals

# +-3, +-2, +-1 sigma Gaussian quantiles and the median (index 3, the point estimate)
QUANTILES = (0.00135, 0.02275, 0.15866, 0.5, 0.84134, 0.97725, 0.99865)
MEDIAN = 3
RESIDUAL_FEATURES = ("seed_du", "seed_dv", "seed_s")


class FourierFeatures(nn.Module):
    """[N, F] in ~[0, 1] -> [N, F * 2 * n_scales] with frequencies pi * 2^k."""

    def __init__(self, n_scales=16):
        super().__init__()
        self.register_buffer("freq", math.pi * 2.0 ** torch.arange(n_scales, dtype=torch.float32))

    def forward(self, x):
        arg = (x[..., None] * self.freq).flatten(-2)
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


def target_scale(seed, resolution):
    """Expected seed error [B, 5] per parameter, hypot(a, b / p) * sin(theta)^-c at the seed
    momentum p = 1 / |q/p| and polar angle.

    resolution: [5, 3] (a, b, c) from data.normalize.fit_resolution ([5, 2] means c = 0);
    (1, 0, 0) means no scaling.
    """
    inv_p = seed[:, QOP].abs().unsqueeze(-1)
    scale = torch.hypot(resolution[:, 0], resolution[:, 1] * inv_p)
    if resolution.shape[1] > 2:
        sin_theta = torch.sin(seed[:, THETA]).abs().clamp(min=1e-3).unsqueeze(-1)
        scale = scale * sin_theta ** -resolution[:, 2]
    return scale


def make_targets(truth, seed, windows, qop_eps, resolution=None):
    """Seed-anchored regression targets [B, 5], scaled by the momentum-dependent expected
    seed error (if resolution is given) and the per-parameter windows."""
    delta = truth - seed
    delta[:, PHI] = wrap_angle(delta[:, PHI])
    delta[:, QOP] = delta[:, QOP] / (seed[:, QOP].abs() + qop_eps)
    if resolution is not None:
        delta = delta / target_scale(seed, resolution)
    return delta / windows


def apply_corrections(seed, delta_scaled, windows, qop_eps, resolution=None):
    """Inverse of make_targets: perigee [B, ..., 5] from scaled corrections."""
    delta = delta_scaled * windows
    if resolution is not None:
        scale = target_scale(seed, resolution)
        delta = delta * (scale.unsqueeze(-2) if delta.dim() == 3 else scale)
    seed = seed.unsqueeze(-2) if delta.dim() == 3 else seed
    out = seed + delta
    out[..., QOP] = seed[..., QOP] + delta[..., QOP] * (seed[..., QOP].abs() + qop_eps)
    out[..., PHI] = wrap_angle(out[..., PHI])
    return out


class TrackFitter(nn.Module):
    def __init__(self, hit_columns, norm, bz, d_model=128, hidden=192, n_layers=2, n_scales=16,
                 head_hidden=128, qop_eps=0.02, residual_scale=0.01):
        super().__init__()
        self.bz = bz
        self.qop_eps = qop_eps
        self.residual_scale = residual_scale
        self.hit_columns = list(hit_columns)
        self.register_buffer("cols", torch.tensor([COL[c] for c in hit_columns]))
        names = self.hit_columns + list(RESIDUAL_FEATURES)
        self.register_buffer("lo", torch.tensor([norm["features"][c][0] for c in names], dtype=torch.float32))
        self.register_buffer("hi", torch.tensor([norm["features"][c][1] for c in names], dtype=torch.float32))
        self.register_buffer("windows", torch.tensor(norm["windows"], dtype=torch.float64))
        # not persistent: rebuilt from norm, so checkpoints from before the momentum scaling still load
        resolution = norm.get("resolution", [[1.0, 0.0, 0.0]] * 5)
        self.register_buffer("resolution", torch.tensor(resolution, dtype=torch.float64), persistent=False)
        n_feat = len(names)
        self.fourier = FourierFeatures(n_scales)
        self.embed = nn.Sequential(
            nn.Linear(n_feat * 2 * n_scales, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU()
        )
        self.encoder = BiMinGRUEncoder(d_model, hidden, n_layers)
        self.head = nn.Sequential(
            nn.RMSNorm(self.encoder.out_dim),
            nn.Linear(self.encoder.out_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 5 * len(QUANTILES)),
        )

    @torch.no_grad()
    def seed(self, pos, seed_idx):
        return estimate_seed(pos[seed_idx[:, 0]], pos[seed_idx[:, 1]], pos[seed_idx[:, 2]], self.bz)

    @torch.no_grad()
    def hit_features(self, pos, feat, cu_seqlens, seed):
        du, dv, s = seed_residuals(pos, cu_seqlens, seed, self.bz)
        res = torch.stack(
            [torch.asinh(du / self.residual_scale), torch.asinh(dv / self.residual_scale), s], dim=-1
        )
        x = torch.cat([feat[:, self.cols], res.float()], dim=-1)
        return ((x - self.lo) / (self.hi - self.lo)).clamp(-0.5, 1.5)

    def forward(self, batch):
        """Returns (quantiles [B, 5, Q] in scaled target units, seed [B, 5] float64)."""
        seed = self.seed(batch["pos"], batch["seed_idx"])
        x = self.hit_features(batch["pos"], batch["feat"], batch["cu_seqlens"], seed)
        h = self.encoder(self.embed(self.fourier(x)), batch["cu_seqlens"])
        q = self.head(h).view(-1, 5, len(QUANTILES))
        return q, seed

    @torch.no_grad()
    def predict(self, batch):
        """Median perigee [B, 5] and all quantile perigees [B, Q, 5], float64."""
        q, seed = self(batch)
        q = torch.sort(q.double(), dim=-1).values  # enforce monotone quantiles
        per_q = apply_corrections(seed, q.transpose(1, 2), self.windows, self.qop_eps, self.resolution)
        return per_q[:, MEDIAN], per_q
