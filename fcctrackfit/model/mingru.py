"""Bidirectional minGRU over packed (unpadded) track sequences.

Per direction and channel:  z_t = sigmoid(W_z m_t),  n_t = W_n m_t,
                            h_t = h_{t-1} + z_t * (n_t - h_{t-1}),  h_0 = 0.
Hit features stay packed ([N, D], track boundaries in cu_seqlens); only the scan
pads to [B, L, H]. The scan is a plain loop over hit positions (L <= ~20 for
silicon trackers), which is exact and runs anywhere; a fused Triton kernel is
the planned replacement for inference throughput.
"""

import torch
from torch import nn


class PackedIndex:
    """Index helpers mapping packed hits to padded forward / reversed layouts."""

    def __init__(self, cu_seqlens):
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        self.lengths = lengths
        self.batch = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), lengths)
        self.pos = torch.arange(int(cu_seqlens[-1]), device=lengths.device) - torch.repeat_interleave(
            cu_seqlens[:-1], lengths
        )
        self.rev = self.lengths[self.batch] - 1 - self.pos
        self.max_len = int(lengths.max())
        self.mask = torch.arange(self.max_len, device=lengths.device)[None, :] < lengths[:, None]

    def pad(self, x, reverse=False):
        out = x.new_zeros(len(self.lengths), self.max_len, x.shape[-1])
        out[self.batch, self.rev if reverse else self.pos] = x
        return out

    def unpad(self, x, reverse=False):
        return x[self.batch, self.rev if reverse else self.pos]


def mingru_scan(k, n, mask):
    """Sequential minGRU scan. k, n: [B, L, H] gate logits / candidates; mask: [B, L]."""
    z = torch.sigmoid(k) * mask[..., None]
    h = torch.zeros_like(n[:, 0])
    out = []
    for t in range(n.shape[1]):
        h = h + z[:, t] * (n[:, t] - h)
        out.append(h)
    return torch.stack(out, dim=1)


class BiMinGRULayer(nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.hidden = hidden
        self.proj = nn.Linear(d_in, 4 * hidden)

    def forward(self, x, index):
        """x: [N, d_in] packed. Returns packed [N, 2H] and final states (fwd, bwd) [B, H]."""
        kf, nf, kb, nb = self.proj(x).chunk(4, dim=-1)
        hf = mingru_scan(index.pad(kf), index.pad(nf), index.mask)
        hb = mingru_scan(index.pad(kb, reverse=True), index.pad(nb, reverse=True), index.mask)
        last = index.lengths - 1
        b = torch.arange(len(last), device=x.device)
        out = torch.cat([index.unpad(hf), index.unpad(hb, reverse=True)], dim=-1)
        return out, (hf[b, last], hb[b, last])


class BiMinGRUEncoder(nn.Module):
    def __init__(self, d_model=128, hidden=192, n_layers=2):
        super().__init__()
        dims = [d_model] + [2 * hidden] * (n_layers - 1)
        self.layers = nn.ModuleList(BiMinGRULayer(d, hidden) for d in dims)
        self.out_dim = 2 * hidden

    def forward(self, x, cu_seqlens):
        """Returns [B, 2H]: forward state at the outermost hit, reverse state at the innermost."""
        index = PackedIndex(cu_seqlens)
        for layer in self.layers:
            x, (fwd, bwd) = layer(x, index)
        return torch.cat([fwd, bwd], dim=-1)
