import torch

from fcctrackfit.model.fitter import QUANTILES


def pinball_loss(pred, target, quantiles=QUANTILES):
    """Mean quantile (pinball) loss. pred: [B, P, Q]; target: [B, P]. Equal weight per parameter."""
    tau = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    err = target.to(pred.dtype)[..., None] - pred
    return torch.maximum(tau * err, (tau - 1.0) * err).sum(-1).mean()
