"""Optimisers and schedules for the two training phases."""

import math

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion (Chen et al. 2023): sign of an interpolated momentum, decoupled weight decay."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(torch.sign(m * beta1 + p.grad * (1 - beta1)), alpha=-group["lr"])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return loss


class MuonAdamW:
    """Muon for the hidden 2-D weight matrices, AdamW for everything else.

    The first embedding layer and the final head layer stay on AdamW, as is
    usual for Muon. Behaves like a single optimiser for the training loop.
    """

    def __init__(self, model, lr_muon=0.02, lr_adamw=3e-4, weight_decay=0.0):
        first = model.embed[0].weight
        last = model.head[-1].weight
        muon, adamw = [], []
        for p in model.parameters():
            (muon if p.ndim == 2 and p is not first and p is not last else adamw).append(p)
        self.optimizers = [
            torch.optim.Muon(muon, lr=lr_muon, weight_decay=weight_decay),
            torch.optim.AdamW(adamw, lr=lr_adamw, weight_decay=weight_decay),
        ]

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, states):
        for o, s in zip(self.optimizers, states):
            o.load_state_dict(s)


def lr_factor(step, total, schedule, warmup_frac=0.05, decay_frac=0.2):
    """Multiplicative LR factor. one_cycle: linear warmup, cosine to zero.
    wsd: warmup, constant, then linear decay over the last decay_frac."""
    warm = max(1, int(warmup_frac * total))
    if step < warm:
        return (step + 1) / warm
    if schedule == "one_cycle":
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    if schedule == "wsd":
        decay_start = int((1 - decay_frac) * total)
        return 1.0 if step < decay_start else max(0.0, (total - step) / max(1, total - decay_start))
    raise ValueError(schedule)


def build_optimizer(model, cfg):
    if cfg["optimizer"] == "lion":
        opt = Lion(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    elif cfg["optimizer"] == "muon_adamw":
        opt = MuonAdamW(model, cfg["lr_muon"], cfg["lr"], cfg.get("weight_decay", 0.0))
    else:
        raise ValueError(cfg["optimizer"])
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return opt


def set_lr(opt, factor):
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * factor
