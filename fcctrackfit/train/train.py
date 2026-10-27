"""Training loop (single process or torchrun DDP), strict fp32."""

import csv
import glob
import os
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
import yaml

from fcctrackfit.data.dataset import PackedTrackStream, to_device
from fcctrackfit.data.normalize import load_norm
from fcctrackfit.eval.evaluate import run_inference, summarize
from fcctrackfit.model.fitter import TrackFitter, make_targets
from fcctrackfit.train.losses import pinball_loss
from fcctrackfit.train.optim import build_optimizer, lr_factor, set_lr


def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def expand(patterns):
    paths = sorted(p for pat in ([patterns] if isinstance(patterns, str) else patterns) for p in glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"no shards match {patterns}")
    return paths


def count_usable(paths):
    n = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            n += int((f["seed_local"][()] >= 0).all(1).sum())
    return n


def build_model(cfg, norm, bz):
    m = cfg["model"]
    return TrackFitter(
        m["hit_columns"], norm, bz, d_model=m["d_model"], hidden=m["hidden"], n_layers=m["n_layers"],
        n_scales=m["n_scales"], head_hidden=m["head_hidden"], qop_eps=norm["qop_eps"],
        residual_scale=norm["residual_scale"],
    )  # fmt: skip


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["cfg"], ckpt["norm"], ckpt["bz"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def train(cfg):
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    strict_fp32()
    torch.manual_seed(cfg.get("seed", 0))

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    norm = load_norm(cfg["data"]["norm"])
    train_paths = expand(cfg["data"]["train"])
    val_paths = expand(cfg["data"]["val"])
    with h5py.File(train_paths[0], "r") as f:
        bz = float(f.attrs["bz"])

    model = build_model(cfg, norm, bz).to(device)
    if cfg.get("init_from"):
        model.load_state_dict(torch.load(cfg["init_from"], map_location=device, weights_only=False)["model"])
    if rank == 0:
        print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device] if device.type == "cuda" else None) \
        if ddp else model  # fmt: skip

    t = cfg["train"]
    opt = build_optimizer(model, t)
    stream = PackedTrackStream(train_paths, t["batch_size"], shuffle=True, rank=rank, world_size=world,
                               seed=cfg.get("seed", 0), max_tracks=t.get("max_tracks_per_epoch"))  # fmt: skip
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=t.get("workers", 0),
                                         pin_memory=device.type == "cuda", persistent_workers=False)  # fmt: skip
    n_tracks = min(count_usable(train_paths), t.get("max_tracks_per_epoch") or np.inf)
    steps_per_epoch = max(1, int(n_tracks // (t["batch_size"] * world)))
    total = steps_per_epoch * t["epochs"]

    start_epoch, step = 0, 0
    resume = os.path.join(out_dir, "last.pt")
    if os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch, step = ckpt["epoch"] + 1, ckpt["step"]
        if rank == 0:
            print(f"resumed from {resume} at epoch {start_epoch}")

    log = open(os.path.join(out_dir, "log.csv"), "a", newline="") if rank == 0 else None
    writer = csv.writer(log) if log else None
    for epoch in range(start_epoch, t["epochs"]):
        stream.set_epoch(epoch)
        net.train()
        t0, running, n_seen = time.time(), 0.0, 0
        for batch in loader:
            batch = to_device(batch, device)
            set_lr(opt, lr_factor(step, total, t["schedule"], t.get("warmup_frac", 0.05), t.get("decay_frac", 0.2)))
            q, seed = net(batch)
            target = make_targets(batch["truth"], seed, model.windows, model.qop_eps)
            loss = pinball_loss(q, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if t.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            step += 1
            running += loss.item()
            n_seen += 1
            if rank == 0 and step % t.get("log_every", 100) == 0:
                print(f"epoch {epoch} step {step}/{total} loss {running / n_seen:.5f} "
                      f"lr {opt.param_groups[0]['lr']:.3g} {(time.time() - t0) / n_seen * 1e3:.1f} ms/step", flush=True)  # fmt: skip

        if rank == 0:
            res = run_inference(model, val_paths, t["batch_size"] * 4, device, max_tracks=cfg["data"].get("val_tracks"))
            summary = summarize(res, bootstrap=0)
            msg = " ".join(f"{p}:{s['model']['clipped_rms']:.3g}(seed {s['seed']['clipped_rms']:.3g})"
                           for p, s in summary["params"].items())  # fmt: skip
            print(f"epoch {epoch} train_loss {running / max(n_seen, 1):.5f} val clipped RMS {msg}", flush=True)
            writer.writerow([epoch, step, running / max(n_seen, 1)] +
                            [s["model"]["clipped_rms"] for s in summary["params"].values()])  # fmt: skip
            log.flush()
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step,
                    "cfg": cfg, "norm": norm, "bz": bz}  # fmt: skip
            torch.save(ckpt, resume + ".tmp")
            os.replace(resume + ".tmp", resume)
        if ddp:
            dist.barrier()
    if rank == 0:
        torch.save(torch.load(resume, weights_only=False), os.path.join(out_dir, "final.pt"))
        log.close()
    if ddp:
        dist.destroy_process_group()


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        key, value = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return cfg
