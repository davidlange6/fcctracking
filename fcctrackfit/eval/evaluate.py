"""Stage 4: inference on test shards and resolution summaries vs the KF baseline."""

import numpy as np
import torch

from fcctrackfit.data.dataset import PackedTrackStream, to_device
from fcctrackfit.eval.metrics import bootstrap_ratio, clipped_rms, rms, rms_error
from fcctrackfit.geometry.perigee import PARAMS, PHI


def perigee_residuals(pred, truth):
    d = pred - truth
    d[:, PHI] = np.remainder(d[:, PHI] + np.pi, 2 * np.pi) - np.pi
    return d


@torch.no_grad()
def run_inference(model, paths, batch_size, device, max_tracks=None):
    """Returns dict of numpy arrays: pred, quantiles, seed, truth, kf, kf_ok."""
    model.eval()
    stream = PackedTrackStream(paths, batch_size, shuffle=False, max_tracks=max_tracks)
    out = {k: [] for k in ("pred", "quantiles", "seed", "truth", "kf", "kf_ok")}
    for batch in stream:
        batch = to_device(batch, device)
        pred, per_q = model.predict(batch)
        out["pred"].append(pred.cpu().numpy())
        out["quantiles"].append(per_q.cpu().numpy())
        out["seed"].append(model.seed(batch["pos"], batch["seed_idx"]).cpu().numpy())
        for k in ("truth", "kf", "kf_ok"):
            out[k].append(batch[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _stats(res):
    c, mean, frac = clipped_rms(res)
    return {"clipped_rms": c, "clipped_rms_err": rms_error(c, len(res)), "clipped_frac": frac, "mean": mean,
            "rms": rms(res), "n": int(len(res))}  # fmt: skip


def summarize(res, bootstrap=400, selection=None):
    """Per-parameter statistics for model, seed and (on tracks fitted by both) the KF.

    When KF results exist, all estimators are evaluated on the common
    (kf_ok) tracks, as in the paper; otherwise on all tracks.
    """
    sel = np.ones(len(res["truth"]), bool) if selection is None else selection.copy()
    has_kf = bool(res["kf_ok"][sel].any())
    if has_kf:
        sel &= res["kf_ok"]
    truth = res["truth"][sel]
    r_model = perigee_residuals(res["pred"][sel], truth)
    r_seed = perigee_residuals(res["seed"][sel], truth)
    r_kf = perigee_residuals(res["kf"][sel], truth) if has_kf else None
    out = {"n_tracks": int(sel.sum()), "has_kf": has_kf, "params": {}}
    for i, p in enumerate(PARAMS):
        entry = {"model": _stats(r_model[:, i]), "seed": _stats(r_seed[:, i])}
        if has_kf:
            entry["kf"] = _stats(r_kf[:, i])
            if bootstrap:
                ratio, err = bootstrap_ratio(r_model[:, i], r_kf[:, i], n_boot=bootstrap)
                ur, uerr = bootstrap_ratio(r_model[:, i], r_kf[:, i], stat=rms, n_boot=bootstrap)
                entry["ratio_clipped"], entry["ratio_clipped_err"] = ratio, err
                entry["ratio_rms"], entry["ratio_rms_err"] = ur, uerr
            else:
                entry["ratio_clipped"] = entry["model"]["clipped_rms"] / entry["kf"]["clipped_rms"]
        out["params"][p] = entry
    # quantile calibration: fraction of truth below each predicted quantile
    q = res["quantiles"][sel]
    below = q > truth[:, None, :]
    out["quantile_coverage"] = {p: below[:, :, i].mean(0).tolist() for i, p in enumerate(PARAMS)}
    return out


def format_table(summary):
    ref = "kf" if summary["has_kf"] else "seed"
    lines = [f"{summary['n_tracks']} tracks; reference = {ref}",
             f"{'param':6s} {'model':>11s} {ref:>11s} {'ratio':>14s} {'model rms':>11s} {ref + ' rms':>11s}"]  # fmt: skip
    for p, e in summary["params"].items():
        m, r = e["model"], e[ref]
        ratio = m["clipped_rms"] / r["clipped_rms"]
        err = f"({e['ratio_clipped_err']:.3f})" if "ratio_clipped_err" in e else ""
        lines.append(f"{p:6s} {m['clipped_rms']:11.4g} {r['clipped_rms']:11.4g} {ratio:7.4f}{err:7s} "
                     f"{m['rms']:11.4g} {r['rms']:11.4g}")  # fmt: skip
    return "\n".join(lines)
