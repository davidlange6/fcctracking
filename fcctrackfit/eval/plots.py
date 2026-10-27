"""Resolution-vs-variable plots with ratio panels (paper Figs. 3-4)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fcctrackfit.eval.evaluate import perigee_residuals  # noqa: E402
from fcctrackfit.eval.metrics import binned_clipped_rms  # noqa: E402
from fcctrackfit.geometry.perigee import PARAMS  # noqa: E402

UNITS = {"d0": (1e3, "µm"), "z0": (1e3, "µm"), "phi": (1e3, "mrad"), "theta": (1e3, "mrad"), "qop": (1e3, "1e-3/GeV")}
COLORS = {"model": "#2563eb", "kf": "#dc2626", "seed": "#6b7280"}


def resolution_vs(res, var_name, edges, path, title=""):
    truth = res["truth"]
    sel = res["kf_ok"] if res["kf_ok"].any() else np.ones(len(truth), bool)
    truth = truth[sel]
    var = {"cos_theta": np.cos(truth[:, 3]), "abs_cos_theta": np.abs(np.cos(truth[:, 3])),
           "pt": np.sin(truth[:, 3]) / np.abs(truth[:, 4])}[var_name]  # fmt: skip
    estimators = {"model": res["pred"][sel]}
    if res["kf_ok"].any():
        estimators["kf"] = res["kf"][sel]
    else:
        estimators["seed"] = res["seed"][sel]
    ref = "kf" if "kf" in estimators else "seed"
    centers = 0.5 * (edges[1:] + edges[:-1])

    fig, axes = plt.subplots(2, 5, figsize=(22, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for i, p in enumerate(PARAMS):
        scale, unit = UNITS[p]
        curves = {}
        for name, est in estimators.items():
            r, e, _ = binned_clipped_rms(perigee_residuals(est, truth)[:, i], var, edges)
            curves[name] = (r, e)
            axes[0, i].errorbar(centers, r * scale, e * scale, fmt="o-", ms=3, color=COLORS[name], label=name)
        axes[0, i].set_title(p)
        axes[0, i].set_ylabel(f"iter-3σ RMS [{unit}]")
        axes[0, i].legend(frameon=False)
        ratio = curves["model"][0] / curves[ref][0]
        axes[1, i].plot(centers, ratio, "o-", ms=3, color=COLORS["model"])
        axes[1, i].axhline(1.0, color="k", lw=0.8, ls=":")
        axes[1, i].set_ylabel(f"model/{ref}")
        axes[1, i].set_xlabel(var_name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
