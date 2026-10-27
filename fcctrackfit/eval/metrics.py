"""Track-fit resolution metrics (numpy)."""

import numpy as np


def clipped_rms(x, nsigma=3.0, max_iter=100):
    """Iterative n-sigma clipped RMS about the mean.

    Returns (rms, mean, clipped_fraction). Residuals outside mean +- nsigma*rms
    are removed and the statistics recomputed until the kept set is stable.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    keep = np.ones(len(x), dtype=bool)
    for _ in range(max_iter):
        mean = x[keep].mean()
        rms = x[keep].std()
        new = np.abs(x - mean) <= nsigma * rms
        if np.array_equal(new, keep):
            break
        keep = new
    return rms, mean, 1.0 - keep.mean()


def rms(x):
    """Un-clipped RMS about the mean."""
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)].std()


def rms_error(rms_value, n):
    """Analytic standard error of an RMS estimate for n (approximately Gaussian) entries."""
    return rms_value / np.sqrt(2.0 * max(n - 1, 1))


def bootstrap_ratio(res_a, res_b, stat=lambda x: clipped_rms(x)[0], n_boot=400, seed=0):
    """Paired bootstrap of stat(res_a) / stat(res_b). Returns (ratio, std)."""
    res_a = np.asarray(res_a)
    res_b = np.asarray(res_b)
    rng = np.random.default_rng(seed)
    n = len(res_a)
    reps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        reps[i] = stat(res_a[idx]) / stat(res_b[idx])
    return stat(res_a) / stat(res_b), reps.std()


def binned_clipped_rms(res, var, edges, min_entries=20):
    """Clipped RMS of res in bins of var. Returns (rms, err, n) arrays per bin."""
    idx = np.digitize(var, edges) - 1
    nb = len(edges) - 1
    out = np.full(nb, np.nan)
    err = np.full(nb, np.nan)
    count = np.zeros(nb, dtype=np.int64)
    for b in range(nb):
        sel = res[idx == b]
        count[b] = len(sel)
        if len(sel) >= min_entries:
            out[b] = clipped_rms(sel)[0]
            err[b] = rms_error(out[b], len(sel))
    return out, err, count
