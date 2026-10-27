import numpy as np

from fcctrackfit.eval.metrics import binned_clipped_rms, bootstrap_ratio, clipped_rms, rms


def test_clipped_rms_gaussian_core_with_tails():
    rng = np.random.default_rng(0)
    core = rng.normal(0.0, 1.0, 200_000)
    tails = rng.normal(0.0, 20.0, 4_000)
    x = np.concatenate([core, tails])
    r, mean, frac = clipped_rms(x)
    # 3-sigma clipping of a pure Gaussian gives ~0.986 sigma; tails add a little back
    assert 0.97 < r < 1.03
    assert abs(mean) < 0.02
    assert 0.015 < frac < 0.03
    assert rms(x) > 2.5


def test_bootstrap_ratio_identity():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 5000)
    ratio, err = bootstrap_ratio(x, 2 * x, n_boot=50)
    assert abs(ratio - 0.5) < 1e-12 and err < 1e-12


def test_binned():
    rng = np.random.default_rng(2)
    var = rng.uniform(0, 2, 20_000)
    res = rng.normal(0, 1, 20_000) * (1 + var)
    out, err, n = binned_clipped_rms(res, var, np.array([0, 1, 2]))
    assert out[0] < out[1] and n.sum() == 20_000 and np.all(err > 0)
