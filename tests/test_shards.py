import h5py
import numpy as np
import pytest

from fcctrackfit.data.shards import COL, read_shard, write_seed_indices, write_shard
from fcctrackfit.toy.sim import simulate


def test_roundtrip_without_kf(tmp_path):
    shard = simulate(200, np.random.default_rng(0), pt_range=(1.0, 20.0))
    path = tmp_path / "s.h5"
    write_shard(path, **shard)
    with h5py.File(path, "r") as f:
        assert "kf" not in f and "hit_drift" not in f and "hit_feat" not in f
    got = read_shard(path)
    # float32 storage: quantisation well below 1 um at tracker radii
    assert np.abs(got["hit_pos"] - shard["hit_pos"]).max() < 2e-4
    np.testing.assert_allclose(got["hit_feat"], shard["hit_feat"], rtol=1e-6, atol=2e-4)
    assert np.array_equal(got["offsets"], shard["offsets"])
    np.testing.assert_array_equal(got["truth"], shard["truth"])
    assert got["kf"].shape == (len(shard["truth"]), 5) and not got["kf_ok"].any()
    assert "seed_idx" not in got


def test_kf_and_seed_indices(tmp_path):
    shard = simulate(50, np.random.default_rng(1), pt_range=(1.0, 20.0))
    n = len(shard["truth"])
    shard["kf"] = shard["truth"] + 1e-3
    shard["kf_ok"] = np.arange(n) % 2 == 0
    shard["kf_cov"] = np.ones((n, 21), np.float32)
    path = tmp_path / "s.h5"
    write_shard(path, **shard)
    off = shard["offsets"]
    seed = np.stack([off[:-1], off[:-1] + 2, off[1:] - 1], 1)
    seed[3] = -1
    write_seed_indices(path, seed, off)
    got = read_shard(path)
    np.testing.assert_array_equal(got["seed_idx"], seed)
    np.testing.assert_array_equal(got["kf_ok"], shard["kf_ok"])
    np.testing.assert_array_equal(got["kf"], shard["kf"])
    assert got["kf_cov"].shape == (n, 21)


def test_drift_columns_kept_when_present(tmp_path):
    shard = simulate(20, np.random.default_rng(2), pt_range=(1.0, 20.0))
    shard["hit_feat"][::3, COL["drift_dist"]] = 1.5
    path = tmp_path / "s.h5"
    write_shard(path, **shard)
    got = read_shard(path)
    np.testing.assert_array_equal(got["hit_feat"][:, COL["drift_dist"]], shard["hit_feat"][:, COL["drift_dist"]])


def test_old_format_rejected(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f.attrs["detector"] = "x"
    with pytest.raises(ValueError, match="format"):
        read_shard(path)
