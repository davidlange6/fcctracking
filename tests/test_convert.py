import numpy as np
import pytest

from fcctrackfit.convert.cellid import BitFieldCoder
from fcctrackfit.data.shards import COL, read_shard
from fcctrackfit.toy.edm4hep_writer import ENCODING, encode_cell
from fcctrackfit.toy.sim import simulate


def test_bitfield_coder():
    coder = BitFieldCoder(ENCODING)
    cells = np.array([encode_cell(5, 3), encode_cell(1, 63, side=-1)], dtype=np.uint64)
    assert coder.get(cells, "system").tolist() == [5, 1]
    assert coder.get(cells, "layer").tolist() == [3, 63]
    assert coder.get(cells, "side").tolist() == [0, -1]
    assert coder.get(cells, "missing").tolist() == [0, 0]


def test_convert_toy_edm4hep_roundtrip(tmp_path):
    pytest.importorskip("edm4hep")
    from fcctrackfit.convert.edm4hep_reader import convert_file
    from fcctrackfit.toy.edm4hep_writer import TOY_EDM4HEP_CONFIG, write_toy_edm4hep

    shard = simulate(30, np.random.default_rng(7), pt_range=(1.0, 20.0))
    src = tmp_path / "toy_edm4hep.root"
    write_toy_edm4hep(src, shard, tracks_per_event=2)
    out = tmp_path / "toy.h5"
    n, n_kf = convert_file(str(src), TOY_EDM4HEP_CONFIG, str(out))
    got = read_shard(out)
    assert n == len(shard["truth"]) and n_kf == n
    np.testing.assert_allclose(got["truth"], shard["truth"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(got["kf"], shard["truth"], atol=1e-5, rtol=1e-5)
    assert np.array_equal(got["offsets"], shard["offsets"])
    # hits were written in reverse; the converter restores time order
    np.testing.assert_allclose(got["hit_pos"], shard["hit_pos"], atol=2e-4)  # float32 storage
    for c in ("system", "layer", "t", "r"):
        np.testing.assert_allclose(got["hit_feat"][:, COL[c]], shard["hit_feat"][:, COL[c]], rtol=1e-6, atol=2e-4)
