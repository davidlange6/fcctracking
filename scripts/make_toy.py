"""Generate toy-tracker shards for CPU development: train/val/test splits."""

import argparse
import os

import numpy as np

from fcctrackfit.data.shards import write_shard
from fcctrackfit.toy.sim import simulate

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="data/toy")
ap.add_argument("--tracks-per-shard", type=int, default=20000)
ap.add_argument("--train-shards", type=int, default=4)
ap.add_argument("--val-shards", type=int, default=1)
ap.add_argument("--test-shards", type=int, default=1)
args = ap.parse_args()

seed = 1000
for split, n_shards in (("train", args.train_shards), ("val", args.val_shards), ("test", args.test_shards)):
    os.makedirs(f"{args.out}/{split}", exist_ok=True)
    for i in range(n_shards):
        # half uniform, half log-uniform in pT, as in the paper's training mixture
        shard = simulate(args.tracks_per_shard, np.random.default_rng(seed), pt_range=(0.5, 100.0), log_pt=i % 2 == 1)
        seed += 1
        write_shard(f"{args.out}/{split}/{split}_{i:04d}.h5", **shard)
        print(split, i, len(shard["truth"]), "tracks")
