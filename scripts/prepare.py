"""Stage 2: add seed triplets to every shard and compute normalisation from training shards."""

import argparse
import glob

from fcctrackfit.data.normalize import add_seed_indices, compute_norm, save_norm
from fcctrackfit.train.train import load_config

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--detector", required=True, help="detector YAML (fcctrackfit/config/detectors)")
ap.add_argument("--shards", nargs="+", required=True, help="glob(s) of all shards (train, val, test)")
ap.add_argument("--train", required=True, help="glob of training shards used for the normalisation")
ap.add_argument("--hit-columns", nargs="+", default="x y z t r phi theta system layer side du dv".split())
ap.add_argument("--out", required=True, help="output norm.json")
ap.add_argument("--max-tracks", type=int, default=200_000)
args = ap.parse_args()

det = load_config(args.detector)
for pat in args.shards:
    for path in sorted(glob.glob(pat)):
        print(f"{path}: {add_seed_indices(path, det['seed']):.4f} of tracks have a seed triplet")
norm = compute_norm(sorted(glob.glob(args.train)), args.hit_columns, det["bz"], det["qop_eps"],
                    det["residual_scale_mm"], max_tracks=args.max_tracks)  # fmt: skip
save_norm(norm, args.out)
print("target windows (d0, z0, phi, theta, scaled q/p):", norm["windows"])
