"""Stage 1: convert edm4hep files to HDF5 shards (one shard per input file).

Usage: python scripts/convert.py --detector fcctrackfit/config/detectors/cld.yaml --out data/cld/train in1.root ...
"""

import argparse
import os

from fcctrackfit.convert.edm4hep_reader import convert_file
from fcctrackfit.train.train import load_config

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("inputs", nargs="+")
ap.add_argument("--detector", required=True)
ap.add_argument("--out", required=True, help="output directory")
ap.add_argument("--max-events", type=int)
args = ap.parse_args()

cfg = load_config(args.detector)
os.makedirs(args.out, exist_ok=True)
for path in args.inputs:
    out = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0] + ".h5")
    n, n_kf = convert_file(path, cfg, out, args.max_events)
    print(f"{path} -> {out}: {n} tracks, {n_kf} with matched KF track")
