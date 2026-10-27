"""Stage 4: evaluate a checkpoint on test shards; writes summary JSON, table and plots."""

import argparse
import glob
import json
import os

import numpy as np
import torch

from fcctrackfit.eval.evaluate import format_table, run_inference, summarize
from fcctrackfit.eval.plots import resolution_vs
from fcctrackfit.train.train import load_model

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("checkpoint")
ap.add_argument("--test", required=True, help="glob of test shards (one sample)")
ap.add_argument("--name", default="test")
ap.add_argument("--out", default="eval")
ap.add_argument("--batch-size", type=int, default=8192)
ap.add_argument("--max-tracks", type=int)
ap.add_argument("--bootstrap", type=int, default=400)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = load_model(args.checkpoint, device)
res = run_inference(model, sorted(glob.glob(args.test)), args.batch_size, device, args.max_tracks)
os.makedirs(args.out, exist_ok=True)
np.savez_compressed(f"{args.out}/{args.name}_predictions.npz", **res)
summary = summarize(res, bootstrap=args.bootstrap)
with open(f"{args.out}/{args.name}_summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=float)
table = format_table(summary)
print(table)
with open(f"{args.out}/{args.name}_table.txt", "w") as f:
    f.write(table + "\n")
resolution_vs(res, "cos_theta", np.linspace(-1, 1, 21), f"{args.out}/{args.name}_vs_costheta.png", args.name)
resolution_vs(res, "pt", np.geomspace(0.5, 100, 15), f"{args.out}/{args.name}_vs_pt.png", args.name)
