# fcctrackfit

A learned track fit on edm4hep data for FCC-ee detectors (CLD, IDEA). It follows the seed-guided bidirectional
minGRU of [arXiv:2609.32479](https://arxiv.org/abs/2609.32479): an analytic 3-hit helix seed, per-hit seed residual features, a
bidirectional minGRU encoder and 7-quantile corrections to the seed perigee `(d0, z0, φ, θ, q/p)`. The reference is
the truth-seeded Kalman filter from the key4hep reconstruction.

Units: mm, GeV, ns, T. Perigee conventions are in `fcctrackfit/geometry/perigee.py`; edm4hep/LCIO states are
converted on input.

## Pipeline

Run everything with the package on the path: `export PYTHONPATH=$PWD:$PYTHONPATH` (bash) or
`setenv PYTHONPATH $PWD:$PYTHONPATH` (tcsh).

| Stage | Command | Where |
|---|---|---|
| 0. samples (separate) | ddsim + digi + truth-tracking/KF → edm4hep files that satisfy the contract below | key4hep |
| 1. convert | `python scripts/convert.py --detector fcctrackfit/config/detectors/cld.yaml --out data/cld/train files...` | key4hep, CPU, one job per file |
| 2. prepare | `python scripts/prepare.py --detector .../cld.yaml --shards 'data/cld/*/*.h5' --train 'data/cld/train/*.h5' --out data/cld/norm.json` | CPU |
| 3. train | `python scripts/train.py fcctrackfit/config/train/phase1.yaml` then `torchrun --nproc_per_node=2 scripts/train.py .../phase2.yaml` | GPU |
| 4. evaluate | `python scripts/evaluate.py runs/cld_phase2/final.pt --test 'data/cld/test_mu10/*.h5' --name mu10 --out eval/cld` | GPU or CPU |

Config values can be overridden on the command line, e.g. `scripts/train.py cfg.yaml train.epochs=5 out_dir=runs/x`.
Training resumes from `out_dir/last.pt` if it exists.

To find the collection names of a production: `python -m fcctrackfit.convert.inspect file.root`.

## Sample contract (input to stage 1)

Per event:

- `MCParticles`
- digitised tracker hits (`TrackerHitPlane`), each with a `TrackerHitSimTrackerHitLink` collection to
  `SimTrackerHit`s (whose `particle` gives the truth assignment)
- the KF baseline `Track` collection with an `AtIP` `TrackState`

The `metadata` frame must hold `<hits>__CellIDEncoding` for every hit collection. `fcctrackfit/toy/edm4hep_writer.py`
writes a minimal file that satisfies this contract and serves as an executable reference.

## CPU development without samples

```
python scripts/make_toy.py --out data/toy
python scripts/prepare.py --detector fcctrackfit/config/detectors/toy.yaml --shards 'data/toy/*/*.h5' \
       --train 'data/toy/train/*.h5' --out data/toy/norm.json
python scripts/train.py fcctrackfit/config/train/smoke.yaml
python scripts/evaluate.py runs/smoke/final.pt --test 'data/toy/test/*.h5' --name toy --out eval/toy
python -m pytest
```

## Not done yet

- Drift-chamber (IDEA DCH, `SenseWireHit`) features and residuals. The converter refuses `use_drift: true`.
- A fused Triton scan kernel; the encoder currently uses an exact Python loop over hit positions.
- Writing predictions back to edm4hep `Track`s; ONNX export; throughput benchmarks.
- Seed triplet selection is a Python loop per track and should be vectorised before converting hundreds of
  millions of tracks.
