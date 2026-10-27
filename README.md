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
| 0+1. CLD samples | `production/launch_local.sh mu- 10 10000 1000 10 data/cld/mu10 1000` (ddsim gun → CLDReconstruction `--truthTracking` → HDF5 shard, per job) | key4hep, CPU |
| 1. convert (other productions) | `python scripts/convert.py --detector fcctrackfit/config/detectors/cld.yaml --out data/cld/train files...` | key4hep, CPU, one job per file |
| 2. prepare | `python scripts/prepare.py --detector .../cld.yaml --shards 'data/cld/*/*.h5' --train 'data/cld/train/*.h5' --out data/cld/norm.json` | CPU |
| 3. train | `python scripts/train.py fcctrackfit/config/train/phase1.yaml` then `torchrun --nproc_per_node=2 scripts/train.py .../phase2.yaml` | GPU |
| 4. evaluate | `python scripts/evaluate.py runs/cld_phase2/final.pt --test 'data/cld/test_mu10/*.h5' --name mu10 --out eval/cld` | GPU or CPU |

Config values can be overridden on the command line, e.g. `scripts/train.py cfg.yaml train.epochs=5 out_dir=runs/x`.
Training resumes from `out_dir/last.pt` if it exists.

To find the collection names of a production: `python -m fcctrackfit.convert.inspect file.root`.

## CLD sample production

`production/make_sample.sh PARTICLE ENERGY N_EVENTS SEED OUTDIR` runs one job. It does a ddsim particle gun
(uniform in cosθ, |cosθ| ≤ 0.99, no crossing-angle boost), then `CLDReconstruction.py --truthTracking --trackingOnly -n -1`
(truth-hit `SiTracks` + KF refit `SiTracks_Refitted`), then conversion to `OUTDIR/h5/<name>.h5`. ENERGY is a fixed energy
in GeV or a momentum range `PMIN:PMAX`. `production/launch_local.sh` splits a sample into such jobs with seeds
`SEED_BASE + i` and runs them in parallel. Give every sample a disjoint seed range.

Measured on 32 cores (10 GeV μ, 1000 events per job, 10 jobs in parallel): sim ~105 s, reco ~290 s, convert ~40 s per job.
10k events take about 7 min wall. With the KF, reconstruction (~0.25 s/event) is the bottleneck. A Gaudi
`AlgTimingAuditor` run (`production/audit_timing.py`) puts ~123 ms/event in `TruthTrackFinder` and ~121 ms/event in
`Refit`; perf shows nearly all of it in DDKalTest/aidaTT helix–surface intersection and DDRec `insideBounds`.

Training samples don't need the KF baseline: `NO_KF=1 production/make_sample.sh ...` runs only the tracker
digitisation (`--digiOnly`, added by `production/patch_cld_reco.py`). That takes 3.6 ms/event, so a job costs essentially
the Geant4 time. Test samples should keep the full `--truthTracking` + `Refit` chain, because `SiTracks_Refitted` is 5–7 %
better in d0/z0/φ/θ than the truth finder's own `SiTracks` fit.

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
