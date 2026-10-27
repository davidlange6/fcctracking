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
| 0+1. CLD samples | `production/run_campaign.sh OUTDIR HOURS` or `production/make_sample.sh ...` (generator → ddsim → CLDReconstruction → HDF5 shard, per job) | key4hep, CPU |
| 1. convert (other productions) | `python scripts/convert.py --detector fcctrackfit/config/detectors/cld.yaml --out data/cld/train files...` | key4hep, CPU, one job per file |
| 2. prepare | `python scripts/prepare.py --detector .../cld.yaml --shards 'data/cld/*/*.h5' --train 'data/cld/train/*.h5' --out data/cld/norm.json` | CPU |
| 3. train | `python scripts/train.py fcctrackfit/config/train/phase1.yaml` then `torchrun --nproc_per_node=2 scripts/train.py .../phase2.yaml` | GPU |
| 4. evaluate | `python scripts/evaluate.py runs/cld_phase2/final.pt --test 'data/cld/test_mu10/*.h5' --name mu10 --out eval/cld` | GPU or CPU |

Config values can be overridden on the command line, e.g. `scripts/train.py cfg.yaml train.epochs=5 out_dir=runs/x`.
Training resumes from `out_dir/last.pt` if it exists.

To find the collection names of a production: `python -m fcctrackfit.convert.inspect file.root`.

## Shard format

One HDF5 file per input file, ragged (flat hit arrays + `offsets`), gzip+shuffle compressed, float32 hit positions
(0.12 µm quantisation at r = 2.1 m). KF datasets are written only when present. That comes to ~385 B/track with the KF
baseline and ~295 B/track without (10 GeV CLD muons, 13 hits/track): ~0.3 GB per million training tracks. Details are
in `fcctrackfit/data/shards.py`. `read_shard` rebuilds the full per-hit feature table in memory.

## CLD sample production

`production/make_sample.sh PARTICLE MOMENTUM N_EVENTS SEED OUTDIR` runs one job:
`production/gen_hepevt.py` (single particle, HEPEvt) → ddsim → `CLDReconstruction.py --trackingOnly` with
`--truthTracking` (truth-hit `SiTracks` + KF refit `SiTracks_Refitted`) or, with `NO_KF=1`, `--digiOnly` → conversion
to `OUTDIR/h5/<name>.h5`.
- **Momentum:** sampled in p, either fixed (`10`) or a 50/50 log-uniform/uniform mixture (`0.2:50`).
- **Direction:** uniform in cosθ (|cosθ| ≤ 0.99) and in φ.
- **Particle:** a name like `mu` gives a random charge per event.
- **Vertex:** Gaussian with σ_xy = 1 mm, σ_z = 20 mm, so the network cannot learn a d0 = z0 = 0 prior.

`production/run_campaign.sh OUTDIR HOURS` runs a whole campaign from one queue of ~30 min jobs (`PARALLEL`, default 30, at
a time). The KF test samples come first (p = 0.5, 2, 10, 45 GeV and the training spectrum), then validation and training
(digitisation only) until the deadline. To run it detached from the session:

    setsid nohup production/run_campaign.sh data/cld/campaign_mu 3 > data/cld/campaign_mu.out 2>&1 < /dev/null &

Progress is in `OUTDIR/campaign.log` (start, failures, final shard counts) and `OUTDIR/<sample>.jobs.log` (one line per job).

Costs per event (32 cores, 10 parallel jobs, 10 GeV μ): Geant4 ~0.07 s; KF reconstruction ~0.27 s (a Gaudi
`AlgTimingAuditor` run with `production/audit_timing.py` puts ~123 ms in `TruthTrackFinder` and ~121 ms in `Refit`, which
perf shows is nearly all DDKalTest/aidaTT helix–surface intersection); digitisation only 3.6 ms; conversion ~0.04 s.
Test samples keep `Refit`, because `SiTracks_Refitted` is 5–7 % better in d0/z0/φ/θ than the truth finder's own fit.

## Training on HTCondor (CERN batch, GPU)

The batch nodes cannot see local disk, so the code goes into the job as a tarball of the committed tree and the data
comes from EOS. The job uses the `LCG_110a_cuda` view (torch with CUDA), not key4hep.

1. Prepare the shards **before** copying them: `prepare.py` writes the seed triplets into every shard and produces
   `norm.json`.

       python scripts/prepare.py --detector fcctrackfit/config/detectors/cld.yaml \
           --shards 'data/cld/campaign_mu/*/h5/*.h5' --train 'data/cld/campaign_mu/train/h5/*.h5' \
           --out data/cld/campaign_mu/norm.json

2. Copy them to EOS with the layout the job expects: `DATA/train/h5/*.h5`, `DATA/val/h5/*.h5`, `DATA/norm.json`.
   The test samples go along as `DATA/test_*/h5`. `production/push_to_eos.sh` does this with rsync through
   lxplus, forwarding the Kerberos ticket so that EOS is writable there. Re-running it copies only new or
   changed shards.

       production/push_to_eos.sh data/cld/campaign_mu /eos/user/d/dlange/fcctracking/cld_mu_v1

3. On lxplus, with a Kerberos ticket, in `condor/`:

       ./make_bundle.sh                 # git HEAD -> fcctrackfit_code.tar.gz (warns about uncommitted changes)
       condor_submit train.sub          # phase 1 with the defaults in train.sub
       condor_submit train.sub CONFIG=fcctrackfit/config/train/phase2.yaml RUN=cld_mu_v1_phase2 NGPU=2 CPUS=16 \
           "MEMORY=32 GB" INIT_FROM=/eos/user/d/dlange/fcctracking/runs/cld_mu_v1_phase1/final.pt
       condor_submit train.sub RUN=lr1e-4 OVERRIDES="train.lr=1e-4"

`condor/train_job.sh` stages the shards to the node's scratch disk, then runs `scripts/train.py`, or `torchrun` when it
has more than one GPU. Every 15 min and at exit it copies `last.pt`, `final.pt`, `log.csv` and `train.out` to
`RUNS_DIR/RUN` on EOS. Checkpoints are written once per epoch. When the job is resubmitted with the same `RUN` it
resumes from that `last.pt`, for example after it hits the 3-day `testmatch` limit. For a very large training set, use
`train.max_tracks_per_epoch` to keep epochs shorter than the sync and eviction time scales. To test the job script
without condor or EOS, set `EOS_MGM=local` and use local paths.

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
