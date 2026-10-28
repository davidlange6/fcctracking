# Status and restart notes (updated 2026-10-28)

What has been done, where everything is, and what comes next. How to run each stage is in
`README.md`; this file holds the state of the work.

## Where things are

| What | Where |
|---|---|
| Code | `git@github.com:davidlange6/fcctracking.git`; working clone `/dlange/tracking` on vocms010; submit clone `~/fcctracking` on lxplus (AFS) |
| Paper | `/dlange/tracking/2609.32479.pdf` (arXiv:2609.32479, not in git) |
| Local samples | `/dlange/tracking/data/cld/campaign_mu/{train,val,test_p0.5,test_p2,test_p10,test_p45,test_spectrum}/h5/` and `norm.json` |
| EOS data | `/eos/user/d/dlange/fcctracking/cld_mu_v1/` (3.6M train tracks, used for phase 1) and `cld_mu_v2/` (31.7M train tracks: v1 + the 15 h campaign), each with val, tests and norm.json |
| EOS runs | `/eos/user/d/dlange/fcctracking/runs/<RUN>/` (`last.pt`, `final.pt`, `log.csv`, `train.out`) |
| Phase-1 model | `runs/cld_mu_v1_phase1/final.pt` (local copy) and on EOS |
| Evaluation | `eval/cld_mu_v1_phase1/` (tables, plots vs cosθ and pT, predictions) |

## Environments
- **Production, conversion, prepare, evaluation:** the key4hep nightly on vocms010 (32 cores, no GPU).
  It has torch 2.11 (CPU), podio, edm4hep and pyarrow, and `K4GEO` / `CLDCONFIG` are set.
- **GPU training:** `LCG_110a_cuda` (torch 2.11, CUDA 12.5), set up by `condor/train_job.sh`.
- **EOS from vocms010:** ssh/rsync to lxplus with Kerberos forwarding (`GSSAPIDelegateCredentials=yes`;
  without forwarding EOS is permission-denied). `production/push_to_eos.sh` does this.
- **lxplus login shell is tcsh:** run remote commands as `ssh lxplus.cern.ch bash -s <<'EOF' ... EOF`.

## Samples (CLD_o2_v07, single μ±)
- **Kinematics:**
  - p 0.2–50 GeV, half log-uniform and half uniform in p;
  - uniform cosθ with |cosθ| ≤ 0.99, uniform φ;
  - vertex σ = 1 mm (xy) / 20 mm (z);
  - no crossing-angle boost.
- **Selection:** 6–30 hits; about 99 % of training events give a track.
- **Training sample:** digitisation only (no KF). 290 shards, 3.59M tracks, seeds 10000–10289.
- **Validation sample:** 8 shards, 99k tracks.
- **Test samples:** KF (`SiTracks_Refitted`) included:
  - p = 0.5, 2, 10, 45 GeV: 24k events each;
  - test_spectrum: 100k events.
- **Second training campaign** (2026-10-27 16:55 to 10-28 07:36, 880 min, no failures): 2,277 shards with
  28.16M tracks, seeds 100000+, in the same `train/h5`. The total training sample is 2,567 shards, 31.7M
  tracks, 8.3 GB. It was prepared and pushed to EOS as `cld_mu_v2`. The refitted resolution parameters
  agree with the phase-1 fit to within about 2 %.
- **Costs:**
  - digitisation-only training job: ~10.5 min per 12.5k events;
  - throughput: ~35k training tracks/min with 30 slots;
  - h5 size: ~300 B/track.

## Phase-1 result (cld_mu_v1, 3.6M tracks)
- **Training run:** 25 epochs with Lion one-cycle, batch 2048, 527k parameters. It took 71 min on an H100
  MIG 1g.12gb slice at ~150 ms/step. Validation was still improving slowly at the end.
- **Model vs KF:** ratio of iterative 3σ-clipped RMS (below 1 means the model is better):

| sample | d0 | z0 | φ | θ | q/p |
|---|---|---|---|---|---|
| p = 0.5 | 0.95 | 1.01 | 0.96 | 1.01 | 0.65 |
| p = 2 | 1.01 | 1.02 | 1.01 | 1.02 | 0.87 |
| p = 10 | 0.99 | 1.01 | 1.05 | 1.11 | 0.85 |
| p = 45 | 0.94 | 0.91 | 0.97 | 0.97 | 0.89 |
| 0.2–50 | 0.97 | 0.98 | 1.01 | 1.08 | 0.93 |

- **Barrel:** the model equals the KF.
- **Forward (|cosθ| > 0.5):** θ is 10–20 % worse than the KF, and φ and z0 a little worse. Only about 10 %
  of training tracks have |cosθ| > 0.9.
- **q/p:** better than the KF everywhere. But the CLD KF's q/p was already worse than the 3-hit seed, so the
  KF q/p is a suspect baseline (see open questions).
- **Tails:** the unclipped RMS of the model is far below the KF's, because the KF has catastrophic fits that
  the model avoids.

## Next steps
1. Done: campaign checked, `prepare.py` run, `cld_mu_v2` pushed to EOS, `request_disk` added to
   `condor/train.sub` (the `DISK` macro, 25 GB).
2. Before submitting: `git pull` and `./make_bundle.sh` on lxplus (done on 2026-10-28).
3. **Submit:** `condor_submit train.sub DATA=/eos/user/d/dlange/fcctracking/cld_mu_v2 RUN=cld_mu_v2_phase1
   OVERRIDES="train.epochs=8"`.
   - This is about 16.6k steps (~42 min) per epoch, so 8 epochs take about 6 h on a MIG slice.
   - The one-cycle schedule fixes the run length, so choose the number of epochs up front.
4. **Evaluate:** copy `final.pt` from EOS, then run `scripts/evaluate.py` on each test sample (see
   `eval/cld_mu_v1_phase1/` for the earlier output) and compare with the table above.
5. **Then:**
   - phase 2 (Muon+AdamW WSD, `INIT_FROM=<phase-1 final.pt>`);
   - possibly oversample the forward region in the next production.

## Open questions
- **KF q/p:** why the CLD truth-seeded KF (`SiTracks_Refitted`, AtIP) has worse q/p than the 3-hit seed.
  Possible causes are its material or energy-loss treatment, or the state or convention we read.
- **Forward θ/φ gap** to the KF: more data, forward oversampling, or a model change?
- **Step time:** ~150–200 ms/step is slow for 0.5M parameters. Is it the sequential minGRU scan or the
  3 loader workers? Profile it before scaling up.
- **Deferred:** IDEA (drift-chamber hits, not implemented); vectorising `select_seed_hits` (Python loop, ~1 min
  per 3.6M tracks).

## Decisions and why
- **Seed rule "spread" with `max_turn` = 0.4π:**
  - "spread" means the first hit, the outermost hit on the outgoing arm, and the hit closest to mid-radius.
  - The arm also ends where the turning angle estimated from the hit time exceeds `max_turn`.
  - Without the time cut, low-p loopers got seed hits on different turns (θ errors up to 0.7 rad).
- **Target scaling:** targets are divided by hypot(a, b/p_seed)·sin(θ_seed)^−c, fitted per parameter in
  `prepare.py` (`norm.json` "resolution"). Without it, a single window let the p < 1 GeV tails dominate.
- **Formats:**
  - HDF5 rather than Parquet: Parquet was 35 % larger, and `prepare.py` rewrites the seed indices in place.
  - Test samples keep their ROOT files; training samples don't.

## Condor gotchas (CERN batch, learned the hard way)
- **Submit from AFS** (`~/fcctracking/condor`), never from `/eos`: the standard schedds reject it.
- **No `stream_output` / `stream_error`:** they're rejected since Nov 2025. Use `condor_tail <id>` or `train.out`
  on EOS (synced every 15 min).
- **No V100s:** the LCG torch builds (108–110a) have only sm_75+ kernels. The job runs a CUDA check first.
- **Reserved GPUs:** the H200s and some H100 NVLs are reserved for the BE-ABP / ATS groups. They show as free but
  never match our group (`group_u_CMS.u_zh.users`).
- **Usable GPUs:** A100, open H100 NVL, H100 MIG slices, T4. A request of 4 CPUs / 12 GB fits the MIG slices.
  Count GPUs with torch, not `nvidia-smi -L`, which lists MIG instances as extra lines.
- **`xrdcp -r`** needs an existing destination directory.
- **GPU status:** `condor_status -constraint 'TotalGpus > 0 && SlotType == "Partitionable"' -af
  GPUs_DeviceName Gpus TotalGpus`; `condor_q -better-analyze <id>` explains why a job is idle.
