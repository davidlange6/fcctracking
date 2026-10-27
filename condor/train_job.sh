#!/bin/bash
# HTCondor GPU training job (CERN batch). Arguments:
#   train_job.sh CONFIG EOS_DATA EOS_OUT [key=value overrides ...]
#     CONFIG    training YAML inside the code bundle, e.g. fcctrackfit/config/train/phase1.yaml
#     EOS_DATA  directory with train/h5/*.h5, val/h5/*.h5 and norm.json (output of the
#               production campaign after scripts/prepare.py), e.g. /eos/user/d/dlange/fcctracking/cld_mu_v1
#     EOS_OUT   run directory on EOS; checkpoints (last.pt, final.pt), log.csv and train.out are
#               synced there every SYNC_SECONDS and at exit. If EOS_OUT/last.pt exists the job resumes.
# Environment (set in the submit file): EOS_MGM (default root://eosuser.cern.ch; "local" uses plain
# cp, for testing), INIT_FROM (EOS path of a checkpoint to start from, e.g. phase 1 final.pt),
# SYNC_SECONDS (default 900), LCG_VIEW (default LCG_110a_cuda el9 gcc13).
set -euo pipefail

config=$1; eos_data=$2; eos_out=$3
shift 3
mgm=${EOS_MGM:-root://eosuser.cern.ch}
view=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt}
scratch=${_CONDOR_SCRATCH_DIR:-$PWD}

if [ "$mgm" = local ]; then
    fetch_dir() { cp -r "$1" "$2"; }
    fetch() { cp "$1" "$2"; }
    put() { cp "$1" "$2"; }
    exists() { [ -e "$1" ]; }
    make_dir() { mkdir -p "$1"; }
else
    fetch_dir() { xrdcp -r -s --parallel 8 "$mgm/$1" "$2"; }
    fetch() { xrdcp -f -s "$mgm/$1" "$2"; }
    put() { xrdcp -f -s "$1" "$mgm/$2"; }
    exists() { xrdfs "${mgm#root://}" stat "$1" > /dev/null 2>&1; }
    make_dir() { xrdfs "${mgm#root://}" mkdir -p "$1"; }
fi

set +u
source "$view/setup.sh"
set -u
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l || true)
echo "host $(hostname), $ngpu GPU(s): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';')"

mkdir -p "$scratch/code" "$scratch/data" "$scratch/run"
tar -xzf "$scratch/fcctrackfit_code.tar.gz" -C "$scratch/code"
echo "code version $(cat "$scratch/code/VERSION")"
export PYTHONPATH="$scratch/code${PYTHONPATH:+:$PYTHONPATH}"

# stage the shards to local disk: training re-reads them every epoch
t0=$(date +%s)
for split in train val; do
    fetch_dir "$eos_data/$split/h5" "$scratch/data/$split.in"
    # flatten: a recursive copy may or may not nest the source directory name
    mkdir -p "$scratch/data/$split/h5"
    find "$scratch/data/$split.in" -name '*.h5' -exec mv -t "$scratch/data/$split/h5" {} +
    rm -rf "$scratch/data/$split.in"
done
fetch "$eos_data/norm.json" "$scratch/data/norm.json"
echo "staged $(ls "$scratch"/data/train/h5 | wc -l) train and $(ls "$scratch"/data/val/h5 | wc -l) val shards" \
     "($(du -sh "$scratch/data" | cut -f1)) in $(( $(date +%s) - t0 )) s"

make_dir "$eos_out"
if exists "$eos_out/last.pt"; then
    echo "resuming from $eos_out/last.pt"
    fetch "$eos_out/last.pt" "$scratch/run/last.pt"
    fetch "$eos_out/log.csv" "$scratch/run/log.csv" || true
    fetch "$eos_out/train.out" "$scratch/run/train.out" || true
fi
init=()
if [ -n "${INIT_FROM:-}" ]; then
    fetch "$INIT_FROM" "$scratch/init.pt"
    init=(init_from="$scratch/init.pt")
fi

sync_out() {
    for f in last.pt final.pt log.csv train.out; do
        if [ -f "$scratch/run/$f" ]; then put "$scratch/run/$f" "$eos_out/$f" || true; fi
    done
}
( while sleep "${SYNC_SECONDS:-900}"; do sync_out; done ) &
syncer=$!
trap 'kill $syncer 2>/dev/null; sync_out' EXIT

overrides=(out_dir="$scratch/run" data.train="$scratch/data/train/h5/*.h5" data.val="$scratch/data/val/h5/*.h5"
           data.norm="$scratch/data/norm.json" "${init[@]}" "$@")
cd "$scratch/code"
if [ "$ngpu" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$ngpu" scripts/train.py "$config" "${overrides[@]}" 2>&1 | tee -a "$scratch/run/train.out"
else
    python3 scripts/train.py "$config" "${overrides[@]}" 2>&1 | tee -a "$scratch/run/train.out"
fi
