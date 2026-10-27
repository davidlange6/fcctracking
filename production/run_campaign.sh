#!/bin/bash
# CLD muon campaign on this machine; survives logout when started with
#   setsid nohup production/run_campaign.sh OUTDIR HOURS > OUTDIR.out 2>&1 < /dev/null &
#
# All jobs are sized to take ~30 min and run from one queue, PARALLEL (default 30)
# at a time; a new job starts whenever a slot frees up.
#   1. KF test samples (full truth tracking + Refit), always run to completion
#   2. validation + training (digitisation only); training jobs are not started
#      later than HOURS - 0.5 h after the campaign start, so the campaign ends
#      after about HOURS.
#
# Kinematics for every sample: one muon per event, random charge, uniform cos(theta)
# with |cos(theta)| <= 0.99, uniform phi, vertex sigma 1 mm (xy) / 20 mm (z).
# Seeds: train 10000+, validation 20000+, tests 30000+ (one block of 1000 per sample).
# Stop it with: kill -- -<PID of run_campaign.sh>  (it is its own process group).
set -euo pipefail

if [ $# -ne 2 ]; then
    sed -n '2,15p' "$0"
    exit 1
fi
outdir=$(realpath -m "$1"); hours=$2
here=$(dirname "$(realpath "$0")")
parallel=${PARALLEL:-30}
kf_events=${KF_EVENTS_PER_JOB:-4000}       # ~0.38 s/event incl. truth tracking + Refit
digi_events=${DIGI_EVENTS_PER_JOB:-12500}  # ~0.11 s/event, digitisation only
spectrum=0.2:50
mkdir -p "$outdir"
start=$(date +%s)
deadline=$(python3 -c "print(int($start + ($hours - 0.5) * 3600))")
log="$outdir/campaign.log"
echo "$(date): campaign start (pid $$), last training job start $(date -d @"$deadline"), $parallel slots" >> "$log"

jobs() {  # jobs SAMPLE MOMENTUM TOTAL PER_JOB SEED_BASE ENV  -> "ENV SAMPLE MOMENTUM N SEED"
    local n_jobs=$(( ($3 + $4 - 1) / $4 ))
    for ((i = 0; i < n_jobs; i++)); do
        echo "$6 $1 $2 $(( i == n_jobs - 1 ? $3 - i * $4 : $4 )) $(( $5 + i ))"
    done
}

KF=NO_KF=0
DIGI="NO_KF=1,DEADLINE=$deadline"
{
    jobs test_p0.5 0.5 24000 "$kf_events" 30000 "$KF"
    jobs test_p2 2 24000 "$kf_events" 31000 "$KF"
    jobs test_p10 10 24000 "$kf_events" 32000 "$KF"
    jobs test_p45 45 24000 "$kf_events" 33000 "$KF"
    jobs test_spectrum "$spectrum" 100000 "$kf_events" 34000 "$KF"
    jobs val "$spectrum" 100000 "$digi_events" 20000 "$DIGI"
    jobs train "$spectrum" 100000000 "$digi_events" 10000 "$DIGI"
} | xargs -P "$parallel" -L 1 bash -c '
    here=$0 out=$1 envs=$2 sample=$3 momentum=$4 n=$5 seed=$6
    # jobs past the deadline exit immediately with a SKIPPED line
    env ${envs//,/ } "$here"/make_sample.sh mu "$momentum" "$n" "$seed" "$out/$sample" >> "$out/$sample.jobs.log" 2>&1 \
        || echo "$(date): FAILED $sample seed $seed" >> "$out/campaign.log"' "$here" "$outdir"

for d in "$outdir"/*/h5; do
    [ -d "$d" ] && echo "$(date): $(basename "$(dirname "$d")"): $(ls "$d" | wc -l) shards" >> "$log"
done
echo "$(date): campaign finished after $(( ($(date +%s) - start) / 60 )) min" >> "$log"
