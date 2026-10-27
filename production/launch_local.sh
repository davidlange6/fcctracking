#!/bin/bash
# Run a sample as parallel make_sample.sh jobs on this machine.
#
# Usage: launch_local.sh PARTICLE ENERGY TOTAL_EVENTS EVENTS_PER_JOB N_PARALLEL OUTDIR [SEED_BASE]
#   e.g. launch_local.sh mu- 10 10000 1000 10 data/cld/mu10
#   Job i uses seed SEED_BASE + i (default SEED_BASE 1000). Use disjoint seed ranges
#   for different samples, so that training and test samples never share seeds.
# The same environment variables as make_sample.sh apply. Per-job summaries go to
# OUTDIR/jobs.log.
set -euo pipefail

if [ $# -lt 6 ]; then
    sed -n '2,10p' "$0"
    exit 1
fi
particle=$1; energy=$2; total=$3; per_job=$4; parallel=$5; outdir=$6; seed_base=${7:-1000}
here=$(dirname "$(realpath "$0")")
n_jobs=$(( (total + per_job - 1) / per_job ))
mkdir -p "$outdir"

echo "$(date): $n_jobs jobs of <= $per_job $particle events at $energy GeV, $parallel in parallel -> $outdir" \
    | tee -a "$outdir/jobs.log"
for ((i = 0; i < n_jobs; i++)); do
    n=$(( i == n_jobs - 1 ? total - i * per_job : per_job ))
    echo "$n $((seed_base + i))"
done | xargs -P "$parallel" -L 1 bash -c \
    '"$0"/make_sample.sh "$1" "$2" "$4" "$5" "$3" >> "$3/jobs.log" 2>&1 || echo "FAILED seed $5" >> "$3/jobs.log"' \
    "$here" "$particle" "$energy" "$outdir"
# (xargs appends "N_EVENTS SEED" to the fixed arguments: $0 here, $1 particle, $2 energy, $3 outdir)
echo "$(date): done; failures: $(grep -c FAILED "$outdir/jobs.log" || true)" | tee -a "$outdir/jobs.log"
