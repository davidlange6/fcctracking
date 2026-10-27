#!/bin/bash
# One CLD single-particle job: ddsim particle gun -> CLDReconstruction (truth tracking
# + KF refit, tracking only) -> conversion to an HDF5 training shard.
#
# Usage: make_sample.sh PARTICLE ENERGY N_EVENTS SEED OUTDIR
#   PARTICLE  Geant4 name, e.g. mu-, mu+, pi-, e-
#   ENERGY    fixed energy in GeV ("10"), or a momentum range "PMIN:PMAX" in GeV
#             (ddsim samples the momentum uniformly in that range)
#   SEED      random seed; also part of the output name, so use one per job
#
# Optional environment: COMPACT (geometry xml), COS_THETA_MAX (default 0.99),
# KEEP_SIM=1 (keep the Geant4 output), DETECTOR_CFG (converter YAML),
# NO_KF=1 (digitisation only: no truth tracking / KF refit; for training samples,
#          which do not need the baseline; ~3x cheaper per event).
#
# Outputs: OUTDIR/raw/<name>_REC.edm4hep.root, OUTDIR/h5/<name>.h5, OUTDIR/logs/<name>_*.log
# Needs the key4hep stack (K4GEO, CLDCONFIG).
set -euo pipefail

if [ $# -ne 5 ]; then
    sed -n '2,17p' "$0"
    exit 1
fi
particle=$1; energy=$2; nevents=$3; seed=$4; outdir=$(realpath -m "$5")

repo=$(cd "$(dirname "$(realpath "$0")")/.." && pwd)
compact=${COMPACT:-$K4GEO/FCCee/CLD/compact/CLD_o2_v07/CLD_o2_v07.xml}
cos_max=${COS_THETA_MAX:-0.99}
detector_cfg=${DETECTOR_CFG:-$repo/fcctrackfit/config/detectors/cld.yaml}

if [[ $energy == *:* ]]; then
    pmin=${energy%%:*}; pmax=${energy##*:}
    gun_energy=(--gun.momentumMin "${pmin}*GeV" --gun.momentumMax "${pmax}*GeV")
    tag="p${pmin}-${pmax}"
else
    gun_energy=(--gun.energy "${energy}*GeV")
    tag="e${energy}"
fi
name="${particle}_${tag}_s${seed}"
theta_min=$(python3 -c "import math; print(math.degrees(math.acos($cos_max)))")
theta_max=$(python3 -c "import math; print(180 - math.degrees(math.acos($cos_max)))")

mkdir -p "$outdir/raw" "$outdir/h5" "$outdir/logs"
work=$(mktemp -d "${TMPDIR:-/tmp}/cldprod_${name}_XXXX")
trap 'rm -rf "$work"' EXIT
cp -r "$CLDCONFIG/share/CLDConfig/." "$work/"
python3 "$repo/production/patch_cld_reco.py" "$work/CLDReconstruction.py" "$work/CLDReconstruction.py"
cd "$work"
if [ "${NO_KF:-0}" = 1 ]; then reco_mode=(--digiOnly); else reco_mode=(--truthTracking); fi

t0=$(date +%s)
# uniform in cos(theta); no crossing-angle boost, so the gun defines the truth exactly
ddsim --steeringFile cld_steer.py --compactFile "$compact" \
      --enableGun --gun.particle "$particle" --gun.distribution "cos(theta)" "${gun_energy[@]}" \
      --gun.thetaMin "${theta_min}*deg" --gun.thetaMax "${theta_max}*deg" \
      --crossingAngleBoost 0 --random.seed "$seed" \
      --numberOfEvents "$nevents" --outputFile "$work/${name}_sim.edm4hep.root" \
      > "$outdir/logs/${name}_sim.log" 2>&1
t1=$(date +%s)
# k4run processes only 3 events unless -n -1 is given
k4run CLDReconstruction.py --inputFiles "$work/${name}_sim.edm4hep.root" \
      --outputBasename "$work/${name}" --trackingOnly "${reco_mode[@]}" \
      --compactFile "$compact" -n -1 \
      > "$outdir/logs/${name}_reco.log" 2>&1
t2=$(date +%s)
mv "$work/${name}_REC.edm4hep.root" "$outdir/raw/"
if [ "${KEEP_SIM:-0}" = 1 ]; then
    mv "$work/${name}_sim.edm4hep.root" "$outdir/raw/"
fi
cd "$outdir"
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" python3 "$repo/scripts/convert.py" \
      --detector "$detector_cfg" --out "$outdir/h5" "$outdir/raw/${name}_REC.edm4hep.root" \
      > "$outdir/logs/${name}_convert.log" 2>&1
mv "$outdir/h5/${name}_REC.edm4hep.h5" "$outdir/h5/${name}.h5"
t3=$(date +%s)
echo "$name: $nevents events, sim $((t1 - t0)) s, reco $((t2 - t1)) s, convert $((t3 - t2)) s;" \
     "$(grep -h -- '->' "$outdir/logs/${name}_convert.log" | sed 's/.*: //')"
