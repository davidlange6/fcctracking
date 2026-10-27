#!/bin/bash
# One CLD single-particle job: gen_hepevt.py -> ddsim -> CLDReconstruction (truth tracking
# + KF refit, or digitisation only) -> conversion to an HDF5 shard.
#
# Usage: make_sample.sh PARTICLE MOMENTUM N_EVENTS SEED OUTDIR
#   PARTICLE  mu, e, pi, k, p (charge sign random per event) or mu-, mu+, e-, e+, pi-, pi+, ...
#   MOMENTUM  fixed momentum in GeV ("10"), or "PMIN:PMAX": a LOG_FRAC log-uniform /
#             (1 - LOG_FRAC) uniform mixture in p over that range
#   SEED      random seed for generator and Geant4; part of the output name, use one per job
#
# Optional environment: VTX_SIGMA_XY / VTX_SIGMA_Z (vertex Gaussian sigmas, mm; default 1 / 20),
# LOG_FRAC (default 0.5), COS_THETA_MAX (default 0.99), NO_KF=1 (digitisation only: no truth
# tracking / KF refit, for training samples), KEEP_SIM=1, COMPACT (geometry xml),
# DETECTOR_CFG (converter YAML), DEADLINE (unix time; the job is skipped if started later).
#
# Outputs: OUTDIR/raw/<name>_REC.edm4hep.root, OUTDIR/h5/<name>.h5, OUTDIR/logs/<name>_*.log
# Needs the key4hep stack (K4GEO, CLDCONFIG).
set -euo pipefail

if [ $# -ne 5 ]; then
    sed -n '2,19p' "$0"
    exit 1
fi
particle=$1; momentum=$2; nevents=$3; seed=$4; outdir=$(realpath -m "$5")

if [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "SKIPPED seed $seed: past deadline"
    exit 0
fi

repo=$(cd "$(dirname "$(realpath "$0")")/.." && pwd)
compact=${COMPACT:-$K4GEO/FCCee/CLD/compact/CLD_o2_v07/CLD_o2_v07.xml}
detector_cfg=${DETECTOR_CFG:-$repo/fcctrackfit/config/detectors/cld.yaml}

case $particle in
    mu) pdg=(--pdg-abs 13) ;; mu-) pdg=(--pdg 13) ;; mu+) pdg=(--pdg -13) ;;
    e) pdg=(--pdg-abs 11) ;; e-) pdg=(--pdg 11) ;; e+) pdg=(--pdg -11) ;;
    pi) pdg=(--pdg-abs 211) ;; pi+) pdg=(--pdg 211) ;; pi-) pdg=(--pdg -211) ;;
    k) pdg=(--pdg-abs 321) ;; k+) pdg=(--pdg 321) ;; k-) pdg=(--pdg -321) ;;
    p) pdg=(--pdg-abs 2212) ;;
    *) echo "unknown particle $particle"; exit 1 ;;
esac
if [[ $momentum == *:* ]]; then
    pmin=${momentum%%:*}; pmax=${momentum##*:}
    spectrum=(--pmin "$pmin" --pmax "$pmax" --log-frac "${LOG_FRAC:-0.5}")
    tag="p${pmin}-${pmax}"
else
    spectrum=(--p "$momentum")
    tag="p${momentum}"
fi
name="${particle}_${tag}_s${seed}"

mkdir -p "$outdir/raw" "$outdir/h5" "$outdir/logs"
work=$(mktemp -d "${TMPDIR:-/tmp}/cldprod_${name}_XXXX")
trap 'rm -rf "$work"' EXIT
cp -r "$CLDCONFIG/share/CLDConfig/." "$work/"
python3 "$repo/production/patch_cld_reco.py" "$work/CLDReconstruction.py" "$work/CLDReconstruction.py"
cd "$work"
if [ "${NO_KF:-0}" = 1 ]; then reco_mode=(--digiOnly); else reco_mode=(--truthTracking); fi

t0=$(date +%s)
python3 "$repo/production/gen_hepevt.py" --out "$work/${name}.hepevt" --n "$nevents" --seed "$seed" \
        "${pdg[@]}" "${spectrum[@]}" --cos-max "${COS_THETA_MAX:-0.99}" \
        --sigma-xy "${VTX_SIGMA_XY:-1.0}" --sigma-z "${VTX_SIGMA_Z:-20.0}"
# no crossing-angle boost: the generated particle is the truth
ddsim --steeringFile cld_steer.py --compactFile "$compact" --inputFiles "$work/${name}.hepevt" \
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
