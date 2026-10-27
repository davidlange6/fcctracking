#!/bin/bash
# Copy a prepared campaign (after scripts/prepare.py) to EOS through lxplus, with the layout
# condor/train_job.sh expects:
#   DEST/norm.json, DEST/<sample>/h5/*.h5   (train, val, test_*; raw ROOT files and logs stay here)
#
# Usage: push_to_eos.sh CAMPAIGN_DIR [DEST]
#   DEST default /eos/user/d/dlange/fcctracking/<basename of CAMPAIGN_DIR>
# Needs a Kerberos ticket here (kinit); it is forwarded so that EOS is writable on lxplus.
# Re-running only copies new or changed shards (e.g. after prepare.py rewrote seed_local).
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    sed -n '2,10p' "$0"
    exit 1
fi
src=$(realpath "$1")
dest=${2:-/eos/user/d/dlange/fcctracking/$(basename "$src")}
host=${LXPLUS:-lxplus.cern.ch}
ssh_cmd="ssh -o GSSAPIAuthentication=yes -o GSSAPIDelegateCredentials=yes"

if [ ! -f "$src/norm.json" ]; then
    echo "no $src/norm.json: run scripts/prepare.py first (it also writes the seed triplets into the shards)"
    exit 1
fi
$ssh_cmd "$host" "mkdir -p '$dest'"
# --whole-file: no delta transfer over EOS fuse; size + mtime decide what changed
rsync -rt --whole-file -e "$ssh_cmd" --info=stats1 \
      --include='/norm.json' --include='/*/' --include='/*/h5/' --include='/*/h5/*.h5' --exclude='*' \
      "$src/" "$host:$dest/"
# the remote login shell may be tcsh: run the summary in bash
$ssh_cmd "$host" bash -s <<EOF
cd '$dest' && for d in */h5; do echo "\$d: \$(ls \$d | wc -l) shards"; done; du -sh .
EOF
