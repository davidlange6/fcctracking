#!/bin/bash
# Pack the committed code (git HEAD: fcctrackfit/ and scripts/) into condor/fcctrackfit_code.tar.gz,
# so every job runs a known version. Uncommitted changes are not included; the commit is recorded
# in the bundle as VERSION.
set -euo pipefail
here=$(dirname "$(realpath "$0")")
cd "$here/.."
if ! git diff --quiet HEAD -- fcctrackfit scripts; then
    echo "warning: uncommitted changes in fcctrackfit/ or scripts/ are NOT in the bundle" >&2
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git archive HEAD fcctrackfit scripts | tar -x -C "$tmp"
git rev-parse HEAD > "$tmp/VERSION"
tar -czf "$here/fcctrackfit_code.tar.gz" -C "$tmp" .
mkdir -p "$here/logs"
echo "bundle $(cat "$tmp/VERSION" | cut -c1-12): $here/fcctrackfit_code.tar.gz ($(du -h "$here/fcctrackfit_code.tar.gz" | cut -f1))"
