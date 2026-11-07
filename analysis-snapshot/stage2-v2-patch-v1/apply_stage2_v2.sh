#!/usr/bin/env bash
# Creates /scratch/zr2245/shared-collapse-stage2-v2 as a COPY of stage2-v1 (built renderer and native modules included)
# and applies stage2-v2.patch (constant vertex_weight + neighbour-context SH index coding).  v1 is never modified.
set -o pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
V1=${STAGE2_V1:-/scratch/zr2245/shared-collapse-stage2-v1}
V2=${STAGE2_V2:-/scratch/zr2245/shared-collapse-stage2-v2}
[ -d "$V1" ] || { echo "stage2-v1 not found: $V1"; exit 1; }
if [ -e "$V2" ]; then echo "$V2 already exists; remove it or set STAGE2_V2 to another path"; exit 1; fi
cp -a "$V1" "$V2" || { echo "copy failed"; exit 1; }
cd "$V2" && patch -p1 --no-backup-if-mismatch < "$HERE/stage2-v2.patch" || { echo "PATCH FAILED (see rejects above)"; exit 1; }
echo "stage2-v2 $(date +%F)" > "$V2/STAGE2_VERSION"
echo "patched: $V2"; ls "$V2/codec/index_ctx.cpp" "$V2/codec/index_context.py" >/dev/null && echo "new files present"
