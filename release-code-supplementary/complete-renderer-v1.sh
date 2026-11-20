#!/usr/bin/env bash
# complete-renderer-v1.sh (2026-11-20): the copy of the renderer in ByteMesh-code.zip lacks the glm *.inl files (the archive it was built from
# did not contain them).  This adds every source file of the renderer on the cluster that the zip lacks, checks that the files both have are
# identical, scans the added files for identity strings, and writes ByteMesh-code-v2.zip.  Nothing is deleted or overwritten.
# Usage: bash /scratch/zr2245/analysis/complete-renderer-v1.sh
set -o pipefail
AN=/scratch/zr2245/analysis; SRC=/scratch/zr2245/shared-collapse-stage2-v2/renderer; OUT=$AN/ByteMesh-code-v2.zip
[ -f $AN/ByteMesh-code.zip ] && [ -d $SRC ] || { echo "FAILED: missing $AN/ByteMesh-code.zip or $SRC"; exit 1; }
[ -e $OUT ] && { echo "FAILED: $OUT exists already"; exit 1; }
W=$(mktemp -d /scratch/zr2245/relfix-XXXXXX); cd $W && unzip -q $AN/ByteMesh-code.zip || { echo "FAILED: unzip"; exit 1; }
D=$W/ByteMesh-code/codec/renderer
(cd $SRC && find . -type f ! -path "./build/*" ! -path "*/__pycache__/*" ! -path "*.egg-info/*" ! -name "*.so" ! -name "*.pyc" ! -name "*.o" ! -name ".ninja*" ! -name "*.ninja" | sort) > $W/hpc.txt
(cd $D && find . -type f | sort) > $W/zip.txt
comm -23 $W/hpc.txt $W/zip.txt > $W/missing.txt; comm -12 $W/hpc.txt $W/zip.txt > $W/common.txt; comm -13 $W/hpc.txt $W/zip.txt > $W/onlyzip.txt
echo "renderer files: cluster $(wc -l < $W/hpc.txt), zip $(wc -l < $W/zip.txt), missing in the zip $(wc -l < $W/missing.txt), only in the zip $(wc -l < $W/onlyzip.txt)"
echo "missing by extension: $(sed 's/.*\.//' $W/missing.txt | sort | uniq -c | tr '\n' ' ')"
grep -v "third_party/glm/" $W/missing.txt | head -20 | sed 's/^/  missing outside glm: /'
nd=0; while read f; do cmp -s "$SRC/$f" "$D/$f" || { nd=$((nd+1)); echo "  differs: $f"; }; done < $W/common.txt; echo "files in both that differ: $nd"
while read f; do mkdir -p "$D/$(dirname "$f")" && cp "$SRC/$f" "$D/$f"; done < $W/missing.txt
echo "identity scan of the added files:"; (cd $D && xargs -a $W/missing.txt grep -l -i -E "zr2245|/scratch|nyu\.edu|torch_pr" 2>/dev/null | head -5); echo "  (end of scan)"
python3 - "$W/ByteMesh-code" "$OUT" <<'PY'
import sys, zipfile
from pathlib import Path
root = Path(sys.argv[1]); n = 0
with zipfile.ZipFile(sys.argv[2], 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(root.rglob('*')):
        if f.is_file():
            info = zipfile.ZipInfo(str(Path(root.name) / f.relative_to(root)), date_time=(2026, 11, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if f.suffix == '.sh' else 0o644) << 16; z.writestr(info, f.read_bytes()); n += 1
print('written %s: %d files, %.2f MB' % (sys.argv[2], n, Path(sys.argv[2]).stat().st_size / 1e6))
PY
