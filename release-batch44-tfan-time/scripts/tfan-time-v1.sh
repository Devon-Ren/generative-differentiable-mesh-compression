#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=tfan-time
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/zr2245/analysis/tfan-time-%j.out
# tfan-time-v1.sh (2026-11-16): separate encode and decode time of MPEG TFAN (Open3DGC), on the same 11 meshes as the codec-time run
# (MeshSplatting with a vertex cap near 1.08 million, fine-tuned; SH at 8 bits).  CPU only.
# The driver o3dgc_driver.cpp is rebuilt with two timers: one around Encode, one around DecodeHeader + DecodePayload.  Nothing else changes.
# Submit:  sbatch /scratch/zr2245/analysis/tfan-time-v1.sh        Result:  cat /scratch/zr2245/analysis/tfan-time.txt
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; O=$AN/o3dgc-v1; G=$AN/generic-v1; OUT=$AN/tfan-time; TAB=$AN/tfan-time.txt; mkdir -p $OUT; : > $TAB
for f in $O/o3dgc_driver.cpp $AN/generic_time_v1.py $G/generic_codec_v1.py; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
SRC=$O/open3dgc; [ -d $SRC/src ] || { echo "FAILED: no Open3DGC source at $SRC"; exit 1; }
python - "$O/o3dgc_driver.cpp" "$OUT/o3dgc_driver_timed.cpp" <<'PY' || { echo "FAILED: the driver could not be patched"; exit 1; }
import sys
s = open(sys.argv[1]).read()
def rep(a, b):
    global s
    if s.count(a) != 1: sys.exit('anchor not found exactly once: ' + a[:50])
    s = s.replace(a, b)
rep('#include <cstdio>', '#include <cstdio>\n#include <chrono>')
rep('    if (encoder.Encode(params, ifs, bstream) != O3DGC_OK) die("Encode");', '    auto t0 = std::chrono::steady_clock::now();\n    if (encoder.Encode(params, ifs, bstream) != O3DGC_OK) die("Encode");\n    auto t1 = std::chrono::steady_clock::now();')
rep('    if (decoder.DecodeHeader(ifs2, bstream) != O3DGC_OK) die("DecodeHeader");', '    auto t2 = std::chrono::steady_clock::now();\n    if (decoder.DecodeHeader(ifs2, bstream) != O3DGC_OK) die("DecodeHeader");')
rep('    if (decode_payload(decoder, ifs2, bstream, 0) != O3DGC_OK) die("DecodePayload");', '    if (decode_payload(decoder, ifs2, bstream, 0) != O3DGC_OK) die("DecodePayload");\n    auto t3 = std::chrono::steady_clock::now();\n    printf("O3DGC TIME encode %.3f s decode %.3f s\\n", std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t3 - t2).count());')
open(sys.argv[2], 'w').write(s)
PY
INC=$(find $SRC/src -type d -name inc | sed 's/^/-I/' | tr '\n' ' '); LIBSRC=$(find $SRC/src -path '*lib*' -name '*.cpp' | grep -v -i test | tr '\n' ' ')
g++ -O2 -std=c++11 -w $INC $OUT/o3dgc_driver_timed.cpp $LIBSRC -o $OUT/o3dgc_driver_timed > $OUT/build.log 2>&1 || { echo "FAILED: build"; tail -8 $OUT/build.log; exit 1; }
echo "tfan-time-v1 | host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) | $(date)"
mid () { case $1 in room) echo 1078000;; kitchen) echo 1086000;; counter) echo 1083000;; bonsai|garden) echo 1073000;; bicycle) echo 1080000;; stump) echo 1069000;; flowers) echo 1071000;; treehill) echo 1079000;; truck) echo 1078000;; train) echo 1085000;; esac; }
qp () { case $1 in room) echo 14;; kitchen|bicycle|stump|treehill) echo 16;; counter|bonsai|garden|flowers) echo 15;; truck|train) echo 18;; esac; }
for S in garden room truck kitchen counter bonsai bicycle stump flowers treehill train; do B=$(mid $S); CK=$ROOT/$S-budget-$B/axisB/v1.0/finetune/best.pt
  [ -f $CK ] || { echo "note: $S: no $CK, skipped"; continue; }
  python $AN/generic_time_v1.py --ckpt $CK --qp $(qp $S) --bits 8 --out $OUT/$S.json --generic $G --driver $OUT/o3dgc_driver_timed --label "$S cap$B" > $OUT/$S.log 2>&1 || { echo "FAILED: $S"; tail -4 $OUT/$S.log; continue; }
  python - "$OUT/$S.json" "$S" <<'PY' | tee -a $TAB
import json, re, sys
r = json.load(open(sys.argv[1])); m = re.search(r'TIME encode ([\d.]+) s decode ([\d.]+) s', r.get('tfan_driver_output', ''))
print('TFANTIME %s | whole call %s s | encode %s s | decode %s s' % (sys.argv[2], r.get('tfan_encode_plus_decode_s'), m[1] if m else 'n/a', m[2] if m else 'n/a'))
PY
done
echo "tfan-time-v1 exit 0  $(date)"
