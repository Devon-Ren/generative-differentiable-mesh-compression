#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=vq-noref
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/zr2245/analysis/vq-noref-%j.out
# vq-noref-v1.sh (2026-11-16): BM-Codec WITHOUT the 400-step refinement of the codewords, on the same 11 meshes as the codec-time table
# (MeshSplatting with a vertex cap near 1.08 million, fine-tuned).  For each scene the existing delivery is repeated with the arguments
# recorded in its config.json and --shared-steps 0 (k-means + index assignment + entropy coding only), into a NEW directory.
# Nothing in the existing run directories is changed.  Rows -> $AN/vq-noref.txt : file size and TEST PSNR / SSIM / LPIPS of
#   the mesh before coding | BM-Codec as delivered (with refinement) | BM-Codec without refinement.
# Submit:  sbatch /scratch/zr2245/analysis/vq-noref-v1.sh        Result:  cat /scratch/zr2245/analysis/vq-noref.txt
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; PKG=$AN/vis_collapse-v12; OUT=$AN/vq-noref; TAB=$AN/vq-noref.txt; mkdir -p $OUT; : > $TAB
[ -f $PKG/deliver_coarse.sh ] || { echo "FAILED: missing $PKG/deliver_coarse.sh"; exit 1; }
mid () { case $1 in room) echo 1078000;; kitchen) echo 1086000;; counter) echo 1083000;; bonsai|garden) echo 1073000;; bicycle) echo 1080000;; stump) echo 1069000;; flowers) echo 1071000;; treehill) echo 1079000;; truck) echo 1078000;; train) echo 1085000;; esac; }
for S in garden room truck kitchen counter bonsai bicycle stump flowers treehill train; do D=$ROOT/$S-budget-$(mid $S)/axisB/v1.0/deliver
  [ -f $D/config.json ] && [ -f $D/summary.json ] || { echo "note: $S: no config.json / summary.json under $D, skipped"; continue; }
  ARGS=$(python - "$D/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = ['--coarse', c['coarse'], '--historical-config', 'none', '--source', c['source'], '--dataset', c['dataset'], '--lab', c['lab'], '--images', c['images'],
     '--k', str(c['k']), '--r', str(c['r']), '--xyz', c['xyz'], '--dc', c['dc'], '--dictionary-rounds', str(c['dictionary_rounds']), '--shared-steps', '0']
if c.get('qat'): a.append('--qat')
print(' '.join(a))
PY
) || { echo "FAILED: $S: config.json could not be read"; continue; }
  rm -rf $OUT/$S; bash $PKG/deliver_coarse.sh $ARGS --output $OUT/$S > $OUT/$S.log 2>&1 || { echo "FAILED: $S"; tail -6 $OUT/$S.log; continue; }
  python - "$S" "$D/summary.json" "$OUT/$S/summary.json" <<'PY' | tee -a $TAB
import json, sys
s, a, b = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3])); f = lambda t: '%.3f %.4f %.4f' % (t['psnr'], t['ssim'], t['lpips'])
print('VQNOREF %-9s | before coding test %s | with refinement %.3f MB test %s | without refinement %.3f MB test %s | steps %s / %s' % (s, f(a['input_test']), a['file_MB'], f(a['delivery_test']), b['file_MB'], f(b['delivery_test']), a.get('shared_steps'), b.get('shared_steps')))
PY
done
python - "$TAB" <<'PY' | tee $OUT/mean.txt
import re, sys, statistics as st
R = [list(map(float, re.findall(r'test ([\d.]+)', l))) + list(map(float, re.findall(r'([\d.]+) MB', l))) for l in open(sys.argv[1]) if l.startswith('VQNOREF')]
if R: print('VQNOREF mean over %d scenes | before coding %.3f dB | with refinement %.2f MB %.3f dB | without refinement %.2f MB %.3f dB' % (len(R), st.mean(r[0] for r in R), st.mean(r[3] for r in R), st.mean(r[1] for r in R), st.mean(r[4] for r in R), st.mean(r[2] for r in R)))
PY
cat $OUT/mean.txt >> $TAB
echo "vq-noref-v1 exit 0  $(date)"
