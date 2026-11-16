#!/usr/bin/env bash
# submit-batch46-v1.sh (2026-11-16): bring the codec-side experiments to all 21 scenes, and the growth-aim ablation to 5 scenes.
#   time     : encode / decode time of BM-Codec, zstd, Draco and TFAN (encode and decode timed separately) on ALL 21 scenes, one CPU job on one
#              node (about 45 min).  Mesh per scene: MeshSplatting with the middle cap (about 1.08 million on the 13 captured scenes).
#   choices  : codec design choices (position predictors, DC prediction, SH index context) on the 10 new scenes, one CPU job (about 40 min)
#   noref    : BM-Codec without the codeword refinement on the 10 new scenes, one GPU job (about 30 min); also the time of each
#              vector-quantization step and the validation PSNR from the existing logs of these scenes
#   collapse : zstd / Draco / TFAN on the meshes from collapse after training, Deep Blending (2 GPU jobs, about 1 h) and the objects (8 GPU jobs,
#              a few minutes each), three vertex counts per scene
#   alpha    : growth aim 1.0 / 1.2 / 1.5 on bicycle, kitchen and truck, target 10 MB (9 GPU jobs of about 1 h)
# Usage: bash submit-batch46-v1.sh time | choices | noref | collapse | alpha | all        Collect: bash submit-batch46-v1.sh collect
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; W=${1:?time | choices | noref | collapse | alpha | all | collect}; ACC=torch_pr_38_general
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; return 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; return 1; }; }
# one file with the scene table, sourced by the job scripts:  run SCENE -> run directory of the mesh used ; qp SCENE -> position bits
cat > $AN/b46-scenes.sh <<'SC'
ROOT=/scratch/zr2245
MAIN="garden room truck kitchen counter bonsai bicycle stump flowers treehill train"; DBS="drjohnson playroom"; OBJS="lego chair drums ficus hotdog materials mic ship"
run () { case $1 in room|truck) echo $ROOT/$1-budget-1078000;; kitchen) echo $ROOT/$1-budget-1086000;; counter) echo $ROOT/$1-budget-1083000;; bonsai|garden) echo $ROOT/$1-budget-1073000;; bicycle) echo $ROOT/$1-budget-1080000;;
  stump) echo $ROOT/$1-budget-1069000;; flowers) echo $ROOT/$1-budget-1071000;; treehill) echo $ROOT/$1-budget-1079000;; train) echo $ROOT/$1-budget-1085000;; drjohnson|playroom) echo $ROOT/$1-budget-1081000;;
  lego) echo $ROOT/$1-budgetb-183000-best;; chair) echo $ROOT/$1-budgetb-178000-best;; drums|materials) echo $ROOT/$1-budgetb-176000-best;; ficus) echo $ROOT/$1-budgetb-67614-best;; hotdog|ship) echo $ROOT/$1-budgetb-177000-best;; mic) echo $ROOT/$1-budgetb-83900-best;; esac; }
qp () { case $1 in room) echo 14;; kitchen|bicycle|stump|treehill) echo 16;; counter|bonsai|garden|flowers|drjohnson|playroom) echo 15;; truck|train) echo 18;; *) echo 14;; esac; }
SC
if [ $W = collect ]; then
  echo "== time (all 21 scenes)"; cat $AN/codec-time21.txt 2>/dev/null
  echo "== choices (10 new scenes)"; cat $AN/codec-choices-b46.txt 2>/dev/null
  echo "== noref (10 new scenes)"; cat $AN/vq-noref-b46.txt 2>/dev/null
  echo "== collapse"; grep -h -E "^(drjohnson|playroom|lego|chair|drums|ficus|hotdog|materials|mic|ship) GENERIC-RD \| mesh [0-9]+ " $AN/generic-rd.txt
  echo "== alpha"; grep -h -E "^(bicycle|kitchen|truck) abl24-holda(100|120|150) \| target 10 MB" $AN/ablation-b24.txt | cut -c1-330
  echo "== unfinished"; for f in $(ls -t $AN/b46-*.out $AN/genrd-*-b46-*.out $AN/abl24-*-b46-*.out 2>/dev/null); do tail -1 $f | grep -q "exit 0" || echo "$f : $(tail -1 $f | cut -c1-140)"; done; exit 0
fi
# ---------------------------------------------------------------- time
if [ $W = time ] || [ $W = all ]; then
cat > $AN/b46-time.sh <<'JOB'
#!/usr/bin/env bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8; AN=/scratch/zr2245/analysis; source $AN/b46-scenes.sh; PKG=$AN/vis_collapse-v12; G=$AN/generic-v1; OUT=$AN/codec-time21; TAB=$AN/codec-time21.txt; mkdir -p $OUT; : > $TAB
DRV=$AN/tfan-time/o3dgc_driver_timed; [ -x $DRV ] || DRV=$AN/o3dgc-v1/o3dgc_driver; DOPT=""; [ -x $DRV ] && DOPT="--driver $DRV" || echo "note: no TFAN driver, that column stays empty"
for f in $AN/codec_time_v1.py $AN/generic_time_v1.py $PKG/stage2_run.sh $G/generic_codec_v1.py; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
python -c "import DracoPy, zstandard" 2>/dev/null || pip install DracoPy zstandard > $OUT/pip.log 2>&1 || echo "note: DracoPy / zstandard not installable"
echo "b46-time | host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) | threads $OMP_NUM_THREADS | driver $DRV | $(date)" | tee -a $TAB
for S in $MAIN $DBS $OBJS; do CG=$(run $S)/axisB/v1.0; Q=$(qp $S)
  [ -f $CG/deliver/shared-final.pt ] && [ -f $CG/finetune/best.pt ] || { echo "note: $S: no delivery / fine-tuned mesh under $CG, skipped" | tee -a $TAB; continue; }
  bash $PKG/stage2_run.sh $AN/codec_time_v1.py --shared $CG/deliver/shared-final.pt --xyz q$Q --dc q7 --out $OUT/$S-ours.json --label "$S ours" > $OUT/$S-ours.log 2>&1 && grep -h "^TIME" $OUT/$S-ours.log | tee -a $TAB || { echo "FAILED: ours $S" | tee -a $TAB; tail -4 $OUT/$S-ours.log; }
  python $AN/generic_time_v1.py --ckpt $CG/finetune/best.pt --qp $Q --bits 8 --out $OUT/$S-generic.json --generic $G $DOPT --label "$S generic" > $OUT/$S-generic.log 2>&1 && grep -h "^TIME" $OUT/$S-generic.log | tee -a $TAB || { echo "FAILED: generic $S" | tee -a $TAB; tail -4 $OUT/$S-generic.log; }
  python - "$OUT/$S-generic.json" "$S" <<'PY' | tee -a $TAB
import json, re, sys, os
if os.path.exists(sys.argv[1]):
    r = json.load(open(sys.argv[1])); m = re.search(r'TIME encode ([\d.]+) s decode ([\d.]+) s', r.get('tfan_driver_output', ''))
    print('TFANTIME %s | whole call %s s | encode %s s | decode %s s' % (sys.argv[2], r.get('tfan_encode_plus_decode_s'), m[1] if m else 'n/a', m[2] if m else 'n/a'))
PY
done
echo "b46-time exit 0  $(date)"
JOB
  j=$(sbatch --parsable --account=$ACC --cpus-per-task=8 --mem=48G --time=02:00:00 --job-name=b46-time --output=$AN/b46-time-%j.out $AN/b46-time.sh) && echo "submitted $j  b46-time (CPU)" || echo "FAILED: sbatch time"
fi
# ---------------------------------------------------------------- choices
if [ $W = choices ] || [ $W = all ]; then
cat > $AN/b46-choices.sh <<'JOB'
#!/usr/bin/env bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8; AN=/scratch/zr2245/analysis; source $AN/b46-scenes.sh; PKG=$AN/vis_collapse-v12; OUT=$AN/codec-choices-b46; TAB=$AN/codec-choices-b46.txt; mkdir -p $OUT; : > $TAB
for f in $AN/codec_choices_v2.py $PKG/stage2_run.sh; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
python -c "import zstandard" 2>/dev/null || pip install zstandard > $OUT/pip.log 2>&1
python $AN/codec_choices_v2.py --selftest || { echo "FAILED: selftest"; exit 1; }
for S in $DBS $OBJS; do CG=$(run $S)/axisB/v1.0
  [ -f $CG/deliver/shared-final.pt ] || { echo "note: $S: no $CG/deliver/shared-final.pt, skipped" | tee -a $TAB; continue; }
  bash $PKG/stage2_run.sh $AN/codec_choices_v2.py --shared $CG/deliver/shared-final.pt --xyz q$(qp $S) --dc q7 --out $OUT/$S.json --label "$S" > $OUT/$S.log 2>&1 && grep -h -E "^(DCPRED|XYZPRED|IDXCTX)" $OUT/$S.log | tee -a $TAB || { echo "FAILED: $S" | tee -a $TAB; tail -6 $OUT/$S.log; }
done
echo "b46-choices exit 0  $(date)"
JOB
  j=$(sbatch --parsable --account=$ACC --cpus-per-task=8 --mem=32G --time=02:00:00 --job-name=b46-choices --output=$AN/b46-choices-%j.out $AN/b46-choices.sh) && echo "submitted $j  b46-choices (CPU)" || echo "FAILED: sbatch choices"
fi
# ---------------------------------------------------------------- noref
if [ $W = noref ] || [ $W = all ]; then
cat > $AN/b46-noref.sh <<'JOB'
#!/usr/bin/env bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; source $AN/b46-scenes.sh; PKG=$AN/vis_collapse-v12; OUT=$AN/vq-noref-b46; TAB=$AN/vq-noref-b46.txt; mkdir -p $OUT; : > $TAB
[ -f $PKG/deliver_coarse.sh ] || { echo "FAILED: missing $PKG/deliver_coarse.sh"; exit 1; }
for S in $DBS $OBJS; do D=$(run $S)/axisB/v1.0/deliver
  [ -f $D/config.json ] && [ -f $D/summary.json ] || { echo "note: $S: no config.json / summary.json under $D, skipped" | tee -a $TAB; continue; }
  # the vector-quantization steps of the EXISTING delivery, from its log: time of each step and validation PSNR
  python - "$S" "$(run $S)/axisB" <<'PY' | tee -a $TAB
import re, glob, sys
sec = lambda t: int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8]); T = {}; P = {}
for f in glob.glob(sys.argv[2] + '/**/*.log', recursive=True):
    for l in open(f, errors='ignore'):
        m = re.match(r'\[(\d\d:\d\d:\d\d)\] (k-means seed|MEASURE (\w+) \d+ ([\d.]+))', l)
        if m:
            T[m[3] or 'kmeans'] = sec(m[1])
            if m[3]: P[m[3]] = float(m[4])
K = ('kmeans', 'coarse_input', 'shared_initial', 'shared_fitted')
if all(k in T for k in K): print('VQTIME %-9s k-means %2d s | assign %2d s | refine %2d s | validation PSNR before %.2f, k-means only %.2f, refined %.2f' % (sys.argv[1], T[K[1]] - T[K[0]], T[K[2]] - T[K[1]], T[K[3]] - T[K[2]], P[K[1]], P[K[2]], P[K[3]]))
else: print('VQTIME %-9s incomplete log: %s' % (sys.argv[1], sorted(T)))
PY
  ARGS=$(python - "$D/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); a = ['--coarse', c['coarse'], '--historical-config', 'none', '--source', c['source'], '--dataset', c['dataset'], '--lab', c['lab'], '--images', c['images'],
     '--k', str(c['k']), '--r', str(c['r']), '--xyz', c['xyz'], '--dc', c['dc'], '--dictionary-rounds', str(c['dictionary_rounds']), '--shared-steps', '0']
if c.get('qat'): a.append('--qat')
print(' '.join(a))
PY
) || { echo "FAILED: $S: config.json could not be read" | tee -a $TAB; continue; }
  # a fresh output directory per scene, never inside an existing run directory (deliver_coarse.py replaces its own delivery/ subfolder)
  NEW=$OUT/$S-$(date +%H%M%S); bash $PKG/deliver_coarse.sh $ARGS --output $NEW > $OUT/$S.log 2>&1 || { echo "FAILED: $S" | tee -a $TAB; tail -6 $OUT/$S.log; continue; }
  python - "$S" "$D/summary.json" "$NEW/summary.json" <<'PY' | tee -a $TAB
import json, sys
s, a, b = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3])); f = lambda t: '%.3f %.4f %.4f' % (t['psnr'], t['ssim'], t['lpips'])
print('VQNOREF %-9s | before coding test %s | with refinement %.3f MB test %s | without refinement %.3f MB test %s' % (s, f(a['input_test']), a['file_MB'], f(a['delivery_test']), b['file_MB'], f(b['delivery_test'])))
PY
done
echo "b46-noref exit 0  $(date)"
JOB
  j=$(sbatch --parsable --account=$ACC --gres=gpu:1 --constraint=l40s --cpus-per-task=6 --mem=48G --time=01:30:00 --job-name=b46-noref --output=$AN/b46-noref-%j.out $AN/b46-noref.sh) && echo "submitted $j  b46-noref (GPU)" || echo "FAILED: sbatch noref"
fi
# ---------------------------------------------------------------- collapse
if [ $W = collapse ] || [ $W = all ]; then
  chk $AN/generic-rd-v4.sbatch 73 a48f1e5f1aed893f3d413030de658301 && python3 - "$AN/generic-rd-v4.sbatch" "$AN/generic-rd-v5.sbatch" <<'PY' && bash -n $AN/generic-rd-v5.sbatch && {
import sys
s = open(sys.argv[1]).read(); A = "  if [ $M = full ]; then MK=$CK; TAG=full; fi\n"
if s.count(A) != 1: sys.exit('FAILED: anchor not found exactly once')
# batch46: a mesh named g0ctrl-<B> is the FP32 model collapsed after training to the vertex count of budget B (fine-tuned); its tag is <B>, as in batch33
s = s.replace(A, A + "  case $M in g0ctrl-*) MK=$ORIG/axisB/$M/finetune/best.pt; TAG=${M#g0ctrl-};; esac\n")
open(sys.argv[2], 'w').write(s); print('written generic-rd-v5.sbatch')
PY
    lv () { case $1 in drjohnson) echo "1370000 532000 224000";; playroom) echo "1201000 441000 191000";; lego) echo "372025 167656 78124";; chair) echo "238956 137117 66755";; drums) echo "258000 159000 68000";;
      ficus) echo "104589 67614 35557";; hotdog) echo "222000 100090 47578";; materials) echo "259000 125118 68793";; mic) echo "170000 83900 46659";; ship) echo "294000 164000 74000";; esac; }
    for S in drjohnson playroom lego chair drums ficus hotdog materials mic ship; do case $S in drjohnson|playroom) O=$ROOT/$S-indoor; TL=01:15:00;; *) O=$ROOT/$S-blender4-best; TL=00:30:00;; esac
      C=""; for B in $(lv $S); do [ -f $O/axisB/g0ctrl-$B/finetune/best.pt ] && C="$C g0ctrl-$B" || echo "note: $S: no $O/axisB/g0ctrl-$B/finetune/best.pt"; done
      [ -n "$C" ] || { echo "skipped: $S (no collapsed meshes found)"; continue; }
      j=$(sbatch --parsable --account=$ACC --cpus-per-task=8 --time=$TL --export=ALL,SCENE=$S,CAPS="$C" --job-name=$S-gencol-b46 --output=$AN/genrd-%x-%j.out $AN/generic-rd-v5.sbatch) && echo "submitted $j  $S-gencol-b46 ($C)" || echo "FAILED: sbatch $S"; done; }
fi
# ---------------------------------------------------------------- alpha
if [ $W = alpha ] || [ $W = all ]; then
  cfg () { case $1 in bicycle) echo "INDOOR=0,IMAGES=images_4,XYZ=q16";; kitchen) echo "INDOOR=1,IMAGES=images_2,XYZ=q16";; truck) echo "INDOOR=0,IMAGES=images,XYZ=q18,DATA=$ROOT/datasets/tandt/truck";; esac; }
  chk $AN/rdgrowth/train_rd_adapt_v9b_abl7.py 1083 1eeb64ab5032e903f69ad1a21466d99f && chk $AN/ablation-b24-r10.sbatch 221 5f4e05f1e2db138ca5341214f0c12ad7 && for V in holda100 holda120 holda150; do for S in bicycle kitchen truck; do
    [ -e $ROOT/$S-abl24-$V-t10mb ] && { echo "exists, skipped: $S $V"; continue; }
    j=$(sbatch --parsable --account=$ACC --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS="10" --job-name=$S-$V-b46 --output=$AN/abl24-%x-%j.out $AN/ablation-b24-r10.sbatch) && echo "submitted $j  $S-$V-b46" || echo "FAILED: sbatch $S $V"; done; done
fi
squeue -u $USER -h -o "%j %T" | grep -E "b46" | sort | uniq -c
