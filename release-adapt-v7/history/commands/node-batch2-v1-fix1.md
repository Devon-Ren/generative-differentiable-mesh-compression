# node-batch2-v1 fix 1（2026-11-05 16:15）：adapt smoke 自检误杀 → adapt-v1b.sbatch，重提 4 个 adapt job

原因：1300 it 的 smoke 里 upstream 的 densify 还没开始（densify_from > 1300），v1 的自检要求必须有 densify 事件，于是训练明明正常（prune 轮的控制器也正常：lam 1.65e-2 → 1.17e-2，投影 4.20 MB = 0.84×target）却被判 FAILED。v1b 只改自检（没有 densify 事件时打印 note，有则照查字段），其余逐字不变。ba-sens / fan 的 12 个 job 不受影响。

```bash
AN=/scratch/zr2245/analysis
cat > $AN/adapt-v1b.sbatch <<'SBEOF_ADAPT1B'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=adapt1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/zr2245/analysis/adapt1-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/adapt1-%x-%j.err
# adapt-v1b.sbatch (2026-11-05 16:15, VERSION v1b): = v1 with the smoke check fixed (no densify event happens within 1300 iterations: upstream densify_from > 1300; v1 asserted one and killed every job).
# adapt-v1.sbatch (2026-11-05, VERSION v1): closed-loop rate control -- lambda is adapted during training so the delivered size hits TARGETS (MB).
# For SCENE and each T in TARGETS: train with rdgrowth/train_rd_adapt_v1.py (= train_rd_v3 + rate controller, see its docstring), then the usual
# level-1.0 protocol (K4096 R256 XYZ/DC stage2-v2).  Smoke stage first (1300 iterations incl. one forced prune round): a crash shows in minutes.
# Output /scratch/zr2245/$SCENE-adapt1-t${T}mb/ ; one row per target in $AN/adapt-v1.txt: target | lam0 -> lam_final | projected | delivered MB (hit ratio) | test metrics.
# Restartable (existing checkpoints / deliveries skipped).  INDOOR=1 passes --indoor (room/kitchen/counter/bonsai).
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; TARGETS=${TARGETS:-"5 10 20"}; INDOOR=${INDOOR:?set INDOOR=0 or 1 via --export}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
GAMMA=${GAMMA:-0.5}; GAMMAP=${GAMMAP:-2.0}; DEAD=${DEAD:-0.1}; SMOKE=${SMOKE:-1}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/adapt-v1.txt; TR=$AN/rdgrowth/train_rd_adapt_v1.py
echo "adapt-v1 | scene $SCENE | IMAGES $IMAGES | targets $TARGETS MB | indoor '$FLAG' | B/v $BPV B/face $BPF | gamma $GAMMA / prune $GAMMAP dead $DEAD | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python -m py_compile $TR || { echo "FAILED: $TR does not compile"; exit 1; }
grep -q "'--target_mb'" $TR || { echo "FAILED: $TR has no --target_mb (wrong file pasted?)"; exit 1; }
lam0_of () { python -c "import sys; t=float(sys.argv[1]); print('%.4e' % min(1e-1, max(1e-4, 6.2e-3 * (9.0 / t) ** 1.67)))" $1; }   # initial guess from log MB ~ -0.6 log lambda
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
if [ "$SMOKE" = "1" ]; then
  SM=$AN/adapt1-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM; echo "=== smoke 1300 it, target 5 MB ($(date)) ==="
  python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam $(lam0_of 5) --target_mb 5 \
    --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx --ctrl_gamma $GAMMA --ctrl_gamma_prune $GAMMAP --ctrl_deadband $DEAD \
    --prune_iters 1250 --allow_any_prune_iter --prune_mu 1e6 --prune_max_frac 0.1 $FLAG > $SM/smoke.log 2>&1 || { echo "FAILED: SMOKE"; tail -40 $SM/smoke.log; exit 1; }
  python - <<PY || { echo "FAILED: SMOKE CHECK"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
if dn: assert all('lam_used' in e and 'desired' in e for e in dn), 'densify events lack controller fields'
else: print('note: no densify event within the 1300-it smoke (upstream densify_from > 1300); phase-A controller is exercised in the real runs')
assert p and 'lam_now' in p[0], 'no prune event with lam_now'; assert fin and 'lam_final' in fin[0], 'no final event with lam_final'
print('smoke densify:', [(e['iteration'], e['lam_used'], e['lam_next'], e['rhat_mb'], e['desired'], e['n_pass'], e['n_split']) for e in dn[:3]], '...', len(dn), 'events') if dn else None
print('smoke prune:', {k: p[0][k] for k in ('iteration','lam_now','rhat_mb','vertices_before','vertices_after','n_pairs')}); print('smoke final:', fin[0])
PY
  grep -h "^\[RD adapt\|^\[RD 1[0-3]00\]" $SM/smoke.log | tail -4
else echo "smoke skipped (SMOKE=0)"; fi
for T in $TARGETS; do
  G=/scratch/zr2245/$SCENE-adapt1-t${T}mb; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G; L0=$(lam0_of $T)
  echo "=== $SCENE target $T MB (lam0 $L0) $FLAG: train with rate control ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L0 --target_mb $T \
      --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx --ctrl_gamma $GAMMA --ctrl_gamma_prune $GAMMAP --ctrl_deadband $DEAD \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF $FLAG > $G/train.log 2>&1 || { echo "FAILED: TRAIN $SCENE target $T"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE target $T"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "target $T -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD adapt\|^\[RD prune\|^Training is done" $G/train.log | tail -8
  echo "=== $SCENE target $T: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  python - "$SCENE" "$T" "$G" "$NA" "$L0" "$XYZ" "$DC" <<'PY' | tee -a $TAB
import json, sys
S, T, G, NA, L0, xyz, dc = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
rs = json.load(open(G + '/original/rd_summary.json')); b = json.load(open(NA + '/report.json')); f = json.load(open(NA + '/finetune/summary.json')); d = json.load(open(NA + '/deliver/summary.json')); t = d['delivery_test']
ev = [json.loads(l) for l in open(G + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']; pr = [e for e in ev if e['event'] == 'prune']
lams = [e['lam_used'] for e in dn]; q = lambda a, p: sorted(a)[int(p * (len(a) - 1))] if a else float('nan')
print(f"{S} | target {T:g} MB | lam0 {L0} -> lamA q10/q50/q90 {q(lams,.1):.2e}/{q(lams,.5):.2e}/{q(lams,.9):.2e} -> lam_final {rs['lam_final']:.2e} | projected {rs['rhat_final_mb']:.2f} MB"
      f" | trained V {b['vertices']} F {b['faces']} | delivered {d['file_MB']:.3f} MB (hit {d['file_MB'] / T:.3f} x target) | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f}"
      f" | val {f['validation_best']['psnr']:.3f} | K4096 R256 {xyz}/{dc} | densify events {len(dn)}, prune rounds {[(e['iteration'], e['vertices_before'], e['vertices_after'], round(e['rhat_mb'], 2)) for e in pr]}")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
echo "adapt-v1 $SCENE exit 0  $(date)"
SBEOF_ADAPT1B
cat > $AN/submit-adapt-v1b.sh <<'SUB'
AN=/scratch/zr2245/analysis; RES=yw523-l40s
n=$(wc -l < $AN/adapt-v1b.sbatch); [ "$n" = 79 ] || { echo "CHECK FAILED: adapt-v1b.sbatch has $n lines, expected 79 (paste truncated?)"; exit 1; }
grep -q "VERSION v1b" $AN/adapt-v1b.sbatch || { echo "CHECK FAILED: not the v1b file"; exit 1; }
grep -q "if dn: assert" $AN/adapt-v1b.sbatch || { echo "CHECK FAILED: smoke fix missing"; exit 1; }
[ -f $AN/rdgrowth/train_rd_adapt_v1.py ] || { echo "CHECK FAILED: MISSING train_rd_adapt_v1.py"; exit 1; }
old=$(squeue --me -h -o "%i %j" | grep -E "adapt1$" | awk '{print $1}'); [ -z "$old" ] || { echo "cancelling leftover v1 adapt jobs: $old"; scancel $old; }
echo "CHECK OK"; k=0; ids=""
sub () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS="5 10 20" --job-name=$1-adapt1b $AN/adapt-v1b.sbatch) && { echo "submitted $1-adapt1b $j"; k=$((k+1)); ids="$ids,$j"; } || echo "FAILED: sbatch $1"; }
sub room 1 images_2 q14; sub kitchen 1 images_2 q16; sub garden 0 images_4 q15; sub bicycle 0 images_4 q16
echo "submitted $k of 4"
scontrol top ${ids#,} 2>&1 && echo "moved adapt jobs to the top of your queue" || echo "note: scontrol top not permitted here; adapt jobs run after the queued ba-sens/fan jobs (GPUs stay full either way)"
SUB
bash $AN/submit-adapt-v1b.sh
squeue --me -o "%.10i %.16j %.3t %.8M %R" | grep -E "adapt1|bpv|lkfan2|JOBID"
```

之后照旧 §4 看结果（adapt 的 .out 文件名变成 `adapt1-<scene>-adapt1b-<job>.out`，§4 的 glob 仍然匹配）。
