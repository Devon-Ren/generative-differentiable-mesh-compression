# node-batch9-v2（2026-11-06）：定版 adapt-v7b 的场景 sweep —— 19 个场景、47 个 job、≈45 GPU-h，4 卡过夜

**v2**：文件名全部换成 batch8-v2 写上去的 v7b 版本（`train_rd_adapt_v7b.py`、`adapt-v7b-blender.sbatch`），Mip-360 / T&T 的 sbatch 改名 `adapt-v7b.sbatch`（VERSION v7-r2 = adapt-v7.sbatch + TR/grep/断言指向 v7b）；逻辑、场景参数、job 顺序与 v1 完全相同。

**只在 batch8（lego/chair × 2.5/4）四条都过了之后才提交。** 代码一个字不改：`train_rd_adapt_v7b.py` 和 `adapt-v7b-blender.sbatch` 就是 batch8 写上去的（提交脚本按 md5 核对），新增的只有 Mip-360 / T&T 的 `adapt-v7b.sbatch`（129 行，VERSION v7 = adapt-v5.sbatch + v7 的同一套改动：train_rd_adapt_v7、OVER 1.0、prune 12000 起每 1000 到 19000 共 8 轮（这些场景每次 probe 30–60 s）、诊断 probe 22000/26000、v7b 的 smoke 断言和结果行）。

**跑什么**（每个 job 一个场景一个目标，按目标分波，先出 5 MB / 2.5 MB 的全场景结果）：
- Mip-360 / T&T 11 场景 × {5, 10, 20} MB = 33 job，每个 ≈70–100 min（20 MB 的大场景最长）；≈38 GPU-h。20 MB 在大场景上预期 `growth-capped True`（v4 已确认是 upstream add_percentage 的上限），照跑照报，它们排在最后一波。
- NeRF-Synthetic 另外 6 场景（drums ficus hotdog materials mic ship）× {2.5, 4} MB = 12 job，每个 ≈30 min；≈5 GPU-h。lego / chair 已在 batch8 里，不重跑。
- 8 场景固定 λ hull（λ 6.2e-4 / 6.2e-3）+ 8 个 FP32 原始模型用 val_select_v3 重交付（不重训）：2 job，≈2 GPU-h。论文 Blender 表的参考字节用这个。
- 每个场景的第一个 job（5 MB / 2.5 MB）带 self-check + smoke，同场景后面的目标 SMOKE=0,SELFCHECK=0（代码路径一样，省 ≈1.5 GPU-h）。
- 顺序：Mip/T&T 5 MB → Blender 2.5 → Mip/T&T 10 MB → Blender 4 → 重交付 → Mip/T&T 20 MB。4 卡 ≈11–12 h 墙钟：早上 8 h 时 5/10 MB 和 Blender 应全部结束，20 MB 大约过半，其余中午前完。

## 1. 写 adapt-v7b.sbatch + 提交 47 个 job（一次粘贴；先核对 batch8 的文件 md5，再提交）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/adapt-v7b.sbatch <<'SBEOF_ADAPT7'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=adapt7
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zr2245/analysis/adapt7-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/adapt7-%x-%j.err
# adapt-v7b.sbatch (2026-11-06, VERSION v7-r2): = adapt-v7.sbatch (VERSION v7) with train_rd_adapt_v7b.py (v7 + the final-probe unpack fix that
# crashed every batch8 smoke at its last line); nothing else changed.  v7 text: Mip-360 / T&T sweep sbatch = adapt-v5.sbatch with train_rd_adapt_v7.py (phase B of v6 + the three fixes
# verified on the Blender testbed: exact kept-aware collapse accounting, prune gated on the extrapolated landing and capped at the excess, growth-capped
# flag; diagnostic probes at EXTRA after the last round), phase-A overshoot 1.0 (OVER default, as v6/v7 Blender), prune rounds every 1000 from 12000 to
# 19000 (PRUNE default; 8 rounds, probe ~30-60 s each on these scenes), the v7b smoke assertions and result row.  Same stages: (0) probe self-check on
# the adapt-v1 10 MB mesh where it exists, (1) smoke 1300 it with a forced prune round at 1250 and --probe_check_scaling, (2) per target T: train 30000 it,
# level-1.0 protocol, one row in $AN/adapt-v7.txt.  Output /scratch/zr2245/$SCENE-adapt7-t${T}mb/ .  Restartable.  INDOOR=1 passes --indoor; T&T scenes
# need DATA=/scratch/zr2245/datasets/tandt/$SCENE and IMAGES=images via --export.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; TARGETS=${TARGETS:-"5 10 20"}; INDOOR=${INDOOR:?set INDOOR=0 or 1 via --export}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"$(seq -s " " 12000 1000 19000)"}
OVER=${OVER:-1.0}; EXTRA=${EXTRA:-"22000 26000"}; FPV=${FPV:-2.5}; PFLOOR=${PFLOOR:-0.9999}; PSCALE=${PSCALE:-4}; SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; STAGE2=/scratch/zr2245/shared-collapse-stage2-v2
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/adapt-v7.txt; TR=$AN/rdgrowth/train_rd_adapt_v7b.py; PROBE_PY=$AN/rdgrowth/probe_bytes_v1.py
PROBE="bash $PKG/stage2_run.sh $PROBE_PY"
echo "adapt-v7 | scene $SCENE | IMAGES $IMAGES | targets $TARGETS MB | indoor '$FLAG' | initial B/v $BPV B/face $BPF faces/vertex $FPV | overshoot $OVER probe floor $PFLOOR probe scaling $PSCALE sigma prior $SPRIOR | prune at $PRUNE mu $MU max_frac $MAXF | extra probes $EXTRA | probe $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $PROBE_PY $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $PKG/stage2_run.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py $STAGE2/STAGE2_VERSION $STAGE2/runtime.py $STAGE2/sharing_after.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python -m py_compile $TR || { echo "FAILED: $TR does not compile"; exit 1; }
python -m py_compile $PROBE_PY || { echo "FAILED: $PROBE_PY does not compile"; exit 1; }
grep -q "version='adapt-v7b'" $TR || { echo "FAILED: $TR is not adapt-v7b (wrong file pasted?)"; exit 1; }
grep -q "PROBE_VERSION = 'probe-v1'" $PROBE_PY || { echo "FAILED: $PROBE_PY is not probe-v1"; exit 1; }
grep -q "def faces_removed_by" $TR || { echo "FAILED: $TR lacks faces_removed_by (v6 pasted instead of v7?)"; exit 1; }
grep -q "growth_capped" $TR && grep -q "probe_extra_iters" $TR || { echo "FAILED: $TR lacks growth_capped / probe_extra_iters"; exit 1; }
grep -q "default=1.0, help='phase A aims at target" $TR || { echo "FAILED: $TR phase_a_overshoot default is not 1.0"; exit 1; }
NPR=$(echo $PRUNE | wc -w); [ "$NPR" -ge 4 ] || { echo "FAILED: PRUNE has $NPR rounds (v7 default is 8, at least 4 expected: $PRUNE)"; exit 1; }
LASTP=$(echo $PRUNE | tr ' ' '\n' | sort -n | tail -1); for x in $EXTRA; do [ "$x" -gt "$LASTP" ] && [ "$x" -lt 30000 ] || { echo "FAILED: EXTRA probe $x must lie after the last prune round $LASTP and before 30000"; exit 1; }; done
grep -q "triangles.scaling" $LAB/upstream/train.py || { echo "FAILED: $LAB/upstream/train.py never sets triangles.scaling (the probe's 4x pass assumes upstream's supersampling attribute)"; exit 1; }
lam0_of () { python -c "import sys; t=float(sys.argv[1]); print('%.4e' % min(1e-1, max(1e-4, 6.2e-3 * (9.0 / t) ** 1.67)))" $1; }   # initial guess only; adapted from the first interval on
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
SM=$AN/adapt7-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
if [ "$SELFCHECK" = "1" ]; then
  PV=/scratch/zr2245/$SCENE-adapt1-t10mb/axisB/v1.0
  if [ -f $PV/finetune/best.pt ] && [ -f $PV/deliver/summary.json ]; then
    echo "=== probe self-check on $PV ($(date)) ==="
    $PROBE --coarse $PV/finetune/best.pt --out $SM/selfcheck.json --k $K --r $R --xyz $XYZ --dc $DC > $SM/selfcheck.log 2>&1 || { echo "FAILED: PROBE SELF-CHECK (probe crashed)"; tail -30 $SM/selfcheck.log; exit 1; }
    python - "$SM/selfcheck.json" "$PV/deliver/summary.json" <<'PY' || { echo "FAILED: PROBE SELF-CHECK (bytes disagree with the delivered file)"; tail -5 $SM/selfcheck.log; exit 1; }
import json, sys
p = json.load(open(sys.argv[1])); d = json.load(open(sys.argv[2])); pm = p['file_bytes'] / 1e6; dm = d['file_MB']; r = pm / dm
big = {k: round(v / 1e6, 3) for k, v in sorted(p['payload_bytes'].items(), key=lambda kv: -kv[1]) if v > 20000}
print(f"self-check: probe {pm:.3f} MB vs delivered {dm:.3f} MB -> probe/delivered {r:.3f} | V {p['vertices']} F {p['faces']} | {p['seconds']['total']} s | streams {big}")
assert abs(r - 1.0) <= 0.08, f'probe/delivered {r:.3f} is outside 0.92..1.08'
PY
  else echo "note: no adapt-v1 10 MB delivery for $SCENE at $PV; self-check skipped (the smoke stage still exercises the probe)"; fi
else echo "self-check skipped (SELFCHECK=0)"; fi
if [ "$SMOKE" = "1" ]; then
  echo "=== smoke 1300 it, target 0.01 MB so the prune round at 1250 must collapse; probe at ${PSCALE}x AND 1x ($(date)) ==="
  python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam $(lam0_of 5) --target_mb 0.01 \
    --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --probe_check_scaling --sigma_prior $SPRIOR \
    --prune_iters 1250 --allow_any_prune_iter --prune_mu $MU --prune_max_frac 0.1 \
    --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $SM/smoke.log 2>&1 || { echo "FAILED: SMOKE"; tail -40 $SM/smoke.log; for l in $SM/original/probes/*.log; do echo "--- $l"; tail -15 $l; done; exit 1; }
  python - <<PY || { echo "FAILED: SMOKE CHECK"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
if dn: assert all('lam_used' in e and 'desired' in e and 'sigma_plan' in e and 'front_loaded' in e and 'v_star' in e for e in dn), 'densify events lack the v4 controller fields'
else: print('note: no densify event within the 1300-it smoke (upstream densify_from > 1300); phase-A controller is exercised in the real runs')
assert p and 'probe_mb' in p[0], 'no prune event with probe_mb (probe did not run)'
assert not p[0].get('skipped') and p[0]['n_pairs'] > 0, 'prune round at 1250 collapsed nothing: %r' % {k: p[0].get(k) for k in ('skipped','desired_pairs','n_pairs','n_cand','max_pairs')}
assert p[0]['source'].startswith('probe@'), 'size model not replaced by the probe: %r' % p[0]['source']
assert p[0]['lam_now'] is not None and p[0].get('mode') in ('count', 'lambda', 'capped'), 'no prune mode / lambda: %r' % p[0].get('mode')
assert 'kept_removed' in p[0] and 0 <= p[0]['kept_removed'] <= p[0]['n_pairs'], 'kept-removed count missing / out of range: %r' % {k: p[0].get(k) for k in ('kept_removed','n_pairs')}
assert 'kept_faces_removed' in p[0] and 0 <= p[0]['kept_faces_removed'] <= p[0]['faces_before'] - p[0]['faces_after'], 'kept-faces-removed count missing / out of range: %r' % {k: p[0].get(k) for k in ('kept_faces_removed','faces_before','faces_after')}
assert 'removed_mb' in p[0] and p[0].get('bytes_per_pair_measured') is not None and p[0].get('excess_pairs', 0) > 0, 'v7 accounting fields missing: %r' % {k: p[0].get(k) for k in ('removed_mb','bytes_per_pair_measured','excess_pairs')}
if p[0]['removed_mb'] == 0: print('note: the smoke round collapsed pairs outside the probe kept set (removed_mb 0): accounting exercised, nothing to charge')
assert abs(p[0]['removed_mb'] * 1e6 - (p[0]['kept_removed'] * p[0]['b_v'] + p[0]['kept_faces_removed'] * p[0]['b_f'])) < 2000, 'removed_mb is not kept vertices x b_v + kept faces x b_f: %r' % {k: p[0].get(k) for k in ('removed_mb','kept_removed','kept_faces_removed','b_v','b_f')}
assert 'probe_est_mb' in p[0] and p[0]['probe_est_mb'] == p[0]['probe_mb'], 'first round must use the probe as is (no slope yet): %r' % {k: p[0].get(k) for k in ('probe_mb','probe_est_mb')}
assert 'probe_kept_frac' in p[0] and p[0]['probe_kept_frac'] > 0, 'probe kept fraction missing'
assert fin and 'probe_final_mb' in fin[0], 'no final event with probe_final_mb'
rs=json.load(open('$SM/original/rd_summary.json')); assert rs['version']=='adapt-v7b' and len(rs['probes'])==2, 'rd_summary lacks the two probes'
assert rs.get('growth_capped') is False and 'limiter_bound_intervals' in rs and rs.get('probe_extra_iters') == [], 'growth-capped / limiter fields wrong in the smoke: %r' % {k: rs.get(k) for k in ('growth_capped','limiter_bound_intervals','densify_intervals','probe_extra_iters')}
assert len(rs['probe_trend'])==1 and rs['extrapolate_to']>18000 and rs['probes'][1].get('faces_kept_1x') is None, 'trend / final-probe check fields wrong: %r' % {k: rs.get(k) for k in ('probe_trend','extrapolate_to')}
pr=rs['probes'][0]; assert pr.get('probe_scaling')==$PSCALE and pr.get('faces_kept_1x') is not None and pr['faces_kept']>pr['faces_kept_1x'], 'probe scaling check: %r' % {k: pr.get(k) for k in ('probe_scaling','faces_kept','faces_kept_1x','faces_model')}
print('smoke probe scaling: kept %d faces at %dx vs %d at 1x (of %d; %.2fx more)' % (pr['faces_kept'], pr['probe_scaling'], pr['faces_kept_1x'], pr['faces_model'], pr['faces_kept']/max(pr['faces_kept_1x'],1)))
print('smoke prune:', {k: p[0].get(k) for k in ('iteration','probe_mb','probe_est_mb','probe_kept_frac','probe_vertices','probe_faces','b_v','b_f','overhead_mb','faces_per_vertex','desired_pairs','excess_pairs','mode','n_lam','rank_k','n_cand','max_pairs','n_pairs','kept_removed','kept_faces_removed','removed_mb','bytes_per_pair_measured','lam_now','shadow_lam','vertices_before','vertices_after','probe_seconds')})
print('smoke final:', {k: fin[0][k] for k in ('vertices','triangles','lam_final','rhat_mb','probe_final_mb','growth_capped','limiter_bound_intervals','densify_intervals')})
PY
  grep -h "^\[RD adapt\|^\[RD prune\|^Training is done" $SM/smoke.log | tail -4; for l in $SM/original/probes/*.log; do grep -h "^\[probe\]" $l; done
else echo "smoke skipped (SMOKE=0)"; fi
for T in $TARGETS; do
  G=/scratch/zr2245/$SCENE-adapt7-t${T}mb; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G; L0=$(lam0_of $T)
  echo "=== $SCENE target $T MB (lam0 $L0) $FLAG: train with closed-loop rate control ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L0 --target_mb $T \
      --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --probe_extra_iters $EXTRA \
      --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $G/train.log 2>&1 || { echo "FAILED: TRAIN $SCENE target $T"; tail -12 $G/train.log; for l in $G/original/probes/*.log; do echo "--- $l"; tail -8 $l; done; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE target $T"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "target $T -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD adapt\|^\[RD prune\|^Training is done" $G/train.log | tail -14
  echo "=== $SCENE target $T: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  python - "$SCENE" "$T" "$G" "$NA" "$L0" "$XYZ" "$DC" <<'PY' | tee -a $TAB
import json, sys
S, T, G, NA, L0, xyz, dc = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
rs = json.load(open(G + '/original/rd_summary.json')); b = json.load(open(NA + '/report.json')); f = json.load(open(NA + '/finetune/summary.json')); d = json.load(open(NA + '/deliver/summary.json')); t = d['delivery_test']
ev = [json.loads(l) for l in open(G + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']; pr = [e for e in ev if e['event'] == 'prune']; px = [e for e in ev if e['event'] == 'probe_extra']
lams = [e['lam_used'] for e in dn]; q = lambda a, p: sorted(a)[int(p * (len(a) - 1))] if a else float('nan')
rounds = [(e['iteration'], round(e['probe_mb'], 3), round(e.get('probe_est_mb', e['probe_mb']), 3), e.get('probe_kept_frac'), e['vertices_before'], e['vertices_after'], e.get('mode'), ('%.2e' % e['lam_now']) if e.get('lam_now') is not None else None, round(e['b_v'], 2), round(e['b_f'], 2), e.get('n_pairs', 0), e.get('kept_removed'), e.get('kept_faces_removed'), round(e.get('removed_mb') or 0.0, 3)) for e in pr]
extra = [(e['iteration'], round(e['probe_mb'], 3), round(e['probe_est_mb'], 3), round(e['probe_slope_mb_per_1k'], 3)) for e in px]
ncoll = sum(e.get('n_pairs', 0) for e in pr); nkept = sum(e.get('kept_removed') or 0 for e in pr); nact = sum(1 for e in pr if e.get('n_pairs', 0) > 0)
sig = [e.get('sigma') for e in dn]; sq = lambda p: sorted(x for x in sig if x is not None)[int(p * (len([x for x in sig if x is not None]) - 1))] if any(x is not None for x in sig) else float('nan')
nb = sum(1 for e in dn if e.get('limiter_bound')); nf = sum(1 for e in dn if e.get('front_loaded')); vs = dn[-1]['v_star'] if dn else None
yl = [e.get('rank_yield') for e in dn if e.get('rank_yield') is not None]; yq = lambda p: sorted(yl)[int(p * (len(yl) - 1))] if yl else float('nan')
print(f"{S} | target {T:g} MB | lam0 {L0} -> lamA q10/q50/q90 {q(lams,.1):.2e}/{q(lams,.5):.2e}/{q(lams,.9):.2e} -> lam_final (shadow) {rs['lam_final']:.2e} | final probe {rs['probe_final_mb']:.3f} MB"
      f" | trained V {b['vertices']} F {b['faces']} | delivered {d['file_MB']:.3f} MB (hit {d['file_MB'] / T:.3f} x target, probe/delivered {rs['probe_final_mb'] / d['file_MB']:.3f}) | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f}"
      f" | val {f['validation_best']['psnr']:.3f} | K4096 R256 {xyz}/{dc} | growth-capped {rs.get('growth_capped')} (last landing est {rs.get('last_probe_est_mb')} MB) | densify events {len(dn)} (limiter-bound {nb}, front-loaded {nf}, V* {vs}), sigma q10/q50/q90 {sq(.1):.2f}/{sq(.5):.2f}/{sq(.9):.2f}, yield q10/q50/q90 {yq(.1):.2f}/{yq(.5):.2f}/{yq(.9):.2f}, prune rounds {len(pr)} active {nact} collapsed {ncoll} (both-kept share {nkept / max(ncoll, 1):.2f}; V {pr[0]['vertices_before'] if pr else None} -> {pr[-1]['vertices_after'] if pr else None}), extra probes (it, MB, est MB, slope) {extra}, rounds (it, probe MB, est MB, kept, V before, V after, mode, lam, b_v, b_f, pairs, kept-removed, kept-faces-removed, removed MB) {rounds}")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
echo "adapt-v7 $SCENE exit 0  $(date)"
SBEOF_ADAPT7
cat > $AN/submit-batch9-v2.sh <<'SUB9'
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (not the file batch8/batch7 wrote?)"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v7b.py 827 24f478cab998b59d1886c2b8b956a111
chk $AN/adapt-v7b-blender.sbatch 139 2d110e69c79c3c2552c64fdbfef49c91
chk $AN/adapt-v7b.sbatch 129 0649962ab5797f7f69dc39128d74a7e9
chk $AN/redeliver-v3-blender.sbatch 66 619743b5f07247f4c61e9def9d4b977d
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v7b.py $AN/blender/val_select_v3.py $AN/rdgrowth/probe_bytes_v1.py || { echo "CHECK FAILED: python files do not compile"; exit 1; }
bash -n $AN/adapt-v7b.sbatch && bash -n $AN/adapt-v7b-blender.sbatch && bash -n $AN/redeliver-v3-blender.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
for f in $AN/vis_collapse-v12/stage2_run.sh $AN/vis_collapse-v12/scripts/run_garden.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py $AN/blender/train_ckpt_v2.py $AN/coarse_finetune.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || { echo "CHECK FAILED: MISSING $f"; exit 1; }; done
n_ok=0; for r in lego-2.5 chair-2.5 lego-4 chair-4; do S=${r%-*}; T=${r#*-}; [ -f /scratch/zr2245/$S-adapt7b-t${T}mb-best/axisB/v1.0/deliver/summary.json ] && n_ok=$((n_ok+1)); done
[ "$n_ok" = 4 ] || { echo "CHECK FAILED: only $n_ok of the 4 batch8 testbed deliveries exist; do not sweep before batch8 is done and judged"; exit 1; }
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in drums ficus hotdog materials mic ship; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
LAMS=""; ORIG=""; for S in chair drums ficus hotdog lego materials mic ship; do
  [ -d /scratch/zr2245/$S-blender2/original/point_cloud ] && ORIG="$ORIG $S-blender2" || echo "note: $S-blender2 has no checkpoints, original not re-delivered"
  [ "$S" = lego ] || [ "$S" = chair ] || for L in 0.00062 0.0062; do R=$S-rd3b4-lam$L-mu1.0; [ -d /scratch/zr2245/$R/original/point_cloud ] && LAMS="$LAMS $R" || echo "note: $R has no checkpoints, not re-delivered"; done; done
echo "re-deliver lambda runs:$LAMS"; echo "re-deliver originals:$ORIG"
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt7|redeliv3"); [ "$busy" = 0 ] || echo "note: $busy adapt7/redeliver jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"
echo "CHECK OK"; k=0
m () { EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5$6"; [ "$5" = 5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:00:00 --export=ALL,$EX --job-name=$1-$5-adapt7 $AN/adapt-v7b.sbatch) && { echo "submitted $1 $5 MB $j"; k=$((k+1)); } || echo "FAILED: sbatch $1 $5"; }
b () { EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:30:00 --export=ALL,$EX --job-name=$1-$2-adapt7b $AN/adapt-v7b-blender.sbatch) && { echo "submitted $1 $2 MB $j"; k=$((k+1)); } || echo "FAILED: sbatch $1 $2"; }
mip () { m room 1 images_2 q14 $1; m kitchen 1 images_2 q16 $1; m counter 1 images_2 q15 $1; m bonsai 1 images_2 q15 $1; m garden 0 images_4 q15 $1; m bicycle 0 images_4 q16 $1; m stump 0 images_4 q16 $1; m flowers 0 images_4 q15 $1; m treehill 0 images_4 q16 $1; m truck 0 images q18 $1 ",DATA=$D/tandt/truck"; m train 0 images q18 $1 ",DATA=$D/tandt/train"; }
bl () { for S in drums ficus hotdog materials mic ship; do b $S $1; done; }
mip 5; bl 2.5; mip 10; bl 4
[ -n "$LAMS" ] && { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --export=ALL,RUNS="$LAMS",XYZ=q14 --job-name=refs-lams-redeliv3 $AN/redeliver-v3-blender.sbatch) && { echo "submitted refs-lams-redeliv3 $j"; k=$((k+1)); } || echo "FAILED: sbatch redeliver lams"; }
[ -n "$ORIG" ] && { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --export=ALL,RUNS="$ORIG",XYZ=q14 --job-name=refs-orig-redeliv3 $AN/redeliver-v3-blender.sbatch) && { echo "submitted refs-orig-redeliv3 $j"; k=$((k+1)); } || echo "FAILED: sbatch redeliver originals"; }
mip 20
echo "submitted $k of 47 (4 GPUs: Slurm starts them in this order as GPUs free up)"
SUB9
bash $AN/submit-batch9-v2.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R" | head -60
```

## 2. 10–15 分钟后：第一波 4 个 job 的自检 / smoke

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R" | head -12
for f in $AN/adapt7-*-adapt7-*.out; do echo "=== $f"; grep -h "self-check\|smoke prune\|smoke final\|note:\|FAILED\|=== .* target" $f | tail -6 | cut -c1-600; done
```

有 FAILED 的 job 把 `.out` 最后 40 行贴我；同场景的其它目标 job 可以先 `scancel`（它们会在第一轮 prune 处以同样的错误停下来）。

## 3. 早上：全部结果

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-14hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt7|redeliv3|JobID"
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
echo "===== FAILED jobs"; grep -l "FAILED" $AN/adapt7-*.out $AN/adapt7b-*.out $AN/redeliv3-*.out 2>/dev/null
echo "===== adapt-v7.txt (Mip-360 / T&T)"; cat $AN/adapt-v7.txt
echo "===== adapt-v7-blender.txt (NeRF-Synthetic, incl. batch8 lego/chair)"; cat $AN/adapt-v7-blender.txt
echo "===== redeliver-v3-blender.txt"; cat $AN/redeliver-v3-blender.txt
```

三张表贴回来（行很长，直接整段贴）。每行自带 `growth-capped`、limiter 计数、8/16 轮 probe/est、两次诊断 probe；我做全 19 场景的命中率表、等字节 ΔPSNR vs 固定 λ hull、growth-capped 清单，并给论文 Table 的 LaTeX。
