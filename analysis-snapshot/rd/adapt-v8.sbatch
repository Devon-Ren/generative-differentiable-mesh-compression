#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=adapt8
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zr2245/analysis/adapt8-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/adapt8-%x-%j.err
# adapt-v8.sbatch (2026-11-07, VERSION v8-r1): = adapt-v7b.sbatch (VERSION v7-r2) with train_rd_adapt_v8.py (phase A calibrated by byte probes at densify
# intervals after start_pruning, --probe_a_every $PAE (1000), 6-7 probes of ~20-45 s on these scenes; phase B unchanged), prune rounds every 500 from
# 12000 to 19500 (PRUNE default, 16 rounds as the Blender sbatch: the v7b sweep never opened the gate on these scenes, v8 aims higher and the
# collapse throughput per round is 4-12 % of the cap, so the trimming authority of 16 rounds is wanted; +8 probes of 30-60 s per run), the smoke
# extended to 1600 iterations so its first densify interval (1500) runs a phase-A probe (--probe_a_smoke) before the forced prune round at 1550,
# smoke assertions on the probe_a event / rd_summary fields, and the result row carrying every phase-A probe (bytes, kept share, b_v, overhead, V*).
# Output /scratch/zr2245/$SCENE-adapt8-t${T}mb/, table $AN/adapt-v8.txt.  Everything else verbatim v7-r2, whose text follows.
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
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"$(seq -s " " 12000 500 19500)"}
OVER=${OVER:-1.0}; EXTRA=${EXTRA:-"22000 26000"}; FPV=${FPV:-2.5}; PFLOOR=${PFLOOR:-0.9999}; PSCALE=${PSCALE:-4}; SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}; PAE=${PAE:-1000}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; STAGE2=/scratch/zr2245/shared-collapse-stage2-v2
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/adapt-v8.txt; TR=$AN/rdgrowth/train_rd_adapt_v8.py; PROBE_PY=$AN/rdgrowth/probe_bytes_v1.py
PROBE="bash $PKG/stage2_run.sh $PROBE_PY"
echo "adapt-v8 | scene $SCENE | IMAGES $IMAGES | targets $TARGETS MB | indoor '$FLAG' | initial B/v $BPV B/face $BPF faces/vertex $FPV | phase-A probes every $PAE it after start_pruning | overshoot $OVER probe floor $PFLOOR probe scaling $PSCALE sigma prior $SPRIOR | prune at $PRUNE mu $MU max_frac $MAXF | extra probes $EXTRA | probe $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $PROBE_PY $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $PKG/stage2_run.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py $STAGE2/STAGE2_VERSION $STAGE2/runtime.py $STAGE2/sharing_after.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python -m py_compile $TR || { echo "FAILED: $TR does not compile"; exit 1; }
python -m py_compile $PROBE_PY || { echo "FAILED: $PROBE_PY does not compile"; exit 1; }
grep -q "version='adapt-v8'" $TR || { echo "FAILED: $TR is not adapt-v8 (wrong file pasted?)"; exit 1; }
grep -q "def update_attributes" $TR && grep -q "probe_a_every" $TR && grep -q "event='probe_a'" $TR || { echo "FAILED: $TR lacks the v8 phase-A probe (update_attributes / probe_a_every / probe_a event)"; exit 1; }
[ "$PAE" -ge 0 ] 2>/dev/null || { echo "FAILED: PAE must be an integer >= 0 (0 = no phase-A probes = v7b), got $PAE"; exit 1; }
grep -q "PROBE_VERSION = 'probe-v1'" $PROBE_PY || { echo "FAILED: $PROBE_PY is not probe-v1"; exit 1; }
grep -q "def faces_removed_by" $TR || { echo "FAILED: $TR lacks faces_removed_by (v6 pasted instead of v7?)"; exit 1; }
grep -q "growth_capped" $TR && grep -q "probe_extra_iters" $TR || { echo "FAILED: $TR lacks growth_capped / probe_extra_iters"; exit 1; }
grep -q "default=1.0, help='phase A aims at target" $TR || { echo "FAILED: $TR phase_a_overshoot default is not 1.0"; exit 1; }
NPR=$(echo $PRUNE | wc -w); [ "$NPR" -ge 4 ] || { echo "FAILED: PRUNE has $NPR rounds (v8 default is 16, at least 4 expected: $PRUNE)"; exit 1; }
LASTP=$(echo $PRUNE | tr ' ' '\n' | sort -n | tail -1); for x in $EXTRA; do [ "$x" -gt "$LASTP" ] && [ "$x" -lt 30000 ] || { echo "FAILED: EXTRA probe $x must lie after the last prune round $LASTP and before 30000"; exit 1; }; done
grep -q "triangles.scaling" $LAB/upstream/train.py || { echo "FAILED: $LAB/upstream/train.py never sets triangles.scaling (the probe's 4x pass assumes upstream's supersampling attribute)"; exit 1; }
lam0_of () { python -c "import sys; t=float(sys.argv[1]); print('%.4e' % min(1e-1, max(1e-4, 6.2e-3 * (9.0 / t) ** 1.67)))" $1; }   # initial guess only; adapted from the first interval on
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
SM=$AN/adapt8-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
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
  echo "=== smoke 1600 it: phase-A probe at the first densify interval (1500), target 0.01 MB so the prune round at 1550 must collapse; probe at ${PSCALE}x AND 1x ($(date)) ==="
  python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1600 --test_iterations 1600 --lam $(lam0_of 5) --target_mb 0.01 \
    --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --probe_check_scaling --sigma_prior $SPRIOR \
    --prune_iters 1550 --allow_any_prune_iter --prune_mu $MU --prune_max_frac 0.1 --probe_a_every $PAE --probe_a_smoke \
    --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $SM/smoke.log 2>&1 || { echo "FAILED: SMOKE"; tail -40 $SM/smoke.log; for l in $SM/original/probes/*.log; do echo "--- $l"; tail -15 $l; done; exit 1; }
  python - <<PY || { echo "FAILED: SMOKE CHECK"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
if dn: assert all('lam_used' in e and 'desired' in e and 'sigma_plan' in e and 'front_loaded' in e and 'v_star' in e for e in dn), 'densify events lack the v4 controller fields'
else: print('note: no densify event within the 1600-it smoke (upstream densify_from >= 1600); phase-A controller is exercised in the real runs')
assert p and 'probe_mb' in p[0], 'no prune event with probe_mb (probe did not run)'
assert not p[0].get('skipped') and p[0]['n_pairs'] > 0, 'prune round at 1550 collapsed nothing: %r' % {k: p[0].get(k) for k in ('skipped','desired_pairs','n_pairs','n_cand','max_pairs')}
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
rs=json.load(open('$SM/original/rd_summary.json')); assert rs['version']=='adapt-v8' and len(rs['probes'])==2, 'rd_summary lacks the two phase-B/final probes'
pa=[e for e in ev if e['event']=='probe_a']
if dn:
    assert pa and pa[0]['iteration']==dn[0]['iteration'] and pa[0]['source'].startswith('probe_a@') and pa[0]['source_before']=='constants', 'v8 phase-A probe missing at the first densify interval: %r' % [(e['iteration'], e.get('source')) for e in pa]
    assert pa[0]['b_v']>0 and pa[0]['overhead_mb']>0 and pa[0]['b_f']==$BPF and abs(pa[0]['faces_per_vertex']-$FPV)<1e-9 and pa[0]['kept_share']>0 and pa[0]['v_star_after']>=1 and pa[0]['probe_vertices']<=pa[0]['vertices_model'], 'v8 phase-A probe fields (topology must keep the prior): %r' % {k: pa[0].get(k) for k in ('b_v','overhead_mb','b_f','faces_per_vertex','kept_share','v_star_before','v_star_after','probe_mb','probe_vertices','vertices_model')}
    assert str(dn[0].get('size_source','')).startswith('probe_a@'), 'densify event after the phase-A probe does not carry size_source probe_a@: %r' % dn[0].get('size_source')
    print('smoke probe-a:', {k: pa[0].get(k) for k in ('iteration','probe_mb','vertices_model','probe_vertices','kept_share','probe_kept_frac','b_v','overhead_mb','soup_b_f','soup_faces_per_vertex','v_star_before','v_star_after','source','probe_seconds','seconds_cleaning')})
else: print('note: no densify event within the 1600-it smoke (upstream densify_from >= 1600): the v8 phase-A probe path is NOT exercised by this smoke')
assert rs.get('probe_a_count')==len(pa) and len(rs.get('probes_a', []))==len(pa) and rs.get('probe_a_smoke') is True and rs.get('probe_a_every')==$PAE, 'rd_summary phase-A probe fields wrong: %r' % {k: rs.get(k) for k in ('probe_a_count','probe_a_smoke','probe_a_every')}
assert rs.get('growth_capped') is False and 'limiter_bound_intervals' in rs and rs.get('probe_extra_iters') == [], 'growth-capped / limiter fields wrong in the smoke: %r' % {k: rs.get(k) for k in ('growth_capped','limiter_bound_intervals','densify_intervals','probe_extra_iters')}
assert len(rs['probe_trend'])==1 and rs['extrapolate_to']>18000 and rs['probes'][1].get('faces_kept_1x') is None, 'trend / final-probe check fields wrong: %r' % {k: rs.get(k) for k in ('probe_trend','extrapolate_to')}
pr=rs['probes'][0]; assert pr.get('probe_scaling')==$PSCALE and pr.get('faces_kept_1x') is not None and pr['faces_kept']>pr['faces_kept_1x'], 'probe scaling check: %r' % {k: pr.get(k) for k in ('probe_scaling','faces_kept','faces_kept_1x','faces_model')}
print('smoke probe scaling: kept %d faces at %dx vs %d at 1x (of %d; %.2fx more)' % (pr['faces_kept'], pr['probe_scaling'], pr['faces_kept_1x'], pr['faces_model'], pr['faces_kept']/max(pr['faces_kept_1x'],1)))
print('smoke prune:', {k: p[0].get(k) for k in ('iteration','probe_mb','probe_est_mb','probe_kept_frac','probe_vertices','probe_faces','b_v','b_f','overhead_mb','faces_per_vertex','desired_pairs','excess_pairs','mode','n_lam','rank_k','n_cand','max_pairs','n_pairs','kept_removed','kept_faces_removed','removed_mb','bytes_per_pair_measured','lam_now','shadow_lam','vertices_before','vertices_after','probe_seconds')})
print('smoke final:', {k: fin[0][k] for k in ('vertices','triangles','lam_final','rhat_mb','probe_final_mb','growth_capped','limiter_bound_intervals','densify_intervals')})
PY
  grep -h "^\[RD probe-a\|^\[RD adapt\|^\[RD prune\|^Training is done" $SM/smoke.log | tail -5; for l in $SM/original/probes/*.log; do grep -h "^\[probe\]" $l; done
else echo "smoke skipped (SMOKE=0)"; fi
for T in $TARGETS; do
  G=/scratch/zr2245/$SCENE-adapt8-t${T}mb; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G; L0=$(lam0_of $T)
  echo "=== $SCENE target $T MB (lam0 $L0) $FLAG: train with closed-loop rate control ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L0 --target_mb $T \
      --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --probe_extra_iters $EXTRA --probe_a_every $PAE \
      --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $G/train.log 2>&1 || { echo "FAILED: TRAIN $SCENE target $T"; tail -12 $G/train.log; for l in $G/original/probes/*.log; do echo "--- $l"; tail -8 $l; done; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE target $T"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "target $T -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD probe-a\|^\[RD adapt\|^\[RD prune\|^Training is done" $G/train.log | tail -22
  echo "=== $SCENE target $T: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  python - "$SCENE" "$T" "$G" "$NA" "$L0" "$XYZ" "$DC" <<'PY' | tee -a $TAB
import json, sys
S, T, G, NA, L0, xyz, dc = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
rs = json.load(open(G + '/original/rd_summary.json')); b = json.load(open(NA + '/report.json')); f = json.load(open(NA + '/finetune/summary.json')); d = json.load(open(NA + '/deliver/summary.json')); t = d['delivery_test']
ev = [json.loads(l) for l in open(G + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']; pr = [e for e in ev if e['event'] == 'prune']; px = [e for e in ev if e['event'] == 'probe_extra']
pa = [e for e in ev if e['event'] == 'probe_a']; pas = [(e['iteration'], round(e['probe_mb'], 3), e['vertices_model'], e['probe_vertices'], e['kept_share'], round(e['b_v'], 2), round(e['overhead_mb'], 3), e['v_star_after'], round(e['probe_seconds'] + e['seconds_cleaning'], 1)) for e in pa]; bva = (round(pa[-1]['b_v'], 2) if pa else None); bv12 = (round(pr[0]['b_v'], 2) if pr else None); vs0 = (pa[0]['v_star_before'] if pa else None)
lams = [e['lam_used'] for e in dn]; q = lambda a, p: sorted(a)[int(p * (len(a) - 1))] if a else float('nan')
rounds = [(e['iteration'], round(e['probe_mb'], 3), round(e.get('probe_est_mb', e['probe_mb']), 3), e.get('probe_kept_frac'), e['vertices_before'], e['vertices_after'], e.get('mode'), ('%.2e' % e['lam_now']) if e.get('lam_now') is not None else None, round(e['b_v'], 2), round(e['b_f'], 2), e.get('n_pairs', 0), e.get('kept_removed'), e.get('kept_faces_removed'), round(e.get('removed_mb') or 0.0, 3)) for e in pr]
extra = [(e['iteration'], round(e['probe_mb'], 3), round(e['probe_est_mb'], 3), round(e['probe_slope_mb_per_1k'], 3)) for e in px]
ncoll = sum(e.get('n_pairs', 0) for e in pr); nkept = sum(e.get('kept_removed') or 0 for e in pr); nact = sum(1 for e in pr if e.get('n_pairs', 0) > 0)
sig = [e.get('sigma') for e in dn]; sq = lambda p: sorted(x for x in sig if x is not None)[int(p * (len([x for x in sig if x is not None]) - 1))] if any(x is not None for x in sig) else float('nan')
nb = sum(1 for e in dn if e.get('limiter_bound')); nf = sum(1 for e in dn if e.get('front_loaded')); vs = dn[-1]['v_star'] if dn else None
yl = [e.get('rank_yield') for e in dn if e.get('rank_yield') is not None]; yq = lambda p: sorted(yl)[int(p * (len(yl) - 1))] if yl else float('nan')
print(f"{S} | target {T:g} MB | lam0 {L0} -> lamA q10/q50/q90 {q(lams,.1):.2e}/{q(lams,.5):.2e}/{q(lams,.9):.2e} -> lam_final (shadow) {rs['lam_final']:.2e} | final probe {rs['probe_final_mb']:.3f} MB"
      f" | trained V {b['vertices']} F {b['faces']} | delivered {d['file_MB']:.3f} MB (hit {d['file_MB'] / T:.3f} x target, probe/delivered {rs['probe_final_mb'] / d['file_MB']:.3f}) | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f}"
      f" | val {f['validation_best']['psnr']:.3f} | K4096 R256 {xyz}/{dc} | growth-capped {rs.get('growth_capped')} (last landing est {rs.get('last_probe_est_mb')} MB) | densify events {len(dn)} (limiter-bound {nb}, front-loaded {nf}, V* {vs}), sigma q10/q50/q90 {sq(.1):.2f}/{sq(.5):.2f}/{sq(.9):.2f}, yield q10/q50/q90 {yq(.1):.2f}/{yq(.5):.2f}/{yq(.9):.2f}, prune rounds {len(pr)} active {nact} collapsed {ncoll} (both-kept share {nkept / max(ncoll, 1):.2f}; V {pr[0]['vertices_before'] if pr else None} -> {pr[-1]['vertices_after'] if pr else None}), extra probes (it, MB, est MB, slope) {extra} | phase-A probes {len(pa)} (last b_v {bva} vs 12k b_v {bv12}, V* {vs0} -> {vs}), probes-a (it, MB, V model, V kept, kept share, b_v, overhead MB, V* after, s) {pas}, rounds (it, probe MB, est MB, kept, V before, V after, mode, lam, b_v, b_f, pairs, kept-removed, kept-faces-removed, removed MB) {rounds}")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
echo "adapt-v8 $SCENE exit 0  $(date)"
