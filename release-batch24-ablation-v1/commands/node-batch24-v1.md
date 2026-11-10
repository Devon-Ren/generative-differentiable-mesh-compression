# node-batch24-v1 (2026-11-10): training-rule ablation on all 11 Mip-360 / T&T scenes AND all 8 NeRF-Synthetic objects, + vertex-cap rerun at the controller's vertex counts

Three pastes on a login node (greene). §1 writes the generator and derives the two run files from the batch14 originals (md5-checked in and out).
§2 writes and runs the submit script (95 jobs: 11 Mip-360/T&T scenes and all 8 NeRF-Synthetic objects, five jobs each; 57 on the 4-GPU reservation, 38 on the public partition, see the GPU budget note under §2). §3 is the collect for me when the jobs are done.

What is run (one variant per job, everything else identical to the paper's runs: adapt-v9b code path, aim 1.25, codec, level-1.0 protocol):

| job | variant | what changes | rows |
|---|---|---|---|
| `<scene>-nocoll-abl24` | no in-training collapse | `--prune_iters` empty, no extra probes | 10 MB |
| `<scene>-const-abl24` | phase-A plan priced from fixed constants, no byte probe during growth | `--probe_a_every 0` (b_v 10, b_f 2, fpv 2.5, overhead 0.6 MB stay) | 10, 20 MB |
| `<scene>-clam-abl24` | constant price, controller off | `--hold_lambda` (patch: lambda never changes, no plan cap, phase B = fixed-lambda prune only) | lam 0.00185, 0.0062 |
| `<scene>-match-abl24` | MeshSplatting's importance-sampled split selection on the nocollapse run's split counts | `--dd_mode importance --split_schedule <nocollapse rd_log> --hold_lambda --lam 0`, no collapse (depends on the nocollapse job) | 10 MB |
| `<scene>-g0r24` | MeshSplatting with a vertex cap at the controller meshes' vertex counts | budget-sweep-v3, `--max_points` = V(controller 20/10/5 MB mesh) / 0.9 | 3 budgets |
| `<object>-*-abl24b`, `<object>-g0rb24` | the same five jobs on NeRF-Synthetic (Blender protocol: aim 1.1, val-selected checkpoint) | nocollapse / matched at 2.5 MB, constants at 2.5 + 1.5 MB, constlam at lam 0.0062 + 0.0185, cap at V/0.9 of the 4 / 2.5 / 1.5 MB meshes | |

The constlam job first runs a 2600-iteration smoke with `--hold_lambda` and asserts from `rd_log.jsonl` that lambda never moved and that the forced prune round was lambda-driven.

## §1 generate the run files (login node, any dir)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/make_batch24_files_v1.py <<'PYGEN24'
#!/usr/bin/env python3
"""make_batch24_files_v1.py (2026-11-10): generate the batch24 ablation files from the batch14/21 originals.

  python make_batch24_files_v1.py --train <train_rd_adapt_v9b.py> --sbatch <adapt-v9b.sbatch> --sbatch_blender <adapt-v9b-blender.sbatch> --out <dir>

Writes <out>/train_rd_adapt_v9b_abl.py (v9b + the --hold_lambda ablation switch, nothing else), <out>/ablation-b24.sbatch
(adapt-v9b.sbatch + the ABL variant switch) and <out>/ablation-b24-blender.sbatch (adapt-v9b-blender.sbatch + the same switch).  Every edit is an exact-string replacement that must match exactly once, and the
inputs must be the files the paper runs used (md5 asserted), so the output is deterministic and its md5 can be checked on the node.
"""
import argparse, hashlib, pathlib, sys

V9B_MD5 = 'e379f9906f639d5d69acf127543ce690'; SBATCH_MD5 = 'c0f90652afe97b76d0d3148fe1243f15'

def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)

def md5(s): return hashlib.md5(s.encode()).hexdigest()

ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--out', required=True)
BLENDER_MD5 = 'c6d80dcb8d3af50754079363ffb96dfa'
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

# ---------------- training script: + --hold_lambda ----------------
t = pathlib.Path(a.train).read_text()
if md5(t) != V9B_MD5: sys.exit(f'FAILED: {a.train} md5 {md5(t)} is not the batch14 train_rd_adapt_v9b.py ({V9B_MD5})')
t = rep(t, "    parser.add_argument('--target_mb', type=float, required=True, help='target delivered size in MB; lambda is adapted so the projected size V*B_v + F*B_f + overhead hits it')",
        "    parser.add_argument('--target_mb', type=float, required=True, help='target delivered size in MB; lambda is adapted so the projected size V*B_v + F*B_f + overhead hits it')\n"
        "    parser.add_argument('--hold_lambda', action='store_true', help='ablation (batch24): constant price. The controller never changes lambda: phase A splits every triangle above --lam (upstream limiter only, no plan cap); phase B keeps only the fixed-lambda prune (edges below lam x mu), no target gate; --target_mb is then a label only')",
        'argparse')
t = rep(t, "rd=dict(lam=args.lam,", "rd=dict(hold_lambda=args.hold_lambda, lam=args.lam,", 'rd dict')
t = rep(t, "max_split_up = max_split; limiter_bound = bool(front or max_split_up < cap); max_split = min(max_split_up, cap)",
        "max_split_up = max_split; limiter_bound = (False if rd.get('hold_lambda') else bool(front or max_split_up < cap)); max_split = (max_split_up if rd.get('hold_lambda') else min(max_split_up, cap))",
        'phase-A cap')
t = rep(t, "lam_now = rank_lambda(ratio_pre, n_allowed, k_asked, rd['lam_min'], rd['lam_max']) if k_asked > 0 else lam_used",
        "lam_now = lam_used if rd.get('hold_lambda') else (rank_lambda(ratio_pre, n_allowed, k_asked, rd['lam_min'], rd['lam_max']) if k_asked > 0 else lam_used)",
        'phase-A price')
t = rep(t, "st.update(event='densify', iteration=iteration,", "st.update(event='densify', hold_lambda=bool(rd.get('hold_lambda')), iteration=iteration,", 'densify event flag')
t = rep(t, "                cap_bound = bool(excess > 0 and excess_pairs < max(desired, float(n_lam)))\n",
        "                cap_bound = bool(excess > 0 and excess_pairs < max(desired, float(n_lam)))\n"
        "                if rd.get('hold_lambda'): wanted = float(n_lam); count_driven = False; cap_bound = False; thr = thr_lam  # ablation (batch24): constant price -> only the fixed-lambda prune of train_rd_v3, no target gate\n",
        'phase-B gate')
t = rep(t, "print('Optimizing ' + args.model_path, '| RD growth+prune adapt-v9b target_mb', args.target_mb,",
        "print('Optimizing ' + args.model_path, '| RD growth+prune adapt-v9b target_mb', args.target_mb, '| hold_lambda (ablation)', args.hold_lambda,", 'banner')
(out / 'train_rd_adapt_v9b_abl.py').write_text(t)

# ---------------- sbatch: + ABL switch ----------------
s = pathlib.Path(a.sbatch).read_text()
if md5(s) != SBATCH_MD5: sys.exit(f'FAILED: {a.sbatch} md5 {md5(s)} is not the batch14 adapt-v9b.sbatch ({SBATCH_MD5})')
s = rep(s, "#SBATCH --job-name=adapt9x", "#SBATCH --job-name=abl24", 'job name')
s = rep(s, "#SBATCH --output=/scratch/zr2245/analysis/adapt9x-%x-%j.out\n#SBATCH --error=/scratch/zr2245/analysis/adapt9x-%x-%j.err\n",
        "#SBATCH --output=/scratch/zr2245/analysis/abl24-%x-%j.out\n#SBATCH --error=/scratch/zr2245/analysis/abl24-%x-%j.err\n"
        "# ablation-b24.sbatch (2026-11-10, VERSION abl24-r1): generated from adapt-v9b.sbatch (VERSION v9x-r1, md5 " + SBATCH_MD5 + ") by make_batch24_files_v1.py.\n"
        "# ABL selects ONE variant of the training rule, everything else (scene settings, codec, level-1.0 protocol, aim 1.25, prune rounds) is the paper's:\n"
        "#   nocollapse : rate-aware growth + controller, NO in-training collapse (--prune_iters empty, no extra probes)      -> $SCENE-abl24-nocollapse-t${T}mb\n"
        "#   matched    : MeshSplatting's importance-sampled split selection on the nocollapse run's per-interval split counts  -> $SCENE-abl24-matched-t${T}mb\n"
        "#                (--dd_mode importance --split_schedule <nocollapse rd_log> --hold_lambda --lam 0; no collapse; no phase-A probes)\n"
        "#   constants  : the phase-A plan priced from the fixed constants (b_v 10, b_f 2, fpv 2.5, overhead 0.6 MB), no byte probe during growth (PAE=0)\n"
        "#                phase B unchanged                                                                                    -> $SCENE-abl24-constants-t${T}mb\n"
        "#   constlam   : constant price: --hold_lambda at each lambda in LAMS, no controller, no plan cap, phase B = fixed-lambda prune -> $SCENE-abl24-constlam-lam$L\n"
        "# Rows go to $AN/ablation-b24.txt with the tag 'abl24-<ABL>'.  The v9x text follows unchanged except where marked abl24.\n", 'header')
s = rep(s, "OVER=${OVER:-1.25}; EXTRA=${EXTRA:-\"22000 26000\"}; FPV=${FPV:-2.5}; PFLOOR=${PFLOOR:-0.9999}; PSCALE=${PSCALE:-4}; SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}; PAE=${PAE:-1000}\n",
        "OVER=${OVER:-1.25}; EXTRA=${EXTRA:-\"22000 26000\"}; FPV=${FPV:-2.5}; PFLOOR=${PFLOOR:-0.9999}; PSCALE=${PSCALE:-4}; SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}; PAE=${PAE:-1000}\n"
        "ABL=${ABL:?set ABL=nocollapse|matched|constants|constlam via --export}; export ABL; DDMODE=error; XARGS=\"\"; RUNS=\"$TARGETS\"   # abl24\n"
        "case $ABL in\n"
        "  nocollapse) PRUNE=\"\"; EXTRA=\"\" ;;\n"
        "  matched) PRUNE=\"\"; EXTRA=\"\"; PAE=0; DDMODE=importance; SCHED=${SCHED:?set SCHED=<rd_log.jsonl of the nocollapse run> via --export}; [ -f $SCHED ] || { echo \"FAILED: MISSING split schedule $SCHED (nocollapse run not finished?)\"; exit 1; }; XARGS=\"--hold_lambda --split_schedule $SCHED\" ;;\n"
        "  constants) PAE=0 ;;\n"
        "  constlam) PAE=0; XARGS=\"--hold_lambda\"; LAMS=${LAMS:?set LAMS=\"0.00185 0.0062\" via --export}; RUNS=\"$LAMS\" ;;\n"
        "  *) echo \"FAILED: unknown ABL $ABL\"; exit 1 ;;\n"
        "esac\n", 'ABL switch')
s = rep(s, "TAB=$AN/adapt-v9b.txt; TR=$AN/rdgrowth/train_rd_adapt_v9b.py;", "TAB=$AN/ablation-b24.txt; TR=$AN/rdgrowth/train_rd_adapt_v9b_abl.py;", 'table / script')
s = rep(s, 'echo "adapt-v9b | scene $SCENE |', 'echo "ablation-b24 | ABL $ABL | runs $RUNS | dd_mode $DDMODE | extra args \'$XARGS\' | scene $SCENE |', 'banner')
s = rep(s, 'grep -q "if k_asked > 0 else lam_used" $TR || { echo "FAILED: $TR lacks the v9b continuous-price fix"; exit 1; }\n',
        'grep -q "if k_asked > 0 else lam_used" $TR || { echo "FAILED: $TR lacks the v9b continuous-price fix"; exit 1; }\n'
        'grep -q "hold_lambda=args.hold_lambda" $TR && grep -q "wanted = float(n_lam); count_driven = False" $TR || { echo "FAILED: $TR lacks the abl24 hold_lambda patch"; exit 1; }\n', 'patch assert')
s = rep(s, 'FIRSTP=$(echo $PRUNE | tr \' \' \'\\n\' | sort -n | head -1); [ "$FIRSTP" -ge 12500 ] || echo "note: first prune round $FIRSTP < 12500 (the 12000 probe right after the Delaunay reads high)"\n',
        'if [ -n "$PRUNE" ]; then FIRSTP=$(echo $PRUNE | tr \' \' \'\\n\' | sort -n | head -1); [ "$FIRSTP" -ge 12500 ] || echo "note: first prune round $FIRSTP < 12500 (the 12000 probe right after the Delaunay reads high)"; else echo "abl24: no collapse rounds ($ABL)"; fi\n', 'first prune note')
s = rep(s, 'NPR=$(echo $PRUNE | wc -w); [ "$NPR" -ge 4 ] || { echo "FAILED: PRUNE has $NPR rounds (v9b default is 15, at least 4 expected: $PRUNE)"; exit 1; }\n',
        'if [ -n "$PRUNE" ]; then NPR=$(echo $PRUNE | wc -w); [ "$NPR" -ge 4 ] || { echo "FAILED: PRUNE has $NPR rounds (v9b default is 15, at least 4 expected: $PRUNE)"; exit 1; }; fi\n', 'NPR assert')
s = rep(s, 'LASTP=$(echo $PRUNE | tr \' \' \'\\n\' | sort -n | tail -1); for x in $EXTRA; do',
        'LASTP=$(echo "${PRUNE:-0}" | tr \' \' \'\\n\' | sort -n | tail -1); for x in $EXTRA; do', 'LASTP')
# constlam smoke + the per-run loop
s = rep(s, 'for T in $TARGETS; do\n  G=/scratch/zr2245/$SCENE-adapt9x-t${T}mb; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G; L0=$(lam0_of $T)\n  echo "=== $SCENE target $T MB (lam0 $L0) $FLAG: train with closed-loop rate control ($(date)) ==="\n',
        'if [ "$ABL" = constlam ] && [ "${SMOKE2:-1}" = 1 ]; then\n'
        '  L1=$(echo $RUNS | awk \'{print $1}\'); echo "=== abl24 constlam smoke 2600 it at lam $L1: lambda must never move, the forced prune round at 2550 must be lambda-driven ($(date)) ==="\n'
        '  python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 2600 --test_iterations 2600 --lam $L1 --target_mb 10 \\\n'
        '    --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \\\n'
        '    --prune_iters 2550 --allow_any_prune_iter --prune_mu $MU --prune_max_frac $MAXF --probe_a_every 0 --hold_lambda \\\n'
        '    --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $SM/smoke.log 2>&1 || { echo "FAILED: CONSTLAM SMOKE"; tail -30 $SM/smoke.log; exit 1; }\n'
        '  python - "$SM/original/rd_log.jsonl" "$L1" <<\'PY\' || { echo "FAILED: CONSTLAM SMOKE CHECK"; grep -h "RD \\|Traceback\\|Error" $SM/smoke.log | tail -12; exit 1; }\n'
        'import json, sys\n'
        'ev = [json.loads(l) for l in open(sys.argv[1])]; L = float(sys.argv[2]); dn = [e for e in ev if e[\'event\'] == \'densify\']; pr = [e for e in ev if e[\'event\'] == \'prune\']\n'
        'assert dn, \'no densify event within 2600 iterations\'\n'
        'assert all(e.get(\'hold_lambda\') is True for e in dn), \'densify events do not carry hold_lambda=True (patch not active)\'\n'
        'bad = [(e[\'iteration\'], e[\'lam_used\'], e[\'lam_next\']) for e in dn if abs(e[\'lam_used\'] - L) > 1e-12 or abs(e[\'lam_next\'] - L) > 1e-12]; assert not bad, \'lambda moved under --hold_lambda: %r\' % bad[:3]\n'
        'assert all(e[\'n_split\'] <= e[\'limiter_max\'] for e in dn), \'split count above the upstream limiter\'\n'
        'assert pr, \'no prune event at 2550\'; p = pr[0]\n'
        'assert p.get(\'skipped\') or (p.get(\'mode\') == \'lambda\' and abs(p[\'lam_now\'] - L) < 1e-12), \'prune round not lambda-driven under --hold_lambda: %r\' % {k: p.get(k) for k in (\'mode\', \'skipped\', \'lam_now\', \'n_lam\', \'n_pairs\')}\n'
        'print(\'constlam smoke ok:\', len(dn), \'densify events, lambda constant at\', L, \'| prune round:\', {k: p.get(k) for k in (\'mode\', \'skipped\', \'n_lam\', \'n_pairs\', \'excess_pairs\')})\n'
        'PY\n'
        'fi\n'
        'for RUN in $RUNS; do\n'
        '  if [ "$ABL" = constlam ]; then T=10; L0=$RUN; G=/scratch/zr2245/$SCENE-abl24-constlam-lam$RUN; elif [ "$ABL" = matched ]; then T=$RUN; L0=0; G=/scratch/zr2245/$SCENE-abl24-matched-t${T}mb; else T=$RUN; L0=$(lam0_of $T); G=/scratch/zr2245/$SCENE-abl24-$ABL-t${T}mb; fi\n'
        '  CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G\n'
        '  echo "=== $SCENE ABL $ABL run $RUN (target label $T MB, lam0 $L0, dd_mode $DDMODE, extra args \'$XARGS\') $FLAG: train ($(date)) ==="\n', 'run loop')
s = rep(s, "--bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \\\n      --prune_iters $PRUNE",
        "--bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode $DDMODE --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \\\n      --prune_iters $PRUNE", 'dd_mode')
s = rep(s, '--probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $G/train.log',
        '--probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $XARGS $FLAG > $G/train.log', 'extra args')
s = rep(s, 'import json, sys\nS, T, G, NA, L0, xyz, dc = sys.argv[1]', 'import json, sys, os\nS, T, G, NA, L0, xyz, dc = sys.argv[1]', 'row imports')
s = rep(s, 'print(f"{S} | target {T:g} MB | lam0 {L0}', 'print(f"{S} abl24-{os.environ[\'ABL\']}{(\'-lam\' + L0) if os.environ[\'ABL\'] == \'constlam\' else \'\'} | target {T:g} MB | lam0 {L0}', 'row tag')
s = rep(s, 'echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB\necho "adapt-v9b $SCENE exit 0  $(date)"',
        'echo "===== $TAB ($SCENE $ABL rows) ====="; grep "^$SCENE abl24-$ABL" $TAB\necho "ablation-b24 $SCENE $ABL exit 0  $(date)"', 'tail')
s = s.replace("echo \"=== $SCENE target $T: level-1.0 protocol ===\"", "echo \"=== $SCENE $ABL run $RUN: level-1.0 protocol ===\"")
(out / 'ablation-b24.sbatch').write_text(s)

# ---------------- Blender sbatch: + ABL switch (same edits, Blender anchors) ----------------
b = pathlib.Path(a.sbatch_blender).read_text()
if md5(b) != BLENDER_MD5: sys.exit(f'FAILED: {a.sbatch_blender} md5 {md5(b)} is not the batch14 adapt-v9b-blender.sbatch ({BLENDER_MD5})')
b = rep(b, "#SBATCH --job-name=adapt9xb", "#SBATCH --job-name=abl24b", 'job name')
b = rep(b, "#SBATCH --output=/scratch/zr2245/analysis/adapt9xb-%x-%j.out\n#SBATCH --error=/scratch/zr2245/analysis/adapt9xb-%x-%j.err\n",
        "#SBATCH --output=/scratch/zr2245/analysis/abl24b-%x-%j.out\n#SBATCH --error=/scratch/zr2245/analysis/abl24b-%x-%j.err\n"
        "# ablation-b24-blender.sbatch (2026-11-10, VERSION abl24b-r1): generated from adapt-v9b-blender.sbatch (VERSION v9xb-r1, md5 " + BLENDER_MD5 + ") by make_batch24_files_v1.py.\n"
        "# Same ABL switch as ablation-b24.sbatch (nocollapse | matched | constants | constlam), Blender protocol unchanged (--indoor, black background, aim 1.1,\n"
        "# checkpoints every 1k from 12k, val_select_v3, level-1.0 on $G-best).  Run dirs $SCENE-abl24b-<ABL>-t${T}mb (constlam: -lam$L), rows -> $AN/ablation-b24-blender.txt tagged 'abl24-<ABL>'.\n", 'header')
b = rep(b, "OVER=${OVER:-1.1}; EXTRA=${EXTRA:-\"22000 26000\"}; FPV=${FPV:-2.5}; PFLOOR=${PFLOOR:-0.9999}; PSCALE=${PSCALE:-4}; SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}; PAE=${PAE:-1000}\n",
        "OVER=${OVER:-1.1}; EXTRA=${EXTRA:-\"22000 26000\"}; FPV=${FPV:-2.5}; PFLOOR=${PFLOOR:-0.9999}; PSCALE=${PSCALE:-4}; SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}; PAE=${PAE:-1000}\n"
        "ABL=${ABL:?set ABL=nocollapse|matched|constants|constlam via --export}; export ABL; DDMODE=error; XARGS=\"\"; RUNS=\"$TARGETS\"   # abl24\n"
        "case $ABL in\n"
        "  nocollapse) PRUNE=\"\"; EXTRA=\"\" ;;\n"
        "  matched) PRUNE=\"\"; EXTRA=\"\"; PAE=0; DDMODE=importance; SCHED=${SCHED:?set SCHED=<rd_log.jsonl of the nocollapse run> via --export}; [ -f $SCHED ] || { echo \"FAILED: MISSING split schedule $SCHED (nocollapse run not finished?)\"; exit 1; }; XARGS=\"--hold_lambda --split_schedule $SCHED\" ;;\n"
        "  constants) PAE=0 ;;\n"
        "  constlam) PAE=0; XARGS=\"--hold_lambda\"; LAMS=${LAMS:?set LAMS=\"0.0062 0.0185\" via --export}; RUNS=\"$LAMS\" ;;\n"
        "  *) echo \"FAILED: unknown ABL $ABL\"; exit 1 ;;\n"
        "esac\n", 'ABL switch')
b = rep(b, "TAB=$AN/adapt-v9b-blender.txt; TR=$AN/rdgrowth/train_rd_adapt_v9b.py;", "TAB=$AN/ablation-b24-blender.txt; TR=$AN/rdgrowth/train_rd_adapt_v9b_abl.py;", 'table / script')
b = rep(b, 'echo "adapt-v9b-blender | scene $SCENE', 'echo "ablation-b24-blender | ABL $ABL | runs $RUNS | dd_mode $DDMODE | extra args \'$XARGS\' | scene $SCENE', 'banner')
b = rep(b, 'grep -q "if k_asked > 0 else lam_used" $TR || { echo "FAILED: $TR lacks the v9b continuous-price fix"; exit 1; }\n',
        'grep -q "if k_asked > 0 else lam_used" $TR || { echo "FAILED: $TR lacks the v9b continuous-price fix"; exit 1; }\n'
        'grep -q "hold_lambda=args.hold_lambda" $TR && grep -q "wanted = float(n_lam); count_driven = False" $TR || { echo "FAILED: $TR lacks the abl24 hold_lambda patch"; exit 1; }\n', 'patch assert')
b = rep(b, 'FIRSTP=$(echo $PRUNE | tr \' \' \'\\n\' | sort -n | head -1); [ "$FIRSTP" -ge 12500 ] || echo "note: first prune round $FIRSTP < 12500 (the 12000 probe right after the Delaunay reads high)"\n',
        'if [ -n "$PRUNE" ]; then FIRSTP=$(echo $PRUNE | tr \' \' \'\\n\' | sort -n | head -1); [ "$FIRSTP" -ge 12500 ] || echo "note: first prune round $FIRSTP < 12500 (the 12000 probe right after the Delaunay reads high)"; else echo "abl24: no collapse rounds ($ABL)"; fi\n', 'first prune note')
b = rep(b, 'NPR=$(echo $PRUNE | wc -w); [ "$NPR" -ge 4 ] || { echo "FAILED: PRUNE has $NPR rounds (v9b default is 15, at least 4 expected: $PRUNE)"; exit 1; }\n',
        'if [ -n "$PRUNE" ]; then NPR=$(echo $PRUNE | wc -w); [ "$NPR" -ge 4 ] || { echo "FAILED: PRUNE has $NPR rounds (v9b default is 15, at least 4 expected: $PRUNE)"; exit 1; }; fi\n', 'NPR assert')
b = rep(b, 'LASTP=$(echo $PRUNE | tr \' \' \'\\n\' | sort -n | tail -1); for x in $EXTRA; do', 'LASTP=$(echo "${PRUNE:-0}" | tr \' \' \'\\n\' | sort -n | tail -1); for x in $EXTRA; do', 'LASTP')
b = rep(b, 'for T in $TARGETS; do\n  G=/scratch/zr2245/$SCENE-adapt9xb-t${T}mb; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G; L0=$(lam0_of $T)\n  echo "=== $SCENE target $T MB (lam0 $L0) $FLAG: train with closed-loop rate control, checkpoints every 1k from 12k ($(date)) ==="\n',
        'if [ "$ABL" = constlam ] && [ "${SMOKE2:-1}" = 1 ]; then\n'
        '  L1=$(echo $RUNS | awk \'{print $1}\'); echo "=== abl24 constlam smoke 2600 it at lam $L1: lambda must never move, the forced prune round at 2550 must be lambda-driven ($(date)) ==="\n'
        '  python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 2600 --test_iterations 2600 --lam $L1 --target_mb 2.5 \\\n'
        '    --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \\\n'
        '    --prune_iters 2550 --allow_any_prune_iter --prune_mu $MU --prune_max_frac $MAXF --probe_a_every 0 --hold_lambda \\\n'
        '    --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $SM/smoke.log 2>&1 || { echo "FAILED: CONSTLAM SMOKE"; tail -30 $SM/smoke.log; exit 1; }\n'
        '  python - "$SM/original/rd_log.jsonl" "$L1" <<\'PY\' || { echo "FAILED: CONSTLAM SMOKE CHECK"; grep -h "RD \\|Traceback\\|Error" $SM/smoke.log | tail -12; exit 1; }\n'
        'import json, sys\n'
        'ev = [json.loads(l) for l in open(sys.argv[1])]; L = float(sys.argv[2]); dn = [e for e in ev if e[\'event\'] == \'densify\']; pr = [e for e in ev if e[\'event\'] == \'prune\']\n'
        'assert dn, \'no densify event within 2600 iterations\'\n'
        'assert all(e.get(\'hold_lambda\') is True for e in dn), \'densify events do not carry hold_lambda=True (patch not active)\'\n'
        'bad = [(e[\'iteration\'], e[\'lam_used\'], e[\'lam_next\']) for e in dn if abs(e[\'lam_used\'] - L) > 1e-12 or abs(e[\'lam_next\'] - L) > 1e-12]; assert not bad, \'lambda moved under --hold_lambda: %r\' % bad[:3]\n'
        'assert all(e[\'n_split\'] <= e[\'limiter_max\'] for e in dn), \'split count above the upstream limiter\'\n'
        'assert pr, \'no prune event at 2550\'; p = pr[0]\n'
        'assert p.get(\'skipped\') or (p.get(\'mode\') == \'lambda\' and abs(p[\'lam_now\'] - L) < 1e-12), \'prune round not lambda-driven under --hold_lambda: %r\' % {k: p.get(k) for k in (\'mode\', \'skipped\', \'lam_now\', \'n_lam\', \'n_pairs\')}\n'
        'print(\'constlam smoke ok:\', len(dn), \'densify events, lambda constant at\', L, \'| prune round:\', {k: p.get(k) for k in (\'mode\', \'skipped\', \'n_lam\', \'n_pairs\', \'excess_pairs\')})\n'
        'PY\n'
        'fi\n'
        'for RUN in $RUNS; do\n'
        '  if [ "$ABL" = constlam ]; then T=2.5; L0=$RUN; G=/scratch/zr2245/$SCENE-abl24b-constlam-lam$RUN; elif [ "$ABL" = matched ]; then T=$RUN; L0=0; G=/scratch/zr2245/$SCENE-abl24b-matched-t${T}mb; else T=$RUN; L0=$(lam0_of $T); G=/scratch/zr2245/$SCENE-abl24b-$ABL-t${T}mb; fi\n'
        '  GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G\n'
        '  echo "=== $SCENE ABL $ABL run $RUN (target label $T MB, lam0 $L0, dd_mode $DDMODE, extra args \'$XARGS\') $FLAG: train, checkpoints every 1k from 12k ($(date)) ==="\n', 'run loop')
b = rep(b, "--bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \\\n      --prune_iters $PRUNE",
        "--bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode $DDMODE --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \\\n      --prune_iters $PRUNE", 'dd_mode')
b = rep(b, '--probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $G/train.log', '--probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $XARGS $FLAG > $G/train.log', 'extra args')
b = rep(b, 'import json, sys\nS, T, G, NA, L0, xyz, dc = sys.argv[1]', 'import json, sys, os\nS, T, G, NA, L0, xyz, dc = sys.argv[1]', 'row imports')
b = rep(b, 'print(f"{S} | target {T:g} MB | best iter', 'print(f"{S} abl24-{os.environ[\'ABL\']}{(\'-lam\' + L0) if os.environ[\'ABL\'] == \'constlam\' else \'\'} | target {T:g} MB | best iter', 'row tag')
b = rep(b, 'echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB\necho "adapt-v9b-blender $SCENE exit 0  $(date)"',
        'echo "===== $TAB ($SCENE $ABL rows) ====="; grep "^$SCENE abl24-$ABL" $TAB\necho "ablation-b24-blender $SCENE $ABL exit 0  $(date)"', 'tail')
(out / 'ablation-b24-blender.sbatch').write_text(b)

for f in ('train_rd_adapt_v9b_abl.py', 'ablation-b24.sbatch', 'ablation-b24-blender.sbatch'):
    p = out / f; txt = p.read_text(); print(f'{f}: {len(txt.splitlines())} lines md5 {md5(txt)}')
PYGEN24
python3 $AN/make_batch24_files_v1.py --train $AN/rdgrowth/train_rd_adapt_v9b.py --sbatch $AN/adapt-v9b.sbatch --sbatch_blender $AN/adapt-v9b-blender.sbatch --out $AN/batch24-build \
  && cp $AN/batch24-build/train_rd_adapt_v9b_abl.py $AN/rdgrowth/train_rd_adapt_v9b_abl.py && cp $AN/batch24-build/ablation-b24.sbatch $AN/ablation-b24.sbatch && cp $AN/batch24-build/ablation-b24-blender.sbatch $AN/ablation-b24-blender.sbatch \
  && md5sum $AN/rdgrowth/train_rd_adapt_v9b_abl.py $AN/ablation-b24.sbatch $AN/ablation-b24-blender.sbatch
```

Expected: `train_rd_adapt_v9b_abl.py: 934 lines md5 2ae0a46b27aa135a02c1b56194e52e6d`, `ablation-b24.sbatch: 202 lines md5 a823daddfa8374bffd8537e0ba896d75`, `ablation-b24-blender.sbatch: 204 lines md5 741bd58b89a9e4b1e410b70a440e8896`. Anything else: stop and send me the output.

## §2 submit (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-batch24-v1.sh <<'SUB24'
#!/usr/bin/env bash
# submit-batch24-v1.sh (2026-11-10): the training-rule ablation on all 11 Mip-360 / T&T scenes + the vertex-cap rerun at the controller
# meshes' vertex counts.  Run on a login node AFTER make_batch24_files_v1.py has written $AN/rdgrowth/train_rd_adapt_v9b_abl.py and
# $AN/ablation-b24.sbatch (md5s checked below).  Everything goes to the yw523-l40s reservation (no fairshare cost).
# Per scene: nocollapse (10 MB), constants (10 + 20 MB), constlam (lam 0.00185 + 0.0062), matched (10 MB, after nocollapse: it replays
# that run's split counts), cap (budget-sweep-v3: --max_points at V_controller/0.9 for the 20/10/5 MB meshes).  55 jobs, ~165 GPU-h.
# NeRF-Synthetic, all 8 objects: the same five jobs per object with the Blender protocol (ablation-b24-blender.sbatch; nocollapse / matched at 2.5 MB,
# constants at 2.5 + 1.5 MB, constlam at lam 0.0062 + 0.0185, cap via budget-sweep-blender-v1 at V_controller/0.9 of the 4 / 2.5 / 1.5 MB meshes).  40 jobs, ~80 GPU-h.
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
# GPU budget: the reservation has 4 L40S and ends 11/12 09:35 (about 130 GPU-h left tonight); the whole batch is about 280 GPU-h.  Job groups listed
# in PUBLIC go to the public L40S partition (same account, fairshare cost, 16+ run concurrently); the rest go to the reservation.  Default split:
# reservation = the four-variant ablation runs that use the new code on Mip/T&T + the Blender nocollapse/matched/constants (~135 GPU-h);
# public = constlam (fixed price, no controller) and the vertex-cap reruns (standard MeshSplatting) on both datasets (~145 GPU-h).  Override: PUBLIC="..." bash submit-batch24-v1.sh
PUBLIC=${PUBLIC:-"mip-constlam mip-cap bl-constlam bl-cap db-cap"}
ro () { case " $PUBLIC " in *" $1 "*) RESOPT="";; *) RESOPT="--reservation=$RES";; esac; }
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl.py 934 2ae0a46b27aa135a02c1b56194e52e6d
chk $AN/ablation-b24.sbatch 202 a823daddfa8374bffd8537e0ba896d75
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
chk $AN/ablation-b24-blender.sbatch 204 741bd58b89a9e4b1e410b70a440e8896
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
bash -n $AN/ablation-b24.sbatch && bash -n $AN/budget-sweep-v3.sbatch && bash -n $AN/ablation-b24-blender.sbatch && bash -n $AN/budget-sweep-blender-v1.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
grep -q "VERSION abl24-r1" $AN/ablation-b24.sbatch && grep -q "VERSION g0r-v3" $AN/budget-sweep-v3.sbatch && grep -q "VERSION abl24b-r1" $AN/ablation-b24-blender.sbatch && grep -q "VERSION g0rb-v1" $AN/budget-sweep-blender-v1.sbatch || { echo "CHECK FAILED: sbatch versions"; exit 1; }
for f in $AN/rdgrowth/train_rd_adapt_v9b.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/probe_bytes_v1.py $AN/vis_collapse-v12/scripts/run_garden.sh $AN/indoor_budget_check_v1.py; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
[ "$(md5sum < $AN/rdgrowth/train_rd_adapt_v9b.py | cut -c1-32)" = e379f9906f639d5d69acf127543ce690 ] || { echo "CHECK FAILED: train_rd_adapt_v9b.py is not the batch14 file"; exit 1; }
grep -q "if acc.mode == 'importance'" $AN/rdgrowth/rd_growth_v2.py || { echo "CHECK FAILED: rd_growth_v2.py has no importance mode (matched variant needs it)"; exit 1; }
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in drjohnson playroom; do [ -f $D/db/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in lego chair drums ficus hotdog materials mic ship; do [ -f $D/nerf_synthetic/$S/transforms_train.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; [ -f /scratch/zr2245/$S-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no val-selected FP32 original for $S (Blender cap job needs it)"; exit 1; }; done
for f in $AN/blender/val_select_v3.py $AN/blender/train_ckpt_v2.py; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
for S in room kitchen counter bonsai drjohnson playroom; do [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no --indoor FP32 original for $S (cap job needs it)"; exit 1; }; done
for S in garden bicycle stump flowers treehill truck train; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no FP32 original for $S (cap job needs it)"; exit 1; }; done
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
python $AN/indoor_budget_check_v1.py || { echo "CHECK FAILED: the indoor preset overwrites max_points on this upstream"; exit 1; }
busy=$(squeue -u $USER -h -o "%j" | grep -cE "abl24|g0r24|g0rb24"); [ "$busy" = 0 ] || echo "note: $busy batch24 jobs already in the queue (restartable: finished stages are skipped)"
scontrol show reservation $RES | grep -o "EndTime=[^ ]*\|NodeCnt=[^ ]*\|TRES=[^ ]*"; echo "public groups: $PUBLIC"
echo "CHECK OK"; k=0
sb () { j=$(sbatch --parsable $RESOPT --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1}) ${RESOPT:-public}" >&2; echo $j; } || { echo "FAILED: sbatch $*" >&2; echo ""; }; }
acc () { [ -n "$1" ] && k=$((k+1)); }
cfg () { case $1 in
  room)     echo "1 images_2 q14 '' '2198000 1020000 477000'";;     kitchen)  echo "1 images_2 q16 '' '1895000 928000 409000'";;
  counter)  echo "1 images_2 q15 '' '2040000 978000 449000'";;  bonsai)   echo "1 images_2 q15 '' '2090000 1005000 473000'";;
  garden)   echo "0 images_4 q15 '' '2015000 936000 405000'";;   bicycle)  echo "0 images_4 q16 '' '2029000 986000 440000'";;
  stump)    echo "0 images_4 q16 '' '2015000 983000 440000'";;    flowers)  echo "0 images_4 q15 '' '2068000 992000 447000'";;
  treehill) echo "0 images_4 q16 '' '2054000 1002000 449000'";; truck)    echo "0 images q18 ',DATA=$D/tandt/truck' '1853000 929000 421000'";;
  train)    echo "0 images q18 ',DATA=$D/tandt/train' '2040000 975000 436000'";; esac; }
for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train; do
  eval set -- $(cfg $S); IN=$1; IM=$2; X=$3; EX=$4; BUD=$5
  B="SCENE=$S,INDOOR=$IN,IMAGES=$IM,XYZ=$X,SMOKE=0,SELFCHECK=0$EX"
  ro mip-nocoll; jn=$(sb --time=03:30:00 --export=ALL,$B,ABL=nocollapse,TARGETS=10 --job-name=$S-nocoll-abl24 $AN/ablation-b24.sbatch); acc $jn
  ro mip-constants; j1=$(sb --time=06:00:00 --export=ALL,$B,ABL=constants,TARGETS="10 20" --job-name=$S-const-abl24 $AN/ablation-b24.sbatch); acc $j1
  ro mip-constlam; j2=$(sb --time=06:00:00 --export=ALL,$B,ABL=constlam,LAMS="0.00185 0.0062" --job-name=$S-clam-abl24 $AN/ablation-b24.sbatch); acc $j2
  ro mip-nocoll; if [ -n "$jn" ]; then j3=$(sb --time=03:30:00 --dependency=afterok:$jn --export=ALL,$B,ABL=matched,TARGETS=10,SCHED=/scratch/zr2245/$S-abl24-nocollapse-t10mb/original/rd_log.jsonl --job-name=$S-match-abl24 $AN/ablation-b24.sbatch); acc $j3
  else echo "SKIP matched $S: nocollapse job not accepted"; fi
  ro mip-cap; j4=$(sb --time=05:30:00 --export=ALL,SCENE=$S,IMAGES=$IM,XYZ=$X,INDOOR=$IN,BUDGETS="$BUD"$EX --job-name=$S-g0r24 $AN/budget-sweep-v3.sbatch); acc $j4
done
cfgb () { case $1 in lego) echo "259000 142000 66000";; chair) echo "262000 142000 67000";; drums) echo "258000 159000 68000";; ficus) echo "115000 113000 75000";; hotdog) echo "222000 146000 69000";; materials) echo "259000 165000 75000";; mic) echo "170000 145000 69000";; ship) echo "294000 164000 74000";; esac; }
for S in lego chair drums ficus hotdog materials mic ship; do
  BUD=$(cfgb $S); B="SCENE=$S,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0"
  ro bl-nocoll; jn=$(sb --time=02:30:00 --export=ALL,$B,ABL=nocollapse,TARGETS=2.5 --job-name=$S-nocoll-abl24b $AN/ablation-b24-blender.sbatch); acc $jn
  ro bl-constants; j1=$(sb --time=04:30:00 --export=ALL,$B,ABL=constants,TARGETS="2.5 1.5" --job-name=$S-const-abl24b $AN/ablation-b24-blender.sbatch); acc $j1
  ro bl-constlam; j2=$(sb --time=04:30:00 --export=ALL,$B,ABL=constlam,LAMS="0.0062 0.0185" --job-name=$S-clam-abl24b $AN/ablation-b24-blender.sbatch); acc $j2
  ro bl-nocoll; if [ -n "$jn" ]; then j3=$(sb --time=02:30:00 --dependency=afterok:$jn --export=ALL,$B,ABL=matched,TARGETS=2.5,SCHED=/scratch/zr2245/$S-abl24b-nocollapse-t2.5mb/original/rd_log.jsonl --job-name=$S-match-abl24b $AN/ablation-b24-blender.sbatch); acc $j3
  else echo "SKIP matched $S: nocollapse job not accepted"; fi
  ro bl-cap; j4=$(sb --time=04:00:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS="$BUD" --job-name=$S-g0rb24 $AN/budget-sweep-blender-v1.sbatch); acc $j4
done
# Deep Blending (transfer set): only the vertex-cap rerun at the controller counts, so that Table E2 is controller-based on all four datasets
ro db-cap
j5=$(sb --time=06:00:00 --export=ALL,SCENE=drjohnson,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/drjohnson,BUDGETS="1772000 927000 393000" --job-name=drjohnson-g0r24 $AN/budget-sweep-v3.sbatch); acc $j5
j5=$(sb --time=06:00:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS="1918000 906000 436000" --job-name=playroom-g0r24 $AN/budget-sweep-v3.sbatch); acc $j5
echo "submitted $k jobs (expected 97: 11 Mip/T&T scenes x 5 + 8 NeRF-Synthetic objects x 5 + 2 Deep Blending cap jobs)"
squeue -u $USER -o "%.10i %.22j %.2t %.10M %.6D %R" | head -60
SUB24
bash $AN/submit-batch24-v1.sh
```

Expected: `CHECK OK`, then 97 `submitted` lines (the last two are the Deep Blending vertex-cap jobs). The matched jobs sit in PD (Dependency) until their nocollapse job finishes. Rough timing: nocollapse / matched ~2 h, constants / constlam ~4 h, cap ~4 h; Blender jobs ~1-3 h. GPU budget: the reservation has 4 GPUs (about 130 GPU-h before it ends 11/12 09:35); the batch is about 280 GPU-h, so by default the constlam and vertex-cap groups (about 155 GPU-h, 40 jobs incl. the two Deep Blending cap jobs) go to the public L40S partition and the rest (57 jobs) to the reservation. Change the split with `PUBLIC="..." bash $AN/submit-batch24-v1.sh` (group names: mip-nocoll mip-constants mip-constlam mip-cap bl-nocoll bl-constants bl-constlam bl-cap db-cap; matched follows nocoll). Jobs that cannot finish before the reservation ends will not start inside it, so do not move more onto it.

## §3 collect (when `squeue -u $USER | grep -cE "abl24|g0r24|g0rb24"` is 0)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch24-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== ablation-b24.txt"; cat $AN/ablation-b24.txt; echo; echo "===== ablation-b24-blender.txt"; cat $AN/ablation-b24-blender.txt; echo; echo "===== g0rb24 rows (Blender cap at the controller counts)"; for S in lego chair drums ficus hotdog materials mic ship; do grep -h "TRAIN-AT-BUDGET-BLENDER\|POST-HOC-COLLAPSE-BLENDER" $AN/g0-refresh-blender-$S.txt 2>/dev/null; done; echo; echo "===== g0r24 rows (TRAIN-AT-BUDGET / POST-HOC-COLLAPSE at the controller counts)"; for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom; do grep -h "budget \(2[0-9][0-9][0-9]000\|1[0-9][0-9][0-9]000\|9[0-9][0-9]000\|4[0-9][0-9]000\) " $AN/g0-refresh-$S.txt 2>/dev/null; done; echo; echo "===== exits"; grep -h "exit 0\|FAILED" $AN/abl24-*.out $AN/abl24b-*.out $AN/g0r-*-g0r24-*.out $AN/g0rb-*-g0rb24-*.out 2>/dev/null | sort | uniq -c; echo; echo "===== constlam smokes"; grep -h "constlam smoke ok\|CONSTLAM SMOKE" $AN/abl24-*clam*.out $AN/abl24b-*clam*.out; echo; echo "===== split counts: matched vs nocollapse (n_split_total per interval)"; for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train lego chair drums ficus hotdog materials mic ship; do for V in nocollapse matched; do f=/scratch/zr2245/$S-abl24-$V-t10mb/original/rd_log.jsonl; [ -f $f ] || f=/scratch/zr2245/$S-abl24b-$V-t2.5mb/original/rd_log.jsonl; [ -f $f ] && python3 -c "import json,sys; d=[json.loads(l) for l in open(sys.argv[1])]; d=[e for e in d if e['event']=='densify']; print(sys.argv[2], sys.argv[3], [e['n_split'] for e in d], 'total', sum(e['n_split'] for e in d))" $f $S $V; done; done; } > $O 2>&1; wc -l $O; echo $O
```

Send me that file (it is small). Failures: `grep -l FAILED $AN/abl24-*.out` and the tail of that .out.
