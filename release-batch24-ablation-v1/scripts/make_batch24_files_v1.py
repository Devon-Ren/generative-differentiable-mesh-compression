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
