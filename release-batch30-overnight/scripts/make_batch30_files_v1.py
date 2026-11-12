#!/usr/bin/env python3
"""make_batch30_files_v1.py (2026-11-12): the final design of the size controller.
  1. Grow with a fixed margin: the growth plan aims at 1.35 x the vertex count the target can hold (every dataset).
  2. Predictive rounds (before the opacity ramp ends at 24,000; acting from 16,000): predicted final size = file now x growth factor
     (ratio form, last 6 probes); they steer to target x (1 + margin) and spread the excess over the remaining predictive rounds.
  3. Measured rounds (24,000 / 25,500 / 27,000): the mesh has stopped changing, so the probe IS the final size; they remove what is left
     above the target.  Nothing is predicted at the end; the landing error is the drift of the file after the last round.
Code change (on top of batch29 v3): --trim_margin and the per-phase count of remaining rounds.  The late rounds use the existing
--allow_any_prune_iter switch (the error accumulator already handles the supersampled id maps used after 20,000).
Outputs train_rd_adapt_v9b_abl6.py and ablation-b24-r7{,-blender}.sbatch with ABL = final
(--trim_exact 5 --trim_start 16000 --trim_margin 0.05 --allow_any_prune_iter, aim 1.35, rounds 12500..19500 step 500 + 21000 22500 24000 25500 27000,
objects: validation checkpoint chosen among 28000 / 29000, i.e. after the last round).
  python3 make_batch30_files_v1.py --train <train_rd_adapt_v9b_abl5.py> --sbatch <ablation-b24-r6.sbatch> --sbatch_blender <ablation-b24-r6-blender.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'train': '3f1f876e372659b9dd8ad0a07610497d', 'sbatch': '4625b51e1cb8f7086c228fda3ec69fc7', 'blender': 'c7e8a016bcabf0391204c07936ceab15'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
s = pathlib.Path(a.train).read_text()
if md5(s) != IN['train']: sys.exit(f'FAILED: {a.train} md5 {md5(s)} != {IN["train"]}')
s = rep(s, "                te5 = (not rd.get('hold_lambda')) and int(rd.get('trim_exact') or 0) == 5; g_fac = 1.0  # batch29 v3\n",
        "                te5 = (not rd.get('hold_lambda')) and int(rd.get('trim_exact') or 0) == 5; g_fac = 1.0  # batch29 v3\n"
        "                predictive = iteration < int(total_iters_opacity)  # batch30: before the ramp ends the final size is predicted, afterwards it is measured\n"
        "                if te5: rounds_left = len([i for i in prune_set if i >= iteration and ((i < int(total_iters_opacity)) == predictive)])  # batch30: each phase spreads over its own rounds\n", 'phase rounds')
s = rep(s, "                excess = (r_est - rd['target_mb']) * 1e6\n",
        "                excess = (r_est - rd['target_mb'] * (1.0 + (float(rd.get('trim_margin') or 0.0) if (te5 and predictive) else 0.0))) * 1e6  # batch30: predictive rounds stop above the target\n", 'margin')
s = rep(s, "ev.update(trim_exact=te, trim_growth=round(g_fac, 4), trim_want_mb=", "ev.update(trim_exact=te, trim_predictive=bool(predictive), trim_rounds_left=rounds_left, trim_growth=round(g_fac, 4), trim_want_mb=", 'log')
s = rep(s, "    parser.add_argument('--trim_start', type=int, default=0,", "    parser.add_argument('--trim_margin', type=float, default=0.0, help='batch30: with --trim_exact 5, rounds before the end of the opacity ramp steer to target x (1 + margin); later rounds steer to the target')\n    parser.add_argument('--trim_start', type=int, default=0,", 'argument')
s = rep(s, "rd=dict(trim_start=args.trim_start,", "rd=dict(trim_margin=args.trim_margin, trim_start=args.trim_start,", 'rd dict')
(out / 'train_rd_adapt_v9b_abl6.py').write_text(s); print(f'train_rd_adapt_v9b_abl6.py: {len(s.splitlines())} lines md5 {md5(s)}')
FIN = "  final) XARGS=\"--trim_exact 5 --trim_start 16000 --trim_margin 0.05 --allow_any_prune_iter\"; OVER=1.35; PRUNE=\"$(seq -s \" \" 12500 500 19500) 21000 22500 24000 25500 27000\"; EXTRA=\"28500\"@@SAVES@@ ;;\n"
for key, src, name, saves in [('sbatch', a.sbatch, 'ablation-b24-r7.sbatch', ''), ('blender', a.sbatch_blender, 'ablation-b24-r7-blender.sbatch', '; SAVES="28000 29000"')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]}')
    s = rep(s, "|ratio|ratio25|ratio35 via --export}", "|ratio|ratio25|ratio35|final via --export}", 'ABL prompt')
    s = rep(s, "  ratio35) XARGS=\"--trim_exact 5 --trim_start 16000\"; OVER=1.35 ;;\n", "  ratio35) XARGS=\"--trim_exact 5 --trim_start 16000\"; OVER=1.35 ;;\n" + FIN.replace('@@SAVES@@', saves), 'final case')
    s = rep(s, "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl5.py;", "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl6.py;", 'TR')
    s = s.replace("VERSION abl24-r6", "VERSION abl24-r7").replace("VERSION abl24b-r6", "VERSION abl24b-r7")
    (out / name).write_text(s); print(f'{name}: {len(s.splitlines())} lines md5 {md5(s)}')
