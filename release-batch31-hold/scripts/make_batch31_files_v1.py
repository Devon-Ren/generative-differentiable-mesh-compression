#!/usr/bin/env python3
"""make_batch31_files_v1.py (2026-11-12): the controller without any prediction ("hold").
Result of batch30 (final design, 21 scenes): captured scenes land within 0.7 % (playroom -3.6 %) at equal or better PSNR; objects land within
0.4 % but lose 0.44 dB on average, because their checkpoint had to be taken after the last round (28,000) while validation prefers 21,000-26,000.
Hold: every 500 iterations from 12,500 to 29,500 the file is measured with the real encoder and whatever is above the target NOW is removed
(same fold-over-safe, byte-targeted selection).  No prediction, no margin, no knowledge of when the mesh stops changing: the file is kept at
the target for the rest of training, so every later checkpoint has the right size and objects keep their validation-selected checkpoint.
Code change: --trim_hold (with --trim_exact 5): estimate = measured file, growth factor 1, the whole excess in this round.
Outputs train_rd_adapt_v9b_abl7.py and ablation-b24-r8{,-blender}.sbatch with ABL = hold (aim 1.35, rounds 12500..29500 step 500, no diagnostic probes).
  python3 make_batch31_files_v1.py --train <train_rd_adapt_v9b_abl6.py> --sbatch <ablation-b24-r7.sbatch> --sbatch_blender <ablation-b24-r7-blender.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'train': '37d3ca515f0df262bdee538c8a0fe162', 'sbatch': 'e9c48dba072b122a671efffc27996380', 'blender': '6d731771709ee222175e858c2a56355d'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
s = pathlib.Path(a.train).read_text()
if md5(s) != IN['train']: sys.exit(f'FAILED: {a.train} md5 {md5(s)} != {IN["train"]}')
s = rep(s, "                else: r_est, slope_k = extrapolate_probe(trend, iteration, r_meas, int(total_iters_opacity)); last_r_est = r_est  # what this probe will read once the opacity ramp has ended\n",
        "                else: r_est, slope_k = extrapolate_probe(trend, iteration, r_meas, int(total_iters_opacity)); last_r_est = r_est  # what this probe will read once the opacity ramp has ended\n"
        "                if te5 and rd.get('trim_hold'): r_est, slope_k, g_fac = r_meas, 0.0, 1.0; last_r_est = r_est; rounds_left = 1; predictive = False  # batch31: no prediction, the measured excess goes now\n", 'hold')
s = rep(s, "    parser.add_argument('--trim_margin', type=float, default=0.0,", "    parser.add_argument('--trim_hold', action='store_true', help='batch31: with --trim_exact 5, no prediction: every round removes what the measured file has above the target')\n    parser.add_argument('--trim_margin', type=float, default=0.0,", 'argument')
s = rep(s, "rd=dict(trim_margin=args.trim_margin,", "rd=dict(trim_hold=args.trim_hold, trim_margin=args.trim_margin,", 'rd dict')
(out / 'train_rd_adapt_v9b_abl7.py').write_text(s); print(f'train_rd_adapt_v9b_abl7.py: {len(s.splitlines())} lines md5 {md5(s)}')
HOLD = "  hold) XARGS=\"--trim_exact 5 --trim_hold --allow_any_prune_iter\"; OVER=1.35; PRUNE=\"$(seq -s \" \" 12500 500 29500)\"; EXTRA=\"\" ;;\n"
for key, src, name in [('sbatch', a.sbatch, 'ablation-b24-r8.sbatch'), ('blender', a.sbatch_blender, 'ablation-b24-r8-blender.sbatch')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]}')
    s = rep(s, "|ratio|ratio25|ratio35|final via --export}", "|ratio|ratio25|ratio35|final|hold via --export}", 'ABL prompt')
    s = rep(s, "  ratio35) XARGS=\"--trim_exact 5 --trim_start 16000\"; OVER=1.35 ;;\n", "  ratio35) XARGS=\"--trim_exact 5 --trim_start 16000\"; OVER=1.35 ;;\n" + HOLD, 'hold case')
    s = rep(s, "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl6.py;", "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl7.py;", 'TR')
    s = s.replace("VERSION abl24-r7", "VERSION abl24-r8").replace("VERSION abl24b-r7", "VERSION abl24b-r8")
    (out / name).write_text(s); print(f'{name}: {len(s.splitlines())} lines md5 {md5(s)}')
