#!/usr/bin/env python3
"""make_batch29_files_v3.py (2026-11-12): the size prediction in ratio form.
What the second version showed (10 runs): the collapse now frees exactly the bytes it is asked for, and the four Mip-NeRF 360 runs land within
1.5 % of the target (kitchen -1.3, garden +1.2, room -1.5, treehill -0.5; before: +13.5, +2.0, +6.1, -6.6) at equal or better PSNR at the same
file size.  Runs that trim a lot end BELOW the target (drjohnson -13 %, objects with aim 1.25 -17 .. -31 %).  Cause: the prediction of the
final size adds the removed bytes back ('virtual size') and extrapolates their growth, but a byte removed at round k is worth more than one
byte at the end (kitchen: 1.22), because the part of the mesh that is gone no longer grows.  The prediction is then too high, the controller
removes more, and nothing can add bytes back.
Change (--trim_exact 5): the natural growth is tracked as a ratio, n_k = n_{k-1} * probe_k / (probe_{k-1} - removed_{k-1}); the predicted
final size is probe_now * G with G = n_hat(end of ramp) / n_now; the bytes to remove now are (prediction - target) / G / rounds left.
--trim_start: rounds before this iteration only measure (the first predictions rest on two or three probes).
Outputs train_rd_adapt_v9b_abl5.py and ablation-b24-r6{,-blender}.sbatch with ABL = ratio (default aim) | ratio25 (aim 1.25) | ratio35 (aim 1.35).
  python3 make_batch29_files_v3.py --train <train_rd_adapt_v9b_abl4.py> --sbatch <ablation-b24-r5.sbatch> --sbatch_blender <ablation-b24-r5-blender.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'train': 'c36e97b1e56f17e51291bec76833c3fe', 'sbatch': '0947c52ac846f35b14141dcd0efa9db8', 'blender': '792b10a43264e763ea2882bbba29a488'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
RATIO = '''# >>> extrapolate_ratio
def extrapolate_ratio(trend, iteration, r_meas, horizon_iter, window=6):
    """Prediction of the final size when a removal scales with the growth that follows it.  trend = list of (iteration, probe_mb,
    mb removed by that round).  Natural series (the file without any collapse): n_0 = first probe, n_k = n_{k-1} * probe_k /
    (probe_{k-1} - removed_{k-1}).  A least-squares line through the last `window` points (iteration, n_k), anchored at the current point, gives the growth factor
    G = n(horizon) / n(now); the prediction is r_meas * G.  Returns (est, slope of the natural series per 1000 it, G)."""
    if not trend or iteration >= horizon_iter: return r_meas, 0.0, 1.0
    xs, ns, n, prev = [], [], None, None
    for it0, p0, rem0 in trend:
        n = float(p0) if n is None else n * float(p0) / max(prev, 1e-9)
        xs.append(float(it0)); ns.append(n); prev = float(p0) - float(rem0)
    n = n * float(r_meas) / max(prev, 1e-9); xs.append(float(iteration)); ns.append(n)
    xs, ns = xs[-window:], ns[-window:]  # the last probes only: growth right after the stitch is not representative (146 logged runs: 6 probes give the smallest error at the last round)
    k = len(xs); mx = sum(xs) / k; my = sum(ns) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ns)) / sxx if sxx > 0 else 0.0
    g = max(0.8, (n + slope * (horizon_iter - iteration)) / max(n, 1e-9))
    return r_meas * g, slope * 1000.0, g
# <<< extrapolate_ratio


'''
ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
s = pathlib.Path(a.train).read_text()
if md5(s) != IN['train']: sys.exit(f'FAILED: {a.train} md5 {md5(s)} != {IN["train"]}')
s = rep(s, "def extrapolate_probe(trend, iteration, r_meas, horizon_iter):", RATIO + "def extrapolate_probe(trend, iteration, r_meas, horizon_iter):", 'ratio function')
s = rep(s, "                r_est, slope_k = extrapolate_probe(trend, iteration, r_meas, int(total_iters_opacity)); last_r_est = r_est  # what this probe will read once the opacity ramp has ended\n",
        "                te5 = (not rd.get('hold_lambda')) and int(rd.get('trim_exact') or 0) == 5; g_fac = 1.0  # batch29 v3\n"
        "                if te5: r_est, slope_k, g_fac = extrapolate_ratio(trend, iteration, r_meas, int(total_iters_opacity)); last_r_est = r_est\n"
        "                else: r_est, slope_k = extrapolate_probe(trend, iteration, r_meas, int(total_iters_opacity)); last_r_est = r_est  # what this probe will read once the opacity ramp has ended\n", 'prediction')
s = rep(s, "  # batch29 v2: any excess opens the round\n", "  # batch29 v2: any excess opens the round\n"
        "                if te == 5 and iteration < int(rd.get('trim_start') or 0): wanted = 0.0  # batch29 v3: measure only before trim_start\n", 'trim_start gate')
s = rep(s, "want_bytes = excess / (rounds_left if te == 3 else min(rounds_left, 3));", "want_bytes = (excess / g_fac / rounds_left) if te == 5 else excess / (rounds_left if te == 3 else min(rounds_left, 3));", 'want_bytes')
s = rep(s, "ev.update(trim_exact=te, trim_want_mb=", "ev.update(trim_exact=te, trim_growth=round(g_fac, 4), trim_want_mb=", 'log')
s = rep(s, "parser.add_argument('--trim_exact', type=int, default=0, choices=[0, 1, 2, 3, 4],", "parser.add_argument('--trim_start', type=int, default=0, help='batch29 v3: with --trim_exact 5, collapse rounds before this iteration only measure')\n    parser.add_argument('--trim_exact', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],", 'arguments')
s = rep(s, "rd=dict(trim_exact=args.trim_exact,", "rd=dict(trim_start=args.trim_start, trim_exact=args.trim_exact,", 'rd dict')
(out / 'train_rd_adapt_v9b_abl5.py').write_text(s); print(f'train_rd_adapt_v9b_abl5.py: {len(s.splitlines())} lines md5 {md5(s)}')
for key, src, name in [('sbatch', a.sbatch, 'ablation-b24-r6.sbatch'), ('blender', a.sbatch_blender, 'ablation-b24-r6-blender.sbatch')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]}')
    s = rep(s, "feas|feasq|feas25|feasq25|feasq35 via --export}", "feas|feasq|feas25|feasq25|feasq35|ratio|ratio25|ratio35 via --export}", 'ABL prompt')
    s = rep(s, "  feasq35) XARGS=\"--trim_exact 4\"; OVER=1.35 ;;\n", "  feasq35) XARGS=\"--trim_exact 4\"; OVER=1.35 ;;\n  ratio) XARGS=\"--trim_exact 5 --trim_start 16000\" ;;\n  ratio25) XARGS=\"--trim_exact 5 --trim_start 16000\"; OVER=1.25 ;;\n  ratio35) XARGS=\"--trim_exact 5 --trim_start 16000\"; OVER=1.35 ;;\n", 'ratio cases')
    s = rep(s, "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl4.py;", "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl5.py;", 'TR')
    s = s.replace("VERSION abl24-r5", "VERSION abl24-r6").replace("VERSION abl24b-r5", "VERSION abl24b-r6")
    (out / name).write_text(s); print(f'{name}: {len(s.splitlines())} lines md5 {md5(s)}')
