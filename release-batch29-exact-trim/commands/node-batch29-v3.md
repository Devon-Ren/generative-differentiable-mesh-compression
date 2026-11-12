# node-batch29-v3 (2026-11-12): ratio-form size prediction (12 jobs, fairshare, about 8 GPU-h)

## What the second version showed (10 finished runs)

| scene, target | current method | second version | PSNR at the same file size (new - current) |
|---|---|---|---|
| kitchen 10 MB | +13.5% | -1.3% (both schedules) | +0.11 dB |
| garden 10 MB | +2.0% | +1.2% | 0.00 dB |
| room 5 MB | +6.1% | -1.5% | +0.20 dB |
| treehill 20 MB (aim 1.35) | -6.6% | -0.5% | -0.22 dB |
| kitchen 10 MB (aim 1.35, fast schedule) | +13.5% | -5.5% | +0.11 dB |
| drjohnson 10 MB (fast schedule) | +11.9% | -13.0% | +0.21 dB |
| hotdog 2.5 MB (aim 1.25) | +9.2% | -17.3% (fast schedule: -31.3%) | +0.30 dB (fast: -0.20 dB) |
| lego 2.5 MB (aim 1.25, fast schedule) | -0.2% | -24.5% | +0.61 dB |

- The collapse frees exactly the bytes it is asked for (per-round log).
- Every miss is now on the low side: too much was removed. Quality at the same file size is not the problem.
- Cause: the prediction adds the removed bytes back and lets them grow, but a byte removed in the middle of the window is worth more than one byte at the end, because the removed part of the mesh no longer grows (kitchen: the untouched file ends at 12.04 MB, 1.78 MB were removed, the file ended at 9.87 MB: 1.22 bytes per removed byte). The prediction is then too high, and removal cannot be undone.

## What changes now

- Prediction in ratio form: the natural growth is tracked as a ratio between probes, predicted final = file now x predicted growth factor; bytes to remove = (prediction - target) / growth factor / rounds left.
- The slope uses the last 6 probes (on 146 logged runs: mean error of the last-round prediction 1.9% instead of 2.3%, 139 within 5%).
- Rounds before 16,000 only measure.
- The fast schedule is dropped.

CPU tests: closed loop on a synthetic file with linear growth lands at 0.0% (the old form: -1.3% to -2.3%); selection test unchanged (100% of the wanted bytes, no fold-over).
Simulation on the 100 logged growth curves, proportional removal: 92 of 100 within 5%, 100 within 10%, range -5.9% .. +7.0% (a simulation, not runs).

Runs: aim 1.35: drjohnson 10, playroom 10, kitchen 10, room 5, treehill 20, garden 10, hotdog 2.5, lego 2.5, mic 2.5. Aim 1.25: kitchen 10, hotdog 2.5, lego 2.5.

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/make_batch29_files_v3.py <<'PYGEN29C'
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
PYGEN29C
python3 $AN/make_batch29_files_v3.py --train $AN/rdgrowth/train_rd_adapt_v9b_abl4.py --sbatch $AN/ablation-b24-r5.sbatch --sbatch_blender $AN/ablation-b24-r5-blender.sbatch --out $AN/batch29-build3 && cp $AN/batch29-build3/train_rd_adapt_v9b_abl5.py $AN/rdgrowth/ && cp $AN/batch29-build3/ablation-b24-r6.sbatch $AN/batch29-build3/ablation-b24-r6-blender.sbatch $AN/ || echo "GENERATOR FAILED: stop here"
cat > $AN/submit-batch29-v3.sh <<'SUB29C'
#!/usr/bin/env bash
# submit-batch29-v3.sh (2026-11-12): pilot of the ratio-form size prediction (12 jobs, public partition, about 8 GPU-h).
#   ratio / ratio25 / ratio35 : fold-over-safe byte-targeted trimming + ratio-form prediction, rounds act from 16,000; growth aim default / 1.25 / 1.35
# Second version, for reference: kitchen -1.3%, garden +1.2%, room -1.5%, treehill -0.5% (aim 1.35); drjohnson -13.0%, hotdog -17.3% (aim 1.25).
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl5.py 1078 3f1f876e372659b9dd8ad0a07610497d
chk $AN/ablation-b24-r6.sbatch 216 4625b51e1cb8f7086c228fda3ec69fc7
chk $AN/ablation-b24-r6-blender.sbatch 218 c7e8a016bcabf0391204c07936ceab15
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl5.py && bash -n $AN/ablation-b24-r6.sbatch && bash -n $AN/ablation-b24-r6-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
echo "CHECK OK"
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && echo "submitted $j  (${*:3:1})" || echo "FAILED: sbatch $*"; }
C="SMOKE=0,SELFCHECK=0"
for S in drjohnson playroom; do sb --time=02:30:00 --export=ALL,SCENE=$S,INDOOR=1,IMAGES=images,XYZ=q15,$C,DATA=$D/db/$S,ABL=ratio35,TARGETS=10 --job-name=$S-ratio35-b29 $AN/ablation-b24-r6.sbatch; done
for S in hotdog lego; do for A in ratio35 ratio25; do sb --time=02:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,$C,ABL=$A,TARGETS=2.5 --job-name=$S-$A-b29 $AN/ablation-b24-r6-blender.sbatch; done; done
for A in ratio35 ratio; do sb --time=02:30:00 --export=ALL,SCENE=kitchen,INDOOR=1,IMAGES=images_2,XYZ=q16,$C,ABL=$A,TARGETS=10 --job-name=kitchen-$A-b29 $AN/ablation-b24-r6.sbatch; done
sb --time=02:30:00 --export=ALL,SCENE=mic,XYZ=q14,INDOOR=1,$C,ABL=ratio35,TARGETS=2.5 --job-name=mic-ratio35-b29 $AN/ablation-b24-r6-blender.sbatch
sb --time=02:30:00 --export=ALL,SCENE=room,INDOOR=1,IMAGES=images_2,XYZ=q14,$C,ABL=ratio35,TARGETS=5 --job-name=room-ratio35-b29 $AN/ablation-b24-r6.sbatch
sb --time=02:30:00 --export=ALL,SCENE=treehill,INDOOR=0,IMAGES=images_4,XYZ=q16,$C,ABL=ratio35,TARGETS=20 --job-name=treehill-ratio35-b29 $AN/ablation-b24-r6.sbatch
sb --time=02:00:00 --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,$C,ABL=ratio35,TARGETS=10 --job-name=garden-ratio35-b29 $AN/ablation-b24-r6.sbatch
squeue -u $USER -o "%.10i %.26j %.2t %.10M %R" | grep -E "b29|JOBID"
SUB29C
bash $AN/submit-batch29-v3.sh
```

Expected: three generator lines (`train_rd_adapt_v9b_abl5.py: 1078 lines md5 3f1f876e372659b9dd8ad0a07610497d`, `ablation-b24-r6.sbatch: 216 lines md5 4625b51e1cb8f7086c228fda3ec69fc7`, `ablation-b24-r6-blender.sbatch: 218 lines md5 c7e8a016bcabf0391204c07936ceab15`), `CHECK OK`, 12 `submitted` lines.

## Watch (rounds act from iteration 16,000)

```bash
for G in drjohnson-abl24-ratio35-t10mb hotdog-abl24b-ratio35-t2.5mb hotdog-abl24b-ratio25-t2.5mb kitchen-abl24-ratio35-t10mb; do echo "== $G"; python3 -c "
import json, sys
for l in open(sys.argv[1]):
    e = json.loads(l)
    if e.get('event') == 'prune': print('it %5d | file now %.3f x target, predicted final %.3f x (growth still to come x%s) | wanted %.3f MB, freed %.3f MB, %6d pairs | V %d -> %d' % (e['iteration'], e['probe_ratio'], e['probe_est_ratio'], e.get('trim_growth', '-'), e.get('trim_want_mb', 0), e.get('trim_planned_mb', 0), e.get('trim_goal', 0), e['vertices_before'], e['vertices_after']))
" /scratch/zr2245/$G/original/rd_log.jsonl 2>/dev/null; done
```

## Collect (when `squeue --me | grep -c b29` is 0)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch29-v3-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== rows"; grep -h "abl24-ratio" $AN/ablation-b24.txt $AN/ablation-b24-blender.txt | sed -E 's/, val_select kept faces.*//; s/ \| phase-A probes.*//'; echo; echo "===== sacct"; sacct -X --starttime=now-2days --format=JobID,JobName%28,State,Elapsed,ExitCode | grep "ratio"; grep -l "FAILED\|Traceback" $AN/abl24*-*ratio*-b29-*.out $AN/abl24*-*ratio*-b29-*.err 2>/dev/null; } > $O 2>&1; wc -l $O; echo $O
```
