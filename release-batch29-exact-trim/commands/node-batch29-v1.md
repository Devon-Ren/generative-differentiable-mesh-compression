# node-batch29-v1 (2026-11-12): pilot of exact trimming (12 jobs, 13 runs, fairshare, about 10 GPU-h)

## What changes

Finding from the batch28 logs: a collapse round removes far less than it plans. Example (lego, aim 1.25, round at 13,000): the estimate is 0.74 MB above the target, each collapsed pair frees 9.6 bytes, so about 26,000 pairs were allowed in that round (15% of the vertices); 4,530 were collapsed. The plan picks candidate edges by rank, then the independent-set matching (no two collapses share a vertex) and the fold-over filter discard most of them, and nothing compensates. The controller sees the excess and cannot remove it; the growth aim had to absorb the difference, which is why it needed a value per capture class and why indoor scenes land 5-14% high.

Change (`--trim_exact`): the round keeps raising the candidate threshold (and the number of matched pairs it asks for) until the planned number of pairs is actually collapsed; the plan itself is the excess spread over the remaining rounds. Nothing else changes (same probe, same growth, same codec). Tested on CPU on synthetic meshes with the real `rd_prune.select_collapses`: delivered pairs 97-100% of the plan (40-95% before).

| variant | trimming | growth aim |
|---|---|---|
| exact | planned pairs delivered | default (1.25 captured, 1.1 objects) |
| exact25 | planned pairs delivered | 1.25 (objects) |
| exact35 | planned pairs delivered | 1.35 |
| exactfl | original count (front-loaded), delivered | default |

Runs: kitchen 10 MB (exact, exact35, exactfl); garden 10 (exact35); treehill 20 (exact35); room 5 (exact35); playroom 10, drjohnson 10 (exact); lego 2.5 and 1.5, hotdog 2.5, mic 2.5 (exact25); hotdog 2.5 (exact).

What would count as success (fixed before the runs): every run within 5% of the target except where the object cannot fill the size; PSNR at the same file size not more than 0.1 dB below the current method; one growth aim (1.35 or 1.25) working for captured scenes and objects.

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
echo "=== how much of the planned collapse was delivered in an existing run (lego, aim 1.25) ==="; grep -h "RD prune" $AN/abl24b-lego-aim125-b28-*.out | grep -v "exact trim" | head -4 | cut -c1-420
cat > $AN/make_batch29_files_v1.py <<'PYGEN29'
#!/usr/bin/env python3
"""make_batch29_files_v1.py (2026-11-12): exact trimming.
Finding (batch28 logs): a collapse round delivers only 20-40 % of the pairs it plans.  The plan asks for `wanted` pairs, the candidates are the
4 x wanted cheapest edges, and the independent-set matching (no two collapses share a vertex) keeps a small part of them because cheap edges
cluster around the same vertices.  The controller therefore sees the excess, plans its removal, and removes a fraction; the growth aim had to
absorb the rest.
Change (--trim_exact, default 0 = unchanged):
  1 : per-round plan wanted = min(excess / rounds_left, excess) pairs, and the candidate threshold is raised (margin 4, 8, 16, ...) until the
      matching finds that many pairs (still at most prune_max_frac of the vertices per round);
  2 : the original wanted (max(plan, edges under lambda), capped by the whole excess), delivered in the same exact way (front-loaded).
Outputs: train_rd_adapt_v9b_abl3.py and ablation-b24-r4{,-blender}.sbatch with the variants
  exact   (--trim_exact 1, default aim)      exact25 (--trim_exact 1, aim 1.25)      exact35 (--trim_exact 1, aim 1.35)      exactfl (--trim_exact 2, default aim)
  python3 make_batch29_files_v1.py --train <train_rd_adapt_v9b_abl2.py> --sbatch <ablation-b24-r3.sbatch> --sbatch_blender <ablation-b24-r3-blender.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'train': '75e92ffc397018fe472fc97801d0b9f9', 'sbatch': '949703fb20c0b3c5ba58a8d3e53eb21c', 'blender': '69614021a3aaf74cc7f999697ed8be54'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
# ---- training script
s = pathlib.Path(a.train).read_text()
if md5(s) != IN['train']: sys.exit(f'FAILED: {a.train} md5 {md5(s)} != {IN["train"]}')
s = rep(s, "                if rd.get('hold_lambda'): wanted = float(n_lam); count_driven = False; cap_bound = False; thr = thr_lam",
        "                te = 0 if rd.get('hold_lambda') else int(rd.get('trim_exact') or 0)  # batch29\n"
        "                if te == 1: wanted = (min(desired, excess_pairs) if excess > 0 else 0.0); cap_bound = False  # batch29: the per-round plan, nothing else\n"
        "                if rd.get('hold_lambda'): wanted = float(n_lam); count_driven = False; cap_bound = False; thr = thr_lam", 'wanted')
s = rep(s, "                    pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac)\n",
        "                    pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac)\n"
        "                    if te:  # batch29: deliver the wanted pairs (the matching and the fold-over filter both lose pairs)\n"
        "                        goal = int(frac * V0); margin = 4.0; frac_eff = frac; tries = 1; first_pairs = int(ps['n_pairs']); n_fin = int(torch.isfinite(score).sum())\n"
        "                        while ps['n_pairs'] < 0.97 * goal and tries < 14:\n"
        "                            tries += 1\n"
        "                            if ps['n_pass_pairs'] < int(frac_eff * V0):  # the matching ran out of candidates: admit more edges\n"
        "                                margin *= 2.0; thr_m, k_m = rank_threshold(score, wanted, margin=margin)\n"
        "                                if thr_m <= thr:\n"
        "                                    if k_m >= n_fin - 1: break\n"
        "                                    continue\n"
        "                                thr = thr_m; count_driven = True\n"
        "                            else:  # enough matched pairs, the fold-over filter removed too many: ask the matching for more\n"
        "                                if frac_eff >= 0.45: break\n"
        "                                frac_eff = min(0.45, frac_eff * max(1.25, goal / max(ps['n_pairs'], 1)))\n"
        "                            pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac_eff)\n"
        "                        if int(pairs.shape[0]) > goal:  # keep the cheapest `goal` pairs\n"
        "                            Vn = int(edges.max()) + 1; key = edges[:, 0] * Vn + edges[:, 1]; order_e = torch.argsort(key); skey = key[order_e]\n"
        "                            pu = torch.minimum(pairs[:, 0], pairs[:, 1]).long(); pv = torch.maximum(pairs[:, 0], pairs[:, 1]).long(); pk = pu * Vn + pv\n"
        "                            pos = torch.searchsorted(skey, pk).clamp_max(skey.numel() - 1); sc = score[order_e[pos]]\n"
        "                            pairs = pairs[torch.argsort(sc)[:goal]]; ps['n_pairs'] = int(pairs.shape[0])\n"
        "                        ev.update(trim_exact=te, trim_goal=goal, trim_margin=margin, trim_frac_eff=round(frac_eff, 4), trim_tries=tries, trim_first_pairs=first_pairs)\n"
        "                        print(f\"[RD prune {iteration}] exact trim (mode {te}): goal {goal} pairs, first pass {first_pairs}, after {tries} tries (margin {margin:g}, pair cap {frac_eff:.3f} V) {ps['n_pairs']} pairs (matching found {ps['n_pass_pairs']}, flip-rejected {ps['n_flip_rejected']})\")\n", 'select loop')
s = rep(s, "    parser.add_argument('--hold_lambda', action='store_true',", "    parser.add_argument('--trim_exact', type=int, default=0, choices=[0, 1, 2], help='batch29: 1 = collapse the per-round plan exactly (candidate threshold raised until the matching delivers it); 2 = the original wanted count, delivered exactly; 0 = unchanged')\n    parser.add_argument('--hold_lambda', action='store_true',", 'argument')
s = rep(s, "rd=dict(hold_lambda=args.hold_lambda, lam=args.lam,", "rd=dict(trim_exact=args.trim_exact, hold_lambda=args.hold_lambda, lam=args.lam,", 'rd dict')
(out / 'train_rd_adapt_v9b_abl3.py').write_text(s); print(f'train_rd_adapt_v9b_abl3.py: {len(s.splitlines())} lines md5 {md5(s)}')
# ---- sbatch files
for key, src, name in [('sbatch', a.sbatch, 'ablation-b24-r4.sbatch'), ('blender', a.sbatch_blender, 'ablation-b24-r4-blender.sbatch')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]}')
    s = rep(s, "constlam|repeat|adaptcap via --export}", "constlam|repeat|adaptcap|exact|exact25|exact35|exactfl via --export}", 'ABL prompt')
    s = rep(s, "  adaptcap) PRUNE=\"\"; EXTRA=\"\"; DDMODE=importance ;;\n",
            "  adaptcap) PRUNE=\"\"; EXTRA=\"\"; DDMODE=importance ;;\n  exact) XARGS=\"--trim_exact 1\" ;;\n  exact25) XARGS=\"--trim_exact 1\"; OVER=1.25 ;;\n  exact35) XARGS=\"--trim_exact 1\"; OVER=1.35 ;;\n  exactfl) XARGS=\"--trim_exact 2\" ;;\n", 'exact cases')
    s = rep(s, "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl2.py;", "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl3.py;", 'TR')
    s = s.replace("VERSION abl24-r3", "VERSION abl24-r4").replace("VERSION abl24b-r3", "VERSION abl24b-r4")
    (out / name).write_text(s); print(f'{name}: {len(s.splitlines())} lines md5 {md5(s)}')
PYGEN29
python3 $AN/make_batch29_files_v1.py --train $AN/rdgrowth/train_rd_adapt_v9b_abl2.py --sbatch $AN/ablation-b24-r3.sbatch --sbatch_blender $AN/ablation-b24-r3-blender.sbatch --out $AN/batch29-build && cp $AN/batch29-build/train_rd_adapt_v9b_abl3.py $AN/rdgrowth/ && cp $AN/batch29-build/ablation-b24-r4.sbatch $AN/batch29-build/ablation-b24-r4-blender.sbatch $AN/ || echo "GENERATOR FAILED: stop here"
cat > $AN/submit-batch29-v1.sh <<'SUB29'
#!/usr/bin/env bash
# submit-batch29-v1.sh (2026-11-12): pilot of exact trimming (12 jobs, 13 runs, public partition, about 10 GPU-h).
#   exact   : collapse rounds deliver the planned pairs exactly, default growth aim (1.25 captured, 1.1 objects)
#   exact25 : same, growth aim 1.25 (objects)          exact35 : same, growth aim 1.35          exactfl : front-loaded variant, default aim
# Reference (current method): kitchen 10 MB +13.5%, garden 10 +2.0%, treehill 20 -6.6%, room 5 +6.1%, playroom 10 +8.3%, drjohnson 10 +12.0%,
#   lego 2.5 -0.2% (aim 1.25: +11.4%), hotdog 2.5 +9.2% (aim 1.25: +20.6%), mic 2.5 -6.6% (aim 1.25: -6.2%), lego 1.5 +1.1%.
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl3.py 962 a30a25faa3c204d0551604ce78255122
chk $AN/ablation-b24-r4.sbatch 208 d16aa0d2f6fd254fa0b680c1f0c9043a
chk $AN/ablation-b24-r4-blender.sbatch 210 79f9ede7b9b0207563c471b486fd965c
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl3.py && bash -n $AN/ablation-b24-r4.sbatch && bash -n $AN/ablation-b24-r4-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
grep -q "n_pass_pairs=n_picked" $AN/rdgrowth/rd_prune.py && grep -q "n_flip_rejected=n_flip" $AN/rdgrowth/rd_prune.py || { echo "CHECK FAILED: $AN/rdgrowth/rd_prune.py does not report n_pass_pairs / n_flip_rejected"; exit 1; }
for G in kitchen-abl24-exact-t10mb kitchen-abl24-exact35-t10mb kitchen-abl24-exactfl-t10mb garden-abl24-exact35-t10mb treehill-abl24-exact35-t20mb room-abl24-exact35-t5mb playroom-abl24-exact-t10mb drjohnson-abl24-exact-t10mb lego-abl24b-exact25-t2.5mb hotdog-abl24b-exact25-t2.5mb mic-abl24b-exact25-t2.5mb hotdog-abl24b-exact-t2.5mb; do [ -e /scratch/zr2245/$G ] && echo "note: /scratch/zr2245/$G already exists (finished stages are skipped)"; done
echo "CHECK OK"
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && echo "submitted $j  (${*:3:1})" || echo "FAILED: sbatch $*"; }
C="SMOKE=0,SELFCHECK=0"
for A in exact exact35 exactfl; do sb --time=02:30:00 --export=ALL,SCENE=kitchen,INDOOR=1,IMAGES=images_2,XYZ=q16,$C,ABL=$A,TARGETS=10 --job-name=kitchen-$A-b29 $AN/ablation-b24-r4.sbatch; done
sb --time=02:00:00 --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,$C,ABL=exact35,TARGETS=10 --job-name=garden-exact35-b29 $AN/ablation-b24-r4.sbatch
sb --time=02:30:00 --export=ALL,SCENE=treehill,INDOOR=0,IMAGES=images_4,XYZ=q16,$C,ABL=exact35,TARGETS=20 --job-name=treehill-exact35-b29 $AN/ablation-b24-r4.sbatch
sb --time=02:30:00 --export=ALL,SCENE=room,INDOOR=1,IMAGES=images_2,XYZ=q14,$C,ABL=exact35,TARGETS=5 --job-name=room-exact35-b29 $AN/ablation-b24-r4.sbatch
for S in playroom drjohnson; do sb --time=02:30:00 --export=ALL,SCENE=$S,INDOOR=1,IMAGES=images,XYZ=q15,$C,DATA=$D/db/$S,ABL=exact,TARGETS=10 --job-name=$S-exact-b29 $AN/ablation-b24-r4.sbatch; done
sb --time=04:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,$C,ABL=exact25,TARGETS="2.5 1.5" --job-name=lego-exact25-b29 $AN/ablation-b24-r4-blender.sbatch
for S in hotdog mic; do sb --time=02:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,$C,ABL=exact25,TARGETS=2.5 --job-name=$S-exact25-b29 $AN/ablation-b24-r4-blender.sbatch; done
sb --time=02:30:00 --export=ALL,SCENE=hotdog,XYZ=q14,INDOOR=1,$C,ABL=exact,TARGETS=2.5 --job-name=hotdog-exact-b29 $AN/ablation-b24-r4-blender.sbatch
squeue -u $USER -o "%.10i %.26j %.2t %.10M %R" | grep -E "b29|JOBID"
SUB29
bash $AN/submit-batch29-v1.sh
```

Expected: the four `RD prune` lines of the old run (they show `wanted N pairs ... -> M collapsed`), three generator lines (`train_rd_adapt_v9b_abl3.py: 962 lines md5 a30a25faa3c204d0551604ce78255122`, `ablation-b24-r4.sbatch: 208 lines md5 d16aa0d2f6fd254fa0b680c1f0c9043a`, `ablation-b24-r4-blender.sbatch: 210 lines md5 79f9ede7b9b0207563c471b486fd965c`), `CHECK OK`, 12 `submitted` lines.

The patched code first runs at iteration 12,500 (about 10 minutes into a job). To see early that it works:

```bash
grep -h "exact trim" /scratch/zr2245/analysis/abl24*-*-b29-*.out | head -8 | cut -c1-260
```

## Collect (when `squeue --me | grep -c b29` is 0)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch29-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== rows"; grep -h "abl24-exact" $AN/ablation-b24.txt $AN/ablation-b24-blender.txt | sed -E 's/, val_select kept faces.*//; s/ \| phase-A probes.*//'; echo; echo "===== exact trim lines (first and last rounds per job)"; for f in $AN/abl24*-*-b29-*.out; do echo "--- $f"; grep -h "exact trim" $f | sed -n '1,3p;$p' | cut -c1-260; done; echo; echo "===== old run"; grep -h "RD prune" $AN/abl24b-lego-aim125-b28-*.out | head -4 | cut -c1-420; echo; echo "===== sacct"; sacct -X --starttime=now-3days --format=JobID,JobName%28,State,Elapsed,ExitCode | grep b29; grep -l "FAILED\|Traceback" $AN/abl24*-*-b29-*.out $AN/abl24*-*-b29-*.err 2>/dev/null; } > $O 2>&1; wc -l $O; echo $O
```
