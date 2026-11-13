# node-batch31-v1 (2026-11-12): the prediction-free controller, "hold" (42 jobs, 63 runs, about 45 GPU-h)

## Why one more variant

The final design of batch30 lands within 0.7% on 12 of 13 captured scenes with equal or better PSNR, and within 0.4% on the 7 objects that can fill the target, but the objects lose 0.44 dB on average: their checkpoint had to be taken after the last round (28,000), while validation prefers 21,000 to 26,000.

Hold removes the cause instead of tuning around it. Every 500 iterations from 12,500 to 29,500 the file is measured with the real encoder and whatever is above the target at that moment is removed. There is no prediction, no margin and no assumption about when the mesh stops changing: the file is kept at the target for the rest of training, so every later checkpoint has the right size and the objects keep their validation-selected checkpoint.

| group | what | jobs | order |
|---|---|---|---|
| H | hold on all 21 scenes at 10 MB (captured) / 2.5 MB (objects); garden and lego first, the other 19 only after those two ended without error | 21 | first |
| HX | hold at the other targets of the main table: 20 and 5 MB (captured), 4 and 1.5 MB (objects), two runs per job, after the same two pilots | 21 | second (lower priority) |

## Step 0: free the queue (the fallback runs still waiting are no longer needed)

```bash
squeue -u $USER -h -o "%i %j" | awk '$2 ~ /-ratio-b30$/ || $2 ~ /-ratio35-b30$/ {print $1}' | xargs -r scancel
```

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/make_batch31_files_v1.py <<'PYGEN31'
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
PYGEN31
python3 $AN/make_batch31_files_v1.py --train $AN/rdgrowth/train_rd_adapt_v9b_abl6.py --sbatch $AN/ablation-b24-r7.sbatch --sbatch_blender $AN/ablation-b24-r7-blender.sbatch --out $AN/batch31-build && cp $AN/batch31-build/train_rd_adapt_v9b_abl7.py $AN/rdgrowth/ && cp $AN/batch31-build/ablation-b24-r8.sbatch $AN/batch31-build/ablation-b24-r8-blender.sbatch $AN/ || echo "GENERATOR FAILED: stop here"
cat > $AN/submit-batch31-v1.sh <<'SUB31'
#!/usr/bin/env bash
# submit-batch31-v1.sh (2026-11-12): the prediction-free controller ("hold": measure every 500 iterations, remove what is above the target now).
#   H   : all 21 scenes at 10 MB (captured) / 2.5 MB (objects) -> abl24-hold.  garden and lego start first; the other 19 only after those two ended without error.
#   HX  : the other targets of the main table: 20 and 5 MB (captured), 4 and 1.5 MB (objects); two runs per job; after the same two pilots; lower priority.
# Usage: bash submit-batch31-v1.sh [GROUPS]   (default "H HX")
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; GR=${1:-"H HX"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl7.py 1083 1eeb64ab5032e903f69ad1a21466d99f
chk $AN/ablation-b24-r8.sbatch 218 82259b73557bfcdb760648f42fad08cc
chk $AN/ablation-b24-r8-blender.sbatch 220 d204fb12d9bc3b5c634ab9870b1f38b4
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl7.py && bash -n $AN/ablation-b24-r8.sbatch && bash -n $AN/ablation-b24-r8-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
echo "CHECK OK (groups: $GR)"; n=0
cfg () { case $1 in room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; kitchen) echo "INDOOR=1,IMAGES=images_2,XYZ=q16";; counter|bonsai) echo "INDOOR=1,IMAGES=images_2,XYZ=q15";;
  garden|flowers) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; bicycle|stump|treehill) echo "INDOOR=0,IMAGES=images_4,XYZ=q16";; truck|train) echo "INDOOR=0,IMAGES=images,XYZ=q18,DATA=$D/tandt/$1";;
  drjohnson|playroom) echo "INDOOR=1,IMAGES=images,XYZ=q15,DATA=$D/db/$1";; esac; }
ctl () { # $1 scene  $2 targets ("10" or "20 5")  $3 time limit  $4 tag for the job name  $5.. extra sbatch options; sets LASTJ
  local S=$1 TG=$2 TL=$3 TAG=$4; shift 4; local j
  case $S in lego|chair|drums|ficus|hotdog|materials|mic|ship) j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=$TL "$@" --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=hold,TARGETS="$TG" --job-name=$S-$TAG-b31 $AN/ablation-b24-r8-blender.sbatch);;
    *) j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=$TL "$@" --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=hold,TARGETS="$TG" --job-name=$S-$TAG-b31 $AN/ablation-b24-r8.sbatch);; esac
  LASTJ=$j; [ -n "$j" ] && { n=$((n+1)); echo "submitted $j  $S-$TAG-b31 (targets $TG)"; } || echo "FAILED: sbatch $S $TAG"; }
MIP="room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom"; OBJ="lego chair drums ficus hotdog materials mic ship"
pilot () { # $1 job name -> job id from the queue, or 'done' when its result row exists, or ''
  local id=$(squeue -u $USER -h -n $1 -o %i | head -1); [ -n "$id" ] && { echo $id; return; }
  case $1 in garden-hold-b31) grep -q "^garden abl24-hold | target 10 MB" $AN/ablation-b24.txt 2>/dev/null && echo done;; lego-hold-b31) grep -q "^lego abl24-hold | target 2.5 MB" $AN/ablation-b24-blender.txt 2>/dev/null && echo done;; esac; }
JG=$(pilot garden-hold-b31); JL=$(pilot lego-hold-b31)
for G in $GR; do case $G in
H)  [ -z "$JG" ] && { ctl garden 10 02:30:00 hold; JG=$LASTJ; }; [ -z "$JL" ] && { ctl lego 2.5 02:30:00 hold; JL=$LASTJ; }
    DG=""; [ "$JG" != done ] && DG="--dependency=afterok:$JG"; DL=""; [ "$JL" != done ] && DL="--dependency=afterok:$JL"
    for S in $MIP; do [ $S = garden ] && continue; [ -e /scratch/zr2245/$S-abl24-hold-t10mb ] && { echo "exists, skipped: $S hold 10"; continue; }; ctl $S 10 02:30:00 hold $DG; done
    for S in $OBJ; do [ $S = lego ] && continue; [ -e /scratch/zr2245/$S-abl24b-hold-t2.5mb ] && { echo "exists, skipped: $S hold 2.5"; continue; }; ctl $S 2.5 02:30:00 hold $DL; done;;
HX) [ -n "$JG" ] && [ -n "$JL" ] || { echo "NOT SUBMITTED: HX needs the two pilots (run group H first)"; continue; }
    DG=""; [ "$JG" != done ] && DG="--dependency=afterok:$JG"; DL=""; [ "$JL" != done ] && DL="--dependency=afterok:$JL"
    for S in $MIP; do ctl $S "20 5" 04:30:00 holdx --nice=200 $DG; done
    for S in $OBJ; do ctl $S "4 1.5" 04:00:00 holdx --nice=200 $DL; done;;
*) echo "FAILED: unknown group $G";;
esac; done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
SUB31
bash $AN/submit-batch31-v1.sh
```

Expected: three generator lines (`train_rd_adapt_v9b_abl7.py: 1083 lines md5 1eeb64ab5032e903f69ad1a21466d99f`, `ablation-b24-r8.sbatch: 218 lines md5 82259b73557bfcdb760648f42fad08cc`, `ablation-b24-r8-blender.sbatch: 220 lines md5 d204fb12d9bc3b5c634ab9870b1f38b4`), `CHECK OK (groups: H HX)`, 42 `submitted` lines, `42 GPU jobs submitted`.

## Watch the two pilots

```bash
for G in garden-abl24-hold-t10mb lego-abl24b-hold-t2.5mb; do echo "== $G"; python3 -c "
import json, sys
for l in open(sys.argv[1]):
    e = json.loads(l)
    if e.get('event') == 'prune' and e['iteration'] % 2000 == 500: print('it %5d | file now %.3f x target | wanted %.3f MB, freed %.3f MB, %6d pairs | V %d -> %d' % (e['iteration'], e['probe_ratio'], e.get('trim_want_mb', 0), e.get('trim_planned_mb', 0), e.get('trim_goal', 0), e['vertices_before'], e['vertices_after']))
" /scratch/zr2245/$G/original/rd_log.jsonl 2>/dev/null; done
```

After the first rounds "file now" should stay between 1.00 and about 1.01 x target.

## Collect (in the morning, whatever has finished)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch31-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== controller rows"; grep -h "abl24-hold\|abl24-final" $AN/ablation-b24.txt $AN/ablation-b24-blender.txt | sed -E 's/, val_select kept faces.*//; s/ \| phase-A probes.*//; s/, rounds \(it, probe MB.*//; s/, extra probes.*//'
  echo; echo "===== failures"; grep -l "FAILED\|Traceback" $AN/abl24*-*-b31-*.out $AN/abl24*-*-b31-*.err 2>/dev/null; for f in $(grep -l "FAILED" $AN/abl24*-*-b31-*.out 2>/dev/null | head -3); do echo "--- $f"; grep -h -A 14 "FAILED" $f | head -30; done
  echo; echo "===== sacct"; sacct -X --starttime=now-2days --format=JobID,JobName%30,State,Elapsed,ExitCode | grep -E "b31"
  echo; echo "===== queue"; squeue -u $USER -h -o "%.10i %.28j %.2t %.10M %R" | grep -E "b31"; } > $O 2>&1; wc -l $O; echo $O
```
