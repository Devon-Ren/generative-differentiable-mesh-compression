# node-batch30-v1 (2026-11-12): the overnight batch (about 72 GPU jobs, about 47 GPU-h; fairshare decides how many finish)

Everything at one target per dataset: 10 MB for the 13 captured scenes, 2.5 MB for the 8 objects.

| group | what | runs | queue order |
|---|---|---|---|
| FIN | the final design of the controller on all 21 scenes | 21 | first |
| CAP | MeshSplatting with a vertex cap from the size prior fitted on the other scenes of the same kind, for the 15 scenes not yet run; one more training if the file misses by more than 10%; plus the two open retries (kitchen, lego third training) | 17 (+ up to 15 retries) | second |
| R35 | fallback: third version without the late rounds, growth aim 1.35 | 14 (7 more exist from the pilot) | third |
| R | fallback: third version, default growth aim | 20 | last |

## The final design (FIN), in three sentences

1. **Grow with a margin.** The growth plan aims at 1.35 times the vertex count the target can hold, the same for every dataset.
2. **Predict and trim.** From iteration 16,000, every round predicts the final size as (file now) x (growth still to come, from the last six measurements) and removes its share of the excess above target x 1.05, by collapsing the edges with the least error per freed byte.
3. **Measure and finish.** The mesh stops changing when the opacity ramp ends (24,000). Three more rounds (24,000 / 25,500 / 27,000) measure the file and remove what is left above the target. Nothing is predicted at the end.

Simulation on the 100 logged growth curves (not runs): mean absolute error 1.0%, 85 within 2%, 97 within 5%, 100 within 10%; the few results near -6% are objects whose reported file came from an early validation checkpoint, which this design no longer allows (objects: the validation checkpoint is chosen among iterations after the last round).

What is new in the code and has not run yet: collapse rounds after iteration 20,000 (the existing `--allow_any_prune_iter` switch; the error accumulator already handles the supersampled renders used there), the 5% margin, and the per-phase round count. garden and lego run first; the other 19 FIN jobs start only if those two end without error, so a crash in the new part costs two jobs.

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/capjob-v1.sh <<'CAPJOB30'
# capjob-v1.sh (2026-11-12): one place that knows how to submit a MeshSplatting vertex-cap run (train with --max_points, fine-tune, encode) for each of the 21 scenes.
# usage: source capjob-v1.sh; [NICE=n] capjob <scene> <cap> <job name>   -> prints the job id
capjob () {
  local S=$1 B=$2 N=$3 AN=/scratch/zr2245/analysis D=/scratch/zr2245/datasets
  local A="--parsable --account=torch_pr_38_general --cpus-per-task=6 --nice=${NICE:-0} --job-name=$N"
  case $S in
    garden|flowers)          sbatch $A --time=02:00:00 --export=ALL,SCENE=$S,IMAGES=images_4,XYZ=q15,INDOOR=0,BUDGETS=$B $AN/budget-sweep-v2.sbatch;;
    bicycle|stump|treehill)  sbatch $A --time=02:00:00 --export=ALL,SCENE=$S,IMAGES=images_4,XYZ=q16,INDOOR=0,BUDGETS=$B $AN/budget-sweep-v2.sbatch;;
    truck|train)             sbatch $A --time=02:00:00 --export=ALL,SCENE=$S,IMAGES=images,XYZ=q18,INDOOR=0,DATA=$D/tandt/$S,BUDGETS=$B $AN/budget-sweep-v2.sbatch;;
    room)                    sbatch $A --time=03:00:00 --export=ALL,SCENE=$S,IMAGES=images_2,XYZ=q14,INDOOR=1,BUDGETS=$B $AN/budget-sweep-v3.sbatch;;
    kitchen)                 sbatch $A --time=03:00:00 --export=ALL,SCENE=$S,IMAGES=images_2,XYZ=q16,INDOOR=1,BUDGETS=$B $AN/budget-sweep-v3.sbatch;;
    counter|bonsai)          sbatch $A --time=03:00:00 --export=ALL,SCENE=$S,IMAGES=images_2,XYZ=q15,INDOOR=1,BUDGETS=$B $AN/budget-sweep-v3.sbatch;;
    drjohnson|playroom)      sbatch $A --time=03:00:00 --export=ALL,SCENE=$S,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/$S,BUDGETS=$B $AN/budget-sweep-v3.sbatch;;
    lego|chair|drums|ficus|hotdog|materials|mic|ship) sbatch $A --time=02:00:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=$B $AN/budget-sweep-blender-v1.sbatch;;
    *) echo "FAILED: capjob: unknown scene $S" >&2; return 1;;
  esac
}
CAPJOB30
cat > $AN/cap-pass2-v3.sbatch <<'PASS30'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=cappass2
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/zr2245/analysis/cappass2v3-%x-%j.out
# cap-pass2-v3.sbatch (2026-11-12, VERSION cappass2-v3): the retry step of the vertex-cap baseline for any of the 21 scenes.
# After the first cap run of scene S: if the file misses the target by more than 10%, one more training with
# CAP2 = CAP1 * (T - C) / (MB1 - C)  (C = fixed bytes of the size prior).  Needs via --export: S, CAP1, T, C.  Prints one line 'PASS2 ...'.
AN=/scratch/zr2245/analysis; source $AN/capjob-v1.sh
S=${S:?}; CAP1=${CAP1:?}; T=${T:?}; C=${C:?}
case $S in lego|chair|drums|ficus|hotdog|materials|mic|ship) TAB=$AN/g0-refresh-blender-$S.txt; TAG="TRAIN-AT-BUDGET-BLENDER";; *) TAB=$AN/g0-refresh-$S.txt; TAG="TRAIN-AT-BUDGET";; esac
MB1=$(grep "^$S budget $CAP1 $TAG " $TAB | tail -1 | sed -n 's/.*| delivered \([0-9.]*\) MB.*/\1/p')
[ -n "$MB1" ] || { echo "FAILED: no '$S budget $CAP1 $TAG' row in $TAB"; exit 1; }
OUT=$(python3 -c "
import sys
c1 = int(sys.argv[1]); mb = float(sys.argv[2]); t = float(sys.argv[3]); c = float(sys.argv[4]); e = mb / t - 1
if abs(e) <= 0.10: print('HIT %+.1f%% 0' % (100 * e))
elif mb <= c: print('BAD %+.1f%% 0' % (100 * e))
else: print('MISS %+.1f%% %d' % (100 * e, int(round(c1 * (t - c) / (mb - c) / 1000.0)) * 1000))
" $CAP1 $MB1 $T $C)
set -- $OUT; ST=$1; ERR=$2; CAP2=$3
case $ST in
  HIT)  echo "PASS2 $S cap1 $CAP1 delivered $MB1 MB (target $T MB, error $ERR): within 10%, no second training"; exit 0;;
  BAD)  echo "PASS2 $S cap1 $CAP1 delivered $MB1 MB (target $T MB, error $ERR): below the fixed bytes $C MB, no update possible"; exit 0;;
  MISS) echo "PASS2 $S cap1 $CAP1 delivered $MB1 MB (target $T MB, error $ERR, fixed bytes $C MB) -> cap2 $CAP2";;
  *)    echo "FAILED: PASS2 $S could not compute the update ('$OUT')"; exit 1;;
esac
[ "$CAP2" != "$CAP1" ] || { echo "cap2 equals cap1: no second training"; exit 0; }
J=$(capjob $S $CAP2 $S-cap2-b30) && echo "submitted $J ($S cap2 $CAP2)" || echo "FAILED: sbatch $S cap2 $CAP2"
PASS30
cat > $AN/make_batch30_files_v1.py <<'PYGEN30'
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
PYGEN30
python3 $AN/make_batch30_files_v1.py --train $AN/rdgrowth/train_rd_adapt_v9b_abl5.py --sbatch $AN/ablation-b24-r6.sbatch --sbatch_blender $AN/ablation-b24-r6-blender.sbatch --out $AN/batch30-build && cp $AN/batch30-build/train_rd_adapt_v9b_abl6.py $AN/rdgrowth/ && cp $AN/batch30-build/ablation-b24-r7.sbatch $AN/batch30-build/ablation-b24-r7-blender.sbatch $AN/ || echo "GENERATOR FAILED: stop here"
cat > $AN/submit-batch30-v1.sh <<'SUB30'
#!/usr/bin/env bash
# submit-batch30-v1.sh (2026-11-12): the overnight batch.  Everything at ONE target per dataset: 10 MB (13 captured scenes), 2.5 MB (8 objects).
#   FIN  : the final design of the controller (make_batch30_files_v1.py: growth aim 1.35 for every scene, predictive rounds with a 5% margin,
#          measured rounds at 24,000 / 25,500 / 27,000) on all 21 scenes -> abl24-final.  garden and lego start first; the other 19 start only
#          after those two have ended without error.
#   CAP  : MeshSplatting with a vertex cap from the leave-one-out size prior, on every scene not yet run (15) + the retry rule (one more training
#          on a miss > 10%) + the two retries still open (kitchen 900000, lego third training 130000)
#   R35  : fallback, third version without the late rounds, growth aim 1.35 (14 runs; 7 exist from the pilot)  -> abl24-ratio35
#   R    : fallback, third version, default growth aim (1.25 captured, 1.1 objects)                           -> abl24-ratio
# Queue order: FIN, then CAP, then R35, then R (nice values).  R35 and R start only after a pilot job of the third version ended without error.
# Usage: bash submit-batch30-v1.sh [GROUPS]   (default "FIN CAP R35 R")
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; GR=${1:-"FIN CAP R35 R"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/rdgrowth/train_rd_adapt_v9b_abl5.py 1078 3f1f876e372659b9dd8ad0a07610497d
chk $AN/rdgrowth/train_rd_adapt_v9b_abl6.py 1081 37d3ca515f0df262bdee538c8a0fe162
chk $AN/ablation-b24-r7.sbatch 217 e9c48dba072b122a671efffc27996380
chk $AN/ablation-b24-r7-blender.sbatch 219 6d731771709ee222175e858c2a56355d
chk $AN/ablation-b24-r6.sbatch 216 4625b51e1cb8f7086c228fda3ec69fc7
chk $AN/ablation-b24-r6-blender.sbatch 218 c7e8a016bcabf0391204c07936ceab15
chk $AN/capjob-v1.sh 17 5a42c14029c6bbb3db8abcf1e921f5ae
chk $AN/cap-pass2-v3.sbatch 32 d96578bca24ef217eae0929e34d46175
bash -n $AN/capjob-v1.sh && bash -n $AN/cap-pass2-v3.sbatch && python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl6.py && bash -n $AN/ablation-b24-r7.sbatch && bash -n $AN/ablation-b24-r7-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for S in room kitchen counter bonsai drjohnson playroom; do [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no --indoor original for $S"; exit 1; }; done
for S in garden bicycle stump flowers treehill truck train; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no original for $S"; exit 1; }; done
for S in lego chair drums ficus hotdog materials mic ship; do [ -f /scratch/zr2245/$S-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no $S-blender4-best original"; exit 1; }; done
echo "CHECK OK (groups: $GR)"; source $AN/capjob-v1.sh; n=0
cfg () { case $1 in room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; kitchen) echo "INDOOR=1,IMAGES=images_2,XYZ=q16";; counter|bonsai) echo "INDOOR=1,IMAGES=images_2,XYZ=q15";;
  garden|flowers) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; bicycle|stump|treehill) echo "INDOOR=0,IMAGES=images_4,XYZ=q16";; truck|train) echo "INDOOR=0,IMAGES=images,XYZ=q18,DATA=$D/tandt/$1";;
  drjohnson|playroom) echo "INDOOR=1,IMAGES=images,XYZ=q15,DATA=$D/db/$1";; esac; }
ctl () { # $1 scene  $2 variant  $3 sbatch generation (r6|r7)  $4.. extra sbatch options; sets LASTJ
  local S=$1 V=$2 G=$3; shift 3; local j
  case $S in lego|chair|drums|ficus|hotdog|materials|mic|ship) j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 "$@" --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS=2.5 --job-name=$S-$V-b30 $AN/ablation-b24-$G-blender.sbatch);;
    *) j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 "$@" --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS=10 --job-name=$S-$V-b30 $AN/ablation-b24-$G.sbatch);; esac
  LASTJ=$j; [ -n "$j" ] && { n=$((n+1)); echo "submitted $j  $S-$V-b30"; } || echo "FAILED: sbatch $S $V"; }
dep () { # $1 pilot job name  $2 result table  $3 row prefix -> prints the dependency option ('' when the pilot already finished); fails when the pilot is unknown
  local id=$(squeue -u $USER -h -n $1 -o %i | head -1)
  if [ -n "$id" ]; then echo "--dependency=afterok:$id"; return 0; fi
  grep -q "^$3 " $2 2>/dev/null && return 0; return 1; }
MIP="room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom"; OBJ="lego chair drums ficus hotdog materials mic ship"
for G in $GR; do case $G in
CAP) for X in room:1078000:2.15 counter:1083000:2.11 bonsai:1073000:2.17 bicycle:1080000:2.14 stump:1069000:2.15 flowers:1071000:2.05 treehill:1079000:2.09 train:1085000:2.12 drjohnson:1081000:2.11 drums:176000:0.89 ficus:171000:1.02 hotdog:177000:0.9 materials:176000:0.92 mic:174000:0.95 ship:177000:0.87; do
       S=${X%%:*}; R=${X#*:}; B=${R%%:*}; C=${R#*:}; case $S in drums|ficus|hotdog|materials|mic|ship) T=2.5;; *) T=10;; esac
       j=$(NICE=100 capjob $S $B $S-cap1-b30) && { n=$((n+1)); echo "submitted $j  $S-cap1-b30 (cap $B)"; sbatch --parsable --dependency=afterok:$j --export=ALL,S=$S,CAP1=$B,T=$T,C=$C --job-name=$S-pass2v3 $AN/cap-pass2-v3.sbatch >/dev/null; } || echo "FAILED: cap $S"; done
     j=$(NICE=100 capjob kitchen 900000 kitchen-cap2-b30) && { n=$((n+1)); echo "submitted $j  kitchen-cap2-b30 (cap 900000)"; }
     j=$(NICE=100 capjob lego 130000 lego-cap3-b30) && { n=$((n+1)); echo "submitted $j  lego-cap3-b30 (cap 130000)"; };;
FIN) ctl garden final r7; JG=$LASTJ; ctl lego final r7; JL=$LASTJ
     [ -n "$JG" ] && for S in $MIP; do [ $S = garden ] && continue; ctl $S final r7 --dependency=afterok:$JG; done
     [ -n "$JL" ] && for S in $OBJ; do [ $S = lego ] && continue; ctl $S final r7 --dependency=afterok:$JL; done;;
R|R35) V=ratio; NI=600; [ $G = R35 ] && { V=ratio35; NI=300; }
     if DC=$(dep garden-ratio35-b29 $AN/ablation-b24.txt "garden abl24-ratio35"); then for S in $MIP; do [ -e /scratch/zr2245/$S-abl24-$V-t10mb ] && { echo "exists, skipped: $S $V"; continue; }; ctl $S $V r6 --nice=$NI $DC; done
     else echo "NOT SUBMITTED: $V on captured scenes (pilot job garden-ratio35-b29 is neither in the queue nor finished with a result row)"; fi
     if DB=$(dep lego-ratio25-b29 $AN/ablation-b24-blender.txt "lego abl24-ratio25"); then for S in $OBJ; do [ -e /scratch/zr2245/$S-abl24b-$V-t2.5mb ] && { echo "exists, skipped: $S $V"; continue; }; ctl $S $V r6 --nice=$NI $DB; done
     else echo "NOT SUBMITTED: $V on objects (pilot job lego-ratio25-b29 is neither in the queue nor finished with a result row)"; fi;;
*) echo "FAILED: unknown group $G";;
esac; done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
SUB30
bash $AN/submit-batch30-v1.sh
```

Expected: three generator lines (`train_rd_adapt_v9b_abl6.py: 1081 lines md5 37d3ca515f0df262bdee538c8a0fe162`, `ablation-b24-r7.sbatch: 217 lines md5 e9c48dba072b122a671efffc27996380`, `ablation-b24-r7-blender.sbatch: 219 lines md5 6d731771709ee222175e858c2a56355d`), `CHECK OK (groups: FIN CAP R35 R)`, `submitted` lines, a few `exists, skipped` lines, and `72 GPU jobs submitted`.

To submit only some groups: `bash $AN/submit-batch30-v1.sh "FIN CAP"`.

## Watch the two FIN pilot jobs (rounds act from 16,000; the new part starts at 21,000)

```bash
for G in garden-abl24-final-t10mb lego-abl24b-final-t2.5mb; do echo "== $G"; python3 -c "
import json, sys
for l in open(sys.argv[1]):
    e = json.loads(l)
    if e.get('event') == 'prune': print('it %5d | file now %.3f x target, %s final %.3f x | wanted %.3f MB, freed %.3f MB, %6d pairs | V %d -> %d' % (e['iteration'], e['probe_ratio'], 'predicted' if e.get('trim_predictive', True) else 'measured ', e['probe_est_ratio'], e.get('trim_want_mb', 0), e.get('trim_planned_mb', 0), e.get('trim_goal', 0), e['vertices_before'], e['vertices_after']))
" /scratch/zr2245/$G/original/rd_log.jsonl 2>/dev/null; done
```

## Collect (in the morning, whatever has finished)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch30-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== controller rows"; grep -h "abl24-final\|abl24-ratio\|abl24-feas" $AN/ablation-b24.txt $AN/ablation-b24-blender.txt | sed -E 's/, val_select kept faces.*//; s/ \| phase-A probes.*//; s/, rounds \(it, probe MB.*//; s/, extra probes.*//'
  echo; echo "===== cap rows"; for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom; do grep -h "TRAIN-AT-BUDGET" $AN/g0-refresh-$S.txt | tail -5; done; for S in lego chair drums ficus hotdog materials mic ship; do grep -h "TRAIN-AT-BUDGET-BLENDER" $AN/g0-refresh-blender-$S.txt | tail -4; done
  echo; echo "===== retry launchers"; grep -h "PASS2\|FAILED\|submitted" $AN/cappass2v3-*.out 2>/dev/null
  echo; echo "===== cap job stages (time stamps)"; grep -h "^=== A:\|^=== B:\|exit 0" $AN/g0r*-*-b30-*.out 2>/dev/null | cut -c1-150
  echo; echo "===== failures"; grep -l "FAILED\|Traceback" $AN/abl24*-*-b30-*.out $AN/abl24*-*-b30-*.err $AN/g0r*-*-b30-*.out 2>/dev/null; for f in $(grep -l "FAILED" $AN/abl24*-*final-b30-*.out 2>/dev/null | head -3); do echo "--- $f"; grep -h -A 14 "FAILED" $f | head -30; done
  echo; echo "===== sacct"; sacct -X --starttime=now-2days --format=JobID,JobName%30,State,Elapsed,ExitCode | grep -E "b29|b30|pass2v3"
  echo; echo "===== queue"; squeue -u $USER -h -o "%.10i %.28j %.2t %.10M %R" | grep -E "b29|b30"; } > $O 2>&1; wc -l $O; echo $O
```
