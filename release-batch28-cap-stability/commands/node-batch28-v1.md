# node-batch28-v1 (2026-11-12): how stable is a vertex cap + fair cap baselines + one growth aim (fairshare, 11 GPU runs, up to 3 retries, about 15-20 GPU-h)

| group | question | runs |
|---|---|---|
| A | Does a vertex cap give a repeatable file size? Caps within 0.1% of caps already run: garden 1,104,000 / 1,106,000 (next to 1,105,000) and 1,074,000 (next to 1,073,000); playroom 1,080,000 / 1,082,000 (next to 1,081,000). BudgetMesh's counterpart exists: garden 10 MB with three seeds landed at 10.21 / 10.16 / 10.16 MB. | 5 |
| B | Fair cap baseline on objects: cap from the NeRF-Synthetic size prior with the object left out (lego 183,000, chair 178,000; target 2.5 MB). A second training is submitted only if the first file misses by more than 10%, with the fixed bytes removed before rescaling. | 2 (+2) |
| C | Cap baseline on the scene where BudgetMesh is worst: kitchen at 10 MB, cap 1,086,000 from the Mip-360 / T&T prior with kitchen left out, same retry rule. | 1 (+1) |
| D | One growth aim for everything: BudgetMesh on lego, hotdog, mic at 2.5 MB with aim 1.25 (the captured-scene value) instead of 1.1. Rows are tagged `abl24-repeat-s0`. | 3 |

To run only some groups: `bash $AN/submit-batch28-v1.sh "A B"`.

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/cap-pass2-v2.sbatch <<'SBP2V2'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=cappass2
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/zr2245/analysis/cappass2v2-%x-%j.out
# cap-pass2-v2.sbatch (2026-11-12, VERSION cappass2-v2): the retry step of the vertex-cap baseline, with the two changes the second review asked for:
#   (1) CONDITIONAL: a second training is submitted only when the first file misses the target by more than 10%;
#   (2) AFFINE update: the fixed bytes C of the size prior (dictionaries, header) are removed before rescaling, CAP2 = CAP1 * (T - C) / (MB1 - C).
# Needs via --export: S, CAP1, T, C.  Prints one line 'PASS2 ...'.  Scenes: kitchen, lego, chair.
AN=/scratch/zr2245/analysis
S=${S:?}; CAP1=${CAP1:?}; T=${T:?}; C=${C:?}
case $S in lego|chair) TAB=$AN/g0-refresh-blender-$S.txt; TAG="TRAIN-AT-BUDGET-BLENDER";; kitchen) TAB=$AN/g0-refresh-$S.txt; TAG="TRAIN-AT-BUDGET";; *) echo "FAILED: unknown scene $S"; exit 1;; esac
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
  BAD)  echo "FAILED: PASS2 $S cap1 $CAP1 delivered $MB1 MB is not above the fixed bytes $C MB"; exit 1;;
  MISS) echo "PASS2 $S cap1 $CAP1 delivered $MB1 MB (target $T MB, error $ERR, fixed bytes $C MB) -> cap2 $CAP2";;
  *)    echo "FAILED: PASS2 $S could not compute the update ('$OUT')"; exit 1;;
esac
[ "$CAP2" != "$CAP1" ] || { echo "cap2 equals cap1: no second training"; exit 0; }
case $S in
  kitchen)    sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=03:00:00 --export=ALL,SCENE=kitchen,IMAGES=images_2,XYZ=q16,INDOOR=1,BUDGETS=$CAP2 --job-name=kitchen-cap2-b28 $AN/budget-sweep-v3.sbatch;;
  lego|chair) sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=02:00:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=$CAP2 --job-name=$S-cap2-b28 $AN/budget-sweep-blender-v1.sbatch;;
esac
SBP2V2
cat > $AN/submit-batch28-v1.sh <<'SUB28'
#!/usr/bin/env bash
# submit-batch28-v1.sh (2026-11-12): how stable is a vertex cap, and fair cap baselines (public partition / fairshare; 11 GPU runs now, up to 3 retries).
#   A  cap stability   : caps within 0.1% of a cap that was already run (garden 1,105,000 and 1,073,000; playroom 1,081,000).  5 runs.
#   B  fair object cap : lego 183,000 and chair 178,000 at 2.5 MB (cap from the NeRF-Synthetic size prior, the object left out) + conditional retry.  2 runs.
#   C  indoor cap      : kitchen 1,086,000 at 10 MB (cap from the Mip-360 / T&T size prior, kitchen left out) + conditional retry.  1 run.
#   D  one growth aim  : BudgetMesh on lego, hotdog, mic at 2.5 MB with the captured-scene aim 1.25 instead of 1.1 (seed 0, rows tagged abl24-repeat-s0).  3 runs.
# Usage: bash submit-batch28-v1.sh [GROUPS]   (default "A B C D")
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; GROUPS_=${1:-"A B C D"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/ablation-b24-r3-blender.sbatch 206 69614021a3aaf74cc7f999697ed8be54
chk $AN/rdgrowth/train_rd_adapt_v9b_abl2.py 938 75e92ffc397018fe472fc97801d0b9f9
chk $AN/cap-pass2-v2.sbatch 36 57ceb08062ae905438f3a607963cc428
bash -n $AN/cap-pass2-v2.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for S in kitchen playroom; do [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no --indoor original for $S"; exit 1; }; done
[ -f /scratch/zr2245/garden-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no garden original"; exit 1; }
for S in lego chair; do [ -f /scratch/zr2245/$S-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no $S-blender4-best original"; exit 1; }; done
for G in garden-budget-1104000 garden-budget-1106000 garden-budget-1074000 playroom-budget-1080000 playroom-budget-1082000 lego-budget-183000 chair-budget-178000 kitchen-budget-1086000 lego-abl24b-repeat0-t2.5mb hotdog-abl24b-repeat0-t2.5mb mic-abl24b-repeat0-t2.5mb; do [ -e /scratch/zr2245/$G ] && echo "note: /scratch/zr2245/$G already exists (finished stages are skipped)"; done
echo "CHECK OK (groups: $GROUPS_)"; k=0
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})" >&2; echo $j; } || { echo "FAILED: sbatch $*" >&2; echo ""; }; }
p2 () { [ -n "$1" ] && sbatch --parsable --dependency=afterok:$1 --export=ALL,S=$2,CAP1=$3,T=$4,C=$5 --job-name=$2-pass2v2 $AN/cap-pass2-v2.sbatch >/dev/null && echo "  retry launcher queued for $2 (after $1)"; }
for G in $GROUPS_; do case $G in
A) for B in 1104000 1106000 1074000; do sb --time=02:00:00 --export=ALL,SCENE=garden,IMAGES=images_4,XYZ=q15,INDOOR=0,BUDGETS=$B --job-name=garden-cap$B-b28 $AN/budget-sweep-v2.sbatch >/dev/null; done
   for B in 1080000 1082000; do sb --time=02:30:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS=$B --job-name=playroom-cap$B-b28 $AN/budget-sweep-v3.sbatch >/dev/null; done;;
B) j=$(sb --time=02:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=183000 --job-name=lego-cap1-b28 $AN/budget-sweep-blender-v1.sbatch); p2 "$j" lego 183000 2.5 0.95
   j=$(sb --time=02:00:00 --export=ALL,SCENE=chair,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=178000 --job-name=chair-cap1-b28 $AN/budget-sweep-blender-v1.sbatch); p2 "$j" chair 178000 2.5 0.88;;
C) j=$(sb --time=03:00:00 --export=ALL,SCENE=kitchen,IMAGES=images_2,XYZ=q16,INDOOR=1,BUDGETS=1086000 --job-name=kitchen-cap1-b28 $AN/budget-sweep-v3.sbatch); p2 "$j" kitchen 1086000 10 2.08;;
D) for S in lego hotdog mic; do sb --time=02:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=repeat,TARGETS=2.5,SEED=0,OVER=1.25 --job-name=$S-aim125-b28 $AN/ablation-b24-r3-blender.sbatch >/dev/null; done;;
*) echo "FAILED: unknown group $G";;
esac; done
squeue -u $USER -o "%.10i %.26j %.2t %.10M %R" | grep -E "b28|pass2v2|JOBID"
SUB28
bash $AN/submit-batch28-v1.sh
```

Expected: `CHECK OK (groups: A B C D)`, 11 `submitted` lines, 3 `retry launcher queued` lines, then the queue.

## Collect (when `squeue --me | grep -c "b28\|pass2v2"` is 0)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch28-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== retry launchers"; grep -h "PASS2\|FAILED" $AN/cappass2v2-*.out; echo; echo "===== cap rows"; for S in garden playroom kitchen; do grep -h "TRAIN-AT-BUDGET" $AN/g0-refresh-$S.txt | tail -6; done; for S in lego chair; do grep -h "TRAIN-AT-BUDGET-BLENDER" $AN/g0-refresh-blender-$S.txt | tail -3; done; echo; echo "===== aim 1.25 rows"; grep -h "abl24-repeat-s0" $AN/ablation-b24-blender.txt; echo; echo "===== vertex count through training (cap runs)"; for G in garden-budget-1104000 garden-budget-1105000 garden-budget-1106000 garden-budget-1073000 garden-budget-1074000 playroom-budget-1080000 playroom-budget-1081000 playroom-budget-1082000 playroom-budget-986000; do echo "--- $G"; grep -ihE "number of (points|vertices|triangles)|nb (points|vertices|triangles)|max_points|prun|clean" /scratch/zr2245/$G/train.log 2>/dev/null | tail -25 | cut -c1-200; done; echo; echo "===== sacct"; sacct -X --starttime=now-3days --format=JobID,JobName%28,State,Elapsed,ExitCode | grep -E "b28|pass2v2"; } > $O 2>&1; wc -l $O; echo $O
```

## How the results will be read (fixed before the runs)

- A: for each group of caps within 0.1% of each other, the spread of the delivered file size and of vertices / cap, next to BudgetMesh's spread over three seeds on garden (0.6%). If the cap files agree within about 2%, the cap is repeatable and the 0.80 / 0.94 difference is a property of those cap values; if they scatter, the cap does not fix the size. Either outcome is reported.
- B, C: size error after the first training and after the retry, trainings used, next to BudgetMesh's single run (lego 2.50 MB, chair and kitchen from Table 2 runs).
- D: size error with aim 1.25 against the aim 1.1 runs (lego 2.50 MB; hotdog, mic from batch24). If all three stay within 10%, the paper states one aim for all datasets; if not, it keeps one aim per capture class and reports these runs.
