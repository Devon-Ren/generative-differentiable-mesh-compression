# node-batch37-v1 (2026-11-14): BudgetMesh (hold, code of batch31 unchanged) at 7.5 and 15 MB on the 13 captured scenes, 26 GPU jobs

Why: three rate points per scene are few for the RD curve and for the standard Bjontegaard fit (cubic, four points).

## One paste

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-batch37-v1.sh <<'SUB37'
#!/usr/bin/env bash
# submit-batch37-v1.sh (2026-11-14): two more rate points of BudgetMesh (hold controller, unchanged code of batch31): 7.5 and 15 MB on the
# 13 captured scenes.  One run per job so that short jobs can be backfilled.  26 GPU jobs.
# Usage: bash submit-batch37-v1.sh [TARGETS]   (default "7.5 15")
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; TG=${1:-"7.5 15"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl7.py 1083 1eeb64ab5032e903f69ad1a21466d99f
chk $AN/ablation-b24-r8.sbatch 218 82259b73557bfcdb760648f42fad08cc
echo "CHECK OK (targets: $TG)"; n=0
cfg () { case $1 in room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; kitchen) echo "INDOOR=1,IMAGES=images_2,XYZ=q16";; counter|bonsai) echo "INDOOR=1,IMAGES=images_2,XYZ=q15";;
  garden|flowers) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; bicycle|stump|treehill) echo "INDOOR=0,IMAGES=images_4,XYZ=q16";; truck|train) echo "INDOOR=0,IMAGES=images,XYZ=q18,DATA=$D/tandt/$1";;
  drjohnson|playroom) echo "INDOOR=1,IMAGES=images,XYZ=q15,DATA=$D/db/$1";; esac; }
for T in $TG; do
  case $T in 7.5) TL=01:30:00;; 15) TL=02:00:00;; *) echo "FAILED: target $T is not 7.5 or 15"; continue;; esac
  for S in garden room truck kitchen counter bonsai bicycle stump flowers treehill train drjohnson playroom; do
    [ -e /scratch/zr2245/$S-abl24-hold-t${T}mb ] && { echo "exists, skipped: $S hold $T"; continue; }
    j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=$TL --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=hold,TARGETS="$T" --job-name=$S-hold$T-b37 $AN/ablation-b24-r8.sbatch) && { n=$((n+1)); echo "submitted $j  $S hold $T MB"; } || echo "FAILED: sbatch $S $T"
  done
done
echo "$n GPU jobs submitted"
SUB37
bash $AN/submit-batch37-v1.sh
```

## Collect

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch37-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== controller rows"; grep -h "abl24-hold |" $AN/ablation-b24.txt | grep -E "target (7.5|15) MB" | sed -E 's/, val_select kept faces.*//; s/ \| phase-A probes.*//; s/, rounds \(it, probe MB.*//; s/, extra probes.*//'
  echo; echo "===== failures"; grep -l "FAILED\|Traceback" $AN/abl24*-*-b37-*.out $AN/abl24*-*-b37-*.err 2>/dev/null
  echo; echo "===== sacct"; sacct -X --starttime=now-4days --format=JobID,JobName%30,State,Elapsed,ExitCode | grep -E "b37"
  echo; echo "===== queue"; squeue -u $USER -h -o "%.10i %.28j %.2t %.10M %R" | grep -E "b37"; } > $O 2>&1; wc -l $O; echo $O
```
