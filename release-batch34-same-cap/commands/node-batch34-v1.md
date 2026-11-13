# node-batch34-v1 (2026-11-13): the same vertex cap on all 13 captured scenes (39 GPU jobs)

## Question

With exactly the same vertex cap, how far apart are the files of different scenes? Caps: 300,000 / 600,000 / 900,000, fixed before the runs.
All 13 scenes already have a run at a cap of 1,069,000-1,086,000 (files 8.3 to 11.6 MB).

## How the result will be read (fixed before the runs)

For each cap: the smallest and the largest file over the 13 scenes, and their ratio. If the ratio is below 1.10 at all three caps, the report
will say that one cap gives about one size on captured scenes.

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-batch34-v1.sh <<'SUB34'
#!/usr/bin/env bash
# submit-batch34-v1.sh (2026-11-13): the SAME vertex cap on all 13 captured scenes: how far apart are the files?
#   MeshSplatting with --max_points = 300,000 / 600,000 / 900,000, fine-tuning and our codec as in every other cap run.  39 GPU jobs.
#   The three caps are fixed here, before any of these runs.  (All 13 scenes already have a run at a cap of 1,069,000-1,086,000.)
# Usage: bash submit-batch34-v1.sh [CAPS]   (default "300000 600000 900000")
AN=/scratch/zr2245/analysis; CAPS=${1:-"300000 600000 900000"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/capjob-v1.sh 17 5a42c14029c6bbb3db8abcf1e921f5ae
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
echo "CHECK OK (caps: $CAPS)"; source $AN/capjob-v1.sh; n=0
for B in $CAPS; do
  case $B in 300000|600000|900000) ;; *) echo "FAILED: cap $B is not one of 300000 600000 900000"; continue;; esac
  for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom; do
    j=$(capjob $S $B $S-cap$B-b34) && { n=$((n+1)); echo "submitted $j  $S cap $B"; } || echo "FAILED: cap $S $B"
  done
done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
SUB34
bash $AN/submit-batch34-v1.sh
```

Expected: `CHECK OK (caps: 300000 600000 900000)`, 39 `submitted` lines, `39 GPU jobs submitted`. One cap only: `bash $AN/submit-batch34-v1.sh 900000`.

## Collect (whatever has finished)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch34-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== cap rows"; for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom; do grep -hE "budget (300000|600000|900000) TRAIN-AT-BUDGET" $AN/g0-refresh-$S.txt; done
  echo; echo "===== cap job stages (time stamps)"; grep -h "^=== A:\|^=== B:\|exit 0" $AN/g0r-*-b34-*.out 2>/dev/null | cut -c1-150
  echo; echo "===== failures"; grep -l "FAILED\|Traceback" $AN/g0r-*-b34-*.out $AN/g0r-*-b34-*.err 2>/dev/null
  echo; echo "===== sacct"; sacct -X --starttime=now-2days --format=JobID,JobName%30,State,Elapsed,ExitCode | grep -E "b34"
  echo; echo "===== queue"; squeue -u $USER -h -o "%.10i %.28j %.2t %.10M %R" | grep -E "b34"; } > $O 2>&1; wc -l $O; echo $O
```
