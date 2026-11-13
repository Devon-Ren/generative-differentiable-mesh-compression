# node-batch32-v1 (2026-11-13): how the file responds to the control parameter (41 GPU jobs, 49 runs, about 24 GPU-h)

## Question

Is the vertex cap a usable control for the file size? A usable control responds predictably: a small change gives a small change, in the expected direction.

| group | what | runs |
|---|---|---|
| CAP | MeshSplatting with the cap on a regular 1% grid from -5% to +5% around the prior-chosen cap: garden (1,105,000), room (1,078,000), lego (183,000). The centre runs exist. | 30 |
| OURS | BudgetMesh (final design) with the target at -5 / -3 / -1 / +1 / +3 / +5 %: garden (10 MB), lego (2.5 MB). The centre runs exist. | 12 (4 jobs) |
| X | the cap chosen without statistics of the same kind: the captured-scene prior applied to the other 7 objects (cap 53,000, as for lego, which landed at -41.8%) | 7 |

The grids are fixed here, before any of these runs. The earlier dense caps on garden were placed next to caps that had already behaved oddly; this grid is not.

## How the result will be read (fixed before the runs)

For each scene and each method, over the 11 grid points:
- the largest change of the file between two neighbouring grid points (1% apart for the cap, 2% for the target);
- how many neighbouring steps move the file in the wrong direction;
- the span of the file over the whole window (a cap window of 10% should give a file span of about 8%; a target window of 10% should give 10%);
- the largest distance of a point from the straight line through the grid.

If the cap curves turn out smooth and monotone, the report will say so and the "jumps" of the earlier runs are local accidents.

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-batch32-v1.sh <<'SUB32'
#!/usr/bin/env bash
# submit-batch32-v1.sh (2026-11-13): how does the file respond to the control parameter?  (41 GPU jobs, 49 runs, about 24 GPU-h)
#   CAP  : MeshSplatting with the vertex cap on a regular 1% grid from -5% to +5% around the prior-chosen cap (the centre run exists):
#          garden (1,105,000), room (1,078,000), lego (183,000).  30 runs.  The grid is fixed here, before any of these runs.
#   OURS : BudgetMesh (final design) with the target on the same kind of grid, -5 / -3 / -1 / +1 / +3 / +5 %: garden (10 MB), lego (2.5 MB).
#          12 runs in 4 jobs.  The centre runs exist (garden 10.045 MB, lego 2.490 MB).
#   X    : the cap chosen WITHOUT statistics of the same kind: the captured-scene prior applied to the other 7 objects (cap 53,000 as for lego).  7 runs.
# Usage: bash submit-batch32-v1.sh [GROUPS]   (default "CAP OURS X")
AN=/scratch/zr2245/analysis; GR=${1:-"CAP OURS X"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/capjob-v1.sh 17 5a42c14029c6bbb3db8abcf1e921f5ae
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/rdgrowth/train_rd_adapt_v9b_abl6.py 1081 37d3ca515f0df262bdee538c8a0fe162
chk $AN/ablation-b24-r7.sbatch 217 e9c48dba072b122a671efffc27996380
chk $AN/ablation-b24-r7-blender.sbatch 219 6d731771709ee222175e858c2a56355d
echo "CHECK OK (groups: $GR)"; source $AN/capjob-v1.sh; n=0
cap () { local j; j=$(capjob $1 $2 $1-cap$2-b32) && { n=$((n+1)); echo "submitted $j  $1 cap $2"; } || echo "FAILED: cap $1 $2"; }
for G in $GR; do case $G in
CAP) for B in 1050000 1061000 1072000 1083000 1094000 1116000 1127000 1138000 1149000 1160000; do cap garden $B; done
     for B in 1024000 1035000 1046000 1056000 1067000 1089000 1100000 1110000 1121000 1132000; do cap room $B; done
     for B in 174000 176000 178000 179000 181000 185000 187000 188000 190000 192000; do cap lego $B; done;;
OURS) for TG in "9.5 9.7 9.9" "10.1 10.3 10.5"; do j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,SMOKE=0,SELFCHECK=0,ABL=final,TARGETS="$TG" --job-name=garden-final-b32 $AN/ablation-b24-r7.sbatch) && { n=$((n+1)); echo "submitted $j  garden final targets $TG"; }; done
     for TG in "2.375 2.425 2.475" "2.525 2.575 2.625"; do j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=final,TARGETS="$TG" --job-name=lego-final-b32 $AN/ablation-b24-r7-blender.sbatch) && { n=$((n+1)); echo "submitted $j  lego final targets $TG"; }; done;;
X)   for S in chair drums ficus hotdog materials mic ship; do cap $S 53000; done;;
*) echo "FAILED: unknown group $G";;
esac; done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
SUB32
bash $AN/submit-batch32-v1.sh
```

Expected: `CHECK OK (groups: CAP OURS X)`, 41 `submitted` lines, `41 GPU jobs submitted`. To submit only some groups: `bash $AN/submit-batch32-v1.sh "CAP OURS"`.

## Collect (whatever has finished)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch32-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== cap rows"; for S in garden room; do grep -h "TRAIN-AT-BUDGET" $AN/g0-refresh-$S.txt | tail -24; done; for S in lego chair drums ficus hotdog materials mic ship; do grep -h "TRAIN-AT-BUDGET-BLENDER" $AN/g0-refresh-blender-$S.txt | tail -16; done
  echo; echo "===== controller rows"; grep -h "abl24-final\|abl24-hold" $AN/ablation-b24.txt $AN/ablation-b24-blender.txt | sed -E 's/, val_select kept faces.*//; s/ \| phase-A probes.*//; s/, rounds \(it, probe MB.*//; s/, extra probes.*//'
  echo; echo "===== failures"; grep -l "FAILED\|Traceback" $AN/*-b32-*.out $AN/*-b32-*.err $AN/abl24*-*-b31-*.out 2>/dev/null
  echo; echo "===== sacct"; sacct -X --starttime=now-2days --format=JobID,JobName%30,State,Elapsed,ExitCode | grep -E "b31|b32"
  echo; echo "===== queue"; squeue -u $USER -h -o "%.10i %.28j %.2t %.10M %R" | grep -E "b31|b32"; } > $O 2>&1; wc -l $O; echo $O
```

This collect also contains the rows of the hold batch (batch31), so one file is enough.
