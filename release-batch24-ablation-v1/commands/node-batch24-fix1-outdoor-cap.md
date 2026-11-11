# node-batch24-fix1 (2026-11-11): resubmit the 7 outdoor / T&T vertex-cap jobs with budget-sweep-v2.sbatch

Why: the 7 `*-g0r24` jobs for garden, bicycle, stump, flowers, treehill, truck, train failed after ~45 s with
`FAILED: this upstream's update_indoor() sets max_points; the CLI budget would be overwritten`. That check in budget-sweep-v3.sbatch is
written as `[ "$INDOOR" = "0" ] || grep ... && { FAILED }`, so with INDOOR=0 it always fails (shell precedence); v3 only ever worked for
indoor scenes, outdoor scenes were run with budget-sweep-v2 (batch19). My submit used v3 for all 11. Cost of the failed jobs: 7 x 1 min.
The four indoor cap jobs (room, kitchen, counter, bonsai) completed. This block resubmits the seven with v2, same budgets, public partition.

```bash
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
grep -q "VERSION g0r-v2" $AN/budget-sweep-v2.sbatch && bash -n $AN/budget-sweep-v2.sbatch || { echo "CHECK FAILED: v2 sbatch"; exit 1; }
for S in garden bicycle stump flowers treehill truck train; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no FP32 original for $S"; exit 1; }; done
squeue -u $USER -h -o "%j" | grep -q "g0r24" && echo "note: g0r24 jobs still in the queue (indoor ones?)"
echo "CHECK OK"; k=0
g () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=05:30:00 --export=ALL,SCENE=$1,IMAGES=$2,XYZ=$3,INDOOR=0,BUDGETS="$4"$5 --job-name=$1-g0r24v2 $AN/budget-sweep-v2.sbatch) && { echo "submitted $j $1 budgets $4"; k=$((k+1)); } || echo "FAILED: sbatch $1"; }
g garden images_4 q15 "2015000 936000 405000"
g bicycle images_4 q16 "2029000 986000 440000"
g stump images_4 q16 "2015000 983000 440000"
g flowers images_4 q15 "2068000 992000 447000"
g treehill images_4 q16 "2054000 1002000 449000"
g truck images q18 "1853000 929000 421000" ",DATA=$D/tandt/truck"
g train images q18 "2040000 975000 436000" ",DATA=$D/tandt/train"
echo "submitted $k of 7 (public partition, ~4.5 h each, rows append to g0-refresh-<scene>.txt)"
```
