AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
grep -q "VERSION g0r-v3" $AN/budget-sweep-v3.sbatch || { echo "CHECK FAILED: sbatch version"; exit 1; }; bash -n $AN/budget-sweep-v3.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
echo "=== smoke (CPU, seconds): does --max_points survive --indoor in THIS upstream? ==="
python $AN/indoor_budget_check_v1.py || { echo "CHECK FAILED: the indoor preset overwrites max_points on this upstream; do NOT submit (needs the train_ckpt_v4 --override path)"; exit 1; }
grep -n "max_points" /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/arguments/__init__.py | head -3
ok=""; for S in room kitchen counter bonsai; do O=/scratch/zr2245/$S-indoor
  if [ -f $O/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then ok="$ok $S"; [ -f $O/footprint-sens.npz ] || echo "note: $S indoor original has no footprint-sens.npz (the job computes it)"; else echo "SKIP $S: no --indoor FP32 original at $O"; fi; done
echo "indoor G0 scenes with an original:$ok"
for S in room kitchen counter bonsai; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "g0r|adapt9|rd3i"); [ "$busy" = 0 ] || echo "note: $busy related jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0
g0i () { S=$1; X=$2; BUD=$3; case " $ok " in *" $S "*) ;; *) return;; esac
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=05:00:00 --export=ALL,SCENE=$S,IMAGES=images_2,XYZ=$X,INDOOR=1,BUDGETS="$BUD" --job-name=$S-g0ri $AN/budget-sweep-v3.sbatch) && { echo "submitted $S-g0ri $j  budgets $BUD"; k=$((k+1)); } || echo "FAILED: sbatch $S"; }
g0i room q14 "1215000 486000 220000"; g0i kitchen q16 "1349000 622000 272000"; g0i counter q15 "1162000 526000 245000"; g0i bonsai q15 "1328000 557000 290000"
echo "submitted $k of up to 4 indoor G0 jobs (~3.5-4 h each, 4 GPUs -> done in ~4 h; rows append to g0-refresh-<scene>.txt with '| --indoor')"
