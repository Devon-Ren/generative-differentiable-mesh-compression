AN=/scratch/zr2245/analysis; RES=yw523-l40s
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/generic-codecs-b22.sbatch 50 c1b75cb09fdc9eff4d9e533323b619e6; bash -n $AN/generic-codecs-b22.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for S in kitchen counter bonsai bicycle stump flowers treehill truck train; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || echo "WARNING: $S original missing at $S-vis-collapse (its rows will FAIL inside the job)"; done
scontrol show node gl030 | grep -oE "State=[^ ]*|AllocTRES=[^ ]*"; busy=$(squeue -u $USER -h -o %j | grep -c b22); [ "$busy" = 0 ] || { echo "CHECK FAILED: b22 jobs already queued"; exit 1; }
echo "CHECK OK"
sb () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 "$@") && echo "submitted $j" || echo "FAILED: sbatch $*"; }
sb --time=02:00:00 --export=ALL,SCENES="kitchen counter bonsai truck train" --job-name=b22-genc-a $AN/generic-codecs-b22.sbatch
sb --time=02:00:00 --export=ALL,SCENES="bicycle stump flowers treehill" --job-name=b22-genc-b $AN/generic-codecs-b22.sbatch
sleep 30; squeue -u $USER -o "%.10i %.16j %.2t %.10M %R" | grep -E "b22|JOBID"
