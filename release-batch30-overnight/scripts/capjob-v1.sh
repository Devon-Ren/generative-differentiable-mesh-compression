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
