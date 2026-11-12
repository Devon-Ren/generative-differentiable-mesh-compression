#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=cappass2
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/zr2245/analysis/cappass2-%x-%j.out
# cap-pass2-v1.sbatch (2026-11-12, VERSION cappass2-v1): second pass of the "measure, then retrain" vertex-cap baseline.  Runs after the first cap
# job of scene S: reads the delivered file size of the cap-CAP1 run from the result table, rescales the cap to the target (CAP2 = CAP1 * T / MB1,
# rounded to 1000) and submits the same cap job with CAP2.  Needs via --export: S, CAP1, T.  Prints one line 'PASS2 S cap1 mb1 -> cap2'.
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
S=${S:?}; CAP1=${CAP1:?}; T=${T:?}
case $S in lego) TAB=$AN/g0-refresh-blender-$S.txt; TAG="TRAIN-AT-BUDGET-BLENDER";; *) TAB=$AN/g0-refresh-$S.txt; TAG="TRAIN-AT-BUDGET";; esac
MB1=$(grep "^$S budget $CAP1 $TAG " $TAB | tail -1 | sed -n 's/.*| delivered \([0-9.]*\) MB.*/\1/p')
[ -n "$MB1" ] || { echo "FAILED: no '$S budget $CAP1 $TAG' row in $TAB"; exit 1; }
CAP2=$(python3 -c "import sys; c=int(sys.argv[1]); mb=float(sys.argv[2]); t=float(sys.argv[3]); print(int(round(c*t/mb/1000.0))*1000)" $CAP1 $MB1 $T)
echo "PASS2 $S cap1 $CAP1 delivered $MB1 MB (target $T MB, error $(python3 -c "print('%+.1f%%' % (100*($MB1/$T-1)))")) -> cap2 $CAP2"
[ "$CAP2" != "$CAP1" ] || { echo "cap2 equals cap1: no second pass needed"; exit 0; }
case $S in
  garden)   sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=02:00:00 --export=ALL,SCENE=garden,IMAGES=images_4,XYZ=q15,INDOOR=0,BUDGETS=$CAP2 --job-name=garden-cap2-b27 $AN/budget-sweep-v2.sbatch;;
  truck)    sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=02:00:00 --export=ALL,SCENE=truck,IMAGES=images,XYZ=q18,INDOOR=0,DATA=$D/tandt/truck,BUDGETS=$CAP2 --job-name=truck-cap2-b27 $AN/budget-sweep-v2.sbatch;;
  playroom) sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS=$CAP2 --job-name=playroom-cap2-b27 $AN/budget-sweep-v3.sbatch;;
  lego)     sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=02:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=$CAP2 --job-name=lego-cap2-b27 $AN/budget-sweep-blender-v1.sbatch;;
  *) echo "FAILED: unknown scene $S"; exit 1;;
esac
