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
