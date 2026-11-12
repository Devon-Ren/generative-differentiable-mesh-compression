#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=cappass2
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/zr2245/analysis/cappass2v3-%x-%j.out
# cap-pass2-v3.sbatch (2026-11-12, VERSION cappass2-v3): the retry step of the vertex-cap baseline for any of the 21 scenes.
# After the first cap run of scene S: if the file misses the target by more than 10%, one more training with
# CAP2 = CAP1 * (T - C) / (MB1 - C)  (C = fixed bytes of the size prior).  Needs via --export: S, CAP1, T, C.  Prints one line 'PASS2 ...'.
AN=/scratch/zr2245/analysis; source $AN/capjob-v1.sh
S=${S:?}; CAP1=${CAP1:?}; T=${T:?}; C=${C:?}
case $S in lego|chair|drums|ficus|hotdog|materials|mic|ship) TAB=$AN/g0-refresh-blender-$S.txt; TAG="TRAIN-AT-BUDGET-BLENDER";; *) TAB=$AN/g0-refresh-$S.txt; TAG="TRAIN-AT-BUDGET";; esac
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
  BAD)  echo "PASS2 $S cap1 $CAP1 delivered $MB1 MB (target $T MB, error $ERR): below the fixed bytes $C MB, no update possible"; exit 0;;
  MISS) echo "PASS2 $S cap1 $CAP1 delivered $MB1 MB (target $T MB, error $ERR, fixed bytes $C MB) -> cap2 $CAP2";;
  *)    echo "FAILED: PASS2 $S could not compute the update ('$OUT')"; exit 1;;
esac
[ "$CAP2" != "$CAP1" ] || { echo "cap2 equals cap1: no second training"; exit 0; }
J=$(capjob $S $CAP2 $S-cap2-b30) && echo "submitted $J ($S cap2 $CAP2)" || echo "FAILED: sbatch $S cap2 $CAP2"
