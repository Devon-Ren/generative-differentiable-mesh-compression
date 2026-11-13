#!/usr/bin/env bash
# backup_hpc_runs.sh: pack the live code, the result tables and the logs of the hold runs into one tar on HPC.  Reads only; writes one tar.
# Usage: bash backup_hpc_runs.sh [WITH_ASSETS]     WITH_ASSETS=1 also packs the encoded files (*.mslb) of the hold runs
ROOT=${ROOT:-/scratch/zr2245}; AN=${AN:-$ROOT/analysis}; PKG=${PKG:-$AN/vis_collapse-v12}; STAGE2=${STAGE2:-$ROOT/shared-collapse-stage2-v2}; WITH=${1:-0}
OUT=${OUT:-$ROOT/budgetmesh-hpc-backup-$(date +%Y-%m-%d-%H%M).tar.gz}; L=$(mktemp)
# code and tables (files under 50 MB only)
for d in $AN/rdgrowth $AN/blender $PKG $STAGE2; do [ -d $d ] && find $d -type f -size -50M ! -name "*.pt" ! -name "*.so" ! -path "*/__pycache__/*" >> $L; done
find $AN -maxdepth 1 -type f \( -name "*.sbatch" -o -name "*.sh" -o -name "*.py" -o -name "*.txt" -o -name "*.md" \) -size -50M >> $L
# hold runs: logs and summaries
RUNS=$(find $ROOT -maxdepth 1 -type d -name "*-abl24*-hold-t*mb" | sort); n=$(echo "$RUNS" | grep -c .)
for G in $RUNS; do
  for f in $G/train.log $G/level.log $G/original/rd_log.jsonl $G/original/rd_summary.json $G/original/cfg_args; do [ -f $f ] && echo $f >> $L; done
  [ -d $G/axisB ] && find $G/axisB -type f -name "*.json" -size -5M >> $L
done
A=$(mktemp); for G in $RUNS; do find $G -type f -name "*.mslb" >> $A; done
na=$(grep -c . $A); sa=$(cat $A | xargs -r du -cb 2>/dev/null | tail -1 | cut -f1); echo "hold runs found: $n | encoded files (*.mslb): $na, $(( ${sa:-0} / 1000000 )) MB in total"
[ "$WITH" = "1" ] && cat $A >> $L
echo "files to pack: $(grep -c . $L)"; tar -czf $OUT --ignore-failed-read -T $L 2>/dev/null; ls -la $OUT; rm -f $L $A
[ "$WITH" = "1" ] || echo "encoded files were NOT packed; to include them: bash backup_hpc_runs.sh 1"
