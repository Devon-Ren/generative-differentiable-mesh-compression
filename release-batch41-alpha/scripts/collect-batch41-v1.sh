#!/usr/bin/env bash
# collect-batch41-v1.sh: the result rows of the growth-aim runs (and the two hold reference rows).
AN=/scratch/zr2245/analysis
grep -h -E "^(garden|room) abl24-(hold|holda100|holda120|holda150) \| target 10 MB" $AN/ablation-b24.txt
for f in $(ls -t $AN/abl24-*-b41-*.out 2>/dev/null); do tail -1 $f | grep -q "exit 0" || echo "not finished or failed: $f : $(tail -1 $f | cut -c1-160)"; done
