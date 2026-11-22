#!/usr/bin/env bash
# F3-1: synthetic Sponza photos + GT depth. GPU node. Smoke (~2 min) first, full run only if smoke passes.
source "$(dirname "$0")/env.sh"; main_env; cd $F3
echo "== F3-1 smoke"
python scripts/run_f3.py --kind synth --data $DATA/sponza-synth-v1 --depth gt --out $OUT/f3-1-smoke --smoke > $OUT/f3-1-smoke.log 2>&1
grep -q "== DONE" $OUT/f3-1-smoke.log || { fail "F3-1 smoke" $OUT/f3-1-smoke.log; return 1 2>/dev/null || exit 1; }
grep "^\[" $OUT/f3-1-smoke.log | tail -n 12
echo "== F3-1 full (log: $OUT/f3-1.log)"
python scripts/run_f3.py --kind synth --data $DATA/sponza-synth-v1 --depth gt --out $OUT/f3-1 > $OUT/f3-1.log 2>&1
grep -q "== DONE" $OUT/f3-1.log || { fail "F3-1 full" $OUT/f3-1.log; return 1 2>/dev/null || exit 1; }
grep "^\[5\]\|^\[2\] surface\|^\[3\]" $OUT/f3-1.log
