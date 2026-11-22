#!/usr/bin/env bash
# F3-2: same synthetic photos, depth from DA3 (pose-conditioned). GPU node, no internet needed after install_login.sh.
source "$(dirname "$0")/env.sh"; cd $F3
echo "== F3-2 DA3 depth (smoke: 4 views)"
da3_env; export HF_HUB_OFFLINE=1
python scripts/depth_da3.py --kind synth --data $DATA/sponza-synth-v1 --out $OUT/da3-sponza-smoke.npz --limit 4 > $OUT/da3-sponza-smoke.log 2>&1
grep -q "DONE" $OUT/da3-sponza-smoke.log || { fail "DA3 smoke" $OUT/da3-sponza-smoke.log; return 1 2>/dev/null || exit 1; }
grep "scale vs sparse" $OUT/da3-sponza-smoke.log
python scripts/depth_da3.py --kind synth --data $DATA/sponza-synth-v1 --out $OUT/da3-sponza.npz > $OUT/da3-sponza.log 2>&1
grep -q "DONE" $OUT/da3-sponza.log || { fail "DA3 full" $OUT/da3-sponza.log; return 1 2>/dev/null || exit 1; }
grep "scale vs sparse\|DONE" $OUT/da3-sponza.log
echo "== F3-2 reconstruction (log: $OUT/f3-2.log)"
main_env
python scripts/run_f3.py --kind synth --data $DATA/sponza-synth-v1 --depth $OUT/da3-sponza.npz --out $OUT/f3-2 > $OUT/f3-2.log 2>&1
grep -q "== DONE" $OUT/f3-2.log || { fail "F3-2" $OUT/f3-2.log; return 1 2>/dev/null || exit 1; }
grep "^\[5\]\|^\[2\] surface\|^\[3\]" $OUT/f3-2.log
