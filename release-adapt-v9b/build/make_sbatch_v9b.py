"""make_sbatch_v9b.py: build adapt-v9b.sbatch / adapt-v9b-blender.sbatch from the v9 sbatch files by exact-count replacements.
HPC names: Mip-360/T&T jobs 'adapt9x', run dirs $SCENE-adapt9x-t${T}mb, table adapt-v9b.txt, VERSION v9x-r1;
Blender jobs 'adapt9xb', run dirs $SCENE-adapt9xb-t${T}mb (+ -best), table adapt-v9b-blender.txt, VERSION v9xb-r1
('adapt9b' is taken by the v9 Blender runs)."""
import sys
which, src_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'[{which}] anchor found {c}x (expected {count}): {old[:100]!r}'
    s = s.replace(old, new); n_rep += 1


COMMON = ("# v9b = train_rd_adapt_v9b.py (v9 + ONE line: the phase-A price is kept when the plan asks for zero splits at the last interval, so phase B\n"
          "# starts from the last real rank price instead of lam_max and the lambda-prune is progressive again) + the ASYMMETRIC AIM: OVER default {over}\n"
          "# (phase A aims at target x {over}; batch12 decomposed the Mip-360 miss into cleaning share 0.92-0.94 x delivered topology 3.7-4.4 B/V vs prior 5\n"
          "# x soup b_v over-read 3-9 %, none observable before the 11k Delaunay, so the error is put on the side phase B can trim) + PRUNE default\n"
          "# 12500 ... 19500 (15 rounds; the 12000 probe right after the Delaunay reads high and poisoned the first slope).  The sbatch also asserts\n"
          "# the v9b fix line and OVER >= 1.  Everything else verbatim the v9 sbatch, whose text follows (names rewritten: adapt9 -> adapt9x,\n"
          "# adapt9b -> adapt9xb, adapt-v9 -> adapt-v9b, v9-r1 -> v9x-r1, v9b-r1 -> v9xb-r1).\n")
if which == 'blender':
    over = '1.1'
    assert 'adapt9b' in s
    rep('adapt9b', 'adapt9xb', s.count('adapt9b'))
    assert s.count('adapt9') == s.count('adapt9xb'), 'a bare adapt9 survives in the Blender sbatch'
    rep('v9b-r1', 'v9xb-r1', s.count('v9b-r1'))
else:
    over = '1.25'
    assert 'adapt9b' not in s
    rep('adapt9', 'adapt9x', s.count('adapt9'))
    rep('v9-r1', 'v9x-r1', s.count('v9-r1'))
rep('adapt-v9', 'adapt-v9b', s.count('adapt-v9'))
rep('train_rd_adapt_v9.py', 'train_rd_adapt_v9b.py', s.count('train_rd_adapt_v9.py'))
assert 'adapt-v9bb' not in s and 'train_rd_adapt_v9bb' not in s
if which == 'blender':
    rep("# adapt-v9b-blender.sbatch (2026-11-07, VERSION v9xb-r1): = adapt-v8-blender.sbatch (VERSION v8b-r1) with train_rd_adapt_v9b.py (v8 + the phase-A plan\n",
        "# adapt-v9b-blender.sbatch (2026-11-07 evening, VERSION v9xb-r1): = adapt-v9-blender.sbatch (VERSION v9b-r1) with train_rd_adapt_v9b.py.\n" + COMMON.format(over=over) +
        "# adapt-v9-blender.sbatch (2026-11-07, VERSION v9b-r1): = adapt-v8-blender.sbatch (VERSION v8b-r1) with train_rd_adapt_v9.py (v8 + the phase-A plan\n")
else:
    rep("# adapt-v9b.sbatch (2026-11-07, VERSION v9x-r1): = adapt-v8.sbatch (VERSION v8-r1) with train_rd_adapt_v9b.py (v8 + the phase-A plan fixes: intervals-left\n",
        "# adapt-v9b.sbatch (2026-11-07 evening, VERSION v9x-r1): = adapt-v9.sbatch (VERSION v9-r1) with train_rd_adapt_v9b.py.\n" + COMMON.format(over=over) +
        "# adapt-v9.sbatch (2026-11-07, VERSION v9-r1): = adapt-v8.sbatch (VERSION v8-r1) with train_rd_adapt_v9.py (v8 + the phase-A plan fixes: intervals-left\n")
rep('OVER=${OVER:-1.0};', 'OVER=${OVER:-' + over + '};')
rep('PRUNE=${PRUNE:-"$(seq -s " " 12000 500 19500)"}', 'PRUNE=${PRUNE:-"$(seq -s " " 12500 500 19500)"}')
rep('grep -q "version=\'adapt-v9b\'" $TR || { echo "FAILED: $TR is not adapt-v9b (wrong file pasted?)"; exit 1; }\n',
    'grep -q "version=\'adapt-v9b\'" $TR || { echo "FAILED: $TR is not adapt-v9b (wrong file pasted?)"; exit 1; }\n'
    'grep -q "if k_asked > 0 else lam_used" $TR || { echo "FAILED: $TR lacks the v9b continuous-price fix"; exit 1; }\n'
    'python -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 1.0 else 1)" $OVER || { echo "FAILED: OVER (phase-A aim factor) must be >= 1.0, got $OVER"; exit 1; }\n'
    'FIRSTP=$(echo $PRUNE | tr \' \' \'\\n\' | sort -n | head -1); [ "$FIRSTP" -ge 12500 ] || echo "note: first prune round $FIRSTP < 12500 (the 12000 probe right after the Delaunay reads high)"\n')
rep("NPR=$(echo $PRUNE | wc -w); [ \"$NPR\" -ge 4 ] || { echo \"FAILED: PRUNE has $NPR rounds (v8 default is 16, at least 4 expected: $PRUNE)\"; exit 1; }",
    "NPR=$(echo $PRUNE | wc -w); [ \"$NPR\" -ge 4 ] || { echo \"FAILED: PRUNE has $NPR rounds (v9b default is 15, at least 4 expected: $PRUNE)\"; exit 1; }")
assert "version='adapt-v9b'" in s and 'OVER=${OVER:-' + over + '}' in s and '12500 500 19500' in s
open(dst_path, 'w').write(s)
print(f'[{which}] {n_rep} replacements -> {dst_path}: {len(s.splitlines())} lines')
