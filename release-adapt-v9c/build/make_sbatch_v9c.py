"""make_sbatch_v9c.py: adapt-v9c.sbatch / adapt-v9c-blender.sbatch from the v9b sbatch files.  Names: adapt9c / adapt9cb, run dirs $SCENE-adapt9c-t${T}mb /
$SCENE-adapt9cb-t${T}mb (+ -best), tables adapt-v9c.txt / adapt-v9c-blender.txt, VERSION v9c-r1 / v9cb-r1; PRUNE default every 250 from 12500 to 19750 (30 rounds)."""
import sys
which, src_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'[{which}] anchor found {c}x (expected {count}): {old[:100]!r}'
    s = s.replace(old, new); n_rep += 1


COMMON = ("# v9c = train_rd_adapt_v9c.py (v9b + ONE line: the landing extrapolation never uses a negative slope, so the phase-B gate opens as soon as the\n"
          "# probe itself exceeds the target instead of 3-9 rounds later) + PRUNE default every 250 from 12500 to 19750 (30 rounds, +15 probes of ~30 s per run:\n"
          "# the v9b over-target indoor rows collapsed 0.8-1 % of V per round, the foldover / independent-set limit, and ran out of rounds).  OVER unchanged\n"
          "# ({over}).  Everything else verbatim the v9b sbatch, whose text follows (names rewritten: adapt9x -> adapt9c, adapt9xb -> adapt9cb, adapt-v9b -> adapt-v9c,\n"
          "# v9x-r1 -> v9c-r1, v9xb-r1 -> v9cb-r1).\n")
if which == 'blender':
    over = '1.1'; assert 'adapt9xb' in s
    rep('adapt9xb', 'adapt9cb', s.count('adapt9xb'))
    if 'adapt9x' in s: rep('adapt9x', 'adapt9c', s.count('adapt9x'))  # only in the v9b header comment ('adapt9 -> adapt9x')
    rep('v9xb-r1', 'v9cb-r1', s.count('v9xb-r1'))
else:
    over = '1.25'
    if 'adapt9xb' in s: rep('adapt9xb', 'adapt9cb', s.count('adapt9xb'))  # only in the v9b header comment of the Mip file
    rep('adapt9x', 'adapt9c', s.count('adapt9x')); rep('v9x-r1', 'v9c-r1', s.count('v9x-r1'))
rep('adapt-v9b', 'adapt-v9c', s.count('adapt-v9b')); rep('train_rd_adapt_v9b.py', 'train_rd_adapt_v9c.py', s.count('train_rd_adapt_v9b.py'))
if which == 'blender':
    rep("# adapt-v9c-blender.sbatch (2026-11-07 evening, VERSION v9cb-r1): = adapt-v9-blender.sbatch (VERSION v9b-r1) with train_rd_adapt_v9c.py.\n",
        "# adapt-v9c-blender.sbatch (2026-11-08, VERSION v9cb-r1): = adapt-v9b-blender.sbatch (VERSION v9xb-r1) with train_rd_adapt_v9c.py.\n" + COMMON.format(over=over) +
        "# adapt-v9b-blender.sbatch (2026-11-07 evening, VERSION v9xb-r1): = adapt-v9-blender.sbatch (VERSION v9b-r1) with train_rd_adapt_v9b.py.\n")
else:
    rep("# adapt-v9c.sbatch (2026-11-07 evening, VERSION v9c-r1): = adapt-v9.sbatch (VERSION v9-r1) with train_rd_adapt_v9c.py.\n",
        "# adapt-v9c.sbatch (2026-11-08, VERSION v9c-r1): = adapt-v9b.sbatch (VERSION v9x-r1) with train_rd_adapt_v9c.py.\n" + COMMON.format(over=over) +
        "# adapt-v9b.sbatch (2026-11-07 evening, VERSION v9x-r1): = adapt-v9.sbatch (VERSION v9-r1) with train_rd_adapt_v9b.py.\n")
rep('PRUNE=${PRUNE:-"$(seq -s " " 12500 500 19500)"}', 'PRUNE=${PRUNE:-"$(seq -s " " 12500 250 19750)"}')
rep('grep -q "if k_asked > 0 else lam_used" $TR || { echo "FAILED: $TR lacks the v9b continuous-price fix"; exit 1; }\n',
    'grep -q "if k_asked > 0 else lam_used" $TR || { echo "FAILED: $TR lacks the v9b continuous-price fix"; exit 1; }\n'
    'grep -q "max(0.0, slope) \\* (horizon_iter - iteration)" $TR || { echo "FAILED: $TR lacks the v9c slope clamp"; exit 1; }\n')
rep('(v9b default is 15, at least 4 expected', '(v9c default is 30, at least 4 expected')
assert "version='adapt-v9c'" in s and '12500 250 19750' in s and 'OVER=${OVER:-' + over + '}' in s
open(dst_path, 'w').write(s)
print(f'[{which}] {n_rep} replacements -> {dst_path}: {len(s.splitlines())} lines')
