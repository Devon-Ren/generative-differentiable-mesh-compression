"""make_sbatch_v9d.py: adapt-v9d.sbatch / adapt-v9d-blender.sbatch from the v9c sbatch files.  Names adapt9d / adapt9db, tables adapt-v9d*.txt,
VERSION v9d-r1 / v9db-r1; the file list and the version greps include rd_prune_v2.py; PRUNE default unchanged (every 250 from 12500)."""
import sys
which, src_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'[{which}] anchor found {c}x (expected {count}): {old[:100]!r}'
    s = s.replace(old, new); n_rep += 1


COMMON = ("# v9d = train_rd_adapt_v9d.py + rd_prune_v2.py: phase-B collapse candidates are PRICED ON THE DELIVERED SET (dR = b_attr if both endpoints\n"
          "# are in the probe's kept set + b_face per kept face on the edge; unpriced edges are not candidates).  In the v9c test the rounds collapsed\n"
          "# 4-8k pairs each and freed 0.002-0.03 MB (both-kept share 0.35-0.57): the cheapest dD/dR edges were the invisible triangles the final\n"
          "# cleaning drops anyway.  PRUNE / OVER unchanged ({over}).  Everything else verbatim the v9c sbatch, whose text follows (names rewritten:\n"
          "# adapt9c -> adapt9d, adapt9cb -> adapt9db, adapt-v9c -> adapt-v9d, v9c-r1 -> v9d-r1, v9cb-r1 -> v9db-r1).\n")
if which == 'blender':
    over = '1.1'; assert 'adapt9cb' in s
    rep('adapt9cb', 'adapt9db', s.count('adapt9cb'))
    if 'adapt9c' in s: rep('adapt9c', 'adapt9d', s.count('adapt9c'))
    rep('v9cb-r1', 'v9db-r1', s.count('v9cb-r1'))
else:
    over = '1.25'
    if 'adapt9cb' in s: rep('adapt9cb', 'adapt9db', s.count('adapt9cb'))
    rep('adapt9c', 'adapt9d', s.count('adapt9c')); rep('v9c-r1', 'v9d-r1', s.count('v9c-r1'))
rep('adapt-v9c', 'adapt-v9d', s.count('adapt-v9c')); rep('train_rd_adapt_v9c.py', 'train_rd_adapt_v9d.py', s.count('train_rd_adapt_v9c.py'))
if which == 'blender':
    rep("# adapt-v9d-blender.sbatch (2026-11-08, VERSION v9db-r1): = adapt-v9b-blender.sbatch (VERSION v9xb-r1) with train_rd_adapt_v9d.py.\n",
        "# adapt-v9d-blender.sbatch (2026-11-08, VERSION v9db-r1): = adapt-v9c-blender.sbatch (VERSION v9cb-r1) with train_rd_adapt_v9d.py + rd_prune_v2.py.\n" + COMMON.format(over=over) +
        "# adapt-v9c-blender.sbatch (2026-11-08, VERSION v9cb-r1): = adapt-v9b-blender.sbatch (VERSION v9xb-r1) with train_rd_adapt_v9c.py.\n")
else:
    rep("# adapt-v9d.sbatch (2026-11-08, VERSION v9d-r1): = adapt-v9b.sbatch (VERSION v9x-r1) with train_rd_adapt_v9d.py.\n",
        "# adapt-v9d.sbatch (2026-11-08, VERSION v9d-r1): = adapt-v9c.sbatch (VERSION v9c-r1) with train_rd_adapt_v9d.py + rd_prune_v2.py.\n" + COMMON.format(over=over) +
        "# adapt-v9c.sbatch (2026-11-08, VERSION v9c-r1): = adapt-v9b.sbatch (VERSION v9x-r1) with train_rd_adapt_v9c.py.\n")
rep("PROBE_PY=$AN/rdgrowth/probe_bytes_v1.py\n", "PROBE_PY=$AN/rdgrowth/probe_bytes_v1.py; PRUNE2=$AN/rdgrowth/rd_prune_v2.py\n")
rep("for f in $TR $PROBE_PY $AN/rdgrowth/train_rd_v3.py", "for f in $TR $PROBE_PY $PRUNE2 $AN/rdgrowth/train_rd_v3.py")
rep('grep -q "max(0.0, slope) \\* (horizon_iter - iteration)" $TR || { echo "FAILED: $TR lacks the v9c slope clamp"; exit 1; }\n',
    'grep -q "max(0.0, slope) \\* (horizon_iter - iteration)" $TR || { echo "FAILED: $TR lacks the v9c slope clamp"; exit 1; }\n'
    'grep -q "from rd_prune_v2 import" $TR && grep -q "kept_v=kept_v, kept_f=kept_f" $TR || { echo "FAILED: $TR lacks the v9d kept-priced collapse"; exit 1; }\n'
    'grep -q "PRUNE_VERSION = \'rd_prune_v2\'" $PRUNE2 && grep -q "kept_v=None, kept_f=None" $PRUNE2 || { echo "FAILED: $PRUNE2 is not rd_prune_v2"; exit 1; }\n'
    'python -m py_compile $PRUNE2 || { echo "FAILED: $PRUNE2 does not compile"; exit 1; }\n')
assert "version='adapt-v9d'" in s and 'PRUNE2' in s and '12500 250 19750' in s
open(dst_path, 'w').write(s)
print(f'[{which}] {n_rep} replacements -> {dst_path}: {len(s.splitlines())} lines')
