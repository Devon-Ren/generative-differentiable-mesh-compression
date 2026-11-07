"""make_sbatch_v9.py: build rd/adapt-v9-blender.sbatch and rd/adapt-v9.sbatch from the v8 sbatch files by exact-count replacements."""
import sys
which, src_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'[{which}] anchor found {c}x (expected {count}): {old[:100]!r}'
    s = s.replace(old, new); n_rep += 1


SMOKE_ADD = """    cfg=[e for e in ev if e['event']=='config'][0]; exp_rem=-(-(cfg['densify_until']-dn[0]['iteration'])//cfg['interval']); assert dn[0].get('remaining')==exp_rem and isinstance(dn[0].get('limiter_max'), int) and dn[0]['limiter_max']>=0, 'v9 intervals-left / limiter_max wrong (expected remaining %d): %r' % (exp_rem, {k: dn[0].get(k) for k in ('iteration','remaining','limiter_max','cap','desired','v_star')})
    print('smoke v9 plan:', {k: dn[0].get(k) for k in ('iteration','remaining','cap','limiter_max','desired','desired_next','v_star','limiter_bound','front_loaded')}, '| expected remaining', exp_rem)
"""
SIZE_SRC_LINE = "    assert str(dn[0].get('size_source','')).startswith('probe_a@'), 'densify event after the phase-A probe does not carry size_source probe_a@: %r' % dn[0].get('size_source')\n"
if which == 'blender':
    rep('adapt8b', 'adapt9b', 6); rep('v8b-r1', 'v9b-r1', s.count('v8b-r1'))
    assert 'adapt8' not in s
    rep('adapt-v8', 'adapt-v9', s.count('adapt-v8')); rep('train_rd_adapt_v8', 'train_rd_adapt_v9', s.count('train_rd_adapt_v8'))
    rep("# adapt-v9-blender.sbatch (2026-11-07, VERSION v9b-r1): = adapt-v7b-blender.sbatch (VERSION v7b-r2) with train_rd_adapt_v9.py (phase A calibrated by\n",
        "# adapt-v9-blender.sbatch (2026-11-07, VERSION v9b-r1): = adapt-v8-blender.sbatch (VERSION v8b-r1) with train_rd_adapt_v9.py (v8 + the phase-A plan\n"
        "# fixes: intervals-left counted as upstream runs them, limiter allowance in triangles as applied; phase-A probes and phase B as in v8), job names\n"
        "# adapt9b, output /scratch/zr2245/$SCENE-adapt9b-t${T}mb (+ -best), table $AN/adapt-v9-blender.txt, the smoke asserting the densify event's\n"
        "# 'remaining' against the config schedule and the presence of limiter_max.  Everything else verbatim v8b-r1, whose text follows (names rewritten).\n"
        "# adapt-v8-blender.sbatch (2026-11-07, VERSION v8b-r1): = adapt-v7b-blender.sbatch (VERSION v7b-r2) with train_rd_adapt_v8.py (phase A calibrated by\n")
else:
    assert 'adapt8b' not in s
    rep('adapt8', 'adapt9', 6); rep('v8-r1', 'v9-r1', s.count('v8-r1'))
    rep('adapt-v8', 'adapt-v9', s.count('adapt-v8')); rep('train_rd_adapt_v8', 'train_rd_adapt_v9', s.count('train_rd_adapt_v8'))
    rep("# adapt-v9.sbatch (2026-11-07, VERSION v9-r1): = adapt-v7b.sbatch (VERSION v7-r2) with train_rd_adapt_v9.py (phase A calibrated by byte probes at densify\n",
        "# adapt-v9.sbatch (2026-11-07, VERSION v9-r1): = adapt-v8.sbatch (VERSION v8-r1) with train_rd_adapt_v9.py (v8 + the phase-A plan fixes: intervals-left\n"
        "# counted as upstream runs them, limiter allowance in triangles as applied; phase-A probes and the 16-round phase B as in v8), job names adapt9,\n"
        "# output /scratch/zr2245/$SCENE-adapt9-t${T}mb, table $AN/adapt-v9.txt, the smoke asserting the densify event's 'remaining' against the config\n"
        "# schedule and the presence of limiter_max.  Everything else verbatim v8-r1, whose text follows (names rewritten).\n"
        "# adapt-v8.sbatch (2026-11-07, VERSION v8-r1): = adapt-v7b.sbatch (VERSION v7-r2) with train_rd_adapt_v8.py (phase A calibrated by byte probes at densify\n")
rep('grep -q "version=\'adapt-v9\'" $TR || { echo "FAILED: $TR is not adapt-v9 (wrong file pasted?)"; exit 1; }\n',
    'grep -q "version=\'adapt-v9\'" $TR || { echo "FAILED: $TR is not adapt-v9 (wrong file pasted?)"; exit 1; }\n'
    'grep -q "max_split_up = max_split" $TR && grep -q "limiter_max=int(max_split_up)" $TR && grep -q "remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))" $TR || { echo "FAILED: $TR lacks the v9 plan fixes"; exit 1; }\n')
rep(SIZE_SRC_LINE, SIZE_SRC_LINE + SMOKE_ADD)
assert all(l.startswith('#') for l in s.splitlines() if ('v8b-r1' in l or 'v8-r1' in l or 'adapt-v8' in l or 'train_rd_adapt_v8' in l)), 'a v8 name survives outside the header comments'
open(dst_path, 'w').write(s)
print(f'[{which}] {n_rep} replacements -> {dst_path}: {len(s.splitlines())} lines')
