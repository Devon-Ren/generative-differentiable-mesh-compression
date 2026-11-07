"""CPU test of v9: (1) intervals-left formula vs a brute-force count of upstream's densify iterations; (2) plan_splits replayed on the six
batch11 trajectories at 9500 (old count 2 vs new count 1): the old plan must reproduce the observed model after the last prune, the new one
must reach V* (or upstream's limiter); (3) whole-phase replay 3500 -> 9500 old vs new."""
import ast, math, sys
src = open(sys.argv[1]).read(); tree = ast.parse(src); ns = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'plan_splits': exec(ast.get_source_segment(src, node), ns)
plan_splits = ns['plan_splits']
assert "remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))" in src
assert "(opt.densify_until_iter - iteration) // opt.densification_interval + 1)" not in src

# (1) the count upstream will still run (this interval included): iteration < densify_until, iteration % interval == 0, iteration > densify_from
def new_rem(until, it, interval): return max(1, -(-(until - it) // interval))
def old_rem(until, it, interval): return max(1, (until - it) // interval + 1)
for until, interval, dfrom in ((10000, 500, 1000), (1600, 500, 1000), (10000, 300, 1000), (9999, 500, 1000), (10001, 500, 1000)):
    its = [i for i in range(1, until) if i % interval == 0 and i > dfrom]
    for i in its:
        truth = len([j for j in its if j >= i]); assert new_rem(until, i, interval) == truth, (until, interval, i, truth, new_rem(until, i, interval))
    if until % interval == 0: assert all(old_rem(until, i, interval) == new_rem(until, i, interval) + 1 for i in its)
print('(1) intervals-left formula == brute-force count for 5 schedules (1500..9500 = 17 on the default schedule:', new_rem(10000, 1500, 500), ')')

# (2) at 9500 (last interval) with the recorded V, V*, sigma(obs), upstream add_pct (limiter = (add_pct - 1) V triangles); M = model after the next upstream prune (the row's prune-phase V)
runs = [('chair 2.5', 95077, 139242, 0.524, 1.23, 112732), ('chair 4', 142991, 256048, 0.485, 1.23, 181331), ('lego 2.5', 101215, 135875, 0.624, 1.23, 116957),
        ('lego 4', 157099, 253415, 0.645, 1.23, 196198), ('bicycle 10', 611700, 777190, 0.760, 1.23, 674455), ('room 10', 686567, 814653, 0.491, 1.27, 740378)]
print('(2) last interval 9500: predicted model after the last prune, old plan (2 intervals left) vs v9 (1), vs observed')
for name, V, Vs, s, add, M_obs in runs:
    lim = add * V - V
    d_old, f_old = plan_splits(V, Vs, [s, s], add); d_new, f_new = plan_splits(V, Vs, [s], add)
    m_old = V + 3 * min(d_old, lim) * s; m_new = V + 3 * min(d_new, lim) * s
    print(f"  {name:11s} V {V:7d} V* {Vs:7d} sigma {s:.2f} | old desired {d_old:8.0f} -> M {m_old:8.0f} ({m_old/Vs:.3f} V*) obs {M_obs} ({M_obs/Vs:.3f}; err {m_old/M_obs-1:+.1%}) | v9 desired {d_new:8.0f}{' LIMITER' if d_new > lim else ''} -> M {m_new:8.0f} ({m_new/Vs:.3f} V*){' front' if f_new else ''}")
    assert abs(m_old / M_obs - 1) < 0.08, 'the old plan does not reproduce the observed shortfall (inputs are the printed sigma/V*, rounded)'
    assert m_new >= 0.995 * Vs or d_new > lim, 'v9 neither reaches V* nor is limiter-bound'

# (3) whole phase A replayed from the recorded V at 3500 with the sigma measured at 9500 (yield 1, cohorts pruned after start_pruning 3000)
print('(3) replay 3500 -> 9500 (13 intervals), constant sigma, yield 1: model after the last prune / V*')
starts = {'chair 2.5': 12078, 'chair 4': 13445, 'lego 2.5': 16490, 'lego 4': 17422, 'bicycle 10': 233256, 'room 10': 248755}
for name, _, Vs, s, add, _ in runs:
    out = {}
    for label, rem_of in (('old', lambda it: old_rem(10000, it, 500)), ('v9', lambda it: new_rem(10000, it, 500))):
        v = starts[name]; nb = 0
        for it in range(3500, 10000, 500):
            r = rem_of(it); d, f = plan_splits(v, Vs, [s] * r, add); lim = add * v - v
            if d > lim: nb += 1
            v += 3 * min(d, lim) * s
        out[label] = (v / Vs, nb)
    print(f"  {name:11s} old {out['old'][0]:.3f} V* (limiter-bound {out['old'][1]}/13) | v9 {out['v9'][0]:.3f} V* (limiter-bound {out['v9'][1]}/13)")
print('ALL OK')
