"""CPU test of v9b: (1) the v9 plan tests still hold (plan_splits is unchanged); (2) the continuous-price line: rank_lambda(k=0) is lam_max
(the v9 behaviour), the v9b expression keeps the price just used; (3) the only code difference to v9 is that line + version strings."""
import ast, difflib, sys
v9_path, v9b_path = sys.argv[1], sys.argv[2]
src = open(v9b_path).read(); v9 = open(v9_path).read()
assert "version='adapt-v9b'" in src and src.count("version='adapt-v9b'") == 2 and "version='adapt-v9'" not in src
assert "lam_now = rank_lambda(ratio_pre, n_allowed, k_asked, rd['lam_min'], rd['lam_max']) if k_asked > 0 else lam_used" in src
# (3) diff = header + 1 code line + 3 version strings
body9 = v9[v9.index('import '):]; body9b = src[src.index('import '):]
d = [l for l in difflib.unified_diff(body9.splitlines(), body9b.splitlines(), lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
changed = [l for l in d if l.startswith('-')]
assert len(changed) == 5, f'expected 5 changed lines below the header, got {len(changed)}: {changed}'
assert sum('rank_lambda(ratio_pre' in l for l in changed) == 1 and sum('adapt-v9' in l for l in changed) == 4
print('(3) code diff v9 -> v9b below the header: 5 lines (1 fix + 4 version strings)')
# (2) semantics of rank_lambda at k = 0 (needs torch only for the k >= 1 branch; k = 0 returns before touching torch)
ns = {}
tree = ast.parse(src)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('rank_lambda', 'plan_splits'): exec(ast.get_source_segment(src, node), ns)
rank_lambda = ns['rank_lambda']
lo, hi = 1e-4, 1e-1
assert rank_lambda(None, 1000, 0, lo, hi) == hi, 'rank_lambda(k=0) must return hi (that is the v9 behaviour the fix bypasses)'
for lam_used, k_asked in ((3.3e-3, 0), (2.2e-2, 0), (7.7e-2, 0)):
    lam_now = rank_lambda(None, 1000, k_asked, lo, hi) if k_asked > 0 else lam_used
    assert lam_now == lam_used
print('(2) k_asked == 0 -> lam_now == lam_used (phase B starts from the last real phase-A price, not lam_max %.0e)' % hi)
# (1) the v9 plan tests, verbatim on v9b's plan_splits
plan_splits = ns['plan_splits']
def new_rem(until, it, interval): return max(1, -(-(until - it) // interval))
runs = [('chair 2.5', 95077, 139242, 0.524, 1.23, 112732), ('chair 4', 142991, 256048, 0.485, 1.23, 181331), ('lego 2.5', 101215, 135875, 0.624, 1.23, 116957),
        ('lego 4', 157099, 253415, 0.645, 1.23, 196198), ('bicycle 10', 611700, 777190, 0.760, 1.23, 674455), ('room 10', 686567, 814653, 0.491, 1.27, 740378)]
for name, V, Vs, s, add, M_obs in runs:
    lim = add * V - V; d_new, f_new = plan_splits(V, Vs, [s], add); m_new = V + 3 * min(d_new, lim) * s
    assert m_new >= 0.995 * Vs or d_new > lim, name
starts = {'chair 2.5': 12078, 'chair 4': 13445, 'lego 2.5': 16490, 'lego 4': 17422, 'bicycle 10': 233256, 'room 10': 248755}
for name, _, Vs, s, add, _ in runs:
    v = starts[name]
    for it in range(3500, 10000, 500):
        r = new_rem(10000, it, 500); dd, f = plan_splits(v, Vs, [s] * r, add); lim = add * v - v; v += 3 * min(dd, lim) * s
    assert abs(v / Vs - 1) < 0.01, (name, v / Vs)
print('(1) v9 plan tests hold on v9b (6 trajectories reach V* at 9500 and 1.000 V* on the 3500 -> 9500 replay)')
# the aim: what the asymmetric margin does to the three batch12 Mip runs (hit before phase B = measured hit x OVER)
for name, hit in (('room 10', 0.877), ('bicycle 10', 0.790), ('room 20', 0.829)):
    print(f'    {name:10s} v9 hit {hit:.3f} -> phase-A landing with OVER 1.25: {hit * 1.25:.3f} (phase B trims {max(0, hit * 1.25 - 1) * 100:.0f} %)')
print('ALL OK')
