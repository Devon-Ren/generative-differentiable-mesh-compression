"""predictor_window_v1.py: error of the final-size prediction by round, for a least-squares slope over ALL probes so far (as in the code)
and over the last K probes, on the logged runs (removals in these runs are small, so probe + removed is the natural series)."""
import pickle, statistics as st, sys
runs = pickle.load(open(sys.argv[1], 'rb'))
BL = {'lego','chair','drums','ficus','hotdog','materials','mic','ship'}; DBS = {'drjohnson','playroom'}
def cls(s): return 'objects' if s in BL else ('deep blending' if s in DBS else 'mip360 / t&t')
def traj(r):
    c = 0.0; v = []
    for row in r['rounds']: v.append(row[1] + c); c += row[-1]
    return v, r['delivered'] + c
def pred(v, j, K, H=24000):
    lo = 0 if K is None else max(0, j - K + 1); xs = [12500 + 500 * i for i in range(lo, j + 1)]; ys = v[lo:j + 1]
    if len(xs) < 2: return v[j]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys); sxx = sum((x - mx) ** 2 for x in xs); slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return v[j] + slope * (H - xs[-1])
sel = {k: r for k, r in runs.items() if k[1] in ('main', 'abl24-repeat-s1', 'abl24-repeat-s2', 'abl24-constants') and len(r['rounds']) == 15}
for c in ('mip360 / t&t', 'deep blending', 'objects'):
    R = [traj(r) for k, r in sel.items() if cls(k[0]) == c]; print('%s (%d runs): predicted final / true final - 1, median [min .. max] in %%' % (c, len(R)))
    for j in (7, 9, 11, 13, 14):
        print('   it %5d | ' % (12500 + 500 * j) + ' | '.join('%s: %+5.1f [%+6.1f .. %+5.1f]' % (('all' if K is None else 'last %d' % K), st.median(e), min(e), max(e)) for K in (None, 8, 6, 4) for e in [[100 * (pred(v, j, K) / F - 1) for v, F in R]]))
allR = [traj(r) for r in sel.values()]
print('all %d runs, last round (19,500): share within 5%% of the true final' % len(allR))
for K in (None, 8, 6, 5, 4, 3): e = [abs(100 * (pred(v, 14, K) / F - 1)) for v, F in allR]; print('   %-7s mean abs %.2f%%  within 3%%: %3d  within 5%%: %3d  worst %.1f%%' % ('all' if K is None else 'last %d' % K, st.mean(e), sum(x <= 3 for x in e), sum(x <= 5 for x in e), max(e)))
