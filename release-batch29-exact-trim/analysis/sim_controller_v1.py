import pickle, statistics as st, math, sys
runs = pickle.load(open(sys.argv[1], 'rb'))
BL = {'lego','chair','drums','ficus','hotdog','materials','mic','ship'}; DBS = {'drjohnson','playroom'}
def traj(r):
    """untouched ('virtual') size at each round, the logged linear prediction of the final size (virtual), and the final virtual size"""
    c = 0.0; v = []; P = []
    for row in r['rounds']:
        it, probe, est, rem = row[0], row[1], row[2], row[-1]
        v.append(probe + c); P.append(est + c); c += rem
    return v, P, r['delivered'] + c, c
main = {k: r for k, r in runs.items() if k[1] in ('main', 'abl24-repeat-s1', 'abl24-repeat-s2', 'abl24-constants') and len(r['rounds']) == 15}
# 1) predictor error by round and class
def cls(s): return 'objects' if s in BL else ('deep blending' if s in DBS else 'mip360/t&t')
print('logged linear predictor: predicted final / true final - 1 (%), median [min .. max] by round')
for c in ('mip360/t&t', 'deep blending', 'objects'):
    R = [traj(r) for k, r in main.items() if cls(k[0]) == c]
    print(' ', c, len(R), 'runs')
    for j in (1, 3, 5, 7, 9, 11, 13, 14):
        e = [100 * (P[j] / F - 1) for v, P, F, c0 in R]; print('    it %5d: median %+5.1f  [%+6.1f .. %+6.1f]' % (12500 + 500 * j, st.median(e), min(e), max(e)))
# 2) closed-loop simulation with exact execution
def simulate(r, T, weight, pred, start=0, cap=0.12):
    v, P, F, c0 = traj(r); N = len(v); c = 0.0
    for j in range(N):
        if j < start: continue
        m = N - j; p = pred(j, v, P); excess = p - c - T
        if excess > 0: c += min(excess * weight(m), cap * (v[j] - c))
    return (F - c) / T - 1
def p_lin(j, v, P): return P[j]
def p_last(j, v, P):   # slope of the last two rounds, extended to the end of the ramp (24k) only
    if j == 0: return v[0]
    return v[j] + max(0.0, v[j] - v[j - 1]) * (24000 - (12500 + 500 * j)) / 500.0
def p_min(j, v, P): return min(p_lin(j, v, P), p_last(j, v, P))
def p_now(j, v, P): return v[j]
def p_blend(j, v, P):  # trust the extrapolation in proportion to the elapsed part of the window
    a = j / 14.0; return v[j] + a * (P[j] - v[j])
W = {'excess/rounds left': lambda m: 1.0 / m, 'excess/min(rounds left,3)': lambda m: 1.0 / min(m, 3), 'last round only': lambda m: 1.0 if m == 1 else 0.0, 'excess/m^2': lambda m: 1.0 / (m * m)}
PR = {'linear (logged)': p_lin, 'last-two slope': p_last, 'min(linear, last-two)': p_min, 'blend(now -> linear)': p_blend}
sel = {k: r for k, r in runs.items() if k[1] == 'main' and len(r['rounds']) == 15}
def report(label, errs):
    a = [abs(100 * e) for e in errs]; return '%-58s n %3d | mean abs %4.1f%% | within 5%%: %3d  within 10%%: %3d | worst %+6.1f%% / %+6.1f%%' % (label, len(a), st.mean(a), sum(x <= 5 for x in a), sum(x <= 10 for x in a), 100 * min(errs), 100 * max(errs))
print('\nwhat the logged runs did (main runs, all 100 incl. capacity-limited):'); print(' ', report('current method (as run)', [r['delivered'] / k[2] - 1 for k, r in sel.items()]))
print('\nsimulated exact execution on the same 100 growth trajectories:')
for wn, w in W.items():
    for pn, p in PR.items():
        for start in (0, 6, 10):
            errs = [simulate(r, k[2], w, p, start) for k, r in sel.items()]
            print(' ', report('%s | %s | from it %d' % (wn, pn, 12500 + 500 * start), errs))
