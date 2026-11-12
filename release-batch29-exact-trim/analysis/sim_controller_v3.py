"""sim_controller_v3.py: which controller would have landed on target on real growth curves when a removal acts proportionally
(a byte removed at round k is worth final/now bytes at the end; measured on kitchen: 1.22).  Curves: the 100 main runs (rescaled so that
every run needs trimming) and the three object runs with growth aim 1.25 (batch28, real target 2.5 MB)."""
import pickle, statistics as st, sys
runs = pickle.load(open(sys.argv[1], 'rb'))
D = {'hotdog aim 1.25': ([(2.46, 2.46, 0.0), (2.525, 3.973, 0.019), (2.567, 3.89, 0.013), (2.603, 3.775, 0.011), (2.636, 3.675, 0.009), (2.669, 3.601, 0.009), (2.695, 3.524, 0.008), (2.712, 3.441, 0.008), (2.744, 3.4, 0.008), (2.75, 3.326, 0.007), (2.762, 3.267, 0.008), (2.776, 3.218, 0.007), (2.792, 3.178, 0.009), (2.816, 3.156, 0.01), (2.814, 3.107, 0.013)], 3.015),
     'lego aim 1.25': ([(2.821, 2.821, 0.064), (2.778, 3.238, 0.043), (2.747, 3.102, 0.03), (2.739, 3.096, 0.024), (2.732, 3.073, 0.022), (2.73, 3.059, 0.02), (2.721, 3.021, 0.018), (2.724, 3.006, 0.016), (2.728, 2.993, 0.015), (2.727, 2.974, 0.014), (2.732, 2.961, 0.013), (2.74, 2.952, 0.012), (2.755, 2.954, 0.012), (2.759, 2.942, 0.011), (2.741, 2.902, 0.015)], 2.784)}
def traj(rounds, delivered):
    c = 0.0; v = []; P = []
    for probe, est, rem in rounds: v.append(probe + c); P.append(est + c); c += rem
    return v, P, delivered + c
def simulate(v, P, F, T, ctrl, start, cap=0.12):
    N = len(v); size = v[0]; hist = []
    for j in range(N):
        if j > 0: size = size * v[j] / v[j - 1]            # the file grows by the natural ratio of this interval
        if j < start: continue
        m = N - j; G = P[j] / v[j]                          # predicted growth factor still to come
        est = size * G if ctrl == 'ratio' else P[j] - (v[j] - size)
        if est > T:
            u = min((est - T) / m / (G if ctrl == 'ratio' else 1.0), cap * size); size -= u
    return size * F / v[N - 1] / T - 1
def rep(errs): a = [abs(100 * e) for e in errs]; return 'mean abs %4.1f%% | within 5%%: %3d  within 10%%: %3d of %d | range %+6.1f%% .. %+5.1f%%' % (st.mean(a), sum(x <= 5 for x in a), sum(x <= 10 for x in a), len(a), 100 * min(errs), 100 * max(errs))
sel = [traj([(r[1], r[2], r[-1]) for r in x['rounds']], x['delivered']) for k, x in runs.items() if k[1] == 'main' and len(x['rounds']) == 15]
for name, (rounds, deliv) in D.items():
    v, P, F = traj(rounds, deliv); print('%-16s untouched final %.2f x target |' % (name, F / 2.5), ' | '.join('%s, from %d: %+5.1f%%' % (c, 12500 + 500 * s, 100 * simulate(v, P, F, 2.5, c, s)) for c in ('add', 'ratio') for s in (0, 7, 10)))
for over in (1.15, 1.25, 1.35):
    print('100 main curves, untouched final %.2f x target' % over)
    for c in ('add', 'ratio'):
        for s in (0, 7, 10): print('   %-5s from %5d | %s' % (c, 12500 + 500 * s, rep([simulate(v, P, F, F / over, c, s) for v, P, F in sel])))
