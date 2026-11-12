import pickle, statistics as st, sys
runs = pickle.load(open(sys.argv[1], 'rb'))
def traj(r):
    c = 0.0; v = []; P = []
    for row in r['rounds']:
        probe, est, rem = row[1], row[2], row[-1]; v.append(probe + c); P.append(est + c); c += rem
    return v, P, r['delivered'] + c
sel = {k: r for k, r in runs.items() if k[1] == 'main' and len(r['rounds']) == 15}
def simulate(r, over, truth, ctrl, start, wfun, cap=0.12):
    """the same growth trajectory, target set so that the untouched file would end at over x target; truth: how a removal propagates to the end"""
    v, P, F = traj(r); T = F / over; N = len(v); size = v[0]; removed_end = 0.0; c = 0.0
    for j in range(N):
        g_true = F / v[j]                                   # growth still to come, untouched
        size = (v[j] - c) if truth == 'add' else size       # measured file now
        if truth == 'mul' and j > 0: size = size * v[j] / v[j - 1]
        if j < start: continue
        m = N - j
        if ctrl == 'add': est = P[j] - (v[j] - size)         # code as it is: prediction of the untouched file minus what was removed
        else: est = size * P[j] / v[j]                       # ratio form: measured file times the predicted growth factor
        excess = est - T
        if excess > 0:
            u = min(excess * wfun(m) / (1.0 if ctrl == 'add' else P[j] / v[j]), cap * size)
            c += u; size -= u
    final = (F - c) if truth == 'add' else size * F / v[N - 1]
    return final / T - 1
def rep(errs): a = [abs(100 * e) for e in errs]; return 'mean abs %4.1f%% | within 5%%: %3d/100  within 10%%: %3d/100 | range %+6.1f%% .. %+5.1f%%' % (st.mean(a), sum(x <= 5 for x in a), sum(x <= 10 for x in a), 100 * min(errs), 100 * max(errs))
SCH = {'excess / rounds left, from 12,500': (0, lambda m: 1.0 / m), 'excess / rounds left, from 16,000': (7, lambda m: 1.0 / m), 'excess / rounds left, from 17,500': (10, lambda m: 1.0 / m), 'last round only': (14, lambda m: 1.0)}
for over in (1.15, 1.35):
    print('\nuntouched file would end at %.2f x target (every run needs trimming)' % over)
    for truth in ('add', 'mul'):
        for ctrl in ('add', 'ratio'):
            for name, (start, w) in SCH.items():
                print('  removal acts %s, controller %-5s | %-34s | %s' % ('additively    ' if truth == 'add' else 'proportionally', ctrl, name, rep([simulate(r, over, truth, ctrl, start, w) for r in sel.values()])))
