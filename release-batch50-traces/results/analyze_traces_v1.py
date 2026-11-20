"""analyze_traces_v1.py (2026-11-20): statistics of the controller traces (traces-b50-2026-11-20.txt).  Run from the repository root.
Per run: R_k = probe size at round k (before collapsing), c_k = bytes removed in that round, T = target.
For every round k with R_k >= T, the argument of the appendix says R_{k+1} - T = drift + model error; we report its distribution."""
import re, statistics as st, numpy as np
CAP = ['bicycle','flowers','garden','stump','treehill','room','counter','kitchen','bonsai','truck','train','drjohnson','playroom']
runs = []
for l in open('release-batch50-traces/results/traces-b50-2026-11-20.txt'):
    m = re.match(r'TRACE (\w+) ([\d.]+) \| (.*?) \| final (\S+) \| growth ?(.*)', l)
    pr = [tuple(float(x) for x in t.split(':')) for t in m[3].split()]; gr = [tuple(float(x) for x in t.split(':')) for t in m[5].split()]
    runs.append(dict(scene=m[1], T=float(m[2]), pr=pr, final=(float(m[4]) if m[4] != 'n/a' else None), gr=gr))
def stats(name, R):
    nxt, first, act, short, fin, peak = [], [], [], [], [], []
    for r in R:
        T, pr = r['T'], r['pr']; first.append(pr[0][1] / T); act.append(sum(1 for p in pr if p[3] > 0)); peak.append(max(p[1] for p in pr) / T)
        if r['final'] is not None: fin.append(r['final'] / T)
        for a, b in zip(pr, pr[1:] + [(30000, r['final'], 0, 0)] if r['final'] is not None else pr[1:]):
            if a[1] >= T:
                nxt.append((b[1] - T) / T)
                if a[2] < 0.9 * (a[1] - T) and (a[1] - T) / T > 0.005: short.append((r['scene'], T, int(a[0]), round(a[1] - T, 3), round(a[2], 3)))
    q = lambda v, p: float(np.percentile(v, p)); a = [abs(x) for x in nxt]
    print('%s: %d runs, %d rounds per run | size at the first round / target: %.2f..%.2f (median %.2f), runs that start below the target: %d' % (name, len(R), len(R[0]['pr']), min(first), max(first), st.median(first), sum(f < 1 for f in first)))
    print('   after a round with R_k >= T (%d cases): (R_k+1 - T)/T median %+.2f %%, 5th..95th percentile %+.2f..%+.2f %%, |.| median %.2f %% 95th %.2f %% max %.2f %%' % (len(nxt), 100 * st.median(nxt), 100 * q(nxt, 5), 100 * q(nxt, 95), 100 * st.median(a), 100 * q(a, 95), 100 * max(a)))
    print('   active rounds per run: %d..%d, mean %.1f | size at 30k / target: %.3f..%.3f, mean %.3f | rounds where the removed bytes were below 90 %% of the excess: %d %s' % (min(act), max(act), st.mean(act), min(fin), max(fin), st.mean(fin), len(short), short[:6]))
stats('captured', [r for r in runs if r['scene'] in CAP]); stats('objects', [r for r in runs if r['scene'] not in CAP])
g = [r for r in runs if r['scene'] in CAP and r['gr']]; last = [r['gr'][-1][1] / r['T'] for r in g]
print('growth probes: last probe of Phase 1 / target %.2f..%.2f (median %.2f) at iteration %s' % (min(last), max(last), st.median(last), sorted({int(r['gr'][-1][0]) for r in g})))
