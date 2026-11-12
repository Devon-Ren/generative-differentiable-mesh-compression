"""sim_controller_v4.py: the final design on the logged growth curves (100 main runs + 2 object runs with aim 1.25), removal acting proportionally.
Design: rounds every 500 it from 12,500 to 19,500 plus 21,000 / 22,500 / 24,000 / 25,500 / 27,000.  Before 16,000: measure only.  Rounds before the
end of the opacity ramp (24,000) PREDICT the final size (file now x growth factor from the last 6 probes) and steer to target x (1 + margin);
rounds from 24,000 on MEASURE (no prediction) and steer to the target.  Each phase spreads its excess over its own remaining rounds.
The natural file between logged probes (19,500 -> 22,000 -> 26,000 -> final) is interpolated linearly."""
import pickle, statistics as st, sys
runs = pickle.load(open(sys.argv[1], 'rb'))
def nat_curve(rounds, extra, delivered):
    c = 0.0; pts = []
    for row in rounds: pts.append((row[0], row[1] + c)); c += row[-1]
    for it, mb, *_ in extra: pts.append((it, mb + c))
    pts.append((30000, delivered + c)); pts.sort()
    def f(it):
        for (a, va), (b, vb) in zip(pts, pts[1:]):
            if a <= it <= b: return va + (vb - va) * (it - a) / (b - a)
        return pts[-1][1]
    return f
ROUNDS = list(range(12500, 20000, 500)) + [21000, 22500, 24000, 25500, 27000]; H = 24000
def simulate(nat, T, margin=0.05, start=16000, late=True, window=6, cap=0.12):
    R = ROUNDS if late else ROUNDS[:15]; size = nat(R[0]); ns = []; xs = []
    for j, it in enumerate(R):
        if j > 0: size = size * nat(it) / nat(R[j - 1])
        xs.append(it); ns.append(nat(it))           # the natural series is recovered exactly from the probe ratios under proportional removal
        if it < start: continue
        if it < H:
            x, y = xs[-window:], ns[-window:]; mx = sum(x) / len(x); my = sum(y) / len(y); sxx = sum((a - mx) ** 2 for a in x)
            slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx if sxx > 0 else 0.0; g = max(0.8, (ns[-1] + slope * (H - it)) / ns[-1])
            est = size * g; tgt = T * (1 + margin); m = len([i for i in R if i >= it and i < H])
        else: est = size; g = 1.0; tgt = T; m = len([i for i in R if i >= it])
        if est > tgt: size -= min((est - tgt) / g / m, cap * size)
    return size * nat(30000) / nat(R[-1]) / T - 1
curves = [nat_curve([(r[0], r[1], r[-1]) for r in x['rounds']], x['extra'], x['delivered']) for k, x in runs.items() if k[1] == 'main' and len(x['rounds']) == 15 and len(x['extra']) == 2]
def rep(errs): a = [abs(100 * e) for e in errs]; return 'mean abs %4.1f%% | within 2%%: %3d  within 5%%: %3d  within 10%%: %3d of %d | range %+5.1f%% .. %+5.1f%%' % (st.mean(a), sum(x <= 2 for x in a), sum(x <= 5 for x in a), sum(x <= 10 for x in a), len(a), 100 * min(errs), 100 * max(errs))
print(len(curves), 'curves with probes at 22,000 and 26,000')
for over in (1.15, 1.35):
    print('untouched file would end at %.2f x target' % over)
    print('   rounds to 19,500 only, no margin   |', rep([simulate(n, n(30000) / over, margin=0.0, late=False) for n in curves]))
    for mg in (0.0, 0.03, 0.05, 0.08): print('   late measured rounds, margin %.2f  |' % mg, rep([simulate(n, n(30000) / over, margin=mg) for n in curves]))
