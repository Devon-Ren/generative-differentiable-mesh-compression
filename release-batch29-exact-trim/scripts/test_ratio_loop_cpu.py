"""test_ratio_loop_cpu.py: closed loop with extrapolate_ratio (text from the generated script) on a synthetic file whose natural growth is
linear and where a removal scales with later growth.  With a perfect growth model the file must land on the target.
  python3 test_ratio_loop_cpu.py <train_rd_adapt_v9b_abl5.py>"""
import sys
src = open(sys.argv[1]).read(); ns = {}
exec(src[src.index('# >>> extrapolate_ratio'):src.index('# <<< extrapolate_ratio')], ns); exec(src[src.index('def extrapolate_probe('):src.index('def ', src.index('def extrapolate_probe(') + 10)], ns)
ratio, additive = ns['extrapolate_ratio'], ns['extrapolate_probe']
def run(mode, T, start, nat):
    its = list(range(12500, 20000, 500)); H = 24000; size = nat(its[0]); trend = []
    for j, it in enumerate(its):
        if j > 0: size = size * nat(it) / nat(its[j - 1])
        m = len(its) - j
        if mode == 'ratio': est, _, g = ratio(trend, it, size, H)
        else: est, _ = additive(trend, it, size, H); g = 1.0
        u = (est - T) / g / m if (est > T and it >= start) else 0.0
        trend.append((it, size, u)); size -= u
    return size * nat(H) / nat(its[-1]) / T - 1
ok = True
for label, nat in (('linear growth to 1.25 x target', lambda it: 9.0 + 3.5 * (it - 12500) / 11500.0), ('growth that slows down (object-like)', lambda it: 9.0 + 3.5 * (1 - 0.5 ** ((it - 12500) / 3000.0)) / (1 - 0.5 ** (11500 / 3000.0)))):
    for start in (0, 16000):
        e_r = run('ratio', 10.0, start, nat); e_a = run('add', 10.0, start, nat)
        print('%-38s start %5d | ratio form lands %+5.1f%% | additive form lands %+5.1f%%' % (label, start, 100 * e_r, 100 * e_a))
        if label.startswith('linear'): ok &= abs(e_r) < 0.005
print('TEST', 'PASSED' if ok else 'FAILED')
