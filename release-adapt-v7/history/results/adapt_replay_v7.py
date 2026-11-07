#!/usr/bin/env python3
"""CPU replay of the adapt-v7 phase-B rules on the adapt-v6 Blender logs (2026-11-06), before any GPU job.

For every v6 run the probe series (16 probes 12k-19.5k + the final probe) and the per-round collapse facts are replayed through
  (a) the kept-aware accounting of removed bytes (kept vertices removed x b_v + faces removed x kept-face share x b_f; the exact
      count in v7 uses both-endpoints-kept and the probe's face mask, which the logs do not carry, so this is the closest replay),
  (b) the same least-squares extrapolation to 24000 on the corrected virtual series,
  (c) the gate: a round collapses only when the extrapolated landing is above target.
Output: per run and round the v6 estimate, the v7 estimate, and whether v7 would have pruned; the v7 estimate at the last round
over the final probe; and the first round at which the gate would have opened.  The trajectory itself cannot be replayed (a
round that does not prune leaves more bytes for the next probe), so the gate column is exact only up to the first divergence.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_analysis_v6b import parse_paste, parse_rows, PASTE, ROWS  # noqa: E402

HORIZON = 24000


def ls_slope(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx if sxx > 0 else 0.0


def main():
    rows = parse_rows(ROWS)
    logs = parse_paste(PASTE)
    print('replay of adapt-v7 phase B on the adapt-v6 logs; est = extrapolated-to-24000 size / target; gate = v7 would collapse this round')
    for key in sorted(rows, key=lambda k: (k[1], k[0] != 'lego')):
        r, lg = rows[key], logs[key]
        T, fin = r['T'], r['probe']
        rd = lg['rounds']
        print('\n%s %.1f MB  (final probe %.3f = %.2f x T; v6 delivered hit %.2f)' % (key[0], key[1], fin, fin / T, r['hit']))
        print('    it   probe  est_v6  est_v7  slope7  gate  pairs  removed_v6 removed_v7')
        xs, ys, cum = [], [], 0.0
        first_open = None
        for x in rd:
            share = x['kF'] / x['Tb']
            rem7 = (x['kept_removed'] * x['b_v'] + (x['Tb'] - x['Ta']) * share * x['b_f']) / 1e6
            xs.append(float(x['it'])); ys.append(x['probe'] + cum)
            slope = ls_slope(xs, ys) if len(xs) > 1 else 0.0
            est7 = x['probe'] + slope * (HORIZON - x['it'])
            gate = est7 > T
            if gate and first_open is None: first_open = x['it']
            print('  %5d  %.3f   %.3f   %.3f  %+.3f   %s  %6d   %.3f      %.3f' % (x['it'], x['probe'], x['est'] / T, est7 / T, slope * 1000.0, 'YES' if gate else ' no', x['pairs'], x['acc_mb'], rem7))
            cum += rem7
            x['est7'] = est7
        print('  est@19.5k / final: v6 %.3f  v7 %.3f | first round the v7 gate opens: %s | v6 pruned every round'
              % (rd[-1]['est'] / fin, rd[-1]['est7'] / fin, first_open if first_open else 'never'))


if __name__ == '__main__':
    main()
