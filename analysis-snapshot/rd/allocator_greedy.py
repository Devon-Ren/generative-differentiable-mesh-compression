"""Greedy RD allocator check: starting from the richest config, repeatedly take the single-lever cut (R one notch down,
K one notch down, vertex level one notch down) with the smallest dB lost per MB saved; compare the greedy path against
the global upper hull of ALL measured points.   python allocator_greedy.py room-codebook-levels-v1.csv
"""
import csv, sys
from plot_codebook_rd import upper_hull
rows = {(float(r['level']), int(r['K']), int(r['R'])): (float(r['MB']), float(r['test_psnr'])) for r in csv.DictReader(open(sys.argv[1]))}
levels = sorted({k[0] for k in rows}); Ks = sorted({k[1] for k in rows}); Rs = sorted({k[2] for k in rows})
def moves(c):
    L, K, R = c; out = []
    if Rs.index(R) > 0: out.append((L, K, Rs[Rs.index(R) - 1]))
    if Ks.index(K) > 0: out.append((L, Ks[Ks.index(K) - 1], R))
    if levels.index(L) > 0: out.append((levels[levels.index(L) - 1], K, R))
    return [m for m in out if m in rows]
cur = max(rows, key=lambda c: rows[c][0]); path = [cur]
while True:
    ms = moves(cur)
    if not ms: break
    # price of each cut = dB lost per MB saved (lower is better); a cut that gains PSNR is free
    best = min(ms, key=lambda m: (rows[cur][1] - rows[m][1]) / max(rows[cur][0] - rows[m][0], 1e-6))
    cur = best; path.append(cur)
hull = upper_hull([(v[0], v[1], c) for c, v in rows.items()]); hull_pts = [(h[0], h[1]) for h in hull]
def hull_at(x):
    for (x1, y1), (x2, y2) in zip(hull_pts, hull_pts[1:]):
        if x1 <= x <= x2: return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return None
print('greedy path (level,K,R -> MB / dB / gap-to-hull dB):')
on = 0
for c in path:
    mb, p = rows[c]; h = hull_at(mb); gap = (h - p) if h is not None else float('nan')
    tag = ' HULL' if any(hc[2] == c for hc in hull) else ''
    on += bool(tag); print(f'  v{c[0]:.1f} K{c[1]:<5d} R{c[2]:<4d} {mb:6.2f} MB {p:.3f} dB  gap {gap:+.3f}{tag}')
print(f'{len(path)} greedy points, {on} on the global hull; hull has {len(hull)} points; max gap {max((hull_at(rows[c][0]) or rows[c][1]) - rows[c][1] for c in path):.3f} dB')
print('global hull:', ' -> '.join(f'v{h[2][0]:.1f}K{h[2][1]}R{h[2][2]}' for h in hull))
