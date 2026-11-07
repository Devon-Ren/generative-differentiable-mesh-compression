"""Old route (shared-SH dictionary + post-hoc collapse, 10/13-10/16) vs new route (rate-controlled growth + phase-B prune, 10/18-10/19).
Left: Room, absolute test PSNR vs delivered MB with the upper convex hull of each route.  Right: 7 Mip-360 scenes, PSNR relative to
each scene's original FP32 model; thin lines = per-scene old-route hulls, dots = new-route lambda points, bold = 7-scene means.
usage: python plot_hull_old_vs_new_v1.py <out-stem>   (reads room-lambda-vs-cap-v2.csv, seven-scene-points-v2.csv, lambda-7scenes-v1.csv)"""
import csv, sys, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
out = sys.argv[1]
OLD, NEW, PRUNE, REF, SURF = '#eb6834', '#1baf7a', '#0f6e4c', '#2b2b2b', '#fcfcfb'
ORIG = {'room': (742, 28.263), 'garden': (740, 24.732), 'bonsai': (680, 27.291), 'counter': (575, 25.944), 'kitchen': (456, 26.454), 'bicycle': (680, 23.033), 'stump': (698, 24.928)}

def upper_hull(pts):
    """upper concave envelope of (x, y) points up to the best-y point (RD frontier)."""
    pts = sorted(set(pts)); h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) >= 0: h.pop()
        h.append(p)
    top = max(range(len(h)), key=lambda i: h[i][1]); return h[:top + 1]

room = list(csv.DictReader(open('room-lambda-vs-cap-v2.csv')))
seven = list(csv.DictReader(open('seven-scene-points-v2.csv')))
lam7 = list(csv.DictReader(open('lambda-7scenes-v1.csv')))
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# ---- left: Room absolute
old_pts = [(float(r['MB']), float(r['test_psnr'])) for r in seven if r['scene'] == 'room' and r['series'] == 'collapse']
new_pts = [(float(r['MB']), float(r['test_psnr'])) for r in room if r['series'] in ('lambda', 'prune')]
oh, nh = upper_hull(old_pts), upper_hull(new_pts)
a.plot(*zip(*old_pts), 'o', ms=3, color=OLD, alpha=0.35, mec='none', zorder=2, label='old route: shared-SH dictionary (K×R) + post-hoc collapse, 49 configs')
a.plot(*zip(*oh), '-', lw=2, color=OLD, zorder=3, label='old route hull')
lam = [(float(r['MB']), float(r['test_psnr'])) for r in room if r['series'] == 'lambda']
pru = [(float(r['MB']), float(r['test_psnr'])) for r in room if r['series'] == 'prune']
a.plot(*zip(*lam), 'o', ms=5, color=NEW, mec=SURF, mew=1, zorder=5, label='new route: rate-controlled growth (λ sweep)')
a.plot(*zip(*pru), 's', ms=5.5, color=PRUNE, mec=SURF, mew=1, zorder=6, label='new route: λ growth + phase-B λ-prune')
a.plot(*zip(*nh), '-', lw=2, color=NEW, zorder=4, label='new route hull')
a.axhline(ORIG['room'][1], color=REF, lw=0.8, ls=(0, (4, 3)), zorder=1)
a.text(0.02, ORIG['room'][1], f'original FP32: {ORIG["room"][1]} dB at {ORIG["room"][0]} MB', transform=a.get_yaxis_transform(), ha='left', va='bottom', fontsize=7, color=REF)
# equal-bytes / equal-PSNR callouts
a.annotate('old default\n25.5 MB / 27.88', (25.513, 27.883), textcoords='offset points', xytext=(8, -26), fontsize=7, color=OLD, ha='left')
a.annotate('13.7 MB / 27.99\n(half the bytes, +0.1 dB)', (13.699, 27.985), textcoords='offset points', xytext=(-6, 9), fontsize=7, color=NEW, ha='right')
a.annotate('5.2 MB / 27.38 (142×)', (5.228, 27.382), textcoords='offset points', xytext=(-8, 10), fontsize=7, color=PRUNE, ha='left')
a.annotate('3.0 MB / 26.51 (246×)', (3.013, 26.509), textcoords='offset points', xytext=(8, -3), fontsize=7, color=NEW)
a.annotate('old route cannot go below ~17 MB\nwithout losing > 1 dB', (16.629, 27.222), textcoords='offset points', xytext=(4, -28), fontsize=7, color=OLD)
a.set_xscale('log'); a.set_xticks([3, 5, 10, 20, 30, 40]); a.xaxis.set_major_formatter(mt.ScalarFormatter()); a.xaxis.set_minor_formatter(mt.NullFormatter())
a.set_xlabel('delivered size (MB, log)'); a.set_ylabel('test PSNR (dB)'); a.set_title('Room', fontsize=10, loc='left')
a.margins(y=0.15)

# ---- right: 7 scenes relative to original
for sc, (mb0, p0) in ORIG.items():
    pts = [(float(r['MB']), float(r['test_psnr']) - p0) for r in seven if r['scene'] == sc and r['series'] == 'collapse' and r['label'].startswith('v')]
    h = upper_hull(pts); b.plot(*zip(*h), '-', lw=1.2, color=OLD, alpha=0.6, zorder=2, label='old route hull, one line per scene' if sc == 'room' else None)
    lp = sorted([(float(r['MB']), float(r['dB_vs_orig'])) for r in lam7 if r['scene'] == sc and r['lam_v2'] in ('0.002', '0.006', '0.00185', '0.0062')])
    b.plot(*zip(*lp), '-o', lw=1.2, ms=4, color=NEW, mec=SURF, mew=0.8, alpha=0.85, zorder=4, label='new route, λ = 2e-3 and 6e-3 per scene' if sc == 'room' else None)
# means
b.plot([23.6], [-0.44], 'D', ms=9, color=OLD, mec=REF, mew=0.8, zorder=6); b.annotate('old route default, 7-scene mean\n23.6 MB / −0.44 dB / 28×', (23.6, -0.44), textcoords='offset points', xytext=(12, -22), fontsize=7.5, color=OLD)
for mb, db, x_, txt, dx in ((19.6, -0.26, 36, 'λ=2e-3, 7-scene mean\n19.6 MB / −0.26 dB / 36×', (-12, 14)), (8.8, -0.88, 80, 'λ=6e-3, 7-scene mean\n8.8 MB / −0.88 dB / 80×', (-12, -34))):
    b.plot([mb], [db], 'D', ms=9, color=NEW, mec=REF, mew=0.8, zorder=6); b.annotate(txt, (mb, db), textcoords='offset points', xytext=dx, fontsize=7.5, color=NEW, ha='right' if dx[0] < 0 else 'left')
b.axhline(0, color=REF, lw=0.8, ls=(0, (4, 3)), zorder=1); b.text(0.98, 0, 'original FP32 (456–742 MB)', transform=b.get_yaxis_transform(), ha='right', va='bottom', fontsize=7, color=REF)
b.set_xscale('log'); b.set_xticks([5, 7, 10, 15, 20, 30, 40]); b.xaxis.set_major_formatter(mt.ScalarFormatter()); b.xaxis.set_minor_formatter(mt.NullFormatter())
b.set_xlabel('delivered size (MB, log)'); b.set_ylabel('PSNR vs original (dB)'); b.set_title('7 Mip-NeRF 360 scenes, relative to each original', fontsize=10, loc='left')
b.margins(y=0.12)
for ax in (a, b):
    ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.suptitle('Same codec, two ways to pick the vertices: deciding them while the mesh grows (λ) dominates deciding them afterwards (collapse) — everywhere, and it reaches sizes the old route cannot', fontsize=9, y=1.01)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, frameon=False, loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
