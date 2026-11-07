"""v2: Seven-scene RD figure with three routes: collapse (faded orange), vertex cap (blue), rate-controlled growth lambda (green, headline).
the global upper hull across both routes, y = test PSNR minus the original FP32 model.
Also writes a markdown paper table (<out>-table.md) with the budget headline points.
usage: python plot_scene_hulls_7.py seven-scene-points-v1.csv <out-stem>"""
import argparse, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_codebook_rd import upper_hull

L = dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', muted='#8a8984', grid='#e6e5e1')
SCENES = [('room', 'Room (indoor)', 28.263, 742), ('garden', 'garden (outdoor)', 24.732, 740), ('bonsai', 'bonsai (indoor)', 27.291, 680),
          ('counter', 'counter (indoor)', 25.944, 575), ('kitchen', 'kitchen (indoor)', 26.454, 456), ('bicycle', 'bicycle (outdoor)', 23.033, 680),
          ('stump', 'stump (outdoor)', 24.928, 698)]
C = {'budget': '#2a78d6', 'collapse': '#eb6834', 'lambda': '#1baf7a'}; HULL = '#2b2b2b'
OLD = {'room': 'v0.8 K4096 R256 q14', 'garden': 'v0.8 K4096 R256 q15', 'bonsai': 'v1.0 K4096 R256 q15', 'counter': 'v0.8 K4096 R256 q15/q7',
       'kitchen': 'v0.8 K4096 R256 q16/q7', 'bicycle': 'v0.8 K4096 R256 q16/q7', 'stump': 'v0.8 K4096 R256 q16/q7'}


def main():
    a = argparse.ArgumentParser(); a.add_argument('csv'); a.add_argument('out'); args = a.parse_args()
    rows = [dict(scene=r['scene'], series=r['series'], label=r['label'], MB=float(r['MB']), p=float(r['test_psnr'])) for r in csv.DictReader(open(args.csv))]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7), dpi=200); axes = axes.ravel(); fig.patch.set_facecolor(L['surface'])
    table = []
    for ax, (key, title, orig, omb) in zip(axes, SCENES):
        pts = [r for r in rows if r['scene'] == key and 'q13' not in r['label']]  # q13 = deliberately too coarse, off the chart
        hull = upper_hull([(r['MB'], r['p'] - orig, r) for r in pts]); hs = {id(h[2]) for h in hull}
        ax.plot([h[0] for h in hull], [h[1] for h in hull], color=HULL, lw=1.8, zorder=3, solid_joinstyle='round')
        for r in pts:
            on = id(r) in hs
            ax.scatter(r['MB'], r['p'] - orig, s=34 if on else 16, color=C[r['series']], edgecolor=L['surface'], linewidth=.8, alpha=1 if on else .3, zorder=4 if on else 2)
        bud = sorted([r for r in pts if r['series'] == 'lambda' and ('0.002' in r['label'] or '0.0019' in r['label'] or '0.006' in r['label'] or '0.0062' in r['label'])], key=lambda r: -r['MB'])
        hi, lo = bud[0], bud[1]
        for r, dy in ((hi, 10), (lo, -26)):
            ax.scatter(r['MB'], r['p'] - orig, s=140, facecolor='none', edgecolor=HULL, linewidth=1.3, zorder=5)
            ax.annotate(f"{r['MB']:.1f} MB, {r['p'] - orig:+.2f} dB ({omb / r['MB']:.0f}x)", (r['MB'], r['p'] - orig), xytext=(14, dy), textcoords='offset points',
                        fontsize=7.5, color=L['text'], arrowprops=dict(arrowstyle='-', color=L['muted'], lw=.8, shrinkB=8), zorder=6)
        old = next(r for r in pts if r['label'] == OLD[key])
        ax.set_title(f"{title}: {omb} MB, {orig:.2f} dB original", loc='left', fontsize=9.5, color=L['text'])
        ax.set_xlabel('delivered size (MB)', color=L['text2']); ax.set_facecolor(L['surface'])
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(L['grid'])
        ax.margins(y=0.16); ax.tick_params(colors=L['text2'], length=0); ax.grid(True, color=L['grid'], lw=1, zorder=0); ax.set_axisbelow(True)
        onhull = sum(1 for r in bud if id(r) in hs)
        print(key, f'hull ({onhull}/{len(bud)} budget points on it):', ' -> '.join(h[2]['label'] for h in hull))
        table.append((key, omb, orig, hi, lo, old))
    for i in (0, 4): axes[i].set_ylabel('test PSNR vs original (dB)', color=L['text2'])
    ax = axes[7]; ax.axis('off')
    ax.scatter([], [], color=C['collapse'], label='dense training, collapsed afterwards (vertex level x K x R x precision)')
    ax.scatter([], [], color=C['budget'], label='vertex cap --max_points (K4096 R256, scene precision)')
    ax.scatter([], [], color=C['lambda'], label='rate-controlled growth, one lambda for all scenes')
    ax.plot([], [], color=HULL, lw=1.8, label='upper hull over both routes')
    ax.scatter([], [], s=140, facecolor='none', edgecolor=HULL, linewidth=1.3, label='headline points (lambda 2e-3 / 6e-3 per Mpx)')
    ax.legend(frameon=False, loc='center', fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(f'{args.out}.{ext}', facecolor=fig.get_facecolor())
    with open(args.out + '-table.md', 'w') as f:
        f.write('| scene | original (FP32) | lambda 2e-3: MB / PSNR / ratio / dB | lambda 6e-3: MB / PSNR / ratio / dB | old collapse default: MB / PSNR / ratio / dB |\n|---|---|---|---|---|\n')
        for key, omb, orig, hi, lo, old in table:
            cell = lambda r: f"{r['MB']:.2f} / {r['p']:.3f} / {omb / r['MB']:.0f}x / {r['p'] - orig:+.2f}"
            f.write(f"| {key} | {omb} MB / {orig:.3f} | {cell(hi)} | {cell(lo)} | {cell(old)} |\n")
        mean = lambda i: (sum(t[i]['p'] - t[2] for t in table) / 7, sum(t[1] / t[i]['MB'] for t in table) / 7, sum(t[i]['MB'] for t in table) / 7)
        for name, i in (('lambda 2e-3', 3), ('lambda 6e-3', 4), ('old collapse default', 5)):
            d, x, mb = mean(i); f.write(f"\nmean over 7 scenes, {name}: {mb:.1f} MB, {d:+.2f} dB, {x:.0f}x")
    print('wrote', args.out + '.png/.pdf/-table.md')


if __name__ == '__main__':
    main()
