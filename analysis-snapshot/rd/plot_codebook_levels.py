"""One figure, several vertex levels: at each level the 12 K x R dictionary points and that level's upper convex hull.
    python plot_codebook_levels.py <points.csv> <out-stem> [--title ...] [--note ...] [--default LEVEL,K,R]
CSV columns: level,K,R,MB,test_psnr[,...]   (level = vertex fraction, e.g. 0.8).  Writes <out-stem>.png and <out-stem>.pdf.
Encoding: colour = vertex level (one-hue sequential ramp, light = fewer vertices), marker shape = K, hull line per level,
direct label at the right end of every hull.  Off-hull points are drawn small and faded so the hulls carry the figure.
"""
import argparse, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LIGHT = dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', muted='#8a8984', grid='#e6e5e1')
RAMP = ['#5894e0', '#407bc5', '#2863ab', '#0f4f95', '#003b7f', '#002669']   # OKLCH blues L .66->.29, all >= 3:1 on the surface; direct labels disambiguate adjacent levels
MARKER = {1024: 'o', 4096: 's', 16384: '^'}


def upper_hull(pts):
    """Upper convex hull, x ascending, slopes decreasing; then drop points lower than their predecessor."""
    pts = sorted(pts, key=lambda p: (p[0], -p[1])); hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2][:2], hull[-1][:2]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0: hull.pop()
            else: break
        hull.append(p)
    out = [hull[0]]
    for p in hull[1:]:
        if p[1] > out[-1][1]: out.append(p)
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('csv'); a.add_argument('out'); a.add_argument('--title', default='')
    a.add_argument('--note', default=''); a.add_argument('--default', default='', help='LEVEL,K,R to ring, e.g. 0.8,4096,256'); args = a.parse_args()
    rows = [dict(level=float(r['level']), K=int(r['K']), R=int(r['R']), MB=float(r['MB']), psnr=float(r['test_psnr'])) for r in csv.DictReader(open(args.csv))]
    levels = sorted({r['level'] for r in rows})
    idx = [round(i * (len(RAMP) - 1) / max(1, len(levels) - 1)) for i in range(len(levels))] if len(levels) > 1 else [len(RAMP) - 1]
    color = {lv: RAMP[j] for lv, j in zip(levels, idx)}       # darkest = most vertices
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200); fig.patch.set_facecolor(LIGHT['surface']); ax.set_facecolor(LIGHT['surface'])
    ends = []
    for lv in levels:
        pts = [r for r in rows if r['level'] == lv]; hull = upper_hull([(r['MB'], r['psnr'], r) for r in pts]); on = {id(h[2]) for h in hull}
        ax.plot([h[0] for h in hull], [h[1] for h in hull], color=color[lv], lw=2, solid_joinstyle='round', solid_capstyle='round', zorder=2)
        for r in pts:
            ax.scatter(r['MB'], r['psnr'], s=46 if id(r) in on else 22, marker=MARKER.get(r['K'], 'o'), color=color[lv], edgecolor=LIGHT['surface'],
                       linewidth=1.2, alpha=1 if id(r) in on else .45, zorder=4 if id(r) in on else 3)
        ends.append((hull[-1][0], hull[-1][1], f'{int(round(lv * 100))} % vertices'))
    # direct label at the right end of each hull; labels pushed apart vertically so close hulls (e.g. 90 % vs 80 %) stay readable
    ys = [r['psnr'] for r in rows]; gap = (max(ys) - min(ys) + .16) * .045; xs_all = [r['MB'] for r in rows]; dx = (max(xs_all) - min(xs_all)) * .012
    ends.sort(key=lambda e: e[1]); placed = []
    for x, y, txt in ends:                                   # label just right of each hull's own end, pushed up only when two ends are close in y
        ty = y if not placed else max(y, placed[-1] + gap); placed.append(ty)
        ax.annotate(txt, (x, y), xytext=(x + dx, ty), textcoords='data', va='center', fontsize=8.5, color=LIGHT['text2'], zorder=5,
                    arrowprops=dict(arrowstyle='-', color=LIGHT['muted'], lw=.7, shrinkA=0, shrinkB=4) if abs(ty - y) > 1e-9 else None)
    if args.default:
        dl, dk, dr = args.default.split(','); d = next(r for r in rows if r['level'] == float(dl) and r['K'] == int(dk) and r['R'] == int(dr))
        ax.scatter(d['MB'], d['psnr'], s=170, facecolor='none', edgecolor='#2b2b2b', linewidth=1.5, zorder=5)
        ax.annotate(f'default  {int(round(d["level"] * 100))} %, K={dk}, R={dr}\n{d["MB"]:.2f} MB, {d["psnr"]:.3f} dB', (d['MB'], d['psnr']), xytext=(-14, 28), textcoords='offset points',
                    ha='right', fontsize=8.5, color=LIGHT['text'], arrowprops=dict(arrowstyle='-', color=LIGHT['muted'], lw=.8, shrinkB=9), zorder=6)
    ax.set_xlabel('delivered size (MB)', color=LIGHT['text2']); ax.set_ylabel('test PSNR (dB)', color=LIGHT['text2'])
    if args.title: ax.set_title(args.title, loc='left', fontsize=11, color=LIGHT['text'], pad=10)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(LIGHT['grid'])
    ax.tick_params(colors=LIGHT['text2'], length=0); ax.grid(True, color=LIGHT['grid'], lw=1, zorder=0); ax.set_axisbelow(True)
    handles = [Line2D([], [], marker=MARKER[k], ls='', color=LIGHT['text2'], markersize=6, label=f'K = {k}') for k in sorted(MARKER) if any(r['K'] == k for r in rows)]
    handles.append(Line2D([], [], color=LIGHT['text2'], lw=2, label='hull per vertex level'))
    ax.legend(handles=handles, frameon=False, loc='lower right', fontsize=8.5, labelcolor=LIGHT['text2'], title='marker = primary codebook K', bbox_to_anchor=(1.0, 0.0), title_fontsize=8.5)
    xs = [r['MB'] for r in rows]; ys = [r['psnr'] for r in rows]
    ax.set_xlim(min(xs) - .8, max(xs) + 3.2); ax.set_ylim(min(ys) - .06, max(ys) + .10)
    if args.note:
        import textwrap; fig.text(.012, .012, '\n'.join(textwrap.wrap(args.note, 125)), fontsize=7.5, color=LIGHT['muted'], va='bottom')
    fig.tight_layout(rect=(0, .07 if args.note else 0, 1, 1))
    for ext in ('png', 'pdf'): fig.savefig(f'{args.out}.{ext}', facecolor=fig.get_facecolor())
    print('levels:', ', '.join(f'{lv:g}' for lv in levels)); print('wrote', args.out + '.png/.pdf')


if __name__ == '__main__':
    main()
