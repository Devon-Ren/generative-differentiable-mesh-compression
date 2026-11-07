"""RD figure: SH codebook sweep (K x R) at one vertex-retention level + the upper convex hull of those points.
    python plot_codebook_rd.py <points.csv> <out-stem> [--title "..."] [--note "..."] [--default K,R]
CSV columns: K,R,MB,test_psnr[,...].  Writes <out-stem>.png and <out-stem>.pdf.
"""
import argparse, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LIGHT = dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', muted='#8a8984', grid='#e6e5e1')
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']          # validated 3-slot categorical palette (all-pairs, light)
HULL = '#2b2b2b'


def upper_hull(pts):
    """Upper convex hull of (x, y) points: x ascending, slopes decreasing (the RD-efficient frontier)."""
    pts = sorted(pts, key=lambda p: (p[0], -p[1])); hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2][:2], hull[-1][:2]
            # pop hull[-1] when it lies on/below the chord hull[-2] -> p (cross >= 0 = left turn)
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0: hull.pop()
            else: break
        hull.append(p)
    # keep only points that dominate the trend: drop any point lower than its predecessor (hull must be non-decreasing)
    out = [hull[0]]
    for p in hull[1:]:
        if p[1] > out[-1][1]: out.append(p)
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('csv'); a.add_argument('out'); a.add_argument('--title', default='')
    a.add_argument('--note', default=''); a.add_argument('--default', default='4096,256'); args = a.parse_args()
    rows = [dict(K=int(r['K']), R=int(r['R']), MB=float(r['MB']), psnr=float(r['test_psnr'])) for r in csv.DictReader(open(args.csv))]
    Ks = sorted({r['K'] for r in rows}); color = {k: SERIES[i] for i, k in enumerate(Ks)}
    hull = upper_hull([(r['MB'], r['psnr'], r) for r in rows]); hull_set = {id(h[2]) for h in hull}
    dk, dr = map(int, args.default.split(','))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titleweight': 'normal'})
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200); fig.patch.set_facecolor(LIGHT['surface']); ax.set_facecolor(LIGHT['surface'])
    ax.plot([h[0] for h in hull], [h[1] for h in hull], color=HULL, lw=2, solid_joinstyle='round', solid_capstyle='round', zorder=2, label='convex hull')
    for r in rows:
        on = id(r) in hull_set
        ax.scatter(r['MB'], r['psnr'], s=64 if on else 40, color=color[r['K']], edgecolor=LIGHT['surface'], linewidth=1.5, zorder=4 if on else 3, alpha=1 if on else .55)
    # selective direct labels: R on hull points, plus the default
    d = next(r for r in rows if r['K'] == dk and r['R'] == dr)
    for h in hull:
        r = h[2]
        if r is d: continue                                   # the default annotation already names its R
        ax.annotate(f"R={r['R']}", (r['MB'], r['psnr']), xytext=(7, -12), textcoords='offset points', ha='left', fontsize=8, color=LIGHT['text2'], zorder=5)
    ax.scatter(d['MB'], d['psnr'], s=160, facecolor='none', edgecolor=HULL, linewidth=1.5, zorder=5)
    ax.annotate(f"default  K={dk}, R={dr}\n{d['MB']:.2f} MB, {d['psnr']:.3f} dB", (d['MB'], d['psnr']), xytext=(-12, 26), textcoords='offset points', ha='right', fontsize=8.5, color=LIGHT['text'],
                arrowprops=dict(arrowstyle='-', color=LIGHT['muted'], lw=.8, shrinkB=8), zorder=6)
    ax.set_xlabel('delivered size (MB)', color=LIGHT['text2']); ax.set_ylabel('test PSNR (dB)', color=LIGHT['text2'])
    if args.title: ax.set_title(args.title, loc='left', fontsize=11, color=LIGHT['text'], pad=10)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(LIGHT['grid'])
    ax.tick_params(colors=LIGHT['text2'], length=0); ax.grid(True, color=LIGHT['grid'], lw=1, zorder=0); ax.set_axisbelow(True)
    handles = [Line2D([], [], marker='o', ls='', color=color[k], markersize=7, label=f'K = {k}') for k in Ks] + [Line2D([], [], color=HULL, lw=2, label='convex hull')]
    ax.legend(handles=handles, frameon=False, loc='lower right', fontsize=9, labelcolor=LIGHT['text2'])
    ys = [r['psnr'] for r in rows]; xs = [r['MB'] for r in rows]
    ax.set_ylim(min(ys) - .06, max(ys) + .10); ax.set_xlim(min(xs) - .8, max(xs) + .8)
    if args.note:
        import textwrap; fig.text(.012, .012, '\n'.join(textwrap.wrap(args.note, 118)), fontsize=7.5, color=LIGHT['muted'], va='bottom')
    fig.tight_layout(rect=(0, .07 if args.note else 0, 1, 1))
    for ext in ('png', 'pdf'): fig.savefig(f'{args.out}.{ext}', facecolor=fig.get_facecolor())
    print('hull:', ' -> '.join(f"K{h[2]['K']}R{h[2]['R']}" for h in hull)); print('wrote', args.out + '.png/.pdf')


if __name__ == '__main__':
    main()
