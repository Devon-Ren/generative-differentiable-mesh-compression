"""Three-scene RD figure: every measured delivery point per scene (faded), the global upper hull, and the quality drop vs the
original 740 MB model on the y axis.   python plot_scene_hulls.py three-scene-points-v1.csv <out-stem>
"""
import argparse, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_codebook_rd import upper_hull

LIGHT = dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', muted='#8a8984', grid='#e6e5e1')
SCENES = [('room', 'Room (indoor)', 28.263, 742), ('garden', 'garden (outdoor)', 24.732, 740), ('bonsai', 'bonsai (indoor)', 27.291, 680)]
COLOR = {'room': '#2a78d6', 'garden': '#1baf7a', 'bonsai': '#eb6834'}; HULL = '#2b2b2b'
DEFAULT = {'room': 'v0.8 K4096 R256 q14', 'garden': 'v0.8 K4096 R256 q15', 'bonsai': 'v1.0 K4096 R256 q14'}


def main():
    a = argparse.ArgumentParser(); a.add_argument('csv'); a.add_argument('out'); a.add_argument('--note', default=''); args = a.parse_args()
    rows = [dict(scene=r['scene'], label=r['label'], MB=float(r['MB']), p=float(r['test_psnr'])) for r in csv.DictReader(open(args.csv))]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=200, sharey=True); fig.patch.set_facecolor(LIGHT['surface'])
    for ax, (key, title, orig, orig_mb) in zip(axes, SCENES):
        pts = [r for r in rows if r['scene'] == key]
        hull = upper_hull([(r['MB'], r['p'] - orig, r) for r in pts]); hs = {id(h[2]) for h in hull}
        ax.plot([h[0] for h in hull], [h[1] for h in hull], color=HULL, lw=2, zorder=3, solid_joinstyle='round')
        for r in pts:
            on = id(r) in hs
            ax.scatter(r['MB'], r['p'] - orig, s=34 if on else 18, color=COLOR[key], edgecolor=LIGHT['surface'], linewidth=.8, alpha=1 if on else .35, zorder=4 if on else 2)
        d = next(r for r in pts if r['label'] == DEFAULT[key])
        ax.scatter(d['MB'], d['p'] - orig, s=150, facecolor='none', edgecolor=HULL, linewidth=1.5, zorder=5)
        ax.annotate(f"{d['MB']:.1f} MB, {d['p'] - orig:+.2f} dB\n({orig_mb / d['MB']:.0f}x)", (d['MB'], d['p'] - orig), xytext=(10, -34), textcoords='offset points', fontsize=8.5, color=LIGHT['text'],
                    arrowprops=dict(arrowstyle='-', color=LIGHT['muted'], lw=.8, shrinkB=8), zorder=6)
        ax.set_title(f"{title}: {orig_mb} MB, {orig:.2f} dB original", loc='left', fontsize=10, color=LIGHT['text'])
        ax.set_xlabel('delivered size (MB)', color=LIGHT['text2']); ax.set_facecolor(LIGHT['surface'])
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(LIGHT['grid'])
        ax.tick_params(colors=LIGHT['text2'], length=0); ax.grid(True, color=LIGHT['grid'], lw=1, zorder=0); ax.set_axisbelow(True)
        print(key, 'hull:', ' -> '.join(h[2]['label'] for h in hull))
    axes[0].set_ylabel('test PSNR vs original (dB)', color=LIGHT['text2'])
    if args.note:
        import textwrap; fig.text(.012, .012, '\n'.join(textwrap.wrap(args.note, 190)), fontsize=7.5, color=LIGHT['muted'], va='bottom')
    fig.tight_layout(rect=(0, .1 if args.note else 0, 1, 1))
    for ext in ('png', 'pdf'): fig.savefig(f'{args.out}.{ext}', facecolor=fig.get_facecolor())
    print('wrote', args.out + '.png/.pdf')


if __name__ == '__main__':
    main()
