"""Natural-stop analysis (10/19): fraction of triangles passing the split rule per densification interval, from rd_log.jsonl of all lambda runs.
usage: python plot_pass_curves.py rd-pass-curve-2026-10-19.txt out-stem"""
import re, sys, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
runs = {}
for line in open(sys.argv[1]):
    m = re.match(r'(\S+) lam (\S+) norm (\S+) \| iter:pass/n_tri\(split\) -> (.*?) \| pass_frac', line)
    name, lam, norm, body = m.groups()
    pts = [(int(a), int(b), int(c), int(d)) for a, b, c, d in re.findall(r'(\d+):(\d+)/(\d+)\((\d+)\)', body)]
    scene = name.split('-')[0]
    runs.setdefault(scene, []).append((name, float(lam), norm, pts))
out = sys.argv[2]
MPX = 1.618  # Room images_2, v1 lambdas -> per-Mpx units
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#999', 'axes.linewidth': 0.6, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
order = ['room', 'garden', 'bonsai', 'counter', 'kitchen', 'bicycle', 'stump']
greens = ['#bfe9d6', '#8fd6b6', '#5cc094', '#1baf7a', '#128a5e', '#0a5c3e']
for ax, scene in zip(axes.flat, order):
    rs = sorted([r for r in runs[scene] if 'rd3' not in r[0]], key=lambda r: r[1])
    for k, (name, lam, norm, pts) in enumerate(rs):
        lam2 = lam / MPX if norm == 'none' else lam
        x = [p[0] for p in pts]; y = [p[1] / p[2] for p in pts]
        if lam2 < 1e-3:
            ax.plot(x, y, '-', color='#bbb', lw=1.2, zorder=2)
            ax.annotate('λ=0' if lam == 0 else f'λ={lam2:.1e}', (x[-1], y[-1]), textcoords='offset points', xytext=(3, -3), fontsize=6.5, color='#888')
        else:
            col = greens[min(5, 2 + k)] if scene == 'room' else ('#1baf7a' if lam2 < 4e-3 else '#0a5c3e')
            ax.plot(x, y, '-o', color=col, lw=1.6, ms=3, mec='#fcfcfb', mew=0.8, zorder=4)
            ax.annotate(f'λ={lam2:.2g}', (x[-1], y[-1]), textcoords='offset points', xytext=(3, -3), fontsize=6.5, color=col)
    ax.axvline(4000, color='#ddd', lw=0.8, zorder=1); ax.axvline(5000, color='#ddd', lw=0.8, zorder=1)
    ax.set_yscale('log'); ax.set_title(scene, fontsize=9, loc='left'); ax.grid(color='#e6e6e3', lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
axes.flat[7].axis('off')
axes.flat[7].text(0, 0.95, 'fraction of triangles with residual / bytes > λ\nat each densification interval (500 steps, 1000–9500).\n\n'
                  'grey: λ ≤ 6e-4 per Mpx: the candidate set GROWS;\nonly the add_percentage limiter / 4M cap stops it.\n\n'
                  'green: λ ≥ 1.85e-3: after the opacity floor starts\n(5000) the candidate set decays geometrically,\nhalf-life ≈ 3000 steps at λ=6e-3 (4.5–7.6 intervals),\n'
                  '≈ 5000–10000 steps at λ=2e-3.\nTriangle count 7500→9500: within ±8 % at λ=6e-3\non 7/7 scenes (growth ≈ pruning: a steady state set\nby λ); at λ=2e-3 flat on 4/7, still +8…+24 % on\nbicycle / counter / kitchen; λ=0: +48 %.\n\n'
                  'vertical lines: start_pruning 4000, opacity floor 5000.', fontsize=7.5, va='top', transform=axes.flat[7].transAxes)
for ax in axes[1]: ax.set_xlabel('iteration')
for ax in axes[:, 0]: ax.set_ylabel('passing fraction')
fig.suptitle('Does λ stop growth by itself?  With a price the candidate set decays geometrically; without one it grows', fontsize=9.5, y=0.995)
fig.tight_layout(); fig.savefig(out + '.png', dpi=200, bbox_inches='tight'); fig.savefig(out + '.pdf', bbox_inches='tight'); print('wrote', out)
