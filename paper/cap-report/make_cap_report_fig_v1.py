"""make_cap_report_fig_v1.py: file size against the vertex cap, all cap runs of four scenes (small multiples, one series each)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, MUTED, GRID, BLUE = '#0b0b0b', '#52514e', '#dddcd6', '#2a78d6'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})
D = {'garden (target 10 MB)': (10, [(919, 10.608), (936, 10.986), (1073, 11.641), (1074, 11.607), (1104, 13.299), (1105, 10.298), (1106, 10.146)], {1104: (-5, 2, 'right'), 1105: (-6, 0, 'right'), 1073: (-5, 5, 'right')}, 'k'),
     'bonsai (target 10 MB)': (10, [(557, 5.919), (1005, 9.571), (1073, 8.315), (1328, 9.55), (1367, 9.861)], {1005: (-5, -4, 'right'), 1073: (5, -9, 'left'), 1328: (4, -10, 'left')}, 'k'),
     'lego (target 2.5 MB)': (2.5, [(91, 1.966), (130, 2.504), (142, 2.395), (154, 2.792), (167.656, 2.95), (183, 2.791), (259, 3.82)], {130: (5, -9, 'left'), 154: (-5, 4, 'right'), 183: (6, -2, 'left')}, 'k'),
     'mic (target 2.5 MB)': (2.5, [(69, 1.483), (83.9, 1.606), (133.718, 2.068), (145, 2.036), (170, 2.053), (174, 2.041), (247, 2.051)], {133.718: (0, 7, 'center'), 247: (0, -11, 'center')}, 'k')}
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95))
for ax, (name, (T, pts, lab, _)) in zip(axes, D.items()):
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    ax.axhline(T, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1); ax.text(0.985, T, 'target', transform=ax.get_yaxis_transform(), va='top' if name.startswith('garden') else 'bottom', ha='right', fontsize=6.5, color=MUTED)
    ax.plot(x, y, '-', color=BLUE, lw=1.2, zorder=2); ax.plot(x, y, 'o', color=BLUE, ms=4.2, mec='white', mew=0.9, zorder=3)
    for cx, (dx, dy, ha) in lab.items():
        cy = dict(pts)[cx]; ax.annotate(f'{cx:,.0f}k: {cy:.2f}', (cx, cy), textcoords='offset points', xytext=(dx, dy), ha=ha, fontsize=6.3, color=INK)
    ax.set_title(name, fontsize=8, color=INK, pad=4, loc='left'); ax.set_xlabel('vertex cap (thousands)', fontsize=7); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.tick_params(length=2, labelsize=6.8); ax.margins(x=0.34 if name.startswith('garden') else 0.2, y=0.2)
    if name.startswith('lego'): ax.set_xlim(-25, 330); ax.set_xticks([100, 200, 300])
axes[0].set_ylabel('file size (MB)', fontsize=7)
fig.tight_layout(w_pad=0.9); fig.savefig('paper/cap-report/fig_cap_sensitivity.pdf'); fig.savefig('paper/cap-report/fig_cap_sensitivity.png', dpi=170); print('ok')
