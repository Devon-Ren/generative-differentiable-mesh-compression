"""Index spatial-locality figure: bits/vertex of the two SH index streams vs dictionary size.
    python plot_index_locality.py index-locality-v1.csv <out-stem> [--note "..."]
Left: base code z vs K (raw log2 K, marginal entropy H(z), neighbour-conditional H(z|nbr), context coder actual).
Right: residual code u vs R at K=4096 (raw log2 R, H(u|nbr), context coder actual).  Lines = mean over Room vertex
levels coded with the ctx format; markers = individual levels.  Rows with fmt != ctx are excluded (their coder column is zstd).
"""
import argparse, csv, math, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LIGHT = dict(surface='#fcfcfb', text='#0b0b0b', text2='#52514e', muted='#8a8984', grid='#e6e5e1')
C_RAW, C_H, C_COND, C_CODER = '#9a9893', '#52514e', '#2a78d6', '#eb6834'
LEVEL_MARK = {0.6: 'v', 0.7: 'D', 0.8: 's', 1.0: 'o'}


def main():
    a = argparse.ArgumentParser(); a.add_argument('csv'); a.add_argument('out'); a.add_argument('--note', default=''); args = a.parse_args()
    rows = [r for r in csv.DictReader(open(args.csv)) if r['scene'] == 'room' and r['fmt'] == 'ctx']
    for r in rows:
        for k in ('level', 'hit_z', 'H_z', 'H_z_given_nbr', 'coder_bits_z'): r[k] = float(r[k])
        r['K'] = int(r['K']); r['R'] = int(r['R'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=200); fig.patch.set_facecolor(LIGHT['surface'])
    levels = sorted({r['level'] for r in rows})

    def series(ax, xs, getter, color, label, ls='-', marker=True):
        means = []
        for x in xs:
            vals = [(r['level'], getter(r)) for r in rows if getter(r) is not None and xsel(r) == x]
            seen = {}
            for lv, v in vals: seen[lv] = v                    # one value per level (duplicates across R collapse)
            means.append(st.mean(seen.values()))
            if marker:
                for lv, v in seen.items():
                    ax.scatter(x, v, marker=LEVEL_MARK[lv], s=26, color=color, edgecolor=LIGHT['surface'], linewidth=.8, zorder=4, alpha=.9)
        ax.plot(xs, means, color=color, lw=2, ls=ls, zorder=3, label=label); return means

    # --- left: base code z vs K -----------------------------------------------------------------------------------
    Ks = sorted({r['K'] for r in rows}); xsel = lambda r: r['K']
    ax1.plot(Ks, [math.log2(k) for k in Ks], color=C_RAW, lw=1.5, ls='--', zorder=2, label='raw  log2 K')
    series(ax1, Ks, lambda r: r['H_z'], C_H, 'H(z)  marginal')
    mc = series(ax1, Ks, lambda r: r['coder_bits_z'], C_CODER, 'context coder (actual)')
    mh = series(ax1, Ks, lambda r: r['H_z_given_nbr'], C_COND, 'H(z | neighbour)')
    ax1.set_xscale('log', base=2); ax1.set_xticks(Ks); ax1.set_xticklabels([str(k) for k in Ks])
    ax1.set_xlabel('base dictionary size K', color=LIGHT['text2']); ax1.set_ylabel('bits per vertex', color=LIGHT['text2'])
    ax1.set_title('base code z: neighbours agree 12-22 % of the time', loc='left', fontsize=10.5, color=LIGHT['text'])
    ax1.annotate(f'{mh[0]:.1f}', (Ks[0], mh[0]), xytext=(9, -11), textcoords='offset points', ha='center', fontsize=8, color=C_COND)
    ax1.annotate(f'{mh[-1]:.1f}', (Ks[-1], mh[-1]), xytext=(-9, -11), textcoords='offset points', ha='center', fontsize=8, color=C_COND)
    ax1.annotate(f'{mc[-1]:.1f}', (Ks[-1], mc[-1]), xytext=(-9, 7), textcoords='offset points', ha='center', fontsize=8, color=C_CODER)
    # --- right: residual code u vs R at K = 4096 -----------------------------------------------------------------
    rows_u = [r for r in rows if r['K'] == 4096 and r['R'] > 1]
    for r in rows_u:
        for k in ('H_u_given_nbr', 'coder_bits_u'): r[k] = float(r[k])
    Rs = sorted({r['R'] for r in rows_u}); xsel = lambda r: r['R']
    rows_bak = rows; rows = rows_u
    ax2.plot(Rs, [math.log2(x) for x in Rs], color=C_RAW, lw=1.5, ls='--', zorder=2, label='raw  log2 R')
    series(ax2, Rs, lambda r: r['coder_bits_u'], C_CODER, 'context coder (actual)')
    mhu = series(ax2, Rs, lambda r: r['H_u_given_nbr'], C_COND, 'H(u | neighbour)')
    rows = rows_bak
    ax2.set_xscale('log', base=2); ax2.set_xticks(Rs); ax2.set_xticklabels([str(x) for x in Rs])
    ax2.set_xlabel('residual dictionary size R  (K = 4096)', color=LIGHT['text2'])
    ax2.set_title('residual code u: neighbours agree 2-5 %', loc='left', fontsize=10.5, color=LIGHT['text'])
    for ax in (ax1, ax2):
        ax.set_facecolor(LIGHT['surface'])
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(LIGHT['grid'])
        ax.tick_params(colors=LIGHT['text2'], length=0); ax.grid(True, color=LIGHT['grid'], lw=1, zorder=0); ax.set_axisbelow(True)
        ax.set_ylim(5, 14.5)
    ax1.legend(frameon=False, loc='upper left', fontsize=8.5, labelcolor=LIGHT['text2'])
    h2 = [Line2D([], [], color=C_RAW, ls='--', lw=1.5, label='raw  log2 R'), Line2D([], [], color=C_CODER, lw=2, label='context coder (actual)'),
          Line2D([], [], color=C_COND, lw=2, label='H(u | neighbour)')] + \
         [Line2D([], [], marker=LEVEL_MARK[lv], ls='', color=LIGHT['text2'], markersize=5, label=f'{int(lv*100)} % vertices') for lv in levels]
    ax2.legend(handles=h2, frameon=False, loc='upper left', fontsize=8.5, labelcolor=LIGHT['text2'], ncol=2)
    if args.note:
        import textwrap; fig.text(.012, .012, '\n'.join(textwrap.wrap(args.note, 150)), fontsize=7.5, color=LIGHT['muted'], va='bottom')
    fig.tight_layout(rect=(0, .15 if args.note else 0, 1, 1))
    for ext in ('png', 'pdf'): fig.savefig(f'{args.out}.{ext}', facecolor=fig.get_facecolor())
    print('wrote', args.out + '.png/.pdf')


if __name__ == '__main__':
    main()
