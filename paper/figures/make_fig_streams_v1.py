"""make_fig_streams_v1.py (2026-11-20): fig_streams_v1 -- composition of the ByteMesh files (streams-b50-2026-11-20.txt, read from the codec reports
of the runs).  Stacked bars: mean bytes per stream over the scenes, per target; left the 13 captured scenes, right the 8 objects.  Also prints the
shares and the wall time of the stages after training.  Run from the repository root."""
import re, collections, statistics as st
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
CAP = ['bicycle','flowers','garden','stump','treehill','room','counter','kitchen','bonsai','truck','train','drjohnson','playroom']
GROUP = collections.OrderedDict([('connectivity', ['top_stream']), ('positions', ['local_residual']), ('SH indices', ['ctx_mapping', 'ctx_residual']), ('DC color', ['features_dc']), ('codebooks', ['app_bank', 'app_residual_bank'])])
COLS = ['#2a78d6', '#1baf7a', '#eda100', '#e87ba4', '#8a8f99']; INK, MUTED, GRID = '#1f2430', '#5b6270', '#e3e6ea'
R = collections.defaultdict(dict); ST = collections.defaultdict(dict)
for l in open('release-batch50-traces/results/streams-b50-2026-11-20.txt'):
    m = re.match(r'STREAMS (\w+) ([\d.]+) \| file (\d+) \| V (\d+) F (\d+) \| (.*?) \| stages s: convert (\S+) finetune (\S+) encode (\S+)', l)
    if not m: continue
    pb = dict((k, int(v)) for k, v in (t.split(':') for t in m[6].split())); d = {g: sum(pb.get(k, 0) for k in ks) for g, ks in GROUP.items()}; d['file'] = int(m[3]); d['other'] = d['file'] - sum(d[g] for g in GROUP)
    d['V'] = int(m[4]); d['F'] = int(m[5]); R[m[1]][float(m[2])] = d; ST[m[1]][float(m[2])] = tuple(float(x) for x in (m[7], m[8], m[9]))
plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'pdf.fonttype': 42})
fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.5), gridspec_kw=dict(width_ratios=[5, 3]))
for ax, (name, S, T) in zip(axs, (('13 captured scenes', CAP, [5, 7.5, 10, 15, 20]), ('8 objects', [s for s in R if s not in CAP], [1.5, 2.5, 4]))):
    bot = [0.0] * len(T); print(name)
    for (g, _), col in zip(GROUP.items(), COLS):
        v = [st.mean(R[s][t][g] for s in S) / 1e6 for t in T]; ax.bar(range(len(T)), v, bottom=bot, color=col, width=0.72, label=g, edgecolor='white', linewidth=0.4)
        tot = [st.mean(R[s][t]['file'] for s in S) / 1e6 for t in T]
        for i, (x, b, tt) in enumerate(zip(v, bot, tot)):
            if x > 0.05 * max(tot): ax.text(i, b + x / 2, '%.0f%%' % (100 * x / tt), ha='center', va='center', fontsize=5.6, color='white')
        print('  %-13s ' % g + ' '.join('%5.2f MB (%4.1f%%)' % (x, 100 * x / tt) for x, tt in zip(v, tot))); bot = [b + x for b, x in zip(bot, v)]
    print('  file          ' + ' '.join('%5.2f MB' % x for x in tot) + ' | other bytes max %d' % max(R[s][t]['other'] for s in S for t in T))
    print('  bits/vertex: ' + ' | '.join('T %g: ' % t + ' '.join('%s %.1f' % (g[:4], st.mean(8 * R[s][t][g] / R[s][t]['V'] for s in S)) for g in list(GROUP)[:4]) + ' ; conn bits/face %.2f' % st.mean(8 * R[s][t]['connectivity'] / R[s][t]['F'] for s in S) for t in T))
    ax.set_xticks(range(len(T))); ax.set_xticklabels(['%g' % t for t in T]); ax.set_xlabel('target (MB)\n' + name, color=MUTED); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=2)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
axs[0].set_ylabel('file size (MB)'); h, l = axs[0].get_legend_handles_labels(); fig.legend(h[::-1], l[::-1], loc='upper center', ncol=5, frameon=False, fontsize=6, handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.8, pad=0.4); fig.savefig('paper/figures/fig_streams_v1.pdf'); fig.savefig('paper/figures/fig_streams_v1.png', dpi=220)
for name, S in (('captured (13), 65 runs', CAP), ('objects (8), 24 runs', [s for s in ST if s not in CAP])):
    v = [ST[s][t] for s in S for t in ST[s]]; print('stages after training, %s: convert %.0f s | fine-tuning %.0f s (%.0f..%.0f) | final encoding with evaluation %.0f s' % (name, st.mean(x[0] for x in v), st.mean(x[1] for x in v), min(x[1] for x in v), max(x[1] for x in v), st.mean(x[2] for x in v)))
