"""make_figures_v9.py (2026-11-15): ByteMesh naming.
  fig_rd_pair_v4      : the two RD panels of v7, each with its own PSNR axis.
  fig_rd_ssim_lpips_v2: appendix, SSIM and LPIPS against file size, baselines optimized with a vertex cap (means over 11 scenes).
  bd-table-2026-11-15b.txt: per-scene Bjontegaard deltas of ByteMesh against cap + {our codec, TFAN, zstd, Draco}, 11 scenes."""
import re, glob, json, math, collections, statistics as st, runpy, sys, io, contextlib
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mt
src = open('paper/figures/make_figures_v7.py').read()
# ---------- 1. pair figure, independent y axes ----------
s = src.replace('BudgetMesh', 'ByteMesh').replace("fig_rd_pair_v2", "fig_rd_pair_v4")
s = s.replace("ax.set_xscale('log'); ax.set_xlim(3.3, 900); ax.set_ylim(17.4, 25.1)", "ax.set_xscale('log'); ax.set_xlim(*((3.3, 520) if ax is a else (3.3, 900))); ax.set_ylim(*((20.3, 24.1) if ax is a else (17.4, 25.1)))")
s = s.replace("a.set_ylabel('PSNR (dB)'); b.tick_params(labelleft=False)", "a.set_ylabel('PSNR (dB)'); b.set_ylabel('PSNR (dB)')")
s = s.replace("a.text(11.0, 21.5, 'TFAN', ha='right', va='center', fontsize=7.5, color=INK); a.text(19.0, 21.5, 'zstd', ha='left', va='center', fontsize=7.5, color=INK); a.text(64.0, 21.5, 'Draco', ha='left', va='center', fontsize=7.5, color=INK)", "a.text(11.2, 21.6, 'TFAN', ha='right', va='center', fontsize=7.5, color=INK); a.text(17.5, 21.6, 'zstd', ha='left', va='center', fontsize=7.5, color=INK); a.text(52.0, 21.6, 'Draco', ha='left', va='center', fontsize=7.5, color=INK)")
assert 'fig_rd_pair_v4' in s and '20.3, 24.1' in s and "b.set_ylabel" in s and '11.2, 21.6' in s
exec(compile(s, 'v7mod', 'exec'))
# ---------- data per scene ----------
MID = {'room': 1078000, 'kitchen': 1086000, 'counter': 1083000, 'bonsai': 1073000, 'garden': 1073000, 'bicycle': 1080000, 'stump': 1069000, 'flowers': 1071000, 'treehill': 1079000, 'truck': 1078000, 'train': 1085000}
def lev(sc, c): return 0 if c == 600000 else (2 if c == 3000000 else (1 if c == MID[sc] else None))
G = collections.defaultdict(dict)   # (scene, level) -> bits -> dict
R = re.compile(r'(\w+) GENERIC-RD \| mesh (\w+) \| V (\d+) F (\d+) \| sh(\d) \| zstd ([\d.]+) MB \| draco ([\d.]+) MB \| tfan ([\d.]+) MB \| test ([\d.]+) ([\d.]+) ([\d.]+)')
for f in ['release-batch36-generic-on-cap/results/batch36-collect-2026-11-13-2342.txt', 'release-batch33-generic-rd/results/batch33-collect-2026-11-13-0109.txt']:
    for l in open(f):
        m = R.match(l)
        if not m or m[1] not in MIP: continue
        L = 3 if m[2] == 'full' else (lev(m[1], int(m[2][3:])) if m[2].startswith('cap') else None)
        if L is not None: G[(m[1], L)][int(m[5])] = dict(zstd=float(m[6]), draco=float(m[7]), tfan=float(m[8]), q=(float(m[9]), float(m[10]), float(m[11])))
Cq = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+) ([\d.]+) ([\d.]+)')
capall = collections.defaultdict(dict); O = {}
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = Cq.match(l)
        if m and m[1] in MIP:
            capall[m[1]][int(m[2])] = (float(m[4]), float(m[5]))
            if lev(m[1], int(m[2])) is not None: O[(m[1], lev(m[1], int(m[2])))] = (float(m[4]), (float(m[5]), float(m[6]), float(m[7])))
H = collections.defaultdict(dict)
for fn in ('release-batch31-hold/results/batch31-collect-2026-11-13-0126.txt', 'release-batch37-more-points/results/batch37-rows-2026-11-14.txt'):
    for l in open(fn):
        m = re.match(r'(\w+) abl24-hold \| target ([\d.]+) MB \|.*?delivered ([\d.]+) MB.*?\| test ([\d.]+) ([\d.]+) ([\d.]+)', l)
        if m and m[1] in MIP: H[m[1]][float(m[2])] = (float(m[3]), (float(m[4]), float(m[5]), float(m[6])))
# ---------- 2. SSIM / LPIPS figure ----------
mean = lambda v: sum(v) / len(v)
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.2))
for ax, qi, lab, sign in ((axs[0], 1, 'SSIM', 1), (axs[1], 2, 'LPIPS (lower is better)', -1)):
    bmq = [(mean([H[sc][T][0] for sc in MIP]), mean([H[sc][T][1][qi] for sc in MIP])) for T in sorted(H['garden'])]
    coq = [(mean([O[(sc, L)][0] for sc in MIP]), mean([O[(sc, L)][1][qi] for sc in MIP])) for L in (0, 1, 2)]
    def hull_s(p):   # upper hull for SSIM, lower hull for LPIPS
        h = hull([(x, sign * y) for x, y in p]); return [(x, sign * y) for x, y in h]
    curve(ax, bmq, C5[0], '-', 'o', lw=2.2, ms=6, z=8, hl=bmq); curve(ax, coq, C5[1], '-', 's', lw=2.0, ms=5.5, z=7, hl=coq)
    for k, name, mk, col in SPEC:
        p = [(mean([G[(sc, L)][b][k] for sc in MIP]), mean([G[(sc, L)][b]['q'][qi] for sc in MIP])) for L in (0, 1, 2, 3) for b in (8, 6, 4) if all((sc, L) in G and b in G[(sc, L)] for sc in MIP)]
        curve(ax, p, col, '-', mk, hl=hull_s(p))
    ax.set_xscale('log'); ax.set_xlim(3.3, 520)
    ax.xaxis.set_major_locator(mt.FixedLocator([5, 10, 20, 50, 100, 200, 500])); ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: '%g' % v)); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.set_xlabel('file size (MB, log scale)', color=MUTED); ax.set_ylabel(lab)
    print(lab, 'ByteMesh', [(round(x, 2), round(y, 3)) for x, y in bmq], 'cap+ours', [(round(x, 2), round(y, 3)) for x, y in coq])
LG = lambda c, mk, lab: Line2D([], [], color=c, marker=mk, ms=5, mec=SURF, mew=0.8, lw=1.8, label=lab)
hs2 = [LG(C5[0], 'o', 'ByteMesh (ours)'), LG(C5[1], 's', 'vertex cap + our codec'), LG(C5[2], 'D', 'vertex cap + TFAN'), LG(C5[3], '^', 'vertex cap + zstd'), LG(C5[4], 'v', 'vertex cap + Draco')]
fig.legend(handles=hs2, loc='upper center', ncol=5, frameon=False, fontsize=7, handlelength=2.4, columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.6); fig.savefig('paper/figures/fig_rd_ssim_lpips_v2.pdf'); fig.savefig('paper/figures/fig_rd_ssim_lpips_v2.png', dpi=170)
# ---------- 3. BD table ----------
def envelope(pts):
    out = []
    for r, p in sorted(pts):
        if not out or p > out[-1][1]: out.append((r, p))
    return out
def bd(a_, b_):
    a_, b_ = sorted(a_), sorted(b_); la, pa = np.log10([x[0] for x in a_]), np.array([x[1] for x in a_]); lb, pb = np.log10([x[0] for x in b_]), np.array([x[1] for x in b_])
    lo, hi = max(la[0], lb[0]), min(la[-1], lb[-1]); dp = None
    if hi - lo > 1e-6: g = np.linspace(lo, hi, 400); dp = float(np.trapezoid(np.interp(g, la, pa) - np.interp(g, lb, pb), g) / (hi - lo))
    plo, phi = max(pa.min(), pb.min()), min(pa.max(), pb.max()); dr = None
    if phi - plo > 1e-6: q = np.linspace(plo, phi, 400); dr = float((10 ** (np.trapezoid(np.interp(q, pa, la) - np.interp(q, pb, lb), q) / (phi - plo)) - 1) * 100)
    return dp, dr
out = io.StringIO(); P = lambda *a: print(*a, file=out)
P('Bjontegaard deltas of ByteMesh (hold, 5 targets) against each baseline, per scene, 11 Mip-NeRF 360 / T&T scenes; piecewise-linear over log10(size); upper envelopes.')
P('BD-PSNR needs an overlap in file size, BD-rate an overlap in PSNR; n = scenes with that overlap.\n')
res = collections.defaultdict(list)
for sc in MIP:
    h = envelope([(v[0], v[1][0]) for v in H[sc].values()]); row = [sc]
    base = {'cap + our codec': envelope(capall[sc].values())}
    for k in ('tfan', 'zstd', 'draco'): base['cap + ' + k] = envelope([(G[(sc, L)][b][k], G[(sc, L)][b]['q'][0]) for L in (0, 1, 2, 3) if (sc, L) in G for b in G[(sc, L)]])
    for name, c in base.items():
        r = bd(h, c); res[name].append(r); row.append('%s: %s dB, %s %%' % (name, '%+.2f' % r[0] if r[0] is not None else 'n/a', '%+.1f' % r[1] if r[1] is not None else 'n/a'))
    P(' | '.join(row))
P()
for name, v in res.items():
    dp = [x[0] for x in v if x[0] is not None]; dr = [x[1] for x in v if x[1] is not None]
    P('%-16s BD-PSNR %s (n %d, range %s) | BD-rate %s (n %d, range %s)' % (name, '%+.2f dB' % st.mean(dp) if dp else 'n/a', len(dp), '%+.2f..%+.2f' % (min(dp), max(dp)) if dp else '-', '%+.1f %%' % st.mean(dr) if dr else 'n/a', len(dr), '%+.1f..%+.1f' % (min(dr), max(dr)) if dr else '-'))
open('paper/figures/bd-table-2026-11-15b.txt', 'w').write(out.getvalue()); print(out.getvalue())
