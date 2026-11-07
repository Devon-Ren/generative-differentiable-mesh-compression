"""Report figure 3 (2026-11-01, v1): Gaussian-class compressors vs ours, Mip-NeRF 360 means.  (a) honest absolute PSNR vs size;
(b) quality kept relative to the full-precision model each method compresses (reference in the label)."""
import sys; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v1 import *
rc(); rows, _ = load_rows(); lam = mean_lambda(rows, M360); l1 = mean_level1(rows, M360); o_mb, o_psnr = mean_orig(M360); B = baselines('m360')
REF = {'HAC++': (27.50, 'Scaffold-GS 254 MB'), 'HAC': (27.50, 'Scaffold-GS 254 MB'), 'ContextGS': (27.50, 'Scaffold-GS 254 MB'), 'RDO-Gaussian': (27.49, '3DGS 751 MB'), 'LightGaussian': (27.46, '3DGS 751 MB'), 'Compact3DGS': (27.46, '3DGS 751 MB')}
fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.9))
# (a) absolute
smooth(a, lam, label='ours (4 λ)'); point(a, l1[0], l1[1], None, color=POST, hollow=False, ms=5.5)
a.plot(o_mb, o_psnr, '*', color=STAR, ms=12, zorder=6); a.annotate('MeshSplatting FP32 (657 MB)', (o_mb, o_psnr), textcoords='offset points', xytext=(0, -13), ha='center', fontsize=7.2, color=INK)
for m, pts in B.items():
    c = GAUSS.get(m, '#888')
    if m in ('3DGS', 'Scaffold-GS'): point(a, pts[0][0], pts[0][1], m, color=c, hollow=True, dx=0, dy=9, ha='center'); continue
    if len(pts) > 1: a.plot([p[0] for p in pts], [p[1] for p in pts], '-o', color=c, lw=1.2, ms=4.5, mec='white', mew=0.7, zorder=3)
    else: a.plot(pts[0][0], pts[0][1], 'o', color=c, ms=4.5, mec='white', mew=0.7, zorder=3)
    lab = {'HAC++': (0, 8, 'center'), 'HAC': (6, -4, 'left'), 'ContextGS': (-6, 4, 'right'), 'RDO-Gaussian': (0, -10, 'center'), 'Compact3DGS': (6, 0, 'left'), 'LightGaussian': (0, 9, 'center')}.get(m, (6, 0, 'left'))
    q = pts[-1] if m != 'ContextGS' else pts[0]; a.annotate(m, q, textcoords='offset points', xytext=lab[:2], ha=lab[2], fontsize=7.2, color=c)
a.axhspan(23, 24.9, color=OURS, alpha=0.05, lw=0); a.text(900, 23.1, 'opaque-mesh class\n(representation gap ≈ 3 dB)', fontsize=7, color=OURS, ha='right', va='bottom')
logx(a, 1.3, 1300, ticks=(3, 10, 30, 100, 300, 1000)); a.set_ylim(22.8, 28.4); a.set_xlabel('delivered size (MB, log)'); a.set_ylabel('test PSNR (dB), Mip-NeRF 360 mean')
a.set_title('(a) absolute quality vs size, published numbers', loc='left'); better(a, text='better')
a.legend(frameon=False, loc='center left')
# (b) relative to own base
b.axhline(0, color=INK, lw=0.8, ls=(0, (4, 3)))
smooth(b, [(x, y - o_psnr) for x, y in lam], label='ours vs our FP32 MeshSplatting (657 MB)')
for i, (x, y) in enumerate(lam): label_ratio(b, x, y - o_psnr, o_mb / x, dy=(9 if i == 0 else -11))
point(b, l1[0], l1[1] - o_psnr, None, color=POST, hollow=False, ms=5.5)
for m in ('HAC++', 'HAC', 'ContextGS', 'RDO-Gaussian', 'LightGaussian', 'Compact3DGS'):
    pts = B[m]; rp, rn = REF[m]; c = GAUSS[m]
    if len(pts) > 1: b.plot([p[0] for p in pts], [p[1] - rp for p in pts], '-o', color=c, lw=1.2, ms=4.5, mec='white', mew=0.7, zorder=3, label=f'{m} vs {rn}')
    else: b.plot(pts[0][0], pts[0][1] - rp, 'o', color=c, ms=4.5, mec='white', mew=0.7, zorder=3, label=f'{m} vs {rn}')
logx(b, 1.3, 130, ticks=(1, 3, 10, 30, 100)); b.set_ylim(-3.4, 0.9); b.set_xlabel('delivered size (MB, log)'); b.set_ylabel('Δ test PSNR vs the model each method compresses (dB)')
b.set_title('(b) quality kept relative to each method\'s own full-precision model', loc='left'); better(b, text='better')
b.legend(frameon=False, loc='lower right', fontsize=7)
fig.tight_layout(); save(fig, 'fig-gauss-class-v1')
