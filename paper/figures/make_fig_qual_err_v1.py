"""make_fig_qual_err_v1.py (2026-11-20): fig_qual_error_v1 -- full test views and error maps at the same file size (about 10 MB) on four captured
scenes.  Per scene two rows: the renders, and |render - ground truth| (mean over RGB, x4).  Columns: ground truth, MeshSplatting FP32,
MeshSplatting-cap + TFAN, collapse after training + BM-Codec, ByteMesh.  Sizes and whole-view PSNR as in make_fig_qual_v4.py.  Run from the repository root."""
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
SC = [('garden', 'DSC08076', dict(FP32=740, TFAN=10.1, SIMP=10.1, OURS=10.0), dict(FP32=24.35, TFAN=21.62, SIMP=22.12, OURS=23.59)),
      ('room', 'DSCF4715', dict(FP32=597, TFAN=8.4, SIMP=10.0, OURS=10.0), dict(FP32=29.94, TFAN=21.71, SIMP=27.13, OURS=29.28)),
      ('bicycle', '_DSC8727', dict(FP32=680, TFAN=10.2, SIMP=10.6, OURS=9.8), dict(FP32=23.29, TFAN=21.15, SIMP=21.37, OURS=22.68)),
      ('truck', '000049', dict(FP32=432, TFAN=9.2, SIMP=9.1, OURS=9.9), dict(FP32=23.75, TFAN=21.42, SIMP=22.54, OURS=23.52))]
COLS = [('gt', 'Ground truth'), ('FP32', 'MeshSplatting\nFP32'), ('TFAN', 'MeshSplatting-cap\n+ TFAN'), ('SIMP', 'Collapse after\ntraining + BM-Codec'), ('OURS', 'ByteMesh\n(ours)')]
W = 7.1; cw = W / 5; gap = 0.02; load = lambda s, v, k: np.asarray(Image.open('paper/figures/qual-src/%s-%s-%s.jpg' % (s, v, k)).convert('RGB'), np.float32) / 255
ims = {(s, k): load(s, v, k) for s, v, _, _ in SC for k, _ in COLS}; hs = [cw * ims[(s, 'gt')].shape[0] / ims[(s, 'gt')].shape[1] for s, _, _, _ in SC]
head = 0.34; H = head + sum(2 * h + 3 * gap for h in hs); fig = plt.figure(figsize=(W, H)); y = H - head
for (s, v, mb, ps), h in zip(SC, hs):
    gt = ims[(s, 'gt')]
    for row in (0, 1):
        for c, (k, lab) in enumerate(COLS):
            ax = fig.add_axes([c * cw / W + 0.002, (y - h) / H, cw / W - 0.004, h / H]); ax.axis('off')
            if row == 0:
                ax.imshow(ims[(s, k)])
                if k != 'gt': ax.text(0.985, 0.97, '%s MB\n%.2f dB' % (('%d' % mb[k]) if mb[k] >= 100 else ('%.1f' % mb[k]), ps[k]), transform=ax.transAxes, ha='right', va='top', fontsize=6, color='white', linespacing=1.05, bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
                else: ax.text(0.015, 0.04, s, transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5, color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
            elif k != 'gt':
                im = ims[(s, k)]
                if im.shape != gt.shape: im = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR), np.float32) / 255
                ax.imshow(np.clip(np.abs(im - gt).mean(2) * 4, 0, 1), cmap='magma', vmin=0, vmax=1)
            else: ax.text(0.5, 0.5, 'error\n$|$render $-$ ground truth$|$ $\\times 4$', transform=ax.transAxes, ha='center', va='center', fontsize=6.5, color='#5b6270', linespacing=1.2)
            if s == SC[0][0] and row == 0: ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha='center', va='bottom', fontsize=7, fontweight='bold' if k == 'OURS' else 'normal', linespacing=1.0)
        y -= h + gap
    y -= gap
fig.savefig('paper/figures/fig_qual_error_v1.pdf', dpi=220); fig.savefig('paper/figures/fig_qual_error_v1.png', dpi=110); print('ok', W, round(H, 2))
