"""make_fig_main_qualitative_v1.py (2026-11-12): the main-paper qualitative figure, cut from the existing montage fig_g0_triptych.jpg
(room, test view DSCF4683): ground truth, FP32 model, BudgetMesh at 5 MB (486k vertices), FP32 model simplified to the same vertex count.
Top row: full view; bottom row: the upper-right region enlarged (where the simplified mesh breaks up)."""
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
im = Image.open('paper/cvpr/figures/fig_g0_triptych.jpg'); CW = 480; Y0, Y1 = 32, 350
cols = [(0, 'Ground truth'), (1, 'FP32 model, 597 MB\n30.79 dB'), (2, 'BudgetMesh, 5 MB\n28.55 dB'), (4, 'FP32 model simplified after training\nto 486k vertices (5 MB), 25.83 dB')]
ZX0, ZX1, ZY0, ZY1 = 300, 480, 0, 120      # zoom region inside one full view (pixels of the 480 x 318 view)
fig, axes = plt.subplots(2, 4, figsize=(7.0, 2.62), gridspec_kw={'height_ratios': [318, 320], 'hspace': 0.03, 'wspace': 0.02})
for j, (c, lab) in enumerate(cols):
    full = im.crop((c * CW, Y0, (c + 1) * CW, Y1)); zoom = full.crop((ZX0, ZY0, ZX1, ZY1)).resize((480, 320), Image.LANCZOS)
    axes[0, j].imshow(full); axes[0, j].add_patch(Rectangle((ZX0 + 1, ZY0 + 1), ZX1 - ZX0 - 3, ZY1 - ZY0 - 2, fill=False, ec='#e03030', lw=0.9))
    axes[1, j].imshow(zoom); axes[0, j].set_title(lab, fontsize=7.5, pad=2.5)
    for a in (axes[0, j], axes[1, j]): a.set_xticks([]); a.set_yticks([]); [sp.set_visible(False) for sp in a.spines.values()]
fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
fig.savefig('paper/cvpr/figures/fig_main_qualitative.pdf', dpi=300); fig.savefig('paper/cvpr/figures/fig_main_qualitative.png', dpi=200); print('written')
