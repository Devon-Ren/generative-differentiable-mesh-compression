"""make_fig_qual_v4.py (2026-11-15, v3 + an object row: lego at about 2.5 MB; a square view is centred on black in the full-view cell): qualitative figure, compact.  Rows from sys.argv (default garden room bicycle), one test view each.
Size and PSNR are written inside each crop (top right); no white band between the rows.
Column 0: the full ground-truth view with the crop marked.  Columns 1-5: the same crop of ground truth, MeshSplatting FP32,
vertex cap 600k + TFAN (4-bit SH), FP32 simplified after training + our codec, ByteMesh at a 10 MB target.
Under each crop: file size and the PSNR of the whole view.  Sources: paper/figures/qual-src (renders of batch42), sizes from
batch36 (TFAN), the POST-HOC-COLLAPSE rows (simplified), batch31 (ByteMesh) and the FP32 table.
Output paper/figures/fig_qualitative_v2.{pdf,png}."""
from PIL import Image
import sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
SRC = 'paper/figures/qual-src'
# scene: (view, crop centre x, centre y (fractions of width / height), crop side as a fraction of the height, sizes MB, view PSNR)
ROWS = [
 ('garden',  'DSC08076', 0.355, 0.47, 0.46, dict(FP32=740, TFAN=10.1, SIMP=10.1, OURS=10.0), dict(FP32=24.35, TFAN=21.62, SIMP=22.12, OURS=23.59)),
 ('room',    'DSCF4715', 0.66, 0.32, 0.46, dict(FP32=597, TFAN=8.4,  SIMP=10.0, OURS=10.0), dict(FP32=29.94, TFAN=21.71, SIMP=27.13, OURS=29.28)),
 ('bicycle', '_DSC8727', 0.38, 0.28, 0.46, dict(FP32=680, TFAN=10.2, SIMP=10.6, OURS=9.8),  dict(FP32=23.29, TFAN=21.15, SIMP=21.37, OURS=22.68)),
 ('lego',    'r_166',    0.58, 0.60, 0.38, dict(FP32=126, TFAN=2.77, SIMP=2.73, OURS=2.51), dict(FP32=35.63, TFAN=31.36, SIMP=27.95, OURS=34.31)),
 ('truck',   '000049',   0.80, 0.36, 0.56, dict(FP32=432, TFAN=9.2,  SIMP=9.1,  OURS=9.9),  dict(FP32=23.75, TFAN=21.42, SIMP=22.54, OURS=23.52)),
]
WANT = sys.argv[1:] or ['garden', 'room', 'lego']; ROWS = [r for w in WANT for r in ROWS if r[0] == w]; TAG = '' if WANT == ['garden', 'room', 'lego'] else '_' + '-'.join(WANT)
COLS = [('gt', 'Ground truth'), ('FP32', 'MeshSplatting\nFP32'), ('TFAN', 'MeshSplatting\nwith cap + TFAN'), ('SIMP', 'Simplified\nwith our codec'), ('OURS', 'ByteMesh\n(ours)')]
FULL_W = 1.5; CW = 1.0                      # widths in units of the row height
W_IN = 7.1; unit = W_IN / (FULL_W + 5 * CW + 0.10); LAB = 0.022; HEAD = 0.27
H_IN = HEAD + len(ROWS) * (unit + LAB) - LAB
fig = plt.figure(figsize=(W_IN, H_IN))
def ax_at(x, y, w, h):
    a = fig.add_axes([x / W_IN, 1 - (y + h) / H_IN, w / W_IN, h / H_IN]); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
    return a
gap = 0.02 * unit
for r, (S, cam, cx, cy, side, mb, ps) in enumerate(ROWS):
    y0 = HEAD + r * (unit + LAB)
    gt = Image.open(f'{SRC}/{S}-{cam}-gt.jpg'); W, H = gt.size; s = side * H; x0 = min(max(cx * W - s / 2, 0), W - s); yy = min(max(cy * H - s / 2, 0), H - s); box = tuple(round(v) for v in (x0, yy, x0 + s, yy + s))
    # full view, centre-cropped to the 1.5 : 1 cell
    if W < H * FULL_W:      # narrower than the cell (objects, black background): centre the whole view on black
        tw, th = H * FULL_W, H; fx, fy = -(tw - W) / 2, 0; canvas = Image.new('RGB', (round(tw), H), 'black'); canvas.paste(gt, (round(-fx), 0)); full = canvas
    else:
        tw = H * FULL_W; th = H; fx = min(max(x0 + s / 2 - tw / 2, 0), W - tw); fy = 0; full = gt.crop((round(fx), 0, round(fx + tw), H))
    a = ax_at(0, y0, FULL_W * unit - gap, unit); a.imshow(full)
    a.add_patch(Rectangle((x0 - fx, yy - fy), s, s, fill=False, ec='#ffd21f', lw=1.1)); a.text(0.012, 0.03, S, transform=a.transAxes, color='white', fontsize=7.5, fontweight='bold', va='bottom', bbox=dict(fc='black', ec='none', alpha=0.45, pad=1.2))
    for k, (c, name) in enumerate(COLS):
        im = Image.open(f'{SRC}/{S}-{cam}-{c}.jpg').crop(box).resize((480, 480), Image.LANCZOS)
        x = (FULL_W + 0.10 + k * CW) * unit; a = ax_at(x, y0, CW * unit - gap, unit); a.imshow(im)
        if c == 'OURS':
            for sp in a.spines.values(): sp.set_visible(True); sp.set_edgecolor('#2a78d6'); sp.set_linewidth(1.3)
        if r == 0: fig.text((x + (CW * unit - gap) / 2) / W_IN, 1 - (HEAD - 0.035) / H_IN, name, ha='center', va='bottom', fontsize=7.5, linespacing=1.05, fontweight='bold' if c == 'OURS' else 'normal')
        if c != 'gt': a.text(0.975, 0.972, ('%.0f MB' % mb[c] if mb[c] >= 100 else ('%.1f MB' % mb[c] if mb[c] >= 5 else '%.2f MB' % mb[c])) + '\n%.2f dB' % ps[c], transform=a.transAxes, ha='right', va='top', fontsize=6.3, color='white', linespacing=1.1, bbox=dict(fc='black', ec='none', alpha=0.55, pad=1.6))
    if r == 0: fig.text((FULL_W * unit - gap) / 2 / W_IN, 1 - (HEAD - 0.035) / H_IN, 'Ground truth (full view)', ha='center', va='bottom', fontsize=7.5)
fig.savefig('paper/figures/fig_qualitative_v4%s.pdf' % TAG, dpi=300); fig.savefig('paper/figures/fig_qualitative_v4%s.png' % TAG, dpi=220); print('ok', W_IN, round(H_IN, 2))
