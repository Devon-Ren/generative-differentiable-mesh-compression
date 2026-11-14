"""make_pipeline_sketch_v2.py: layout sketch of the pipeline figure, organised along the training timeline (SVG, to be redrawn by hand).
Each block sits above the iterations in which it happens.  Facts checked against train_rd_adapt_v9b_abl7.py and the job script."""
W, H = 1560, 640
INK, MUTED, LINE, GREY, BLUE, ORANGE, NEW = '#0b0b0b', '#52514e', '#8a8983', '#e9e8e3', '#2a78d6', '#eb6834', '#e6f0fc'
o = []
def rect(x, y, w, h, fill='white', stroke=LINE, r=7, sw=1.3, dash=None, op=None): o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else '') + (f' fill-opacity="{op}"' if op else '') + '/>')
def text(x, y, s, size=12, color=INK, anchor='middle', weight='normal', style='normal'): o.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}" font-style="{style}">{s}</text>')
def lines(x, y, ss, size=11, color=MUTED, anchor='middle', lh=14, weight='normal'):
    for i, s in enumerate(ss): text(x, y + i * lh, s, size, color, anchor, weight)
def arrow(x1, y1, x2, y2, color=INK, sw=1.5, dash=None): o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#a{color[1:]})"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
def path(d, color=INK, sw=1.5, dash=None, end=True): o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"' + (f' marker-end="url(#a{color[1:]})"' if end else '') + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
def tri(pts, fill='#9c9b95', stroke=MUTED, op=1.0, sw=1): o.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="%g" fill-opacity="%g" stroke-opacity="%g"/>' % (' '.join('%g,%g' % p for p in pts), fill, stroke, sw, op, min(1, op + 0.25)))
def trap(x, y, w, h, flip=False):
    d = h * 0.22; p = [(x, y), (x + w, y + d), (x + w, y + h - d), (x, y + h)] if not flip else [(x, y + d), (x + w, y), (x + w, y + h), (x, y + h - d)]
    o.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="1.5"/>' % (' '.join('%g,%g' % q for q in p), NEW, BLUE))
X0, X30 = 40, 1120; tx = lambda k: X0 + (X30 - X0) * k / 30.0
o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">')
o.append('<defs>' + ''.join(f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in (INK, BLUE, ORANGE, MUTED))
         + f'<linearGradient id="op" x1="0" x2="1"><stop offset="0" stop-color="#9c9b95" stop-opacity="0.25"/><stop offset="1" stop-color="#9c9b95" stop-opacity="1"/></linearGradient></defs>')
o.append(f'<rect width="{W}" height="{H}" fill="white"/>')
rect(30, 8, 16, 14, NEW, BLUE, 3); text(52, 20, 'ours', 12, INK, 'start'); rect(100, 8, 16, 14, GREY, LINE, 3); text(122, 20, 'as in MeshSplatting', 12, INK, 'start')
text(X30, 20, 'input: images + target size in MB', 12, MUTED, 'end')
# ================= GROW  (0 .. 10k)
gx0, gx1 = tx(0), tx(10); rect(gx0, 40, gx1 - gx0 - 6, 430, '#f7f7f4', '#c9c8c2', 10)
text(gx0 + 12, 60, 'A  Grow', 14, INK, 'start', 'bold'); text(gx1 - 18, 60, 'every 500 iterations', 11, MUTED, 'end')
# soup: semi-transparent, unconnected
for p, a in (([(58, 110), (92, 92), (84, 134)], .35), ([(104, 100), (150, 90), (136, 122)], .5), ([(60, 160), (88, 142), (104, 176)], .3), ([(118, 140), (166, 132), (154, 168)], .55), ([(70, 190), (98, 184), (90, 208)], .4), ([(126, 184), (160, 190), (140, 206)], .3)): tri(p, op=a)
lines(110, 226, ['triangle soup:', 'semi-transparent,', 'not connected'], 11)
# A-loop boxes
bx = [(190, 86, 196, 62), (190, 176, 196, 62), (190, 266, 196, 74)]
rect(*bx[0], NEW, BLUE); text(288, 106, '1  measure  (from 4k, every 1k)', 11.5, INK, 'middle', 'bold'); lines(288, 122, ['block M: how big would the file be', 'if training stopped now?'], 10.5)
rect(*bx[1], NEW, BLUE); text(288, 196, '2  plan', 11.5, INK, 'middle', 'bold'); lines(288, 212, ['how many triangles to split now so the', 'mesh ends 35% above the target'], 10.5)
rect(*bx[2], NEW, BLUE); text(288, 286, '3  split', 11.5, INK, 'middle', 'bold'); lines(288, 302, ['only the triangles whose split lowers', 'the error most per byte; the rest', 'stay as they are'], 10.5)
arrow(288, 150, 288, 174, BLUE); arrow(288, 240, 288, 264, BLUE); path('M 388 302 L 391 302 L 391 116 L 389 116', BLUE)
# split illustration: one splits, one stays, one faint is removed
tri([(56, 372), (112, 348), (104, 404)], '#b7d3f6', BLUE, .7)
for q in ([(56, 372), (84, 360), (80, 388)], [(84, 360), (112, 348), (108, 376)], [(80, 388), (108, 376), (104, 404)], [(84, 360), (108, 376), (80, 388)]): o.append('<polygon points="%s" fill="none" stroke="%s" stroke-width="1.1"/>' % (' '.join('%g,%g' % p for p in q), BLUE))
text(84, 422, 'split', 10.5, BLUE)
tri([(150, 352), (196, 362), (164, 398)], op=.55); text(172, 422, 'kept', 10.5, MUTED)
tri([(232, 356), (272, 350), (256, 392)], op=.12); o.append(f'<path d="M 236 352 L 270 392 M 270 352 L 236 392" stroke="{MUTED}" stroke-width="1.4"/>'); text(254, 422, 'removed', 10.5, MUTED)
rect(300, 352, 88, 74, GREY, LINE); lines(344, 372, ['MeshSplatting', 'also removes', 'faint or huge', 'triangles'], 10.2)
text((gx0 + gx1) / 2, 458, 'no collapse here: the soup has no shared edges to collapse', 10.5, INK, 'middle', 'normal', 'italic')
# ================= CONNECT (11k)
cx0, cx1 = tx(10) + 2, tx(12.5) - 4; rect(cx0, 40, cx1 - cx0, 430, GREY, LINE, 10); text((cx0 + cx1) / 2, 60, 'B  Connect', 13, INK, 'middle', 'bold')
P = [(cx0 + 10, 200), (cx0 + 38, 120), (cx0 + 70, 190), (cx0 + 50, 250), (cx0 + 16, 130)]
for a, b, c in ((0, 1, 2), (0, 2, 3), (4, 1, 0)): tri([P[a], P[b], P[c]], op=.75)
lines((cx0 + cx1) / 2, 290, ['restricted', 'Delaunay', 'at 11k:', 'one mesh', 'with shared', 'vertices', 'and edges'], 10.5)
# ================= HOLD (12.5k .. 29.5k)
hx0, hx1 = tx(12.5), tx(29.5) + 14; rect(hx0, 40, hx1 - hx0, 430, '#f7f7f4', '#c9c8c2', 10)
text(hx0 + 12, 60, 'C  Hold at the target', 14, INK, 'start', 'bold'); text(hx1 - 12, 60, 'every 500 iterations, 35 rounds; nothing is grown any more', 11, MUTED, 'end')
# block M, drawn once in detail
rect(hx0 + 12, 76, hx1 - hx0 - 24, 150, 'white', BLUE, 8, 1.2, '5 3'); text(hx0 + 24, 94, 'M  measure the size (used in A and in C)', 11.5, BLUE, 'start', 'bold')
mx = hx0 + 30
for p, a in (([(mx, 150), (mx + 30, 116), (mx + 50, 160)], .5), ([(mx + 30, 116), (mx + 74, 128), (mx + 50, 160)], .5), ([(mx + 8, 176), (mx + 50, 160), (mx + 60, 196)], .2)): tri(p, op=a)
lines(mx + 38, 214, ['copy of the mesh'], 10.5)
arrow(mx + 86, 150, mx + 112, 150, BLUE)
for p in ([(mx + 120, 150), (mx + 150, 116), (mx + 170, 160)], [(mx + 150, 116), (mx + 194, 128), (mx + 170, 160)]): tri(p, op=1)
tri([(mx + 128, 176), (mx + 170, 160), (mx + 180, 196)], op=.1); o.append(f'<path d="M {mx + 140} 166 L {mx + 172} 196 M {mx + 172} 166 L {mx + 140} 196" stroke="{MUTED}" stroke-width="1.3"/>')
lines(mx + 158, 214, ['rendered fully opaque;', 'hidden triangles dropped'], 10.5, lh=12)
arrow(mx + 206, 150, mx + 232, 150, BLUE)
trap(mx + 236, 112, 62, 78); text(mx + 267, 148, 'the real', 10.5); text(mx + 267, 161, 'encoder', 10.5)
arrow(mx + 302, 150, mx + 328, 150, BLUE)
rect(mx + 332, 126, 96, 48, NEW, BLUE); text(mx + 380, 146, 'file size', 11.5, INK, 'middle', 'bold'); text(mx + 380, 162, 'in bytes', 10.5, MUTED)
lines(mx + 512, 134, ['in A (soup): only the', 'bytes per vertex are used;', 'in C: the whole file'], 10.5, lh=13)
# C loop
cy = 250
rect(hx0 + 24, cy, 150, 64, NEW, BLUE); text(hx0 + 99, cy + 20, '1  measure (M)', 11.5, INK, 'middle', 'bold'); lines(hx0 + 99, cy + 36, ['file above the target?', 'by how many bytes?'], 10.5, lh=13)
rect(hx0 + 214, cy, 220, 150, NEW, BLUE); text(hx0 + 324, cy + 20, '2  collapse edges worth those bytes', 11.5, INK, 'middle', 'bold')
ex = hx0 + 232; ey = cy + 38   # edge collapse, before -> after
A = [(ex, ey + 40), (ex + 34, ey + 6), (ex + 70, ey + 40), (ex + 34, ey + 76), (ex + 26, ey + 40), (ex + 46, ey + 40)]
for a, b, c in ((0, 1, 4), (1, 5, 4), (1, 2, 5), (0, 4, 3), (4, 5, 3), (5, 2, 3)): tri([A[a], A[b], A[c]], op=.8)
o.append(f'<line x1="{A[4][0]}" y1="{A[4][1]}" x2="{A[5][0]}" y2="{A[5][1]}" stroke="{ORANGE}" stroke-width="3"/>'); arrow(ex + 80, ey + 40, ex + 104, ey + 40, BLUE)
Bp = [(ex + 112, ey + 40), (ex + 146, ey + 6), (ex + 182, ey + 40), (ex + 146, ey + 76), (ex + 148, ey + 40)]
for a, b, c in ((0, 1, 4), (1, 2, 4), (0, 4, 3), (4, 2, 3)): tri([Bp[a], Bp[b], Bp[c]], op=.8)
lines(hx0 + 324, cy + 128, ['cheapest first: least error per byte freed;', 'never a collapse that folds the surface'], 10.2, lh=12)
rect(hx0 + 474, cy, 136, 64, GREY, LINE); text(hx0 + 542, cy + 20, '3  train on', 11.5, INK, 'middle', 'bold'); lines(hx0 + 542, cy + 36, ['500 iterations, the mesh', 'adapts to the removal'], 10.5, lh=13)
arrow(hx0 + 176, cy + 32, hx0 + 212, cy + 32, BLUE); text(hx0 + 194, cy + 24, 'yes', 10.5, BLUE); arrow(hx0 + 436, cy + 32, hx0 + 472, cy + 32, BLUE)
path(f'M {hx0 + 542} {cy + 66} L {hx0 + 542} {cy + 176} L {hx0 + 99} {cy + 176} L {hx0 + 99} {cy + 68}', BLUE)
path(f'M {hx0 + 60} {cy + 66} L {hx0 + 60} {cy + 196} L {hx0 + 574} {cy + 196} L {hx0 + 574} {cy + 68}', BLUE, 1.2, '4 3'); text(hx0 + 150, cy + 210, 'no: nothing is removed this round', 10.5, BLUE, 'start')
# ================= timeline
ty = 520
o.append(f'<line x1="{X0}" y1="{ty}" x2="{X30}" y2="{ty}" stroke="{INK}" stroke-width="1.4"/>')
for k, s in ((0, '0'), (4, '4k'), (10, '10k'), (11, '11k'), (12.5, '12.5k'), (20, '20k'), (24, '24k'), (29.5, '29.5k'), (30, '30k')):
    o.append(f'<line x1="{tx(k)}" y1="{ty - 6}" x2="{tx(k)}" y2="{ty + 6}" stroke="{INK}" stroke-width="1.2"/>'); text(tx(k), ty + 20, s, 10.5, MUTED, 'end' if k == 29.5 else ('start' if k == 30 else 'middle'))
text(X0, ty - 14, 'training iterations', 11, MUTED, 'start')
for (a, b, lab) in ((0, 10, 'A  grow'), (10, 12.5, 'B'), (12.5, 29.5, 'C  hold')):
    o.append(f'<line x1="{tx(a) + 3}" y1="{ty - 26}" x2="{tx(b) - 3}" y2="{ty - 26}" stroke="{BLUE if lab[0] != "B" else LINE}" stroke-width="4"/>'); text((tx(a) + tx(b)) / 2, ty - 32, lab, 10.5, INK)
o.append(f'<rect x="{tx(0)}" y="{ty + 34}" width="{tx(5) - tx(0)}" height="12" fill="#9c9b95" fill-opacity="0.25"/><rect x="{tx(5)}" y="{ty + 34}" width="{tx(24) - tx(5)}" height="12" fill="url(#op)"/><rect x="{tx(24)}" y="{ty + 34}" width="{tx(30) - tx(24)}" height="12" fill="#9c9b95"/>')
text(tx(0), ty + 62, 'opacity of the triangles: semi-transparent', 10.5, MUTED, 'start'); text(tx(14.5), ty + 62, 'raised step by step from 5k', 10.5, MUTED); text(tx(30), ty + 62, 'fully opaque from 24k', 10.5, MUTED, 'end')
text(tx(20), ty + 82, 'rendering at 2x resolution from 20k, 4x from 25k (as in MeshSplatting)', 10.5, MUTED)
# ================= ENCODE / DECODE
ex0 = X30 + 30; rect(ex0, 40, 268, 430, '#f7f7f4', '#c9c8c2', 10); text(ex0 + 12, 60, 'D  Encode', 14, INK, 'start', 'bold'); text(ex0 + 256, 60, 'after training', 11, MUTED, 'end')
rect(ex0 + 16, 80, 120, 44, GREY, LINE); text(ex0 + 76, 98, 'fine-tune', 11.5); text(ex0 + 76, 113, '3,000 steps', 10.5, MUTED)
arrow(ex0 + 76, 126, ex0 + 76, 146)
rect(ex0 + 16, 148, 120, 150, NEW, BLUE); text(ex0 + 76, 166, 'shared colour', 11.5, INK, 'middle', 'bold'); text(ex0 + 76, 180, 'codebook', 11.5, INK, 'middle', 'bold')
rect(ex0 + 26, 190, 100, 34, 'white', BLUE); text(ex0 + 76, 205, 'main: 4,096 entries', 10.2); 
text(ex0 + 76, 218, '', 10)
rect(ex0 + 26, 230, 100, 34, 'white', BLUE); text(ex0 + 76, 245, 'residual: 256', 10.2); text(ex0 + 76, 257, 'entries', 10.2)
lines(ex0 + 76, 280, ['vertex = base colour', '+ 2 indices'], 10.2, lh=12)
arrow(ex0 + 138, 223, ex0 + 156, 223)
trap(ex0 + 158, 96, 62, 254); text(ex0 + 189, 150, 'Encoder', 11.5, INK, 'middle', 'bold')
lines(ex0 + 189, 178, ['connec-', 'tivity', '', 'positions', '', 'base', 'colour', '', 'indices'], 10, lh=12)
arrow(ex0 + 222, 223, ex0 + 236, 223); rect(ex0 + 238, 196, 24, 54, 'white', INK, 3); text(ex0 + 250, 220, 'one', 9.5); text(ex0 + 250, 232, 'file', 9.5)
lines(ex0 + 134, 388, ['the same encoder as in block M;', 'M skips only the codebook', 'refit and the check render'], 10.5, ORANGE, lh=13)
path(f'M {ex0 + 189} 352 L {ex0 + 189} 366', ORANGE, 1.3, '5 3', end=False)
dx0 = ex0 + 280; rect(dx0, 40, 110, 430, '#f7f7f4', '#c9c8c2', 10); text(dx0 + 10, 60, 'E  Decode', 13, INK, 'start', 'bold')
arrow(ex0 + 264, 223, dx0 + 18, 223); trap(dx0 + 20, 170, 52, 106, flip=True); text(dx0 + 46, 227, 'Decoder', 10.5, INK, 'middle', 'bold')
arrow(dx0 + 46, 280, dx0 + 46, 304)
for a, b, c in (((dx0 + 16, 356), (dx0 + 46, 312), (dx0 + 62, 362)), ((dx0 + 46, 312), (dx0 + 94, 330), (dx0 + 62, 362)), ((dx0 + 16, 356), (dx0 + 62, 362), (dx0 + 50, 396))): tri([a, b, c], op=1)
lines(dx0 + 55, 418, ['opaque mesh;', 'only the file', 'is needed'], 10.5, lh=13)
o.append('</svg>'); open('paper/pipeline-sketch/pipeline-sketch-v2.svg', 'w').write('\n'.join(o)); print('ok')
