"""make_pipeline_sketch_v1.py: a layout sketch of the pipeline figure (SVG, to be redrawn by hand in draw.io).  Wide, two-column format."""
W, H = 1500, 470
INK, MUTED, LINE, GREY, BLUE, ORANGE, FILL, NEW = '#0b0b0b', '#52514e', '#8a8983', '#e9e8e3', '#2a78d6', '#eb6834', '#fcfcfb', '#e6f0fc'
o = []
def rect(x, y, w, h, fill=FILL, stroke=LINE, r=8, sw=1.3, dash=None): o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
def text(x, y, s, size=13, color=INK, anchor='middle', weight='normal', style='normal'): o.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}" font-style="{style}">{s}</text>')
def lines(x, y, ss, size=12, color=MUTED, anchor='middle', lh=15):
    for i, s in enumerate(ss): text(x, y + i * lh, s, size, color, anchor)
def arrow(x1, y1, x2, y2, color=INK, sw=1.6, dash=None): o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#a{color[1:]})"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
def path(d, color=INK, sw=1.6, dash=None, end=True): o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"' + (f' marker-end="url(#a{color[1:]})"' if end else '') + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
def tri(pts, fill='#cfcec8', stroke=MUTED): o.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="1"/>' % (' '.join('%g,%g' % p for p in pts), fill, stroke))
def trap(x, y, w, h, flip=False, fill=NEW, stroke=BLUE):
    d = h * 0.22; p = [(x, y), (x + w, y + d), (x + w, y + h - d), (x, y + h)] if not flip else [(x, y + d), (x + w, y), (x + w, y + h), (x, y + h - d)]
    o.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="1.5"/>' % (' '.join('%g,%g' % q for q in p), fill, stroke))
o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">')
o.append('<defs>' + ''.join(f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in (INK, BLUE, ORANGE, MUTED)) + '</defs>')
o.append(f'<rect width="{W}" height="{H}" fill="white"/>')
# ---- stage panels
rect(14, 40, 930, 400, fill='#f7f7f4', stroke='#c9c8c2', r=12); text(28, 62, '1  Training at a target size', 15, INK, 'start', 'bold'); text(930, 62, 'input: images + target size (MB)', 12, MUTED, 'end')
rect(962, 40, 356, 400, fill='#f7f7f4', stroke='#c9c8c2', r=12); text(976, 62, '2  Encoding', 15, INK, 'start', 'bold')
rect(1336, 40, 150, 400, fill='#f7f7f4', stroke='#c9c8c2', r=12); text(1350, 62, '3  Decoding', 15, INK, 'start', 'bold')
# ---- 1a soup
rect(30, 96, 150, 150, fill=GREY, stroke=LINE)
for p in ([(48, 130), (78, 112), (70, 150)], [(96, 122), (140, 110), (128, 140)], [(52, 190), (76, 168), (92, 204)], [(112, 170), (160, 160), (150, 196)], [(100, 214), (126, 206), (120, 232)], [(142, 214), (168, 224), (150, 238)]): tri(p)
text(105, 266, 'Triangle soup', 13, INK, 'middle', 'bold'); lines(105, 283, ['as in MeshSplatting'], 11.5)
arrow(182, 171, 212, 171)
# ---- 1b grow
rect(214, 96, 196, 150, fill=NEW, stroke=BLUE)
tri([(236, 150), (300, 118), (292, 190)], '#cfcec8'); arrow(306, 154, 326, 154, MUTED, 1.2)
for p in ([(332, 150), (364, 134), (360, 170)], [(364, 134), (396, 118), (392, 154)], [(360, 170), (392, 154), (388, 190)], [(364, 134), (392, 154), (360, 170)]): tri(p, '#b7d3f6', BLUE)
text(312, 222, 'split where error drops', 11.5, MUTED); text(312, 236, 'most per byte', 11.5, MUTED)
text(312, 266, 'Grow', 13, INK, 'middle', 'bold'); lines(312, 283, ['iterations 0 to 10k', 'how many: planned from', 'the measured size'], 11.5)
arrow(412, 171, 442, 171)
# ---- 1c connect
rect(444, 96, 150, 150, fill=GREY, stroke=LINE)
P = [(462, 200), (500, 128), (540, 196), (576, 132), (520, 232), (470, 150)]
for a, b, c in ((0, 1, 2), (1, 3, 2), (0, 2, 4), (5, 1, 0)): tri([P[a], P[b], P[c]])
text(519, 266, 'Connect', 13, INK, 'middle', 'bold'); lines(519, 283, ['restricted Delaunay', 'at 11k, as in MeshSplatting'], 11.5)
arrow(596, 171, 626, 171)
# ---- 1d hold loop
rect(628, 84, 300, 262, fill=NEW, stroke=BLUE); text(778, 104, 'Hold at the target', 13, INK, 'middle', 'bold'); text(778, 119, 'every 500 iterations, 12.5k to 29.5k', 11.5, MUTED)
rect(644, 134, 118, 44, fill='white', stroke=BLUE); text(703, 152, 'encode the', 11.5); text(703, 167, 'current mesh', 11.5)
rect(794, 134, 118, 44, fill='white', stroke=BLUE); text(853, 152, 'file size', 11.5); text(853, 167, 'above target?', 11.5)
rect(794, 222, 118, 56, fill='white', stroke=BLUE); text(853, 240, 'collapse edges', 11.5); text(853, 255, 'worth that many', 11.5); text(853, 270, 'bytes', 11.5)
rect(644, 222, 118, 56, fill='white', stroke=BLUE); text(703, 246, 'keep training', 11.5); text(703, 261, '500 iterations', 11.5)
arrow(764, 156, 792, 156, BLUE); arrow(853, 180, 853, 220, BLUE); arrow(792, 250, 764, 250, BLUE); arrow(703, 220, 703, 180, BLUE)
text(862, 204, 'yes', 11, BLUE, 'start'); path('M 794 168 Q 760 200 742 220', BLUE, 1.2, '4 3'); text(752, 198, 'no', 11, BLUE, 'end')
lines(778, 300, ['cheapest edges first: least error per byte freed', 'only collapses that do not fold the surface'], 11.5)
text(778, 336, 'the mesh stays at the target size while it trains', 11.5, INK, 'middle', 'normal', 'italic')
# training timeline
o.append(f'<line x1="30" y1="392" x2="928" y2="392" stroke="{LINE}" stroke-width="1.2"/>')
for x, s in ((30, '0'), (411, '10k'), (444, '11k'), (628, '12.5k'), (928, '30k')):
    o.append(f'<line x1="{x}" y1="386" x2="{x}" y2="398" stroke="{LINE}" stroke-width="1.2"/>'); text(x, 414, s, 11, MUTED, 'start' if x == 30 else ('end' if x == 928 else 'middle'))
text(479, 430, 'training iterations (the mesh becomes fully opaque by 24k)', 11.5, MUTED)
# ---- 2 encoding
rect(978, 96, 150, 46, fill=GREY, stroke=LINE); text(1053, 115, 'fine-tune', 12); text(1053, 131, '3,000 steps', 11.5, MUTED)
arrow(1053, 144, 1053, 164)
rect(978, 166, 150, 120, fill=NEW, stroke=BLUE); text(1053, 184, 'shared colour codebook', 12, INK, 'middle', 'bold')
rect(990, 196, 58, 40, fill='white', stroke=BLUE); text(1019, 213, 'main', 11); text(1019, 228, '4,096', 11, MUTED)
rect(1058, 196, 58, 40, fill='white', stroke=BLUE); text(1087, 213, 'residual', 11); text(1087, 228, '256', 11, MUTED)
text(1053, 256, 'per vertex: base colour', 11, MUTED); text(1053, 271, '+ two indices', 11, MUTED)
arrow(1130, 226, 1160, 226)
trap(1162, 110, 86, 232); text(1205, 222, 'Encoder', 13, INK, 'middle', 'bold')
lines(1205, 242, ['connectivity', 'positions', 'base colour', 'indices'], 11)
arrow(1250, 226, 1272, 226)
rect(1274, 196, 34, 60, fill='white', stroke=INK, r=3); text(1291, 222, 'one', 10.5); text(1291, 236, 'file', 10.5)
lines(1145, 372, ['every stream has its own lossless coder;', 'the same encoder measures the size during training'], 11.5)
path('M 1205 344 L 1205 356 L 703 356 L 703 372', ORANGE, 1.4, '5 3', end=False); path('M 703 372 L 703 350', ORANGE, 1.4, '5 3'); text(954, 351, 'same encoder', 11, ORANGE)
# ---- 3 decoding
arrow(1310, 226, 1350, 226)
trap(1352, 150, 70, 152, flip=True); text(1387, 231, 'Decoder', 12.5, INK, 'middle', 'bold')
arrow(1387, 304, 1387, 330); rect(1350, 332, 122, 46, fill=GREY, stroke=LINE); text(1411, 351, 'mesh, rendered', 11.5); text(1411, 366, 'as usual', 11.5)
lines(1411, 96, ['needs only the file,', 'CPU only'], 11.5)
# ---- legend
rect(30, 8, 16, 14, fill=NEW, stroke=BLUE, r=3); text(52, 20, 'ours', 12, INK, 'start'); rect(100, 8, 16, 14, fill=GREY, stroke=LINE, r=3); text(122, 20, 'as in MeshSplatting', 12, INK, 'start')
o.append('</svg>'); open('paper/pipeline-sketch/pipeline-sketch-v1.svg', 'w').write('\n'.join(o)); print('ok')
