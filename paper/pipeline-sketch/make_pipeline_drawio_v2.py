"""make_pipeline_drawio_v2.py: the pipeline figure (layout of sketch v4) as an editable draw.io file.
Boxes, arrows, labels, encoder / decoder, byte bands, codebook and the iteration bar are native draw.io cells.  The triangle pictures are small
SVG images embedded in the file (page 1); page 2 holds the same pictures built from individual polygon shapes for editing."""
import base64, math, random, xml.etree.ElementTree as ET
INK, MUTED, LINE, BLUE, ORANGE, TRI = '#0b0b0b', '#52514e', '#8a8983', '#2a78d6', '#eb6834', '#8f8e88'
random.seed(4); G = 4
P0 = [[(i + 0.28 * (random.random() - .5) * (0 < i < G - 1), j + 0.28 * (random.random() - .5) * (0 < j < G - 1)) for j in range(G)] for i in range(G)]
PTS = [P0[i][j] for i in range(G) for j in range(G)]; vid = lambda i, j: i * G + j; FACES = []
for i in range(G - 1):
    for j in range(G - 1):
        if (i + j) % 2 == 0: FACES += [(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)), (vid(i, j), vid(i + 1, j + 1), vid(i, j + 1))]
        else: FACES += [(vid(i, j), vid(i + 1, j), vid(i, j + 1)), (vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1))]
ALPHA = [random.uniform(.22, .6) for _ in FACES]; JIT = [(random.uniform(-.07, .07), random.uniform(-.07, .07), random.uniform(-.25, .25)) for _ in FACES]
SPLIT = {2, 7, 9, 14}; HIDDEN = {4, 12}; EU, EV = vid(1, 1), vid(1, 2)
def soup_tri(k):
    a, b, c = [PTS[v] for v in FACES[k]]; cx, cy = (a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3; dx, dy, r = JIT[k]; out = []
    for p in (a, b, c):
        ux, uy = (p[0] - cx) * .74, (p[1] - cy) * .74; out.append((cx + dx + ux * math.cos(r) - uy * math.sin(r), cy + dy + ux * math.sin(r) + uy * math.cos(r)))
    return out
m_ = ((PTS[EU][0] + PTS[EV][0]) / 2, (PTS[EU][1] + PTS[EV][1]) / 2); CP = list(PTS); CP[EU] = m_; CP[EV] = m_
CF = [f for f in [tuple(EU if v == EV else v for v in f) for f in FACES] if len(set(f)) == 3]
def state(kind):
    """list of (triangle in unit coordinates 0..3, fill, stroke, opacity, dash) and optional marked segment"""
    T = []; mark = None
    if kind.startswith('soup'):
        for k in range(len(FACES)):
            t = soup_tri(k)
            if 'clean' in kind and k in HIDDEN: T.append((t, 'none', LINE, 0, 1)); continue
            T.append((t, TRI, MUTED, 100 if 'opaque' in kind else int(100 * ALPHA[k]), 0))
            if 'split' in kind and k in SPLIT:
                mm = [((t[i][0] + t[(i + 1) % 3][0]) / 2, (t[i][1] + t[(i + 1) % 3][1]) / 2) for i in range(3)]
                T[-1] = (t, '#b7d3f6', BLUE, 75, 0); T.append((mm, 'none', BLUE, 0, 0))
    else:
        pts, faces = (CP, CF) if 'collapsed' in kind else (PTS, FACES)
        for k, f in enumerate(faces):
            t = [pts[v] for v in f]
            if 'clean' in kind and k in HIDDEN: T.append((t, 'none', LINE, 0, 1)); continue
            T.append((t, TRI, INK if 'opaque' in kind else MUTED, 100 if 'opaque' in kind else 60, 0))
        if 'marked' in kind: mark = (PTS[EU], PTS[EV])
    return T, mark
def svg(kind, px=132):
    T, mark = state(kind); s = (px - 8) / 3.0; f = lambda p: '%.1f,%.1f' % (4 + p[0] * s, 4 + p[1] * s); o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{px}" height="{px}" viewBox="0 0 {px} {px}">']
    for t, fill, stroke, op, dash in T: o.append(f'<polygon points="{" ".join(f(p) for p in t)}" fill="{fill}" stroke="{stroke}" stroke-width="0.9" fill-opacity="{op / 100:.2f}"' + (' stroke-dasharray="3 2"' if dash else '') + '/>')
    if mark: o.append(f'<line x1="{4 + mark[0][0] * s:.1f}" y1="{4 + mark[0][1] * s:.1f}" x2="{4 + mark[1][0] * s:.1f}" y2="{4 + mark[1][1] * s:.1f}" stroke="{ORANGE}" stroke-width="4" stroke-linecap="round"/>')
    return 'data:image/svg+xml,' + base64.b64encode(('\n'.join(o) + '</svg>').encode()).decode()
root = ET.Element('mxfile', host='app.diagrams.net')
def page(name, w, h):
    d = ET.SubElement(root, 'diagram', name=name, id=name.replace(' ', '')); mdl = ET.SubElement(d, 'mxGraphModel', dx='1600', dy='900', grid='1', gridSize='10', guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1', pageWidth=str(w), pageHeight=str(h), math='0', shadow='0')
    r = ET.SubElement(mdl, 'root'); ET.SubElement(r, 'mxCell', id='0'); ET.SubElement(r, 'mxCell', id='1', parent='0'); return r
N = [1]
def cell(R, value, x, y, w, h, style):
    N[0] += 1; c = ET.SubElement(R, 'mxCell', id=f'c{N[0]}', value=value, style=style, vertex='1', parent='1'); ET.SubElement(c, 'mxGeometry', x=f'{x:.1f}', y=f'{y:.1f}', width=f'{w:.1f}', height=f'{h:.1f}').set('as', 'geometry')
def line(R, x1, y1, x2, y2, color=INK, sw=1.6, dash=False, arrow=True, pts=None, curved=False, label=''):
    N[0] += 1; st = f'endArrow={"block" if arrow else "none"};endFill=1;html=1;rounded=0;strokeColor={color};strokeWidth={sw};fontSize=11;fontColor={color};' + ('dashed=1;' if dash else '') + ('curved=1;' if curved else '')
    c = ET.SubElement(R, 'mxCell', id=f'c{N[0]}', value=label, style=st, edge='1', parent='1'); g = ET.SubElement(c, 'mxGeometry', relative='1'); g.set('as', 'geometry')
    ET.SubElement(g, 'mxPoint', x=f'{x1:.1f}', y=f'{y1:.1f}').set('as', 'sourcePoint'); ET.SubElement(g, 'mxPoint', x=f'{x2:.1f}', y=f'{y2:.1f}').set('as', 'targetPoint')
    if pts:
        a = ET.SubElement(g, 'Array'); a.set('as', 'points')
        for px, py in pts: ET.SubElement(a, 'mxPoint', x=f'{px:.1f}', y=f'{py:.1f}')
PANEL = 'rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor=#f7f7f4;strokeColor=#c9c8c2;verticalAlign=top;align=left;spacingLeft=12;spacingTop=6;fontSize=15;fontStyle=1;'
TXT = lambda size=12, color=INK, bold=0, align='center': f'text;html=1;whiteSpace=wrap;align={align};verticalAlign=middle;fontSize={size};fontColor={color};fontStyle={bold};'
IMG = lambda kind, px: f'shape=image;html=1;imageAspect=0;aspect=fixed;image={svg(kind, px)};'
ENC = 'shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=%d;direction=%s;fillColor=%s;strokeColor=#2a78d6;strokeWidth=1.5;fontSize=%s;'
P = page('pipeline', 1840, 600); S, s = 132, 78
def loop_module(x0, w, title, kinds, lab_a, lab_b, goal, back_lab):
    cell(P, title, x0, 20, w, 440, PANEL); ax, bx, ty = x0 + 46, x0 + w - 46 - S, 78
    cell(P, '', ax, ty, S, S, IMG(kinds[0], S)); cell(P, '', bx, ty, S, S, IMG(kinds[1], S))
    cell(P, lab_a, ax - 40, ty + S + 6, S + 80, 22, TXT()); cell(P, lab_b, bx - 40, ty + S + 6, S + 80, 22, TXT(12, INK, 1))
    line(P, ax + S + 14, ty + S / 2, bx - 14, ty + S / 2, INK, 2.2); cell(P, goal, ax + S + 10, ty + S / 2 - 30, bx - ax - S - 20, 22, TXT(11.5, BLUE))
    line(P, bx + S / 2, ty - 6, ax + S / 2, ty - 6, MUTED, 1.3, True, pts=[(bx + S / 2 - 30, ty - 34), (ax + S / 2 + 30, ty - 34)], curved=True); cell(P, 'next round', x0 + w / 2 - 50, ty - 46, 100, 18, TXT(11, MUTED))
    my = 318; xs = [x0 + 40 + k * (w - 80 - s) / 4 for k in range(5)]
    cell(P, 'measure the file size', x0 + 22, my - 30, w - 44, s + 74, f'rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={BLUE};dashed=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=2;fontSize=11.5;fontStyle=1;fontColor={BLUE};')
    for k in range(3): cell(P, '', xs[k], my, s, s, IMG(kinds[2 + k], s))
    cell(P, 'real<br>encoder', xs[3] + 6, my + 4, s - 22, s - 8, ENC % (12, 'south', '#e6f0fc', '10.5'))
    cell(P, '<b style="font-size:15px">R</b><br><span style="color:#52514e">bytes</span>', xs[4] + 2, my + 16, s - 4, s - 32, f'rounded=1;whiteSpace=wrap;html=1;fillColor=#e6f0fc;strokeColor={BLUE};strokeWidth=1.3;fontSize=10.5;')
    for k in range(4): line(P, xs[k] + s + 3, my + s / 2, xs[k + 1] - 3, my + s / 2, BLUE, 1.4)
    for k, lab in enumerate(('copy', 'made opaque<br>(final opacity)', 'hidden faces<br>dropped')): cell(P, lab, xs[k] - 16, my + s + 4, s + 32, 30, TXT(10.5, MUTED))
    line(P, ax + S / 2, ty + S + 30, ax + S / 2, my - 32, BLUE, 1.4); cell(P, 'copy', ax + S / 2 + 6, ty + S + 36, 60, 18, TXT(11, BLUE, 0, 'left'))
    line(P, xs[4] + s / 2, my + 14, xs[4] + s / 2, ty + S + 30, BLUE, 1.4); cell(P, back_lab, xs[4] + s / 2 - 176, ty + S + 36, 170, 18, TXT(11, BLUE, 0, 'right'))
A0, AW = 20, 600
loop_module(A0, AW, 'A&nbsp; Budgeted growth', ['soup', 'soup split', 'soup', 'soup opaque', 'soup opaque clean'], 'semi-transparent triangle soup', 'split the n worst triangles', 'growth goal V* set by the target size T', 'n = how many to split')
B0, BW = A0 + AW + 22, 190; cell(P, 'B&nbsp; Connect', B0, 20, BW, 440, PANEL); cell(P, '', B0 + (BW - S) / 2, 78, S, S, IMG('mesh', S))
cell(P, 'one connected mesh', B0 + 10, 78 + S + 6, BW - 20, 22, TXT()); cell(P, 'restricted Delaunay triangulation<br>(as in MeshSplatting)', B0 + 14, 290, BW - 28, 50, TXT(11.5, MUTED))
line(P, A0 + AW + 2, 78 + S / 2, B0 + (BW - S) / 2 - 8, 78 + S / 2, INK, 2.2)
C0, CW = B0 + BW + 22, 600
loop_module(C0, CW, 'C&nbsp; Rate-locked refinement', ['mesh marked', 'mesh collapsed', 'mesh', 'mesh opaque', 'mesh opaque clean'], 'cheapest edges marked', 'collapsed', 'remove exactly the excess E = R &minus; T', 'E = bytes to remove')
line(P, B0 + BW + 2, 78 + S / 2, C0 + 46 - 8, 78 + S / 2, INK, 2.2)
D0 = C0 + CW + 22; DW = 1840 - D0 - 20; cell(P, 'D&nbsp; Encode and decode', D0, 20, DW, 440, PANEL); dx = D0 + 24
cell(P, '', dx, 78, 102, 102, IMG('mesh collapsed opaque', 102)); cell(P, 'final opaque mesh', dx - 14, 184, 130, 20, TXT(11.5)); line(P, C0 + CW + 2, 78 + S / 2, dx - 6, 78 + S / 2, INK, 2.2)
ex, hy, hh = dx + 150, 70, 150; cell(P, '', ex, hy, 150, hh, ENC % (30, 'south', '#ffffff', '11')); cell(P, 'Encoder', ex + 25, hy - 22, 100, 18, TXT(12.5, INK, 1)); line(P, dx + 108, 78 + 51, ex - 6, 78 + 51, INK, 1.8)
yy = hy + 34; span = hh - 68
for name, share, col in (('connectivity', 39, '#2a78d6'), ('positions', 21, '#5598e7'), ('colour indices', 19, '#86b6ef'), ('base colour', 15, '#b7d3f6'), ('codebooks', 7, '#cde2fb')):
    bh = span * share / 100; cell(P, f'{name} {share}%' if bh > 11 else '', ex + 14, yy, 122, bh - 1.5, f'rounded=0;whiteSpace=wrap;html=1;fillColor={col};strokeColor=none;fontSize=9.5;fontColor={"#ffffff" if share > 30 else INK};'); yy += bh
fx, fy = ex + 20, 268; cell(P, 'one file', fx, fy, 110, 26, 'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#0b0b0b;strokeWidth=1.4;fontSize=11.5;'); line(P, ex + 75, hy + hh + 4, ex + 75, fy - 4, INK, 1.8)
cell(P, 'Decoder', fx + 20, 320, 70, 86, ENC % (17, 'north', '#e6f0fc', '11.5') + 'fontStyle=1;'); line(P, fx + 55, fy + 28, fx + 55, 322, INK, 1.8)
cell(P, '', dx, 322, 90, 90, IMG('mesh collapsed opaque', 90)); line(P, fx + 18, 363, dx + 96, 363, INK, 1.8); cell(P, 'rendered as usual', dx - 20, 414, 130, 20, TXT(11, MUTED))
cbx, cby = dx + 4, 226; cell(P, 'colour codebook', cbx - 10, cby - 20, 120, 16, TXT(10.5, MUTED))
for r_, (n, lab) in enumerate(((12, 'main 4,096'), (6, 'residual 256'))):
    for i in range(n): cell(P, '', cbx + i * 8.4, cby + r_ * 22, 7, 12, f'rounded=0;html=1;strokeColor=none;fillColor={["#eb6834", "#eda100", "#1baf7a", "#2a78d6", "#e87ba4", "#6250d6"][i % 6]};opacity={85 if r_ == 0 else 45};')
    cell(P, lab, cbx + 104, cby + r_ * 22 - 2, 70, 16, TXT(9.5, MUTED, 0, 'left'))
by = 540; line(P, A0, by, C0 + CW, by, INK, 1.5, arrow=False); ax_ = lambda k: A0 + 16 + (AW - 32) * k / 10.0; cx_ = lambda k: C0 + 16 + (CW - 32) * (k - 12.5) / 17.5
for k in range(1, 21): line(P, ax_(k * .5), by - 2, ax_(k * .5), 466, BLUE, 1.1)
line(P, B0 + BW / 2, by - 2, B0 + BW / 2, 466, MUTED, 1.3)
for k in range(35): line(P, cx_(12.5 + k * .5), by - 2, cx_(12.5 + k * .5), 466, BLUE, 1.1)
for x, lab in ((ax_(0), '0'), (ax_(4), '4k'), (ax_(10), '10k'), (B0 + BW / 2, '11k'), (cx_(12.5), '12.5k'), (cx_(20), '20k'), (cx_(24), '24k'), (cx_(29.5), '29.5k')):
    line(P, x, by - 5, x, by + 5, INK, 1.3, arrow=False); cell(P, lab, x - 25, by + 8, 50, 18, TXT(11, MUTED))
cell(P, 'training iterations: one arrow = one round, every 500 iterations (in A the size is measured every 1,000 from 4k)', A0, by + 30, 760, 20, TXT(11.5, MUTED, 0, 'left'))
cell(P, '35 rounds; triangles fully opaque from 24k', C0 + CW - 420, by + 30, 420, 20, TXT(11.5, MUTED, 0, 'right'))
# ---- page 2: the pictures as individual polygon shapes (editable); if they show as plain rectangles, use the images of page 1
Q = page('triangles editable', 1840, 420); kinds = ['soup', 'soup split', 'mesh', 'mesh marked', 'mesh collapsed', 'mesh collapsed opaque', 'soup opaque clean', 'mesh opaque clean']
for n, kind in enumerate(kinds):
    T, mark = state(kind); ox, oy, sc = 40 + n * 225, 70, 60.0; cell(Q, kind, ox - 10, 30, 200, 20, TXT(12, INK, 1, 'left'))
    for t, fill, stroke, op, dash in T:
        xs_ = [p[0] for p in t]; ys_ = [p[1] for p in t]; x0, y0 = min(xs_), min(ys_); w = max(max(xs_) - x0, 1e-3); h = max(max(ys_) - y0, 1e-3)
        pc = '[' + ','.join('[%.3f,%.3f]' % ((p[0] - x0) / w, (p[1] - y0) / h) for p in t) + ']'
        cell(Q, '', ox + x0 * sc, oy + y0 * sc, w * sc, h * sc, f'shape=mxgraph.basic.polygon;polyCoords={pc};polyline=0;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1;fillOpacity={op};' + ('dashed=1;' if dash else ''))
    if mark: line(Q, ox + mark[0][0] * sc, oy + mark[0][1] * sc, ox + mark[1][0] * sc, oy + mark[1][1] * sc, ORANGE, 4, arrow=False)
ET.ElementTree(root).write('paper/pipeline-sketch/pipeline-v4.drawio', encoding='utf-8', xml_declaration=True)
t = ET.parse('paper/pipeline-sketch/pipeline-v4.drawio'); pages = t.getroot().findall('diagram'); print('ok pages', len(pages), 'cells', [len(p.findall('.//mxCell')) for p in pages])
import re
imgs = re.findall(r'image=data:image/svg\+xml,([A-Za-z0-9+/=]+)', open('paper/pipeline-sketch/pipeline-v4.drawio').read()); [ET.fromstring(base64.b64decode(x)) for x in imgs]; print('embedded pictures', len(imgs), 'all valid SVG')
