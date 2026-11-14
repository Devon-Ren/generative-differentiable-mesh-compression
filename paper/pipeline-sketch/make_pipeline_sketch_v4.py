"""make_pipeline_sketch_v4.py: pipeline figure, first version in the layout the author described: the main line is the triangles changing
shape; under each loop hangs the measuring branch; the iteration bar at the bottom sends one arrow per round into the module."""
import math, random
W, H = 1840, 600
INK, MUTED, LINE, BLUE, ORANGE, PANEL, TRI = '#0b0b0b', '#52514e', '#8a8983', '#2a78d6', '#eb6834', '#f7f7f4', '#8f8e88'
o = []
def add(s): o.append(s)
def text(x, y, s, size=12, color=INK, anchor='middle', weight='normal', style='normal'): add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}" font-style="{style}">{s}</text>')
def arrow(x1, y1, x2, y2, color=INK, sw=1.6, dash=None): add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}" marker-end="url(#a{color[1:]})"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
def path(d, color=INK, sw=1.6, dash=None, end=True): add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"' + (f' marker-end="url(#a{color[1:]})"' if end else '') + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
def poly(pts, fill=TRI, stroke=MUTED, op=1.0, sw=0.9, dash=None): add('<polygon points="%s" fill="%s" stroke="%s" stroke-width="%g" fill-opacity="%g" stroke-opacity="%g"%s/>' % (' '.join('%.1f,%.1f' % p for p in pts), fill, stroke, sw, op, min(1, op + 0.35), f' stroke-dasharray="{dash}"' if dash else ''))
def panel(x, y, w, h, title, sub=''):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{PANEL}" stroke="#c9c8c2" stroke-width="1.3"/>'); text(x + 14, y + 24, title, 15, INK, 'start', 'bold')
    if sub: text(x + w - 14, y + 24, sub, 11.5, MUTED, 'end')
def trap(x, y, w, h, flip=False, fill='#e6f0fc'):
    d = h * 0.2; p = [(x, y), (x + w, y + d), (x + w, y + h - d), (x, y + h)] if not flip else [(x, y + d), (x + w, y), (x + w, y + h), (x, y + h - d)]
    add('<polygon points="%s" fill="%s" stroke="%s" stroke-width="1.5"/>' % (' '.join('%.1f,%.1f' % q for q in p), fill, BLUE))
# ---- one small surface patch, reused in every state so the reader sees the same object change
random.seed(4); G = 4
P0 = [[(i + 0.28 * (random.random() - .5) * (0 < i < G - 1), j + 0.28 * (random.random() - .5) * (0 < j < G - 1)) for j in range(G)] for i in range(G)]
PTS = [P0[i][j] for i in range(G) for j in range(G)]; vid = lambda i, j: i * G + j
FACES = []
for i in range(G - 1):
    for j in range(G - 1):
        if (i + j) % 2 == 0: FACES += [(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1)), (vid(i, j), vid(i + 1, j + 1), vid(i, j + 1))]
        else: FACES += [(vid(i, j), vid(i + 1, j), vid(i, j + 1)), (vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1))]
def place(p, x, y, s): return (x + p[0] * s, y + p[1] * s)
ALPHA = [random.uniform(.22, .6) for _ in FACES]; JIT = [(random.uniform(-.07, .07), random.uniform(-.07, .07), random.uniform(-.25, .25)) for _ in FACES]
SPLIT = {2, 7, 9, 14}; HIDDEN = {4, 12}
def soup_tri(k):   # each face shrunk, shifted and turned a little: an unconnected soup
    a, b, c = [PTS[v] for v in FACES[k]]; cx, cy = (a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3; dx, dy, r = JIT[k]; out = []
    for p in (a, b, c):
        ux, uy = (p[0] - cx) * .74, (p[1] - cy) * .74; out.append((cx + dx + ux * math.cos(r) - uy * math.sin(r), cy + dy + ux * math.sin(r) + uy * math.cos(r)))
    return out
def draw_soup(x, y, s, split=False, opaque=False, clean=False):
    for k in range(len(FACES)):
        t = [place(p, x, y, s) for p in soup_tri(k)]
        if clean and k in HIDDEN: poly(t, 'none', LINE, 0.0, 0.8, '3 2'); continue
        poly(t, TRI, MUTED, 1.0 if opaque else ALPHA[k])
        if split and k in SPLIT:
            m = [((t[i][0] + t[(i + 1) % 3][0]) / 2, (t[i][1] + t[(i + 1) % 3][1]) / 2) for i in range(3)]
            poly(t, '#b7d3f6', BLUE, .75, 1.3); add('<polygon points="%s" fill="none" stroke="%s" stroke-width="1.2"/>' % (' '.join('%.1f,%.1f' % q for q in m), BLUE))
def draw_mesh(x, y, s, pts=None, faces=None, alpha=.6, mark=None, clean=False):
    pts = pts or PTS; faces = faces or FACES
    for k, f in enumerate(faces):
        t = [place(pts[v], x, y, s) for v in f]
        if clean and k in HIDDEN: poly(t, 'none', LINE, 0.0, 0.8, '3 2'); continue
        poly(t, TRI, INK if alpha >= 1 else MUTED, alpha, 1.0)
    if mark:
        a, b = place(pts[mark[0]], x, y, s), place(pts[mark[1]], x, y, s); add(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{ORANGE}" stroke-width="4" stroke-linecap="round"/>')
EU, EV = vid(1, 1), vid(1, 2)
def collapsed():
    pts = list(PTS); m = ((PTS[EU][0] + PTS[EV][0]) / 2, (PTS[EU][1] + PTS[EV][1]) / 2); pts[EU] = m; pts[EV] = m
    faces = [tuple(EU if v == EV else v for v in f) for f in FACES]; return pts, [f for f in faces if len(set(f)) == 3]
CP, CF = collapsed()
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">')
add('<defs>' + ''.join(f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in (INK, BLUE, ORANGE, MUTED)) + '</defs>')
add(f'<rect width="{W}" height="{H}" fill="white"/>')
S, s = 44, 26   # scale of the big and the small pictures
def loop_module(x0, w, title, sub, state_a, state_b, lab_a, lab_b, goal, back_lab, mlabs):
    panel(x0, 20, w, 440, title, sub)
    ax, bx, ty = x0 + 46, x0 + w - 46 - 3 * S, 78
    state_a(ax, ty, S); state_b(bx, ty, S)
    text(ax + 1.5 * S, ty + 3 * S + 22, lab_a, 12); text(bx + 1.5 * S, ty + 3 * S + 22, lab_b, 12, INK, 'middle', 'bold')
    arrow(ax + 3 * S + 14, ty + 1.5 * S, bx - 14, ty + 1.5 * S, INK, 2.2)                                     # the main line
    path(f'M {bx + 1.5 * S} {ty - 8} C {bx + 1.5 * S} {ty - 34}, {ax + 1.5 * S} {ty - 34}, {ax + 1.5 * S} {ty - 8}', MUTED, 1.3, '5 3'); text(x0 + w / 2, ty - 30, 'next round', 11, MUTED)
    text(x0 + w / 2, ty + 1.5 * S - 10, goal, 11.5, BLUE)
    # the measuring branch underneath: copy -> opaque -> hidden faces dropped -> encoder -> R
    my = 318; xs = [x0 + 40 + k * (w - 80 - 3 * s) / 4 for k in range(5)]
    add(f'<rect x="{x0 + 22}" y="{my - 30}" width="{w - 44}" height="{3 * s + 74}" rx="10" fill="white" stroke="{BLUE}" stroke-width="1.2" stroke-dasharray="6 4"/>'); text(x0 + 34, my - 12, 'measure the file size', 11.5, BLUE, 'start', 'bold')
    for k, (f, kw) in enumerate(mlabs['draw']): f(xs[k], my, s, **kw)
    trap(xs[3] + 6, my + 4, 3 * s - 22, 3 * s - 8); text(xs[3] + 1.5 * s - 5, my + 1.5 * s - 2, 'real', 10.5); text(xs[3] + 1.5 * s - 5, my + 1.5 * s + 11, 'encoder', 10.5)
    add(f'<rect x="{xs[4] + 2}" y="{my + 16}" width="{3 * s - 4}" height="{3 * s - 32}" rx="6" fill="#e6f0fc" stroke="{BLUE}" stroke-width="1.3"/>'); text(xs[4] + 1.5 * s, my + 1.5 * s - 3, 'R', 15, INK, 'middle', 'bold'); text(xs[4] + 1.5 * s, my + 1.5 * s + 13, 'bytes', 10.5, MUTED)
    for k in range(4): arrow(xs[k] + 3 * s + 3, my + 1.5 * s, xs[k + 1] - 3, my + 1.5 * s, BLUE, 1.4)
    for k, lab in enumerate(mlabs['text']): text(xs[k] + 1.5 * s, my + 3 * s + 16, lab[0], 10.5, MUTED); text(xs[k] + 1.5 * s, my + 3 * s + 29, lab[1], 10.5, MUTED)
    path(f'M {ax + 1.5 * S} {ty + 3 * S + 30} L {ax + 1.5 * S} {my - 34} ', BLUE, 1.4); text(ax + 1.5 * S + 8, ty + 3 * S + 48, 'copy', 11, BLUE, 'start')
    path(f'M {xs[4] + 1.5 * s} {my + 12} L {xs[4] + 1.5 * s} {ty + 3 * S + 30}', BLUE, 1.4); text(xs[4] + 1.5 * s - 8, ty + 3 * S + 48, back_lab, 11, BLUE, 'end')
# ================= A
A0, AW = 20, 600
loop_module(A0, AW, 'A  Budgeted growth', '', lambda x, y, z: draw_soup(x, y, z), lambda x, y, z: draw_soup(x, y, z, split=True), 'semi-transparent triangle soup', 'split the n worst triangles',
            'growth goal V* set by the target size T', 'n = how many to split',
            dict(draw=[(draw_soup, {}), (draw_soup, dict(opaque=True)), (draw_soup, dict(opaque=True, clean=True))], text=[('copy', ''), ('made opaque', '(final opacity)'), ('hidden faces', 'dropped'), ('', ''), ('', '')]))
# ================= B
B0, BW = A0 + AW + 22, 190; panel(B0, 20, BW, 440, 'B  Connect')
draw_mesh(B0 + (BW - 3 * S) / 2, 78, S, alpha=.55); text(B0 + BW / 2, 78 + 3 * S + 22, 'one connected mesh', 12)
text(B0 + BW / 2, 300, 'restricted Delaunay', 11.5, MUTED); text(B0 + BW / 2, 316, 'triangulation', 11.5, MUTED); text(B0 + BW / 2, 332, '(as in MeshSplatting)', 11.5, MUTED)
arrow(A0 + AW + 2, 78 + 1.5 * S, B0 + (BW - 3 * S) / 2 - 8, 78 + 1.5 * S, INK, 2.2)
# ================= C
C0, CW = B0 + BW + 22, 600
loop_module(C0, CW, 'C  Rate-locked refinement', '', lambda x, y, z: draw_mesh(x, y, z, alpha=.6, mark=(EU, EV)), lambda x, y, z: draw_mesh(x, y, z, CP, CF, alpha=.6), 'cheapest edges marked', 'collapsed',
            'remove exactly the excess E = R &#8722; T', 'E = bytes to remove',
            dict(draw=[(draw_mesh, dict(alpha=.6)), (draw_mesh, dict(alpha=1.0)), (draw_mesh, dict(alpha=1.0, clean=True))], text=[('copy', ''), ('made opaque', '(final opacity)'), ('hidden faces', 'dropped'), ('', ''), ('', '')]))
arrow(B0 + BW + 2, 78 + 1.5 * S, C0 + 46 - 8, 78 + 1.5 * S, INK, 2.2)
# ================= D / E
D0, DW = C0 + CW + 22, W - (C0 + CW + 22) - 20; panel(D0, 20, DW, 440, 'D  Encode and decode')
dx = D0 + 24; draw_mesh(dx, 78, 34, CP, CF, alpha=1.0); text(dx + 51, 78 + 102 + 20, 'final opaque mesh', 11.5)
arrow(C0 + CW + 2, 78 + 1.5 * S, dx - 6, 78 + 1.5 * S, INK, 2.2)
ex = dx + 150; bands = [('connectivity', 39, '#2a78d6'), ('positions', 21, '#5598e7'), ('colour indices', 19, '#86b6ef'), ('base colour', 15, '#b7d3f6'), ('codebooks', 7, '#cde2fb')]
hy = 70; hh = 150; trap(ex, hy, 150, hh, fill='white'); yy = hy + 34; span = hh - 68
for name, share, col in bands:
    bh = span * share / 100; add(f'<rect x="{ex + 14}" y="{yy:.1f}" width="122" height="{bh - 1.5:.1f}" fill="{col}"/>'); text(ex + 75, yy + bh / 2 + 3.5, f'{name} {share}%' if bh > 11 else '', 9.5, INK if share < 30 else 'white'); yy += bh
text(ex + 75, hy - 6, 'Encoder', 12.5, INK, 'middle', 'bold'); arrow(dx + 108, 78 + 51, ex - 6, 78 + 51, INK, 1.8)
fx, fy = ex + 20, 268; add(f'<rect x="{fx}" y="{fy}" width="110" height="26" rx="4" fill="white" stroke="{INK}" stroke-width="1.4"/>'); text(fx + 55, fy + 17, 'one file', 11.5)
arrow(ex + 75, hy + hh + 20, ex + 75, fy - 4, INK, 1.8)
trap(fx + 20, 320, 70, 86, flip=True); text(fx + 55, 368, 'Decoder', 11.5, INK, 'middle', 'bold'); arrow(fx + 55, fy + 28, fx + 55, 326, INK, 1.8)
draw_mesh(dx, 322, 30, CP, CF, alpha=1.0); arrow(fx + 18, 363, dx + 96, 363, INK, 1.8); text(dx + 45, 322 + 90 + 18, 'rendered as usual', 11, MUTED)
# colour codebook picture
cbx, cby = dx + 4, 226; text(cbx + 50, cby - 6, 'colour codebook', 10.5, MUTED)
for r_, (n, lab) in enumerate(((12, 'main 4,096'), (6, 'residual 256'))):
    for i in range(n): add(f'<rect x="{cbx + i * 8.4:.1f}" y="{cby + r_ * 22}" width="7" height="12" fill="{["#eb6834", "#eda100", "#1baf7a", "#2a78d6", "#e87ba4", "#6250d6"][i % 6]}" fill-opacity="{0.85 if r_ == 0 else 0.45}"/>')
    text(cbx + 106, cby + r_ * 22 + 10, lab, 9.5, MUTED, 'start')
# ================= iteration bar: one arrow per round into the module above
by = 540; add(f'<line x1="{A0}" y1="{by}" x2="{C0 + CW}" y2="{by}" stroke="{INK}" stroke-width="1.5"/>')
ax_ = lambda k: A0 + 16 + (AW - 32) * k / 10.0; cx_ = lambda k: C0 + 16 + (CW - 32) * (k - 12.5) / 17.5
for k in range(1, 21): arrow(ax_(k * .5), by - 2, ax_(k * .5), 466, BLUE, 1.1)
arrow(B0 + BW / 2, by - 2, B0 + BW / 2, 466, MUTED, 1.3)
for k in range(35): arrow(cx_(12.5 + k * .5), by - 2, cx_(12.5 + k * .5), 466, BLUE, 1.1)
for x, lab in ((ax_(0), '0'), (ax_(4), '4k'), (ax_(10), '10k'), (B0 + BW / 2, '11k'), (cx_(12.5), '12.5k'), (cx_(20), '20k'), (cx_(24), '24k'), (cx_(29.5), '29.5k')):
    add(f'<line x1="{x:.1f}" y1="{by - 5}" x2="{x:.1f}" y2="{by + 5}" stroke="{INK}" stroke-width="1.3"/>'); text(x, by + 20, lab, 11, MUTED)
text(A0, by + 42, 'training iterations: one arrow = one round, every 500 iterations (in A the size is measured every 1,000 from 4k)', 11.5, MUTED, 'start')
text(C0 + CW, by + 42, '35 rounds; triangles fully opaque from 24k', 11.5, MUTED, 'end')
add('</svg>'); open('paper/pipeline-sketch/pipeline-sketch-v4.svg', 'w').write('\n'.join(o)); print('ok')
