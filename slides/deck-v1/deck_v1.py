"""MeshSplatting compression deck v1 (manim 0.22).  Four clips:
  S1_Pipeline   original asset -> codec -> decoded asset, results across 7 scenes
  S2_Collapse   old route: post-hoc vertex collapse, animated, RD points appear
  S3_RateAware  new route: rate-controlled growth, error-vs-price split rule, scene-adaptive stop
  S4_Hull       old vs new convex hull (Room absolute, 7-scene relative), smoothed
render: python -m manim -qh deck_v1.py S1_Pipeline S2_Collapse S3_RateAware S4_Hull
numbers come from analysis/rd/*.csv (copied into DATA below, 2026-10-22)."""
import math, random, csv, os
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import Delaunay
from manim import *

SURF, REF, MUTED, GRID = '#fcfcfb', '#2b2b2b', '#7a7a76', '#e3e3df'
CAP, OLD, NEW, PRUNE, CTRL = '#2a78d6', '#eb6834', '#1baf7a', '#0f6e4c', '#8a5cd6'
config.background_color = SURF
Text.set_default(font='Inter', color=REF)

RD = '/mnt/project-files/analysis/rd'
ORIG = {'room': (742, 28.263), 'garden': (740, 24.732), 'bonsai': (680, 27.291), 'counter': (575, 25.944),
        'kitchen': (456, 26.454), 'bicycle': (680, 23.033), 'stump': (698, 24.928)}
SCENES = ['room', 'garden', 'bonsai', 'counter', 'kitchen', 'bicycle', 'stump']
# level-1.0 codec (all vertices kept, K4096 R256, scene precision) -- seven-scene-points-v2.csv
LEVEL1 = {'room': (30.932, 28.061), 'garden': (32.539, 24.458), 'bonsai': (25.502, 26.987), 'counter': (25.369, 25.593),
          'kitchen': (21.356, 26.094), 'bicycle': (28.372, 22.838), 'stump': (29.210, 24.648)}
# rate-controlled growth, lambda 2e-3 / 6e-3 -- lambda-7scenes-v1.csv
LAM = {'room': ((13.699, 27.985), (5.695, 27.370)), 'garden': ((24.781, 24.445), (10.685, 23.637)),
       'bonsai': ((14.796, 26.727), (6.914, 26.066)), 'counter': ((14.899, 25.761), (6.940, 25.263)),
       'kitchen': ((15.894, 26.144), (7.604, 25.312)), 'bicycle': ((30.067, 23.018), (13.831, 22.703)),
       'stump': ((23.317, 24.729), (10.044, 24.166))}
# Room post-hoc collapse route (shared-SH K4096 R256 q14, geo-only collapse + finetune) -- room-lambda-vs-cap-v2.csv
ROOM_COLLAPSE = [('3.19M', 30.932, 28.061), ('2.55M', 25.513, 27.883), ('1.92M', 19.434, 27.452), ('1.35M', 13.926, 26.660), ('0.97M', 10.139, 25.665)]
ROOM_LAMBDA = [('λ=6.2e-4', 28.380, 28.272), ('1.85e-3', 13.699, 27.985), ('6.2e-3', 5.695, 27.370), ('1.85e-2', 3.013, 26.509)]
ROOM_PRUNE = [('prune', 12.572, 28.004), ('prune', 5.228, 27.382)]
MEANS = {'old': (23.6, -0.44, 28), 'l2': (19.6, -0.26, 36), 'l6': (8.8, -0.88, 80)}


def upper_hull(pts):
    pts = sorted(set(pts)); h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) >= 0: h.pop()
        h.append(p)
    top = max(range(len(h)), key=lambda i: h[i][1]); return h[:top + 1]


def smooth_hull(pts, n=120):
    """monotone (PCHIP) curve through the hull vertices in log-MB; stays concave-ish and passes through every measured hull point."""
    h = upper_hull(pts); x = np.log10([p[0] for p in h]); y = [p[1] for p in h]
    f = PchipInterpolator(x, y); xs = np.linspace(x[0], x[-1], n); return [(10 ** a, float(f(a))) for a in xs]


def read_seven():
    rows = list(csv.DictReader(open(os.path.join(RD, 'seven-scene-points-v2.csv'))))
    return {sc: [(float(r['MB']), float(r['test_psnr'])) for r in rows if r['scene'] == sc and r['series'] == 'collapse'] for sc in SCENES}


# ------------------------------------------------------------------ mesh toy model (2-D, for illustration only)
class ToyMesh:
    """vertices in a disc; triangles from a jittered grid; supports edge collapse and centroid split."""
    def __init__(self, n=7, seed=3, radius=1.45):
        rnd = random.Random(seed); self.V = []; idx = {}
        for i in range(n + 1):
            for j in range(n + 1):
                x = -radius + 2 * radius * i / n; y = -radius + 2 * radius * j / n
                if 0 < i < n: x += rnd.uniform(-0.25, 0.25) * radius / n
                if 0 < j < n: y += rnd.uniform(-0.25, 0.25) * radius / n
                idx[(i, j)] = len(self.V); self.V.append([x, y])
        self.T = []
        for i in range(n):
            for j in range(n):
                a, b, c, d = idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]
                if (i + j) % 2: self.T += [[a, b, c], [a, c, d]]
                else: self.T += [[a, b, d], [b, c, d]]
        self.rnd = rnd

    def alive(self):
        used = {v for t in self.T for v in t}; return used

    def collapse(self, k):
        """collapse k random interior edges: move i to midpoint, delete j."""
        done = 0; tries = 0
        while done < k and tries < 4000:
            tries += 1
            edges = [(self.elen(a, b) * self.rnd.uniform(0.8, 1.2), a, b) for t in self.T for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])) if not (self.boundary(a) or self.boundary(b))]
            if not edges: break
            _, i, j = min(edges)
            m = [(self.V[i][0] + self.V[j][0]) / 2, (self.V[i][1] + self.V[j][1]) / 2]; self.V[i] = m
            self.T = [tt for tt in self.T if not (i in tt and j in tt)]
            self.T = [[i if v == j else v for v in tt] for tt in self.T]
            self.T = [tt for tt in self.T if len(set(tt)) == 3]; done += 1
        return done

    def elen(self, a, b):
        return math.hypot(self.V[a][0] - self.V[b][0], self.V[a][1] - self.V[b][1])

    def boundary(self, v):
        x, y = self.V[v]; return abs(x) > 1.4 or abs(y) > 1.4

    def split(self, tri_ids):
        new = []
        for t in tri_ids:
            a, b, c = self.T[t]; cx = sum(self.V[v][0] for v in (a, b, c)) / 3; cy = sum(self.V[v][1] for v in (a, b, c)) / 3
            m = len(self.V); self.V.append([cx, cy]); new += [[a, b, m], [b, c, m], [c, a, m]]
        self.T = [t for k, t in enumerate(self.T) if k not in set(tri_ids)] + new

    def delaunay(self):
        self.T = [list(map(int, t)) for t in Delaunay(np.array(self.V)).simplices]

    def area(self, t):
        (x1, y1), (x2, y2), (x3, y3) = (self.V[v] for v in self.T[t]); return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2

    def centroid(self, t):
        return np.mean([self.V[v] for v in self.T[t]], axis=0)


def mesh_mobject(M, origin, scale=1.0, face_color=None, stroke=1.2, vert_r=0.035, colors=None):
    faces = VGroup(); verts = VGroup()
    for k, t in enumerate(M.T):
        pts = [origin + scale * np.array([M.V[v][0], M.V[v][1], 0]) for v in t]
        col = colors[k] if colors is not None else (face_color or '#dfe9f6')
        faces.add(Polygon(*pts, fill_color=col, fill_opacity=0.85, stroke_color='#6b7f99', stroke_width=stroke))
    for v in sorted(M.alive()):
        verts.add(Dot(origin + scale * np.array([M.V[v][0], M.V[v][1], 0]), radius=vert_r, color='#2b5c99'))
    return VGroup(faces, verts)


def detail_field(x, y):
    """where the image has detail (toy): two blobs; error mass of a triangle = area * detail at centroid."""
    return 0.03 + 1.0 * math.exp(-((x - 0.45) ** 2 + (y + 0.35) ** 2) / 0.22) + 0.7 * math.exp(-((x + 0.7) ** 2 + (y - 0.6) ** 2) / 0.09)


def heat(v):
    """0..1 -> light grey to orange-red."""
    v = max(0.0, min(1.0, v)); return interpolate_color(ManimColor('#e9ecef'), ManimColor('#e4572e'), v)


def title(txt, sub=None):
    t = Text(txt, font_size=34, weight=SEMIBOLD).to_edge(UP, buff=0.35).to_edge(LEFT, buff=0.6)
    g = VGroup(t)
    if sub: g.add(Text(sub, font_size=20, color=MUTED).next_to(t, DOWN, buff=0.12, aligned_edge=LEFT))
    return g


def small(txt, size=18, color=REF, **kw): return Text(txt, font_size=size, color=color, **kw)


# ================================================================== S1
class S1_Pipeline(Scene):
    def construct(self):
        tt = title('MeshSplatting asset → codec → decoded asset', 'one opaque triangle mesh in, one out; the codec is lossy by design')
        self.play(FadeIn(tt)); self.wait(0.3)
        # original asset
        M = ToyMesh(n=9, seed=5, radius=1.2); L = LEFT * 4.9 + DOWN * 0.4
        mesh = mesh_mobject(M, L, scale=0.9)
        cap1 = VGroup(small('original asset', 20, weight=SEMIBOLD), small('3.2 M vertices, triangle soup', 16, MUTED), small('XYZ + faces + 48 SH coefficients / vertex', 16, MUTED), small('FP32: 742 MB', 20, OLD, weight=SEMIBOLD)).arrange(DOWN, buff=0.08).next_to(mesh, DOWN, buff=0.25)
        self.play(Create(mesh[0], run_time=1.6), FadeIn(mesh[1])); self.play(FadeIn(cap1))
        # encoder
        enc = RoundedRectangle(corner_radius=0.15, width=3.5, height=2.9, stroke_color=REF, stroke_width=1.5, fill_color='#f1f3f6', fill_opacity=1).move_to(DOWN * 0.4 + LEFT * 0.95)
        enc_t = small('encoder', 20, weight=SEMIBOLD).next_to(enc.get_top(), DOWN, buff=0.12)
        streams = VGroup(*[small(s, 14) for s in ['vertex reorder (spatial locality)', 'XYZ  →  14–16 bit grid', 'faces  →  context coder, ~2 bit/face', 'SH (48 floats)  →  codebook index', '     K = 4096 + residual R = 256', 'per-vertex weight  →  constant']]).arrange(DOWN, buff=0.09, aligned_edge=LEFT).next_to(enc_t, DOWN, buff=0.15)
        a1 = Arrow(mesh.get_right() + RIGHT * 0.05, enc.get_left(), buff=0.1, color=REF, stroke_width=2, max_tip_length_to_length_ratio=0.12)
        self.play(GrowArrow(a1), FadeIn(enc), FadeIn(enc_t)); self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.2) for s in streams], lag_ratio=0.25, run_time=2.2))
        # bitstream
        bits = VGroup(*[Square(0.11, fill_color=REF if random.Random(k).random() < 0.5 else SURF, fill_opacity=1, stroke_color=REF, stroke_width=0.8) for k in range(24)]).arrange_in_grid(rows=3, buff=0.03).next_to(enc, RIGHT, buff=0.3)
        bits_t = VGroup(small('bitstream', 16, MUTED), small('30.9 MB  (24×)', 20, NEW, weight=SEMIBOLD)).arrange(DOWN, buff=0.06).next_to(bits, DOWN, buff=0.15)
        self.play(LaggedStart(*[FadeIn(b) for b in bits], lag_ratio=0.03, run_time=1.0), FadeIn(bits_t))
        # decoder + decoded mesh
        R = RIGHT * 5.3 + DOWN * 0.4; mesh2 = mesh_mobject(M, R, scale=0.9)
        a2 = Arrow(bits.get_right(), mesh2.get_left(), buff=0.15, color=REF, stroke_width=2, max_tip_length_to_length_ratio=0.12)
        dec_t = small('decoder', 16, MUTED).next_to(a2, UP, buff=0.08)
        cap2 = VGroup(small('decoded asset', 20, weight=SEMIBOLD), small('same mesh, SH from codebook', 16, MUTED), small('28.26 → 28.06 dB  (−0.20)', 20, NEW, weight=SEMIBOLD)).arrange(DOWN, buff=0.08).next_to(mesh2, DOWN, buff=0.25)
        self.play(GrowArrow(a2), FadeIn(dec_t)); self.play(Create(mesh2[0], run_time=1.4), FadeIn(mesh2[1])); self.play(FadeIn(cap2))
        note = small('Room, Mip-NeRF 360 · any rasterizer renders the output unchanged', 16, MUTED).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note)); self.wait(1.5)
        # ---- results across scenes
        self.play(*[FadeOut(m) for m in (mesh, cap1, enc, enc_t, streams, a1, bits, bits_t, a2, dec_t, mesh2, cap2, note)])
        t2 = title('Codec alone, all vertices kept: 7 Mip-NeRF 360 scenes', 'bars = delivered size (log scale); labels = PSNR change vs the FP32 original')
        self.play(Transform(tt, t2))
        ax = Axes(x_range=[0, 8, 1], y_range=[0, 3.2, 1], x_length=11.0, y_length=4.4, axis_config={'include_ticks': False, 'include_numbers': False, 'stroke_color': MUTED, 'stroke_width': 1}, tips=False).shift(DOWN * 0.55)
        ylab = VGroup(*[small(s, 14, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v, s in ((0, '1 MB'), (1, '10 MB'), (2, '100 MB'), (3, '1 GB'))])
        self.play(Create(ax), FadeIn(ylab))
        bars = VGroup(); labs = VGroup()
        for k, sc in enumerate(SCENES):
            mb0, p0 = ORIG[sc]; mb1, p1 = LEVEL1[sc]
            b0 = Rectangle(width=0.42, height=0.001, fill_color=MUTED, fill_opacity=0.35, stroke_width=0).move_to(ax.c2p(k + 0.72, 0), aligned_edge=DOWN)
            b1 = Rectangle(width=0.42, height=0.001, fill_color=NEW, fill_opacity=0.95, stroke_width=0).move_to(ax.c2p(k + 1.2, 0), aligned_edge=DOWN)
            bars.add(b0, b1)
            labs.add(small(sc, 16).next_to(ax.c2p(k + 0.96, 0), DOWN, buff=0.15))
            b0.target_h = ax.c2p(0, math.log10(mb0))[1] - ax.c2p(0, 0)[1]; b1.target_h = ax.c2p(0, math.log10(mb1))[1] - ax.c2p(0, 0)[1]
            b0.lab = small(f'{mb0} MB', 13, MUTED); b1.lab = VGroup(small(f'{mb1:.0f} MB', 13, NEW, weight=SEMIBOLD), small(f'{p1 - p0:+.2f} dB', 13, REF)).arrange(DOWN, buff=0.03)
        self.play(FadeIn(labs))
        anims = []
        for b in bars: anims.append(b.animate.stretch_to_fit_height(b.target_h).move_to(b.get_bottom(), aligned_edge=DOWN))
        self.play(LaggedStart(*anims, lag_ratio=0.08, run_time=2.4))
        self.play(LaggedStart(*[FadeIn(b.lab.next_to(b, UP, buff=0.06)) for b in bars], lag_ratio=0.05, run_time=1.5))
        leg = VGroup(VGroup(Square(0.18, fill_color=MUTED, fill_opacity=0.35, stroke_width=0), small('FP32 original', 15, MUTED)).arrange(RIGHT, buff=0.12),
                     VGroup(Square(0.18, fill_color=NEW, fill_opacity=0.95, stroke_width=0), small('codec, all vertices kept: 21–33 MB, −0.20 to −0.35 dB, 22–29×', 15)).arrange(RIGHT, buff=0.12)).arrange(RIGHT, buff=0.6).to_edge(DOWN, buff=0.28)
        self.play(FadeIn(leg)); self.wait(2.5)


# ================================================================== S2
class S2_Collapse(Scene):
    def construct(self):
        tt = title('Old route: post-hoc vertex collapse', 'train dense, then collapse edges on the trained mesh, fine-tune, apply the same codec · Room')
        self.play(FadeIn(tt))
        M = ToyMesh(n=11, seed=7, radius=1.55); L = LEFT * 3.9 + DOWN * 0.55
        mesh = mesh_mobject(M, L, scale=1.0, vert_r=0.03, stroke=1.0)
        counter = VGroup(small('vertices', 16, MUTED), small('3.19 M', 30, weight=SEMIBOLD)).arrange(DOWN, buff=0.05).next_to(mesh, DOWN, buff=0.3)
        self.play(Create(mesh[0], run_time=1.5), FadeIn(mesh[1]), FadeIn(counter))
        # RD chart right
        ax = Axes(x_range=[math.log10(8), math.log10(40), 1], y_range=[25.2, 28.6, 1], x_length=5.2, y_length=4.0,
                  x_axis_config={'scaling': LogBase(10), 'include_ticks': False, 'include_numbers': False}, y_axis_config={'include_ticks': False, 'include_numbers': False},
                  axis_config={'stroke_color': MUTED, 'stroke_width': 1}, tips=False).shift(RIGHT * 3.6 + DOWN * 0.45)
        xt = VGroup(*[small(f'{v}', 13, MUTED).next_to(ax.c2p(v, 25.2), DOWN, buff=0.08) for v in (10, 15, 20, 30)])
        yt = VGroup(*[small(f'{v}', 13, MUTED).next_to(ax.c2p(8, v), LEFT, buff=0.08) for v in (26, 27, 28)])
        xl = small('delivered size (MB, log)', 14, MUTED).next_to(ax, DOWN, buff=0.35); yl = small('test PSNR (dB)', 14, MUTED).rotate(PI / 2).next_to(ax, LEFT, buff=0.45)
        orig = DashedLine(ax.c2p(8, 28.263), ax.c2p(40, 28.263), color=REF, stroke_width=1.2, dash_length=0.08)
        orig_t = small('original FP32 · 28.26 dB · 742 MB', 13, REF).next_to(orig, UP, buff=0.05).align_to(ax.c2p(8.3, 0), LEFT)
        self.play(Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl), Create(orig), FadeIn(orig_t))
        pts = VGroup(); curve = None
        name0, mb, ps = ROOM_COLLAPSE[0]
        d = Dot(ax.c2p(mb, ps), radius=0.08, color=OLD); lab = small(f'{mb:.1f} MB · {ps:.2f}', 12, OLD).next_to(d, LEFT, buff=0.1)
        self.play(FadeIn(d, scale=0.5), FadeIn(lab)); pts.add(d)
        steps = [(22, '2.55 M'), (24, '1.92 M'), (22, '1.35 M'), (18, '0.97 M')]
        for (k, (nk, name)) in enumerate(steps):
            M.collapse(nk); new_mesh = mesh_mobject(M, L, scale=1.0, vert_r=0.03, stroke=1.0)
            new_counter = VGroup(small('vertices', 16, MUTED), small(name, 30, weight=SEMIBOLD)).arrange(DOWN, buff=0.05).next_to(mesh, DOWN, buff=0.3)
            _, mb, ps = ROOM_COLLAPSE[k + 1]
            d = Dot(ax.c2p(mb, ps), radius=0.08, color=OLD)
            lab = small(f'{mb:.1f} MB · {ps:.2f}  ({ps - 28.263:+.2f})', 13, OLD).next_to(d, DOWN if k < 2 else DR, buff=0.08)
            self.play(Transform(mesh, new_mesh, run_time=1.1), Transform(counter, new_counter)); self.play(FadeIn(d, scale=0.5), FadeIn(lab), run_time=0.5); pts.add(d)
            if k >= 1 and lab is not None and k < 3: pass
        line = VMobject(color=OLD, stroke_width=2.5).set_points_smoothly([ax.c2p(mb, ps) for _, mb, ps in ROOM_COLLAPSE])
        self.play(Create(line))
        msg = VGroup(small('the vertices kept were chosen after training, without rate in the loop:', 17), small('−1.6 dB at 1.35 M, −2.6 dB at 0.97 M · the hull cannot go below ~17 MB without losing > 1 dB', 17, OLD, weight=SEMIBOLD)).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(msg)); self.wait(2.5)


# ================================================================== S3
def grow(M, lam, rounds, bpv=10.0):
    """toy rate-controlled growth: split triangle t iff error_mass(t) / (3 * bpv) > lam.  returns list of (triangle count) per round."""
    hist = [len(M.alive())]
    for _ in range(rounds):
        ids = [t for t in range(len(M.T)) if M.area(t) * detail_field(*M.centroid(t)) / (3 * bpv) > lam]
        if ids: M.split(ids); M.delaunay()
        hist.append(len(M.alive()))
    return hist


class S3_RateAware(Scene):
    def construct(self):
        tt = title('New route: rate-controlled growth', 'compression-aware mesh generation: split a triangle only if its error is worth its bytes')
        self.play(FadeIn(tt))
        rule = VGroup(small('at each densification step, for every triangle t:', 16, MUTED),
                      MarkupText('split  t   ⟺   <b>ΔD<sub>t</sub> / ΔR<sub>t</sub>  &gt;  λ</b>', font='Inter', font_size=26, color=REF),
                      small('ΔD = rendered error mass on t  ·  ΔR = bytes of the new vertex  ·  λ = price', 15, MUTED)).arrange(DOWN, buff=0.1).to_edge(RIGHT, buff=0.5).shift(UP * 1.35)
        self.play(FadeIn(rule))
        M = ToyMesh(n=4, seed=11, radius=1.5); C = LEFT * 3.6 + DOWN * 0.6; bpv = 10.0; lam = 0.0005; YMAX = max(grow(ToyMesh(n=4, seed=11, radius=1.5), lam * 0.35, 6, bpv)) * 1.15
        mesh = mesh_mobject(M, C, 1.0, colors=[heat(0) for _ in M.T], vert_r=0.035)
        cnt = VGroup(small('vertices', 15, MUTED), small(f'{len(M.alive())}', 26, weight=SEMIBOLD)).arrange(DOWN, buff=0.04).next_to(mesh, DOWN, buff=0.28)
        self.play(Create(mesh[0], run_time=1.0), FadeIn(mesh[1]), FadeIn(cnt))
        # legend for heat
        heat_bar = VGroup(*[Square(0.16, fill_color=heat(v / 7), fill_opacity=1, stroke_width=0) for v in range(8)]).arrange(RIGHT, buff=0)
        heat_t = VGroup(small('low error mass', 13, MUTED), heat_bar, small('high', 13, MUTED)).arrange(RIGHT, buff=0.12).next_to(mesh, UP, buff=0.22)
        self.play(FadeIn(heat_t))
        # vertex-count curve (right-bottom)
        ax = Axes(x_range=[0, 6, 1], y_range=[0, YMAX, 100], x_length=4.6, y_length=2.1, axis_config={'include_ticks': False, 'include_numbers': False, 'stroke_color': MUTED, 'stroke_width': 1}, tips=False).to_edge(RIGHT, buff=0.7).shift(DOWN * 1.55)
        axl = VGroup(small('densification rounds →', 13, MUTED).next_to(ax, DOWN, buff=0.1), small('vertices', 13, MUTED).next_to(ax, LEFT, buff=0.1).rotate(PI / 2))
        self.play(Create(ax), FadeIn(axl))
        hist = [len(M.alive())]; curve_pts = [ax.c2p(0, hist[0])]
        for r in range(1, 6):
            err = [M.area(t) * detail_field(*M.centroid(t)) for t in range(len(M.T))]; mx = max(err)
            colored = mesh_mobject(M, C, 1.0, colors=[heat(e / mx) for e in err], vert_r=0.035)
            self.play(Transform(mesh, colored, run_time=0.7))
            ids = [t for t in range(len(M.T)) if err[t] / (3 * bpv) > lam]
            if ids:
                flash = VGroup(*[mesh[0][t].copy().set_stroke(NEW, 3).set_fill(opacity=0) for t in ids])
                self.play(FadeIn(flash, run_time=0.3))
                M.split(ids); M.delaunay()
                err2 = [M.area(t) * detail_field(*M.centroid(t)) for t in range(len(M.T))]
                new = mesh_mobject(M, C, 1.0, colors=[heat(e / mx) for e in err2], vert_r=0.035)
                hist.append(len(M.alive()))
                nc = VGroup(small('vertices', 15, MUTED), small(f'{hist[-1]}', 26, weight=SEMIBOLD)).arrange(DOWN, buff=0.04).next_to(mesh, DOWN, buff=0.28)
                self.play(FadeOut(flash, run_time=0.2), Transform(mesh, new, run_time=0.8), Transform(cnt, nc))
            else:
                hist.append(len(M.alive()))
                stop = small('no triangle is worth its bytes → growth stops', 16, NEW, weight=SEMIBOLD).move_to(RIGHT * 3.4 + UP * 0.1)
                self.play(FadeIn(stop))
            curve_pts.append(ax.c2p(r, hist[-1]))
            seg = Line(curve_pts[-2], curve_pts[-1], color=NEW, stroke_width=2.5); self.play(Create(seg, run_time=0.3))
        self.wait(0.6)
        # second lambda: compare final meshes
        M2 = ToyMesh(n=4, seed=11, radius=1.5); h2 = grow(M2, lam * 3.2, 6, bpv)
        M3 = ToyMesh(n=4, seed=11, radius=1.5); h3 = grow(M3, lam * 0.35, 6, bpv)
        c_hi = VMobject(color=OLD, stroke_width=2).set_points_as_corners([ax.c2p(k, h) for k, h in enumerate(h2[:6])])
        c_lo = VMobject(color=CAP, stroke_width=2).set_points_as_corners([ax.c2p(k, h) for k, h in enumerate(h3[:6])])
        l_hi = small('3× larger λ: stops earlier', 12, OLD).next_to(c_hi.get_end(), DOWN, buff=0.06); l_lo = small('3× smaller λ: grows further', 12, CAP).next_to(c_lo.get_end(), UP, buff=0.06)
        l_mid = small('λ', 12, NEW).next_to(curve_pts[-1], RIGHT, buff=0.06)
        self.play(Create(c_hi), FadeIn(l_hi), Create(c_lo), FadeIn(l_lo), FadeIn(l_mid))
        msg = VGroup(small('λ is the only knob: one value per target rate, the vertex count is an output (41–100 % of dense at λ = 2e-3, 15–43 % at 6e-3)', 14),
                     small('phase B: the same price applied in reverse removes vertices that stopped paying · the codec after training is unchanged', 14, MUTED)).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.28)
        self.play(FadeIn(msg)); self.wait(2.5)
        # ---- result across scenes: lambda vs collapse at equal bytes (bars of dB vs original)
        self.play(*[FadeOut(m) for m in (mesh, cnt, heat_t, ax, axl, rule, msg, c_hi, c_lo, l_hi, l_lo, l_mid)], *[FadeOut(m) for m in self.mobjects if isinstance(m, (Line, VMobject)) and m not in (tt,)])
        t2 = title('Rate-controlled growth across 7 scenes', 'PSNR change vs the FP32 original · same codec after training · λ = 2e-3 and 6e-3')
        self.play(Transform(tt, t2))
        ax2 = Axes(x_range=[0, 8, 1], y_range=[-1.4, 0.2, 0.5], x_length=11.0, y_length=4.2, axis_config={'include_ticks': False, 'include_numbers': False, 'stroke_color': MUTED, 'stroke_width': 1}, tips=False).shift(DOWN * 0.5)
        zero = Line(ax2.c2p(0, 0), ax2.c2p(8, 0), color=REF, stroke_width=1.2); zt = small('original', 13, REF).next_to(zero, RIGHT, buff=0.05)
        yl = VGroup(*[small(f'{v:+.1f} dB' if v else '0', 13, MUTED).next_to(ax2.c2p(0, v), LEFT, buff=0.08) for v in (-1.0, -0.5, 0)])
        self.play(Create(ax2), Create(zero), FadeIn(zt), FadeIn(yl))
        bars = VGroup(); labs = VGroup(); txts = VGroup()
        for k, sc in enumerate(SCENES):
            p0 = ORIG[sc][1]
            for j, ((mb, ps), col, dx) in enumerate(((LAM[sc][0], NEW, 0.75), (LAM[sc][1], PRUNE, 1.2))):
                h = abs(ax2.c2p(0, ps - p0)[1] - ax2.c2p(0, 0)[1])
                b = Rectangle(width=0.42, height=max(h, 0.01), fill_color=col, fill_opacity=0.95, stroke_width=0).move_to(ax2.c2p(k + dx, 0), aligned_edge=UP)
                bars.add(b); txts.add(VGroup(small(f'{ps - p0:+.2f}', 12, col, weight=SEMIBOLD), small(f'{mb:.1f} MB', 12, MUTED), small(f'{ORIG[sc][0] / mb:.0f}×', 12, MUTED)).arrange(DOWN, buff=0.02).next_to(b, DOWN, buff=0.06))
            labs.add(small(sc, 16).next_to(ax2.c2p(k + 0.97, 0.2), UP, buff=0.05))
        self.play(FadeIn(labs)); self.play(LaggedStart(*[GrowFromEdge(b, UP) for b in bars], lag_ratio=0.06, run_time=2.0)); self.play(FadeIn(txts))
        leg = VGroup(VGroup(Square(0.18, fill_color=NEW, fill_opacity=0.95, stroke_width=0), small('λ = 2e-3 · 7-scene mean 19.6 MB / −0.26 dB / 36×', 15)).arrange(RIGHT, buff=0.12),
                     VGroup(Square(0.18, fill_color=PRUNE, fill_opacity=0.95, stroke_width=0), small('λ = 6e-3 · 8.8 MB / −0.88 dB / 80×', 15)).arrange(RIGHT, buff=0.12)).arrange(RIGHT, buff=0.7).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(leg)); self.wait(2.5)


# ================================================================== S4
class S4_Hull(Scene):
    def construct(self):
        tt = title('Old vs new route: the rate–distortion hull', 'same codec; the only difference is when the vertices are chosen · Room')
        self.play(FadeIn(tt))
        seven = read_seven(); old_pts = seven['room']
        new_pts = [(mb, ps) for _, mb, ps in ROOM_LAMBDA + ROOM_PRUNE]
        ax = Axes(x_range=[math.log10(2.5), math.log10(45), 1], y_range=[26.2, 28.7, 1], x_length=10.5, y_length=4.6,
                  x_axis_config={'scaling': LogBase(10), 'include_ticks': False, 'include_numbers': False}, y_axis_config={'include_ticks': False, 'include_numbers': False},
                  axis_config={'stroke_color': MUTED, 'stroke_width': 1}, tips=False).shift(DOWN * 0.45 + RIGHT * 0.3)
        grid = VGroup(*[Line(ax.c2p(2.5, v), ax.c2p(45, v), color=GRID, stroke_width=0.8) for v in (26.5, 27, 27.5, 28, 28.5)])
        xt = VGroup(*[small(f'{v}', 13, MUTED).next_to(ax.c2p(v, 26.2), DOWN, buff=0.08) for v in (3, 5, 10, 20, 30, 40)])
        yt = VGroup(*[small(f'{v}', 13, MUTED).next_to(ax.c2p(2.5, v), LEFT, buff=0.08) for v in (26.5, 27, 27.5, 28, 28.5)])
        xl = small('delivered size (MB, log)', 14, MUTED).next_to(ax, DOWN, buff=0.3); yl = small('test PSNR (dB)', 14, MUTED).rotate(PI / 2).next_to(ax, LEFT, buff=0.45)
        orig = DashedLine(ax.c2p(2.5, 28.263), ax.c2p(45, 28.263), color=REF, stroke_width=1.2, dash_length=0.08)
        orig_t = small('original FP32 · 28.26 dB at 742 MB', 13, REF).next_to(orig, UP, buff=0.04).align_to(ax.c2p(2.6, 0), LEFT)
        self.play(FadeIn(grid), Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl), Create(orig), FadeIn(orig_t))
        # old points + smoothed hull
        od = VGroup(*[Dot(ax.c2p(mb, ps), radius=0.045, color=OLD, fill_opacity=0.35) for mb, ps in old_pts])
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in od], lag_ratio=0.02, run_time=1.5))
        oh = smooth_hull(old_pts); ocurve = VMobject(color=OLD, stroke_width=3).set_points_smoothly([ax.c2p(*p) for p in oh])
        ol = small('old route hull: K×R dictionary + post-hoc collapse (49 configs)', 13, OLD).next_to(ax.c2p(oh[0][0], oh[0][1]), DOWN, buff=0.1).shift(RIGHT * 0.8)
        self.play(Create(ocurve, run_time=1.5), FadeIn(ol))
        o_def = Dot(ax.c2p(25.513, 27.883), radius=0.08, color=OLD); o_def_t = small('old default · 25.5 MB / 27.88 (29×)', 13, OLD).next_to(o_def, DOWN, buff=0.1)
        self.play(FadeIn(o_def, scale=0.5), FadeIn(o_def_t)); self.wait(0.8)
        # new points
        nd = VGroup(); nl = VGroup()
        for name, mb, ps in ROOM_LAMBDA:
            d = Dot(ax.c2p(mb, ps), radius=0.085, color=NEW, stroke_color=SURF, stroke_width=1.5); nd.add(d)
        for name, mb, ps in ROOM_PRUNE:
            d = Square(0.17, fill_color=PRUNE, fill_opacity=1, stroke_color=SURF, stroke_width=1.5).move_to(ax.c2p(mb, ps)); nd.add(d)
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in nd], lag_ratio=0.15, run_time=1.6))
        nh = smooth_hull(new_pts); ncurve = VMobject(color=NEW, stroke_width=3).set_points_smoothly([ax.c2p(*p) for p in nh])
        self.play(Create(ncurve, run_time=1.5))
        ann = VGroup(small('13.7 MB / 27.99: half the bytes of the old default, +0.1 dB', 13, NEW).next_to(ax.c2p(13.699, 27.985), DOWN, buff=0.14).shift(LEFT * 0.6),
                     small('5.2 MB / 27.38  (142×)', 13, PRUNE).next_to(ax.c2p(5.228, 27.382), UP, buff=0.12).shift(LEFT * 0.3),
                     small('3.0 MB / 26.51  (246×)', 13, NEW).next_to(ax.c2p(3.013, 26.509), DOWN, buff=0.12).shift(RIGHT * 0.9))
        self.play(FadeIn(ann))
        leg = VGroup(VGroup(Dot(radius=0.07, color=NEW), small('rate-controlled growth (λ sweep)', 14)).arrange(RIGHT, buff=0.1),
                     VGroup(Square(0.14, fill_color=PRUNE, fill_opacity=1, stroke_width=0), small('+ phase-B λ-prune', 14)).arrange(RIGHT, buff=0.1),
                     VGroup(Line(ORIGIN, RIGHT * 0.4, color=NEW, stroke_width=3), small('new route hull', 14)).arrange(RIGHT, buff=0.1)).arrange(RIGHT, buff=0.5).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(leg)); self.wait(2.5)
        # ---- 7 scenes relative
        self.play(*[FadeOut(m) for m in (od, ocurve, ol, o_def, o_def_t, nd, ncurve, ann, leg, grid, ax, xt, yt, xl, yl, orig, orig_t)])
        t2 = title('Seven scenes, relative to each FP32 original', 'thin lines = per-scene old-route hulls · dots = λ points · diamonds = 7-scene means')
        self.play(Transform(tt, t2))
        ax = Axes(x_range=[math.log10(4.5), math.log10(45), 1], y_range=[-1.5, 0.25, 0.5], x_length=10.5, y_length=4.6,
                  x_axis_config={'scaling': LogBase(10), 'include_ticks': False, 'include_numbers': False}, y_axis_config={'include_ticks': False, 'include_numbers': False},
                  axis_config={'stroke_color': MUTED, 'stroke_width': 1}, tips=False).shift(DOWN * 0.45 + RIGHT * 0.3)
        grid = VGroup(*[Line(ax.c2p(4.5, v), ax.c2p(45, v), color=GRID, stroke_width=0.8) for v in (-1.5, -1, -0.5)])
        xt = VGroup(*[small(f'{v}', 13, MUTED).next_to(ax.c2p(v, -1.5), DOWN, buff=0.08) for v in (5, 7, 10, 15, 20, 30, 40)])
        yt = VGroup(*[small(f'{v:+.1f}' if v else '0', 13, MUTED).next_to(ax.c2p(4.5, v), LEFT, buff=0.08) for v in (-1.5, -1, -0.5, 0)])
        xl = small('delivered size (MB, log)', 14, MUTED).next_to(ax, DOWN, buff=0.3); yl = small('PSNR vs original (dB)', 14, MUTED).rotate(PI / 2).next_to(ax, LEFT, buff=0.45)
        zero = DashedLine(ax.c2p(4.5, 0), ax.c2p(45, 0), color=REF, stroke_width=1.2, dash_length=0.08); zt = small('original FP32 (456–742 MB)', 13, REF).next_to(zero, UP, buff=0.04).align_to(ax.c2p(4.7, 0), LEFT)
        self.play(FadeIn(grid), Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl), Create(zero), FadeIn(zt))
        olds = VGroup(); news = VGroup()
        rows = list(csv.DictReader(open(os.path.join(RD, 'seven-scene-points-v2.csv'))))
        for sc in SCENES:
            p0 = ORIG[sc][1]
            pts = [(float(r['MB']), float(r['test_psnr']) - p0) for r in rows if r['scene'] == sc and r['series'] == 'collapse' and r['label'].startswith('v')]
            pts = [p for p in pts if p[0] >= 4.5 and p[1] >= -1.5]
            if len(upper_hull(pts)) >= 2:
                h = smooth_hull(pts); olds.add(VMobject(color=OLD, stroke_width=1.6, stroke_opacity=0.6).set_points_smoothly([ax.c2p(*p) for p in h]))
            (mb1, p1), (mb2, p2) = LAM[sc]
            news.add(VGroup(Line(ax.c2p(mb2, p2 - p0), ax.c2p(mb1, p1 - p0), color=NEW, stroke_width=1.6, stroke_opacity=0.8), Dot(ax.c2p(mb1, p1 - p0), radius=0.06, color=NEW), Dot(ax.c2p(mb2, p2 - p0), radius=0.06, color=NEW)))
        self.play(LaggedStart(*[Create(c) for c in olds], lag_ratio=0.1, run_time=1.6))
        self.play(LaggedStart(*[FadeIn(g) for g in news], lag_ratio=0.1, run_time=1.6))
        dia = VGroup(); dl = VGroup()
        for key, col, txt, off in (('old', OLD, 'old route default\n23.6 MB / −0.44 dB / 28×', RIGHT * 1.3 + DOWN * 0.35), ('l2', NEW, 'λ = 2e-3\n19.6 MB / −0.26 dB / 36×', UP * 0.45 + LEFT * 0.9), ('l6', NEW, 'λ = 6e-3\n8.8 MB / −0.88 dB / 80×', DOWN * 0.55 + LEFT * 0.4)):
            mb, db, x = MEANS[key]; d = Square(0.22, fill_color=col, fill_opacity=1, stroke_color=REF, stroke_width=1).rotate(PI / 4).move_to(ax.c2p(mb, db)); dia.add(d)
            dl.add(small(txt, 13, col, line_spacing=0.9).move_to(d.get_center() + off))
        self.play(LaggedStart(*[FadeIn(d, scale=0.4) for d in dia], lag_ratio=0.3, run_time=1.2)); self.play(FadeIn(dl))
        msg = small('choosing the vertices while the mesh grows dominates choosing them afterwards on every scene, and reaches sizes the old route cannot', 14, weight=SEMIBOLD).to_edge(DOWN, buff=0.28)
        self.play(FadeIn(msg)); self.wait(3)
