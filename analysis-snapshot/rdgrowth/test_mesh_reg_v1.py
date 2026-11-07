"""CPU test for mesh_reg_v1 (no GPU, no upstream): term values and gradient flow on a sphere / noisy sphere, Laplacian = 0 on a flat grid
interior (boundary excluded), jump filter + training helpers (AdjCache, apply_jump_filter, write_report) on a duck-typed TriangleModel,
degenerate-face clamp.  python test_mesh_reg_v1.py  -> prints 'MESH_REG_TEST OK' (any assertion = failure)."""
import json, math, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_reg_v1 import (sliver_loss, laplacian_loss, normal_loss, face_adjacency, jump_face_mask, mesh_report, boundary_vertices,
                         AdjCache, apply_jump_filter, write_report, VERSION)


class Fake:
    def __init__(self, x, f):
        self.vertices = torch.tensor(x, dtype=torch.float32).requires_grad_(True)
        self._triangle_indices = torch.tensor(f, dtype=torch.int32)
        self.importance_score = torch.zeros(len(f))

    def prune_triangles(self, mask):
        self._triangle_indices = self._triangle_indices[mask].to(torch.int32); self.importance_score = self.importance_score[mask]


class Acc:
    def __init__(self): self.n = None
    def reset(self, n): self.n = int(n)


class Log:
    def __init__(self): self.rows = []
    def write(self, **kw): json.dumps(kw); self.rows.append(kw)


def grid(n):
    x = [[i, j, 0.0] for i in range(n) for j in range(n)]; f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j; b = a + 1; c = a + n; d = c + 1; f += [[a, b, c], [b, d, c]]
    return x, f


def sphere(nu=24, nv=12, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = [[0, 0, 1.0]]; f = []
    for i in range(1, nv):
        th = math.pi * i / nv
        for j in range(nu):
            ph = 2 * math.pi * j / nu
            x.append([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    x.append([0, 0, -1.0]); S = len(x) - 1
    for j in range(nu):
        f.append([0, 1 + j, 1 + (j + 1) % nu])
    for i in range(nv - 2):
        for j in range(nu):
            a = 1 + i * nu + j; b = 1 + i * nu + (j + 1) % nu; c = a + nu; d = b + nu
            f += [[a, c, b], [b, c, d]]
    base = 1 + (nv - 2) * nu
    for j in range(nu):
        f.append([S, base + (j + 1) % nu, base + j])
    pos = torch.tensor(x); pos = pos + noise * torch.randn(pos.shape, generator=g)
    return pos, torch.tensor(f, dtype=torch.int32)


# 1 sphere: smooth -> small terms; noisy -> larger; gradients finite
pos, faces = sphere(); pos.requires_grad_(True)
adj = face_adjacency(faces)
s0, l0, n0 = float(sliver_loss(pos, faces)), float(laplacian_loss(pos, faces)), float(normal_loss(pos, faces, adj))
assert not boundary_vertices(faces, pos.shape[0]).any(), 'closed sphere has no boundary'
posn, _ = sphere(noise=0.05); posn.requires_grad_(True)
s1, l1, n1 = float(sliver_loss(posn, faces)), float(laplacian_loss(posn, faces)), float(normal_loss(posn, faces, adj))
assert l1 > 2 * l0 and n1 > 2 * n0, (l0, l1, n0, n1)
tot = sliver_loss(posn, faces) + laplacian_loss(posn, faces) + normal_loss(posn, faces, adj)
g = torch.autograd.grad(tot, posn)[0]
assert torch.isfinite(g).all() and float(g.abs().sum()) > 0
# 2 flat grid: interior Laplacian exactly 0 once boundary vertices are excluded; with boundary included it is > 0
x, f = grid(6); gp = torch.tensor(x); gf = torch.tensor(f, dtype=torch.int32)
assert float(laplacian_loss(gp, gf)) < 1e-10, float(laplacian_loss(gp, gf))
assert float(laplacian_loss(gp, gf, exclude_boundary=False)) > 0.05
assert int(boundary_vertices(gf, gp.shape[0]).sum()) == 20
assert float(normal_loss(gp, gf, face_adjacency(gf))) < 1e-6
# 3 jump filter + helpers on a grid with one spike face
x, f = grid(5); x.append([40.0, 0.5, 0.0]); f.append([0, 1, len(x) - 1])
m = Fake(x, f); acc = Acc(); log = Log(); cache = AdjCache(); T0 = len(f)
keep, st = jump_face_mask(m.vertices.detach(), m._triangle_indices, 20.0, 6.0)
assert st['dropped'] == 1 and not bool(keep[-1])
d = apply_jump_filter(m, acc, log, 20.0, 6.0, 'delaunay', 11001, cache)
assert d == 1 and m._triangle_indices.shape[0] == T0 - 1 and acc.n == T0 - 1 and log.rows[-1]['event'] == 'jumpfilter'
rep = write_report(m, log, 'delaunay', 11001, 20.0, cache)
assert rep['faces'] == T0 - 1 and abs(rep['normal']) < 1e-6 and rep['lap'] < 1e-10, rep
assert apply_jump_filter(m, acc, log, 20.0, 6.0, 'prune', 12000, cache) == 0          # nothing left to drop
assert apply_jump_filter(m, acc, log, 0.0, 6.0, 'final', 30000, cache) == 0            # filter off
a1 = cache.get(m); assert cache.get(m) is a1 and cache.sig == (26, 32) and a1.shape[0] == 40       # adjacency cached; 4x4-cell grid has 40 interior edges
assert int(cache.get_boundary(m).sum()) == 16 and abs(float(laplacian_loss(m.vertices, m._triangle_indices, boundary=cache.get_boundary(m)))) < 1e-10
# 4 degenerate face: finite, capped loss and finite gradient
m2 = Fake([[0, 0, 0], [1, 0, 0], [2, 0, 1e-9]], [[0, 1, 2]]); l2 = sliver_loss(m2.vertices, m2._triangle_indices); l2.backward()
assert math.isfinite(float(l2)) and float(l2) <= 8.0 + 1e-6 and torch.isfinite(m2.vertices.grad).all()
print(f'MESH_REG_TEST OK ({VERSION}) | sphere sliver {s0:.3f} lap {l0:.4f} normal {n0:.4f} | noisy lap {l1:.3f} normal {n1:.3f} | grid lap 0 | jump dropped 1 | degenerate loss {float(l2):.1f}')
