"""CPU plumbing test for rd_rebuild_v1.late_rebuild: a stand-in TriangleModel (same members as upstream), a stand-in
rdel.run that returns a random subset of the interior Delaunay faces (so some vertices become unused and the fixed-point
loop has work), duplicate grid points injected on purpose.  python test_rd_rebuild_cpu.py"""
import numpy as np, torch
from delaunay_canon_v1 import quantize_xyz, dequantize_xyz, canonical_points, delaunay_tets, interior_faces, face_rows_to_ids
from rd_rebuild_v1 import late_rebuild


class Fake:
    def __init__(self, x, f):
        self.vertices = torch.tensor(x, dtype=torch.float32).requires_grad_(True)
        self._features_dc = torch.zeros(len(x), 3).requires_grad_(True)
        self._triangle_indices = torch.tensor(f, dtype=torch.int32)
        T = len(f); self.image_size = torch.zeros(T); self.importance_score = torch.zeros(T); self.pixel_count = torch.zeros(T, dtype=torch.int)
        self.optimizer = torch.optim.Adam([{'params': [self.vertices], 'lr': 1e-3, 'name': 'vertices'}, {'params': [self._features_dc], 'lr': 1e-3, 'name': 'f_dc'}], lr=0.0)
        for p in (self.vertices, self._features_dc):
            self.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))

    def prune_triangles(self, mask):
        self._triangle_indices = self._triangle_indices[mask].to(torch.int32)
        self.image_size = self.image_size[mask]; self.importance_score = self.importance_score[mask]; self.pixel_count = self.pixel_count[mask]

    def _prune_vertex_optimizer(self, mask):
        for g in self.optimizer.param_groups:
            p = g['params'][0]; st = self.optimizer.state.pop(p)
            new = p.data[mask].clone().requires_grad_(True)
            self.optimizer.state[new] = dict(step=st['step'], exp_avg=st['exp_avg'][mask], exp_avg_sq=st['exp_avg_sq'][mask])
            g['params'][0] = new
            setattr(self, 'vertices' if g['name'] == 'vertices' else '_features_dc', new)

    def _prune_vertices(self, vertex_mask):  # copy of upstream logic
        oldV = vertex_mask.numel(); new_id = torch.full((oldV,), -1, dtype=torch.long); kept = torch.nonzero(vertex_mask, as_tuple=True)[0]
        new_id[kept] = torch.arange(kept.numel())
        if self._triangle_indices.numel() > 0:
            remapped = new_id[self._triangle_indices.long()]; valid = (remapped >= 0).all(dim=1)
            self._triangle_indices = remapped[valid].to(torch.int32); self.image_size = self.image_size[valid]
            self.importance_score = self.importance_score[valid]; self.pixel_count = self.pixel_count[valid]
        self._prune_vertex_optimizer(vertex_mask)
        V = self.vertices.shape[0]
        ref = torch.zeros(V, dtype=torch.bool)
        if self._triangle_indices.numel() > 0: ref[torch.unique(self._triangle_indices).long()] = True
        if not ref.all():
            self._prune_vertex_optimizer(ref)
            nid = torch.full((V,), -1, dtype=torch.long); k2 = torch.nonzero(ref, as_tuple=True)[0]; nid[k2] = torch.arange(k2.numel())
            self._triangle_indices = nid[self._triangle_indices.long()].to(torch.int32)


def fake_rdel(P32, faces, frac=0.08, seed=0):
    tets = delaunay_tets(P32); cand, _ = interior_faces(tets)
    rng = np.random.default_rng(seed); keep = rng.random(len(cand)) < frac
    return cand[keep]


rng = np.random.default_rng(1)
x = (rng.random((2000, 3)) * 3).astype(np.float32)
x[100:140] = x[0:40]                                     # 40 exact duplicates -> same grid cell
x[140:160] = x[40:60] + np.float32(1e-6)                 # near duplicates -> same cell at 10 bits
f = rng.integers(0, 2000, (6000, 3)); f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
tri = Fake(x, f)
rd = dict(xyz='q10', rebuild_jitter=0.02, rebuild_max_rounds=4)
st = late_rebuild(tri, rd, None, 0.0, rdel_run=fake_rdel, clean_fn=lambda: None)
print({k: v for k, v in st.items() if k != 'rounds'}); [print('  ', r) for r in st['rounds']]
assert st['rounds'][0]['duplicates_merged'] >= 60, 'duplicates not merged'
# guarantee: every face is an interior Delaunay face of the canonical points of the final vertex set
q, low, scale = tri._v4_grid
xv = tri.vertices.detach().numpy()
assert np.array_equal(dequantize_xyz(q, low, scale), xv), 'vertices are not the dequantised grid points'
q2, low2, scale2 = quantize_xyz(xv, 10); assert np.array_equal(q2, q), 'quantiser not idempotent on the saved vertices'
perm, P = canonical_points(q2, low2, scale2, 0.02); inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
cand, _ = interior_faces(delaunay_tets(P))
ids = face_rows_to_ids(inv[tri._triangle_indices.numpy().astype(np.int64)], cand)
cov = float((ids >= 0).mean()); print('coverage', cov, '| unused kept', st['unused_kept'], '| V', st['vertices'], 'F', st['faces'])
assert cov == 1.0, 'a delivered face is not a Delaunay face'
assert st['unused_kept'] == 0 or len(st['rounds']) == rd['rebuild_max_rounds'] + 1
assert all(len(tri.optimizer.state[g['params'][0]]['exp_avg']) == st['vertices'] for g in tri.optimizer.param_groups), 'optimizer state size'
print('test_rd_rebuild_cpu PASS')
