"""CPU plumbing test for rd_rebuild_v2.rebuild_event (no GPU, no effrdel): a stand-in TriangleModel (same members as
upstream) and a stand-in FixedSoupRestriction that keeps the interior Delaunay faces whose centroid lies within a radius
of a centroid of the FIXED soup (so the result depends on the soup, some vertices end unused and the round loop has
work).  Runs the three stages in the v5 order: early -> final, and snap_only on a fresh copy.  python test_rd_rebuild_cpu_v2.py"""
import numpy as np, torch
from scipy.spatial import cKDTree
from delaunay_canon_v1 import quantize_xyz, dequantize_xyz, canonical_points, delaunay_tets, interior_faces, face_rows_to_ids
from rd_rebuild_v2 import rebuild_event, _face_keys, _n_common


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


class FakeRestriction:
    """Stand-in for rd_rebuild_v2.FixedSoupRestriction: same contract, 'hits the soup' = centroid within RADIUS of a soup centroid."""
    RADIUS = 0.12

    def __init__(self, soup_verts, soup_faces):
        self.soup_verts = np.array(soup_verts, dtype=np.float32, copy=True); self.soup_faces = np.array(soup_faces, dtype=np.int64, copy=True)
        self.tree = cKDTree(self.soup_verts[self.soup_faces].mean(1)); self.calls = 0

    def __call__(self, P32):
        self.calls += 1
        assert self.tree.n == len(self.soup_faces), 'soup changed between rounds'
        cand, _ = interior_faces(delaunay_tets(P32))
        d, _ = self.tree.query(np.asarray(P32)[cand].mean(1))
        return cand[d < self.RADIUS]


def clean_faces(tri, drop_every=50):           # stand-in for the face-only importance clean: drop every k-th face
    keep = torch.ones(tri._triangle_indices.shape[0], dtype=torch.bool); keep[::drop_every] = False; tri.prune_triangles(keep)


def clean_full(tri):
    clean_faces(tri); used = torch.zeros(tri.vertices.shape[0], dtype=torch.bool); used[tri._triangle_indices.flatten().long()] = True; tri._prune_vertices(used)


def coverage(tri, bits, jitter):
    q, low, scale = tri._v4_grid; xv = tri.vertices.detach().numpy()
    assert np.array_equal(dequantize_xyz(q, low, scale), xv), 'vertices are not the dequantised grid points'
    q2, low2, scale2 = quantize_xyz(xv, bits); assert np.array_equal(q2, q), 'quantiser not idempotent on the saved vertices'
    perm, P = canonical_points(q2, low2, scale2, jitter); inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    cand, _ = interior_faces(delaunay_tets(P))
    ids = face_rows_to_ids(inv[tri._triangle_indices.numpy().astype(np.int64)], cand)
    return float((ids >= 0).mean())


def opt_ok(tri, st):
    return all(len(tri.optimizer.state[g['params'][0]]['exp_avg']) == st['vertices'] for g in tri.optimizer.param_groups)


rng = np.random.default_rng(1)
x = (rng.random((2000, 3)) * 3).astype(np.float32)
x[100:140] = x[0:40]                                     # 40 exact duplicates -> same grid cell at the final event
x[140:160] = x[40:60] + np.float32(1e-6)                 # near duplicates -> same cell at 10 bits
inside = (x[:, 0] < 2.0)                                 # the soup lives in x < 2: faces whose Delaunay dual is far from it must be dropped
f = rng.integers(0, 2000, (6000, 3)); f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
f = f[inside[f].all(1)]
rd = dict(xyz='q10', rebuild_jitter=0.02, rebuild_max_rounds=4)
tri = Fake(x, f)
x_before = tri.vertices.detach().numpy().copy()

# --- early event: faces rebuilt against the fixed cleaned soup, positions untouched (no snap), unused vertices dropped
e = rebuild_event(tri, rd, None, 0.0, 'early', 20000, restriction_factory=FakeRestriction, clean_full_fn=lambda: clean_full(tri), clean_faces_fn=lambda: clean_faces(tri))
print('early:', {k: v for k, v in e.items() if k != 'rounds'}); [print('  ', r) for r in e['rounds']]
assert e['stage'] == 'early' and e['restriction_calls'] == len(e['rounds']) and e['rounds'][0]['faces_out'] > 0
assert 0.0 <= e['rounds'][0]['soup_faces_retained'] <= 1.0 and 0.0 <= e['rounds'][0]['output_faces_new'] <= 1.0
assert not hasattr(tri, '_v4_grid'), 'early event must not set the delivery grid'
assert e['faces_removed_by_post_clean'] > 0 and opt_ok(tri, e)
xe = tri.vertices.detach().numpy(); qe, le, se = quantize_xyz(xe, 10)
assert not np.array_equal(dequantize_xyz(qe, le, se), xe), 'early event must not snap the positions'
used = np.zeros(e['vertices'], bool); used[tri._triangle_indices.numpy().reshape(-1)] = True; assert used.all(), 'early event must drop unused vertices'

# --- final event: snap + canonical restricted Delaunay against the snapped soup until every vertex is used, faces cleaned, vertices kept
st = rebuild_event(tri, rd, None, 0.0, 'final', 26000, restriction_factory=FakeRestriction, clean_full_fn=lambda: clean_full(tri), clean_faces_fn=lambda: clean_faces(tri))
print('final:', {k: v for k, v in st.items() if k != 'rounds'}); [print('  ', r) for r in st['rounds']]
assert st['restriction_calls'] == len(st['rounds']) and st['duplicates_merged'] >= 0
cov = coverage(tri, 10, 0.02)
print('coverage', cov, '| unused kept', st['unused_kept'], '| after post clean', st['unused_final_event'], '| V', st['vertices'], 'F', st['faces'])
assert cov == 1.0, 'a delivered face is not a Delaunay face of the canonical points'
assert st['unused_kept'] == 0 or len(st['rounds']) == rd['rebuild_max_rounds'] + 1
assert st['faces_removed_by_post_clean'] > 0 and st['unused_final_event'] >= st['unused_kept'], 'post clean must keep vertices'
assert opt_ok(tri, st)
# the sibling-face bookkeeping of the check script (H0/H1) is exercised by delaunay_canon_v1.selftest; here the contract only

# --- snap_only control on a fresh copy: duplicates merged, snapped, faces otherwise unchanged
tri2 = Fake(x, f)
keys0 = _face_keys(tri2._triangle_indices.numpy())
c = rebuild_event(tri2, rd, None, 0.0, 'snap_only', 26000, restriction_factory=FakeRestriction, clean_full_fn=None, clean_faces_fn=lambda: clean_faces(tri2))
print('snap_only:', {k: v for k, v in c.items() if k != 'rounds'})
assert c['duplicates_merged'] >= 60, 'duplicates not merged'
assert c['rounds'] == [] and 'restriction_calls' not in c, 'snap_only must not run the restriction'
q, low, scale = tri2._v4_grid; assert np.array_equal(dequantize_xyz(q, low, scale), tri2.vertices.detach().numpy())
assert c['faces'] <= len(keys0) and c['faces'] >= 0.9 * len(keys0), 'snap_only changed the faces beyond duplicate merging'
assert opt_ok(tri2, c)
print('test_rd_rebuild_cpu_v2 PASS')
