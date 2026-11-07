# rd-topo-v5: 方向 A 第二版 — fixed soup + 两次重建 + snap-only 对照（train_rd_v5 + rd_rebuild_v2），2026-10-26

**v4 的问题**（Room λ6.2e-3：coverage 100 % 但 test 25.61 vs v3 27.382，−1.8 dB）：(1) rd_rebuild_v1 每一轮把上一轮 Delaunay 的输出当作下一轮的 restriction soup，曲面逐轮漂移（1.34 M → 1.04 M 面，7 轮不收敛）；(2) 只有 26000 一次事件、之后 4000 步只训外观，吸收不了 ~30 % 新面。
**v5 改法**（全部新文件，不改上游 / v3 / v4）：
- `rdgrowth/rd_rebuild_v2.py`：用 effrdel 自己的零件（`lb.build_lbvh` + `upload_bvh_and_mesh` 一次、每轮 scipy Delaunay + `Grid` + `compute_cc` + `query_segments_any_hit`）拼出 **固定 soup** 的 restricted Delaunay；每轮只改 Delaunay 的点集（删未用顶点），soup 不变。三种事件：`early`（不 snap、不冻结，几何继续训）、`final`（snap 到交付网格 + canonical 点 + 固定 soup + 冻结，顶点集保持）、`snap_only`（只 snap + 冻结，不做 Delaunay）。每次事件记录 soup 面保留率 / 新面占比。
- `rdgrowth/train_rd_v5.py`：= v3 + `--rebuild_iters 20000 26000`（第一次 early，最后一次 final）+ `--snap_only` 对照 + 重建后一次只删面的 importance cleaning（`--no_post_clean` 关闭）。
- `rdgrowth/test_rd_rebuild_cpu_v2.py`：CPU 测试（假 restriction，无 GPU），sbatch 开头也会在训练环境里跑一遍。
- `rd-topo-v5.sbatch`：smoke（1300 步，1240 early / 1250 final，check 必须 100 %）→ MODES="v5 snap" 两跑 Room λ6.2e-3：训练 → checkpoint check → level-1.0 protocol → delivered check → 汇总行。
**GPU 成本**：smoke 3 min + 每跑约 28 min（训练 ~18 min + protocol 10 min）≈ 1 GPU-h；walltime 2.5 h，48 GB，general。
**判据**（delivered 行，对 v3 的 27.382 @ 5.228 MB）：test ≥ 27.28（损失 ≤ 0.1 dB）→ 很可行；26.98–27.28 → 把 (est MB, test) 点放到 v3 RD 曲线上，需高出曲线 ≥ 0.15 dB（曲线在 3.5 MB 附近约 26.7–26.8）；< 26.98（损失 > 0.4 dB）→ 放弃。snap 行 = 单纯量化+冻结的代价，用来拆分 v5 的损失来自冻结还是拓扑。

## §1 安装（任意 shell，整块粘贴；不会退出你的终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/rd_rebuild_v2.py <<'PYEOF_RB2'
"""rd_rebuild_v2.py (2026-10-26): the v5 rebuild events used by train_rd_v5.py (v4's rd_rebuild_v1.py is kept unchanged).

What v4 got wrong (Room probe: coverage 100 % but -1.8 dB): every round after the first used the PREVIOUS round's Delaunay
output as the restriction soup, so the surface drifted (1.34 M -> 1.04 M faces over 7 rounds, never converged), and the
single event at 26000 left only 4000 appearance-only steps to absorb a topology that is ~30 % new faces.
v5: the restriction target is built ONCE per event from the cleaned mesh (effrdel's own pieces: LBVH on that soup,
uploaded once); a round only changes the Delaunay vertex set (unused vertices removed) and queries the SAME soup.
Events (train_rd_v5.py --rebuild_iters 20000 26000):
  early     clean (faces + vertices) -> restricted Delaunay of the float positions against the cleaned mesh ->
            face-only clean -> unused vertices dropped; XYZ keeps training (nothing is frozen)
  final     clean -> snap XYZ to the delivery grid (--xyz qN, stage2 quantiser), merge duplicate grid points ->
            canonical jittered points (delaunay_canon_v1) -> restricted Delaunay against the snapped cleaned mesh until
            every vertex is used -> face-only clean (vertices kept) -> XYZ frozen by the caller (vertex lr = 0)
  snap_only clean -> snap -> freeze, faces unchanged: the control that isolates the cost of quantise + freeze
The stats of each event (rebuild-<iter>.json) include the fraction of soup faces retained by the restriction and the
fraction of output faces that are new.  CPU unit test: python test_rd_rebuild_cpu_v2.py (fake restriction, no GPU)."""
import time
import numpy as np
import torch
from scipy.spatial import Delaunay
from delaunay_canon_v1 import quantize_xyz, dequantize_xyz, canonical_points, VERSION as CANON_VERSION

VERSION = 'rd_rebuild_v2 2026-10-26'
STAGES = ('early', 'final', 'snap_only')


def _reset_face_trackers(triangles):
    T = triangles._triangle_indices.shape[0]
    dev = triangles.vertices.device
    triangles.image_size = torch.zeros((T,), dtype=torch.float, device=dev)
    triangles.importance_score = torch.zeros((T,), dtype=torch.float, device=dev)
    triangles.pixel_count = torch.zeros((T,), dtype=torch.int, device=dev)


def _set_faces(triangles, faces_np):
    triangles._triangle_indices = torch.as_tensor(np.asarray(faces_np, dtype=np.int64), device=triangles.vertices.device).contiguous().to(torch.int32)
    _reset_face_trackers(triangles)


def _vertices_np(triangles):
    return triangles.vertices.detach().cpu().numpy().astype(np.float32)


def _faces_np(triangles):
    return triangles._triangle_indices.long().cpu().numpy()


def _face_keys(f):
    """Orientation-free unique face rows (sorted vertex triples)."""
    f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
    return np.unique(np.sort(f, axis=1), axis=0) if len(f) else f


def _n_common(keys_a, keys_b):
    if len(keys_a) == 0 or len(keys_b) == 0:
        return 0
    _, cnt = np.unique(np.concatenate([keys_a, keys_b]), axis=0, return_counts=True)
    return int((cnt == 2).sum())


def _used_mask(triangles):
    V = int(triangles.vertices.shape[0])
    used = np.zeros(V, bool)
    f = _faces_np(triangles)
    if len(f):
        used[f.reshape(-1)] = True
    return used


class FixedSoupRestriction:
    """Restricted Delaunay against a FIXED triangle soup, assembled from effrdel's own pieces (rdel.run is exactly this
    with soup == Delaunay input).  __call__(P32) returns the interior Delaunay faces of P32 whose dual segment (between the
    circumcentres of the two incident tets) hits the soup, as (N, 3) int64 rows in P32's index space."""

    def __init__(self, soup_verts, soup_faces):
        import rdel  # effrdel; its namespace carries lb, Grid, compute_cc, ti_init, upload_bvh_and_mesh, query_segments_any_hit
        self.rdel = rdel
        v = np.ascontiguousarray(np.asarray(soup_verts, dtype=np.float32))
        f = np.ascontiguousarray(np.asarray(soup_faces, dtype=np.int32))
        rdel.ti_init()
        bvh = rdel.lb.build_lbvh(v, f)
        rdel.upload_bvh_and_mesh(bvh, v, f)
        self.n_soup_faces = int(len(f))
        self.calls = 0

    def __call__(self, P32):
        rdel = self.rdel
        self.calls += 1
        P = np.ascontiguousarray(np.asarray(P32, dtype=np.float32))
        tets_np = Delaunay(P).simplices.astype(np.int64)
        verts = torch.from_numpy(P).to('cuda')
        tets = torch.from_numpy(tets_np).to('cuda')
        grid = rdel.Grid(verts, tets)
        valid = grid.face_tet[:, 1] >= 0
        face_tet = grid.face_tet[valid].clone()
        faces = grid.faces[valid].clone()
        del grid
        torch.cuda.empty_cache()
        cc = rdel.compute_cc(verts, tets)
        torch.cuda.empty_cache()
        hit = rdel.query_segments_any_hit(cc[face_tet[:, 0]], cc[face_tet[:, 1]])
        out = faces[hit.to(faces.device)].cpu().numpy().astype(np.int64).reshape(-1, 3)
        del cc, faces, face_tet, verts, tets
        torch.cuda.empty_cache()
        return out


def snap_to_grid(triangles, bits):
    """Snap XYZ to the delivery grid, merging vertices that share a grid cell (faces remapped, degenerate / duplicate faces
    dropped).  Returns (q, low, scale, n_duplicates) with the positions set to dequantize(q) and the quantiser verified
    idempotent on them (exactly what the decoder reconstructs)."""
    dev = triangles.vertices.device
    x = _vertices_np(triangles)
    q, low, scale = quantize_xyz(x, bits)
    uq, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    n_dup = int(len(q) - len(uq))
    if n_dup:
        rep = first[inv.reshape(-1)]
        f = rep[_faces_np(triangles)]
        degenerate = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
        f = f[~degenerate]
        f = np.unique(np.sort(f, axis=1), axis=0) if len(f) else f
        _set_faces(triangles, f)
        keep = np.zeros(len(q), bool)
        keep[first] = True
        triangles._prune_vertices(torch.from_numpy(keep).to(dev))  # also drops now-unused vertices
        x = _vertices_np(triangles)
        q, low, scale = quantize_xyz(x, bits)
    xq = dequantize_xyz(q, low, scale)
    q2, low2, scale2 = quantize_xyz(xq, bits)
    if not np.array_equal(q2, q):
        raise RuntimeError(f'delivery quantiser is not idempotent at {bits} bits ({int((q2 != q).any(1).sum())} vertices move)')
    triangles.vertices.data = torch.from_numpy(xq).to(dev)
    return q2, low2, scale2, n_dup


def assert_on_grid(triangles, bits):
    """Positions must be exactly the dequantised grid points; returns the grid tuple the decoder will reconstruct."""
    x = _vertices_np(triangles)
    q, low, scale = quantize_xyz(x, bits)
    xq = dequantize_xyz(q, low, scale)
    drift = float(np.abs(xq - x).max()) if len(x) else 0.0
    if drift != 0.0:
        raise RuntimeError(f'vertices are not on the q{bits} grid (max drift {drift})')
    return q, low, scale


def rebuild_event(triangles, rd, log, t0, stage, iteration, restriction_factory=None, clean_full_fn=None, clean_faces_fn=None, post_clean=True):
    """One v5 event (stage in STAGES).  rd keys: xyz (qN), rebuild_jitter, rebuild_max_rounds.  Returns the stats dict
    (also written to log).  After 'final' / 'snap_only' the caller freezes XYZ and triangles._v4_grid holds (q, low, scale)."""
    if stage not in STAGES:
        raise ValueError(stage)
    if restriction_factory is None:
        restriction_factory = FixedSoupRestriction
    bits = int(str(rd['xyz'])[1:])
    jitter = float(rd['rebuild_jitter'])
    max_rounds = int(rd['rebuild_max_rounds'])
    dev = triangles.vertices.device
    V0, T0 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    if clean_full_fn is not None:
        clean_full_fn()
    V1, T1 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    stats = dict(event='rebuild', stage=stage, iteration=int(iteration), version='v5', module=VERSION, canon=CANON_VERSION, xyz=f'q{bits}', jitter=jitter,
                 vertices_before=V0, faces_before=T0, vertices_after_clean=V1, faces_after_clean=T1, duplicates_merged=0, rounds=[])
    if stage in ('final', 'snap_only'):
        _, _, _, n_dup = snap_to_grid(triangles, bits)
        stats['duplicates_merged'] = n_dup
        stats['vertices_after_snap'], stats['faces_after_snap'] = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    if stage != 'snap_only':
        soup_v, soup_f = _vertices_np(triangles), _faces_np(triangles)
        soup_keys = _face_keys(soup_f)
        restrict = restriction_factory(soup_v, soup_f)        # built ONCE: the same soup for every round
        stats['soup_vertices'], stats['soup_faces'] = int(len(soup_v)), int(len(soup_f))
        for r in range(max_rounds + 1):
            if stage == 'final':
                x = _vertices_np(triangles)
                q, low, scale = quantize_xyz(x, bits)
                xq = dequantize_xyz(q, low, scale)
                q2, low2, scale2 = quantize_xyz(xq, bits)
                if not np.array_equal(q2, q):
                    raise RuntimeError('delivery quantiser is not idempotent after vertex removal')
                triangles.vertices.data = torch.from_numpy(xq).to(dev)   # re-snapped (low/scale may change when extremes are removed)
                perm, P32 = canonical_points(q2, low2, scale2, jitter)
                new_faces = perm[restrict(P32)]
            else:
                new_faces = restrict(_vertices_np(triangles))
            faces_in = int(triangles._triangle_indices.shape[0])
            _set_faces(triangles, new_faces)
            used = _used_mask(triangles)
            n_unused = int((~used).sum())
            row = dict(round=r, vertices=int(len(used)), faces_in=faces_in, faces_out=int(len(new_faces)), unused=n_unused)
            if r == 0:
                new_keys = _face_keys(new_faces)
                common = _n_common(soup_keys, new_keys)
                row['soup_faces_retained'] = round(common / max(1, len(soup_keys)), 4)
                row['output_faces_new'] = round(1.0 - common / max(1, len(new_keys)), 4)
            stats['rounds'].append(row)
            extra = f" | soup retained {row['soup_faces_retained']:.3f}, new faces {row['output_faces_new']:.3f}" if r == 0 else ''
            print(f'\n[RD rebuild {stage} round {r}] V {len(used)} faces {faces_in} -> {len(new_faces)} | unused vertices {n_unused}{extra}')
            if n_unused == 0 or r == max_rounds:
                break
            triangles._prune_vertices(torch.from_numpy(used).to(dev))
        stats['restriction_calls'] = int(getattr(restrict, 'calls', len(stats['rounds'])))
        stats['unused_kept'] = stats['rounds'][-1]['unused']
        if post_clean and clean_faces_fn is not None:
            Tb = int(triangles._triangle_indices.shape[0])
            clean_faces_fn()
            stats['faces_removed_by_post_clean'] = Tb - int(triangles._triangle_indices.shape[0])
            print(f"[RD rebuild {stage}] post clean removed {stats['faces_removed_by_post_clean']} faces")
        if stage == 'early':
            used = _used_mask(triangles)
            if not used.all():
                triangles._prune_vertices(torch.from_numpy(used).to(dev))
    if stage in ('final', 'snap_only'):
        q, low, scale = assert_on_grid(triangles, bits)
        triangles._v4_grid = (q.copy(), np.asarray(low).copy(), np.asarray(scale).copy())
        stats['unused_final_event'] = int((~_used_mask(triangles)).sum())
    stats['vertices'], stats['faces'] = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    stats['elapsed_s'] = round(time.time() - t0)
    if log is not None:
        log.write(**stats)
    return stats
PYEOF_RB2
cat > $AN/rdgrowth/test_rd_rebuild_cpu_v2.py <<'PYEOF_TEST2'
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
PYEOF_TEST2
cat > $AN/rdgrowth/train_rd_v5.py <<'PYEOF_V5'
"""train_rd_v5.py (v5, 2026-10-26 = v3 + decoder-reproducible topology, second design; v4 kept as train_rd_v4.py).

v5 = v3 + TWO rebuild events (--rebuild_iters 20000 26000, both after the last prune round; rd_rebuild_v2.py):
  early (20000): upstream importance cleaning, then restricted Delaunay of the float positions against the cleaned mesh
     (the soup is FIXED for all rounds; v4 fed each round the previous round's output and drifted), face-only cleaning,
     unused vertices dropped.  XYZ keeps training for the 6000 steps to the final event, so the geometry adapts to the
     Delaunay topology (upstream trains 19000 steps after its own rebuild; v4 gave appearance-only 4000).
  final (26000): cleaning, XYZ snapped to the delivery grid (--xyz qN = stage2 quantiser), duplicate grid points merged,
     restricted Delaunay on the CANONICAL jittered grid points (delaunay_canon_v1) against the snapped cleaned mesh until
     every vertex is used, face-only cleaning (vertex set kept = the decoder's Delaunay input), then XYZ frozen
     (vertex lr = 0); appearance (DC, SH, weight) keeps training.  Final cleaning prunes faces only; the save asserts the
     positions are still on the grid.
  --snap_only: control = cleaning + snap + freeze at the last of --rebuild_iters, no Delaunay (faces unchanged), isolating
     the cost of quantise + freeze from the cost of the topology change.
Each event writes rebuild-<iter>.json (the final one also rebuild.json) with the retained / new face fractions; the
delivered connectivity can then be regenerated by the decoder from the quantised vertices plus one selection bit per
candidate Delaunay face (delaunay_check_v1.py measures coverage and the label entropy).
--rebuild_iters with no values disables both events (then v5 == v3).

v3 (2026-10-19 = v2m + phase-B rate-controlled edge-collapse pruning; v2m kept as train_rd_v2m.py): MeshSplatting training with rate-distortion controlled growth AND pruning.

v3 adds, at each iteration in --prune_iters (default 12000 14000 16000 18000: after the restricted Delaunay rebuild, before
supersampling), one round of rd_prune.select_collapses / collapse_edges: collapse edge (u,v) iff the residual mass of the
faces touching u or v, per byte the collapse saves (one vertex + the 1/2/3+ faces that disappear), is below lam * prune_mu.
Same lam and same per-view per-megapixel residual units as the split rule; the residual keeps being attributed through
rend_ids in phase B.  At most prune_max_frac of the vertices go per round (pruning twin of the growth limiter).
--prune_iters with no values disables pruning (then v3 == v2m).

v2m adds ONE option, --split_schedule <rd_log.jsonl of a reference run>: at every densification interval the number of
triangles chosen by the selection rule is capped at the number the reference run chose at the same iteration (its
"n_split" field).  With --dd_mode importance this gives the upstream importance-sampled selection on exactly the
growth schedule of a lambda run, so the two runs differ only in WHICH triangles split (ranking), not in how many or
when (the stop).  Without --split_schedule v2m behaves exactly like v2.

A copy of upstream `train.training` with ONE change: the densification call `add_new_gs(cap)` is replaced by the
Lagrangian split rule of rd_growth.py (dD_t > lam * dR_t).  Everything else (losses, pruning, opacity schedule,
restricted Delaunay at densify_until+1000, final importance cleaning) is upstream code, imported, not edited.

Run from anywhere:
  python train_rd.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 -s <scene> -i images_2 -r 1 --eval \
      --data_device cpu -m <out>/original --iterations 30000 --lam 1e-3 --bytes_per_vertex 10.0 --dd_mode error
Outputs the usual <out>/original/point_cloud/iteration_30000/point_cloud_state_dict.pt plus <out>/original/rd_log.jsonl
(one line per densification interval: vertices, triangles, candidates passing, splits, ratio quantiles).
"""
import argparse
import json
import os
import sys
import time
from random import randint

import torch
import torch.nn.functional as F

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--lab', required=True, help='root containing upstream/ (train.py)')
_lab, _rest = _pre.parse_known_args()
UP = os.path.join(os.path.abspath(_lab.lab), 'upstream')
if not os.path.isdir(UP):
    sys.exit(f'upstream dir not found: {UP}')
sys.path.insert(0, UP)
os.chdir(UP)  # upstream resolves relative paths (output/, submodules) from its own dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train as up  # noqa: E402  (upstream train.py; its __main__ block is guarded)
from arguments import ModelParams, PipelineParams, OptimizationParams, update_indoor  # noqa: E402
from scene import Scene, TriangleModel  # noqa: E402
from triangle_renderer import render  # noqa: E402
from utils.loss_utils import l1_loss, ssim, vertex_depth_loss_hr  # noqa: E402
from utils.general_utils import safe_state, get_expon_lr_func  # noqa: E402
from tqdm import tqdm  # noqa: E402
import lpips  # noqa: E402
from rd_growth_v2 import RDAccumulator, RDLog, rd_select, rd_split  # noqa: E402
from rd_prune import select_collapses, collapse_edges  # noqa: E402
import numpy as np  # noqa: E402

from delaunay_canon_v1 import dequantize_xyz, VERSION as CANON_VERSION  # noqa: E402
from rd_rebuild_v2 import rebuild_event, VERSION as REBUILD_VERSION  # noqa: E402


def load_split_schedule(path):
    """iteration -> n_split (triangles chosen by the rule, before the largest-area top-k) of a reference rd_log.jsonl."""
    sched = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get('event') == 'densify':
                sched[int(e['iteration'])] = int(e['n_split'])
    if not sched:
        raise RuntimeError(f'no densify events in split schedule {path}')
    return sched


def importance_clean(triangles, scene, pipe, bg):
    """Upstream final cleaning: keep faces whose max blending weight over the training views exceeds 0.5, drop unused vertices."""
    stack = scene.getTrainCameras().copy()
    triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device='cuda')
    while stack:
        cam = stack.pop(0)
        imp = render(cam, triangles, pipe, bg)['max_blending'].detach()
        m = imp > triangles.importance_score
        triangles.importance_score[m] = imp[m]
    keep = (triangles.importance_score > 0.5).squeeze()
    triangles.prune_triangles(keep)
    used = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=triangles.vertices.device)
    if triangles._triangle_indices.numel() > 0:
        used[triangles._triangle_indices.flatten().long()] = True
    triangles._prune_vertices(used)


def importance_clean_faces(triangles, scene, pipe, bg):
    """Face-only variant: faces with max blending weight <= 0.5 removed, vertex set untouched (the Delaunay input)."""
    stack = scene.getTrainCameras().copy()
    triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device='cuda')
    while stack:
        cam = stack.pop(0)
        imp = render(cam, triangles, pipe, bg)['max_blending'].detach()
        m = imp > triangles.importance_score
        triangles.importance_score[m] = imp[m]
    keep = (triangles.importance_score > 0.5).squeeze()
    triangles.prune_triangles(keep)


def training(dataset, opt, pipe, testing_iterations, checkpoint, debug_from, scene_name, rd):
    tb_writer = up.prepare_output_and_logger(dataset)
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage
    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_training_views = len(viewpoint_stack)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc='Training progress (RD growth)')
    first_iter += 1

    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until
    init_opacity = 0.1
    final_opacity = .9999
    total_iters_opacity = opt.final_opacity_iter
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space
    need_delaunay = False
    run_restricted_delaunay = opt.densify_until_iter + 1000
    depth_l1_weight = get_expon_lr_func(opt.depth_lambda_init, opt.depth_lambda_final, max_steps=opt.iterations)

    # --- RD growth state ---------------------------------------------------------------------------------------
    acc = RDAccumulator(triangles._triangle_indices.shape[0], mode=rd['dd_mode'], norm=rd['dd_norm'])
    log = RDLog(os.path.join(dataset.model_path, 'rd_log.jsonl'))
    bytes_per_split = 3.0 * rd['bytes_per_vertex']  # three edge midpoints per split (upper bound; shared midpoints cost less)
    sched = load_split_schedule(rd['split_schedule']) if rd['split_schedule'] else None
    prune_set = sorted(set(int(i) for i in rd['prune_iters']))
    if prune_set:
        if rd['dd_mode'] != 'error':
            raise SystemExit('pruning needs --dd_mode error (residual attribution); got ' + rd['dd_mode'])
        bad = [i for i in prune_set if not (run_restricted_delaunay + 1 < i < start_upsampling)]
        if bad and not rd['allow_any_prune_iter']:
            raise SystemExit(f'prune_iters {bad} must lie after the Delaunay rebuild ({run_restricted_delaunay}) and before supersampling ({start_upsampling}); --allow_any_prune_iter overrides (smoke only)')
    last_prune = max(prune_set) if prune_set else -1
    rebuild_set = sorted(set(int(i) for i in rd['rebuild_iters']))
    final_iter = max(rebuild_set) if rebuild_set else 0
    snap_only = bool(rd['snap_only'])
    frozen = False
    if rebuild_set:
        bad = [i for i in rebuild_set if not (i > max(last_prune, run_restricted_delaunay) and i <= opt.iterations)]
        if bad and not rd['allow_any_rebuild_iter']:
            raise SystemExit(f'rebuild_iters {bad} must lie after the last prune round ({last_prune}) / Delaunay ({run_restricted_delaunay}) and <= iterations; --allow_any_rebuild_iter overrides (smoke only)')
        print('[RD] v5 rebuild events at', rebuild_set, '(final =', final_iter, ')', '| snap_only' if snap_only else '| fixed-soup restricted Delaunay', '| grid', rd['xyz'],
              '| jitter', rd['rebuild_jitter'], '| max rounds', rd['rebuild_max_rounds'], '| post clean', not rd['no_post_clean'], '|', REBUILD_VERSION, '|', CANON_VERSION)
    log.write(event='config', version='v5', rebuild_iters=rebuild_set, snap_only=snap_only, post_clean=(not rd['no_post_clean']), xyz=rd['xyz'], rebuild_jitter=rd['rebuild_jitter'], rebuild_max_rounds=rd['rebuild_max_rounds'], lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'],
              bytes_per_split=bytes_per_split, max_points=opt.max_points, add_percentage=opt.add_percentage,
              densify_from=opt.densify_from_iter, densify_until=opt.densify_until_iter, interval=opt.densification_interval,
              split_schedule=rd['split_schedule'], schedule_total=(sum(sched.values()) if sched else None),
              prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'])
    if prune_set:
        print('[RD] phase-B lambda-prune at', prune_set, '| mu', rd['prune_mu'], 'max_frac/round', rd['prune_max_frac'], 'B/face', rd['bytes_per_face'])
    if sched:
        print('[RD] matched-count control: schedule from', rd['split_schedule'], '|', len(sched), 'intervals, total chosen', sum(sched.values()))
    t0 = time.time()

    for iteration in range(first_iter, opt.iterations + 1):
        if need_delaunay:
            with torch.no_grad():
                triangles.run_restricted_delaunay()
            need_delaunay = False
            log.write(event='delaunay', iteration=iteration, vertices=int(triangles.vertices.shape[0]),
                      triangles=int(triangles._triangle_indices.shape[0]), elapsed_s=round(time.time() - t0))
            acc.reset(triangles._triangle_indices.shape[0])  # new face set after the rebuild
        if iteration in rebuild_set and not frozen:
            is_final = iteration == final_iter
            stage = ('snap_only' if snap_only else 'final') if is_final else 'early'
            if snap_only and not is_final:
                print(f'[RD rebuild] snap_only control: early event at {iteration} skipped')
            else:
                with torch.no_grad():
                    rb = rebuild_event(triangles, rd, log, t0, stage, iteration,
                                       clean_full_fn=lambda: importance_clean(triangles, scene, pipe, background),
                                       clean_faces_fn=lambda: importance_clean_faces(triangles, scene, pipe, background),
                                       post_clean=not rd['no_post_clean'])
                acc.reset(triangles._triangle_indices.shape[0])
                with open(os.path.join(dataset.model_path, f'rebuild-{iteration}.json'), 'w') as fh:
                    json.dump(rb, fh, indent=1)
                if is_final:
                    with open(os.path.join(dataset.model_path, 'rebuild.json'), 'w') as fh:
                        json.dump(rb, fh, indent=1)
                r0 = rb['rounds'][0] if rb['rounds'] else {}
                print(f"[RD rebuild done] {stage} @ {iteration}: V {rb['vertices_before']} -> {rb['vertices']} faces {rb['faces_before']} -> {rb['faces']} "
                      f"| rounds {len(rb['rounds'])} | dup {rb['duplicates_merged']} | soup retained {r0.get('soup_faces_retained')} new faces {r0.get('output_faces_new')} "
                      f"| unused kept {rb.get('unused_kept', rb.get('unused_final_event'))} | post clean -{rb.get('faces_removed_by_post_clean', 0)} faces"
                      + (' | XYZ frozen from here' if is_final else ' | XYZ keeps training'))
            if is_final:
                frozen = True
        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)
        if frozen:
            for g in triangles.optimizer.param_groups:
                if g['name'] == 'vertices':
                    g['lr'] = 0.0
        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = min((iteration - sigma_start) / (total_iters - sigma_start), 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device='cuda') if opt.random_background else background

        if not viewpoint_stack or len(scene.getTrainCameras()) + iteration == opt.iterations:
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.getTrainCameras()) + iteration == opt.iterations:
                triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device='cuda')
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg['render']
        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(viewpoint_cam, 'normal_map', None) is not None:
            gt_normal = viewpoint_cam.normal_map.cuda()
            gt_normal = F.interpolate(gt_normal.unsqueeze(0), size=(gt_image.shape[1], gt_image.shape[2]), mode='area').squeeze(0)
        else:
            gt_normal = None

        pixel_loss = l1_loss(image, gt_image)

        image_size = render_pkg['scaling'].detach()
        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        importance_score = render_pkg['max_blending'].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]
        pixel_count = render_pkg['triangle_was_rendered'].detach()
        mask = pixel_count > triangles.pixel_count
        triangles.pixel_count[mask] = pixel_count[mask]

        if up.FUSED_SSIM_AVAILABLE:
            ssim_value = up.fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)

        lambda_weight = opt.lambda_weight if iteration < opt.start_opacity_floor else 0
        if lambda_weight > 0:
            mask_out = triangles.vertices.shape[0]
            vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            loss += lambda_weight * vertex_weights.mean()

        lambda_vertex = opt.lambda_vertex if iteration > opt.start_vertex_opt else 0
        if lambda_vertex > 0:
            loss += lambda_vertex * vertex_depth_loss_hr(render_pkg['vertex_depth_out'], render_pkg['image_2D'], render_pkg['vertex_rendered'],
                                                         render_pkg['surf_depth'], max_diff_threshold=opt.max_diff_threshold)

        if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, 'invdepthmap', None) is not None:
            invDepth = 1.0 / (render_pkg['expected_depth'] + 1e-6)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            loss += depth_l1_weight(iteration) * torch.abs((invDepth - mono_invdepth) * depth_mask).mean()

        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        if lambda_normal > 0:
            loss += lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0))[None].mean()
        if gt_normal is not None:
            lambda_normals_super = opt.lambda_normals_super if iteration > opt.iteration_mesh else 0
            loss += lambda_normals_super * (1 - (rend_normal * gt_normal).sum(dim=0))[None].mean()

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # RD: attribute this view's residual to triangles (phase A for growth; phase B up to the last prune round)
            if iteration < run_restricted_delaunay or iteration <= last_prune:
                acc.accumulate(render_pkg, image, gt_image, triangles._triangle_indices,
                               vertex_grad=triangles.vertices.grad if rd['dd_mode'] == 'grad' else None)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({'Loss': f'{ema_loss_for_log:.5f}', 'V': int(triangles.vertices.shape[0])})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            up.training_report(tb_writer, scene_name, iteration, pixel_loss, loss, l1_loss, iter_start.elapsed_time(iter_end),
                               testing_iterations, scene, render, (pipe, background))

            if iteration % 500 == 0 and iteration < run_restricted_delaunay:
                triangle_vertex_weights = triangles.opacity_activation(triangles.vertex_weight[triangles._triangle_indices])
                min_weights = triangle_vertex_weights.min(dim=1).values
                mask_opacity = (min_weights <= prune_triangles).squeeze()
                mask_importance = (triangles.importance_score <= prune_triangles).squeeze()
                mask_size = (triangles.image_size > prune_size).squeeze()
                delete_mask = mask_opacity | mask_size
                if number_of_training_views < 500:
                    delete_mask = delete_mask | mask_importance
                keep_mask = ~delete_mask
                if iteration > opt.start_pruning:
                    triangles.prune_triangles(keep_mask)
                    acc.apply_mask(keep_mask)

                device = triangles.vertices.device
                used_vertex_mask = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=device)
                if triangles._triangle_indices.numel() > 0:
                    used_vertex_mask[triangles._triangle_indices.flatten()] = True
                weight_mask = (triangles.get_vertex_weight.squeeze() >= prune_triangles)
                mask_out = triangles.vertices.shape[0]
                vertex_mask = weight_mask[:mask_out] | used_vertex_mask
                valid_tris = vertex_mask[triangles._triangle_indices.long()].all(dim=1) if triangles._triangle_indices.numel() > 0 else None
                triangles._prune_vertices(vertex_mask)
                if valid_tris is not None:
                    acc.apply_mask(valid_tris)

                needs_densification = (iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0
                                       and iteration > opt.densify_from_iter)
                if needs_densification:
                    V = int(triangles.vertices.shape[0])
                    max_split = max(0, min(opt.max_points, int(opt.add_percentage * V)) - V)  # upstream growth limiter
                    k_ref = None
                    if sched is not None:  # matched-count control: never choose more than the reference run did here
                        k_ref = sched.get(iteration, 0)
                        max_split = min(max_split, k_ref)
                    idx, st = rd_select(acc, triangles, rd['lam'], bytes_per_split, max_split,
                                        triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    n_split = rd_split(triangles, idx, iteration, splitt_large_triangles)
                    st.update(event='densify', iteration=iteration, vertices_before=V, vertices_after=int(triangles.vertices.shape[0]),
                              triangles_after=int(triangles._triangle_indices.shape[0]), n_split_total=n_split, k_ref=k_ref, elapsed_s=round(time.time() - t0))
                    log.write(**st)
                    print(f"\n[RD {iteration}] V {V}->{st['vertices_after']} pass {st['n_pass']}/{st['n_tri']} split {n_split} "
                          f"(limit {max_split}) ratio q50/q90/q99 {st['ratio_q50']:.2e}/{st['ratio_q90']:.2e}/{st['ratio_q99']:.2e} "
                          f"lam {rd['lam']:.2e} valid {st['valid_frac']} k_ref {k_ref}")
                acc.reset(triangles._triangle_indices.shape[0])

                if iteration > opt.start_opacity_floor:
                    a = min(1.0, max(0.0, (iteration - opt.start_opacity_floor) / max(1, total_iters_opacity - opt.start_opacity_floor)))
                    triangles.update_min_weight(min(init_opacity + (final_opacity - init_opacity) * a, final_opacity))
                    prune_triangles += 0.01
            elif iteration == run_restricted_delaunay:
                need_delaunay = True
            elif iteration % 500 == 0 and iteration > run_restricted_delaunay + 1000:
                if iteration > opt.start_opacity_floor:
                    a = min(1.0, max(0.0, (iteration - opt.start_opacity_floor) / max(1, total_iters_opacity - opt.start_opacity_floor)))
                    triangles.update_min_weight(min(init_opacity + (final_opacity - init_opacity) * a, final_opacity))
                    prune_triangles += 0.01

            if iteration in prune_set:
                V0, T0 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
                dD = acc.distortion()
                if dD.numel() != T0:
                    raise RuntimeError(f'accumulator/face mismatch at prune {iteration}: {dD.numel()} vs {T0}')
                n_views = acc.views
                pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, rd['lam'], rd['prune_mu'],
                                             rd['bytes_per_vertex'], rd['bytes_per_face'], rd['prune_max_frac'])
                cs = collapse_edges(triangles, pairs)
                acc.reset(triangles._triangle_indices.shape[0])
                ev = dict(event='prune', iteration=iteration, views=n_views, elapsed_s=round(time.time() - t0))
                ev.update(ps)
                ev.update(cs)
                log.write(**ev)
                print(f"\n[RD prune {iteration}] V {V0}->{cs['vertices_after']} T {T0}->{cs['faces_after']} | cand {ps['n_cand']}/{ps['edges']} edges "
                      f"({ps['cand_frac']:.3f}) pairs {ps['n_pairs']} (cap {ps['max_pairs']}, flip-rejected {ps['n_flip_rejected']}, passes {ps['passes']}) "
                      f"| score q10/q50/q90 {ps['score_q10']:.2e}/{ps['score_q50']:.2e}/{ps['score_q90']:.2e} thr {ps['thr']:.2e} "
                      f"| faces-removed share sel {ps['faces_removed_share_sel']} all {ps['faces_removed_share_all']} | views {n_views}")
            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)

    # final cleaning, verbatim upstream
    viewpoint_stack = scene.getTrainCameras().copy()
    triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device='cuda')
    while viewpoint_stack:
        viewpoint_cam = viewpoint_stack.pop(0)
        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        importance_score = render_pkg['max_blending'].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]
    mask_importance = (triangles.importance_score <= 0.5).squeeze()
    triangles.prune_triangles(~mask_importance)
    used_vertex_mask = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=triangles.vertices.device)
    if triangles._triangle_indices.numel() > 0:
        used_vertex_mask[triangles._triangle_indices.flatten()] = True
    unused_final = int((~used_vertex_mask).sum())
    if frozen:
        q, low, scale = triangles._v4_grid
        xq = dequantize_xyz(q, low, scale)
        drift = float(np.abs(triangles.vertices.detach().cpu().numpy().astype(np.float32) - xq).max()) if len(xq) == triangles.vertices.shape[0] else float('inf')
        if drift != 0.0:
            raise RuntimeError(f'vertices moved after the rebuild (max drift {drift}); the Delaunay guarantee is broken')
        print(f'[RD v5] final cleaning: faces pruned, {unused_final} unused vertices KEPT (Delaunay vertex set unchanged), drift 0')
    else:
        triangles._prune_vertices(used_vertex_mask)
    scene.save(iteration)
    log.write(event='final', vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0]), unused_vertices=unused_final, rebuilt=frozen, rebuild_iters=rebuild_set, snap_only=snap_only,
              elapsed_s=round(time.time() - t0))
    with open(os.path.join(dataset.model_path, 'rd_summary.json'), 'w') as f:
        json.dump(dict(version='v5', rebuild_iters=rebuild_set, snap_only=snap_only, post_clean=(not rd['no_post_clean']), xyz=rd['xyz'], rebuilt=frozen, unused_vertices_final=unused_final, lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'], split_schedule=rd['split_schedule'],
                       prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'],
                       vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0])), f, indent=1)
    print('Training is done; vertices', int(triangles.vertices.shape[0]), 'triangles', int(triangles._triangle_indices.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MeshSplatting training with RD-controlled growth', parents=[_pre])
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[7_000, 30_000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[7_000, 30_000])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--scene_name', default='Garden', type=str)
    parser.add_argument('--indoor', action='store_true', default=False)
    parser.add_argument('--lam', type=float, required=True, help='price of a byte in per-view pixel-L1 units (0 = error-ranked growth, no price)')
    parser.add_argument('--bytes_per_vertex', type=float, default=10.0, help='measured delivered bytes per vertex for this scene/precision')
    parser.add_argument('--dd_mode', choices=['error', 'grad', 'importance'], default='error')
    parser.add_argument('--dd_norm', choices=['mpx', 'none'], default='mpx', help='mpx: dD per megapixel of the loss image (resolution-independent lambda); none: v1 raw pixel units')
    parser.add_argument('--split_schedule', type=str, default=None, help='rd_log.jsonl of a reference run: cap the per-interval split count at its n_split (matched-count control)')
    parser.add_argument('--prune_iters', nargs='*', type=int, default=[12000, 14000, 16000, 18000], help='phase-B lambda-prune rounds (no values = no pruning)')
    parser.add_argument('--prune_mu', type=float, default=1.0, help='collapse iff ring-residual / bytes-saved < lam * prune_mu')
    parser.add_argument('--prune_max_frac', type=float, default=0.15, help='at most this fraction of vertices collapsed per round')
    parser.add_argument('--bytes_per_face', type=float, default=2.0, help='topology bytes per face in the delivered file (bytes_per_vertex - 2*bytes_per_face = attribute bytes per vertex)')
    parser.add_argument('--allow_any_prune_iter', action='store_true', help='smoke only: allow prune rounds outside (Delaunay, supersampling)')
    parser.add_argument('--rebuild_iters', nargs='*', type=int, default=[20000, 26000], help='v5: rebuild events; the last one snaps + freezes XYZ (no values = off, then v5 == v3)')
    parser.add_argument('--snap_only', action='store_true', help='v5 control: at the last rebuild iteration clean + snap + freeze only, no Delaunay (earlier events skipped)')
    parser.add_argument('--no_post_clean', action='store_true', help='v5: skip the face-only importance cleaning after each rebuild')
    parser.add_argument('--xyz', type=str, default='q14', help='v5: delivery XYZ grid (q13..q18, from xyz_rule_v1.py); the final event snaps vertices to it')
    parser.add_argument('--rebuild_jitter', type=float, default=0.02, help='v5: symbolic jitter of the Delaunay input, in grid steps (delaunay_canon_v1)')
    parser.add_argument('--rebuild_max_rounds', type=int, default=6, help='v5: max Delaunay/unused-vertex-removal rounds per event')
    parser.add_argument('--allow_any_rebuild_iter', action='store_true', help='smoke only')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path, '| RD growth+prune+rebuild v5 lam', args.lam, 'mode', args.dd_mode, 'norm', args.dd_norm, 'B/v', args.bytes_per_vertex,
          'schedule', args.split_schedule, '| prune', args.prune_iters, 'mu', args.prune_mu, 'max_frac', args.prune_max_frac, 'B/face', args.bytes_per_face, '| v5 rebuild', args.rebuild_iters, 'snap_only', args.snap_only, args.xyz)
    if args.split_schedule and not os.path.isfile(args.split_schedule):
        sys.exit(f'split schedule not found: {args.split_schedule}')
    up.lpips_fn = lpips.LPIPS(net='vgg').to(device='cuda')  # training_report reads this module global
    safe_state(args.quiet)
    lps, ops, pps = lp.extract(args), op.extract(args), pp.extract(args)
    if args.indoor:
        ops = update_indoor(ops)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps, ops, pps, args.test_iterations, args.start_checkpoint, args.debug_from, args.scene_name,
             rd=dict(lam=args.lam, bytes_per_vertex=args.bytes_per_vertex, dd_mode=args.dd_mode, dd_norm=args.dd_norm, split_schedule=args.split_schedule,
                     prune_iters=args.prune_iters, prune_mu=args.prune_mu, prune_max_frac=args.prune_max_frac, bytes_per_face=args.bytes_per_face,
                     allow_any_prune_iter=args.allow_any_prune_iter, rebuild_iters=args.rebuild_iters, snap_only=args.snap_only, no_post_clean=args.no_post_clean,
                     xyz=args.xyz, rebuild_jitter=args.rebuild_jitter, rebuild_max_rounds=args.rebuild_max_rounds, allow_any_rebuild_iter=args.allow_any_rebuild_iter))
PYEOF_V5
for f in rd_rebuild_v2.py test_rd_rebuild_cpu_v2.py train_rd_v5.py delaunay_canon_v1.py delaunay_check_v1.py; do [ -s $AN/rdgrowth/$f ] && echo "ok $f ($(wc -l < $AN/rdgrowth/$f) lines)" || echo "FAILED: missing $AN/rdgrowth/$f (delaunay_canon_v1/delaunay_check_v1 come from rd-topo-v4 §1)"; done
python3 -c "import ast,sys; [ast.parse(open('$AN/rdgrowth/'+f).read()) for f in ('rd_rebuild_v2.py','test_rd_rebuild_cpu_v2.py','train_rd_v5.py')]; print('compiled OK')" || echo "FAILED: compile"
# CPU test needs the training env (torch/scipy); if this shell is not in it the sbatch repeats the test anyway:
(cd $AN/rdgrowth && python test_rd_rebuild_cpu_v2.py 2>&1 | tail -2) || echo "FAILED: CPU test (non-blocking here if torch is missing; sbatch re-runs it)"
```

## §2 提交 Room λ6.2e-3 两跑（v5 + snap 对照；整块粘贴）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/rd-topo-v5.sbatch <<'SBEOF_V5'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=room-topo5
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/topo5-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/topo5-%x-%j.err
# rd-topo-v5.sbatch (2026-10-26): direction A, second design.  For SCENE / LAM and each MODE in MODES:
#   v5   train_rd_v5 = v3 + early fixed-soup restricted Delaunay at REBUILDS[0] (XYZ keeps training) + final canonical
#        rebuild on the delivery grid at REBUILDS[-1] (XYZ frozen) -> decoder-side check -> level-1.0 protocol -> check on
#        the delivered asset (coverage %, label entropy, estimated MB with labels)
#   snap control = v3 + clean + snap + freeze at REBUILDS[-1], no Delaunay (isolates the quantise+freeze cost)
# -> /scratch/zr2245/$SCENE-rd5-$MODE-lam$LAM/, rows in /scratch/zr2245/analysis/rd-topo-v5-$SCENE.txt.  Restartable.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAM=${LAM:-0.0062}; MODES=${MODES:-"v5 snap"}
XYZ=${XYZ:?set XYZ=q14 via --export (xyz_rule_v1 value of this scene)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}; REBUILDS=${REBUILDS:-"20000 26000"}; JIT=${JIT:-0.02}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; S2=/scratch/zr2245/shared-collapse-stage2-v2
DATA=/scratch/zr2245/datasets/m360/$SCENE; TAB=$AN/rd-topo-v5-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v5.py; CK=$AN/rdgrowth/delaunay_check_v1.py
echo "scene $SCENE | IMAGES $IMAGES | lam $LAM | modes $MODES | rebuilds $REBUILDS jitter $JIT | $XYZ/$DC K$K R$R | prune $PRUNE mu $MU | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $CK $AN/rdgrowth/rd_rebuild_v2.py $AN/rdgrowth/test_rd_rebuild_cpu_v2.py $AN/rdgrowth/delaunay_canon_v1.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py $S2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
(cd $AN/rdgrowth && python delaunay_canon_v1.py && python test_rd_rebuild_cpu_v2.py | tail -1) || { echo "CPU SELFTEST FAILED"; exit 1; }
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
# --- smoke: 1300 iterations, prune at 1200 (forced), early rebuild at 1240, final at 1250, 50 frozen iterations + save; the check must give 100 % ---
SM=$AN/topo5-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
echo "=== smoke ($(date)) ==="
python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam 0.0062 --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
  --prune_iters 1200 --allow_any_prune_iter --prune_mu 1e6 --prune_max_frac 0.1 --rebuild_iters 1240 1250 --allow_any_rebuild_iter --xyz $XYZ --rebuild_jitter $JIT > $SM/smoke.log 2>&1 \
  || { echo "SMOKE FAILED"; tail -40 $SM/smoke.log; exit 1; }
grep -h "^\[RD rebuild\|^\[RD v5\|^\[RD prune" $SM/smoke.log | tail -12
python $CK --state $SM/original/point_cloud/iteration_1300/point_cloud_state_dict.pt --xyz $XYZ --jitter $JIT --label "smoke" --out $SM/check.json || { echo "SMOKE CHECK FAILED"; exit 1; }
python - <<PYSMOKE || { echo "SMOKE CHECK FAILED (coverage)"; exit 1; }
import json; c=json.load(open('$SM/check.json')); e=json.load(open('$SM/original/rebuild-1240.json')); r=json.load(open('$SM/original/rebuild.json'))
print('smoke early:', {k:e[k] for k in ('stage','vertices_before','vertices_after_clean','vertices','faces','restriction_calls')}, e['rounds'][0])
print('smoke final:', {k:r[k] for k in ('stage','vertices_before','vertices_after_clean','duplicates_merged','vertices','faces','unused_kept','restriction_calls')}, '| rounds', len(r['rounds']), r['rounds'][0], '| check coverage', c['coverage'], 'on_grid', c['on_grid'], 'H1', c['H1_bits_per_vertex'])
assert e['stage'] == 'early' and r['stage'] == 'final', 'stages'
assert c['on_grid'] and c['quantiser_idempotent'], 'saved vertices are not on the grid'
assert c['coverage'] == 1.0, 'smoke coverage %s: delivered faces are not all Delaunay faces' % c['coverage']
PYSMOKE
for MODE in $MODES; do
  case $MODE in v5) EXTRA="";; snap) EXTRA="--snap_only";; *) echo "unknown MODE $MODE"; exit 1;; esac
  G=/scratch/zr2245/$SCENE-rd5-$MODE-lam$LAM; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $LAM mode $MODE: train v5 (rebuilds $REBUILDS $EXTRA) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $LAM --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --rebuild_iters $REBUILDS --xyz $XYZ --rebuild_jitter $JIT $EXTRA > $G/train.log 2>&1 \
      || { echo "TRAIN FAILED $SCENE $MODE"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE $MODE"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "$MODE -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD prune\|^\[RD rebuild\|^\[RD v5" $G/train.log; grep -h "ITER 30000\] Evaluating test" $G/train.log | tail -1
  python $CK --state $CB --xyz $XYZ --jitter $JIT --label "$SCENE $MODE lam$LAM checkpoint" --out $G/check-ckpt.json | tee -a $TAB || { echo "CHECK FAILED (checkpoint)"; exit 1; }
  echo "=== $SCENE $MODE: level-1.0 protocol ($(date)) ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE $MODE"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE $MODE"; tail -8 $G/level.log; exit 1; }
  python $CK --state $NA/deliver/shared-final.pt --xyz $XYZ --jitter $JIT --label "$SCENE $MODE lam$LAM delivered" --asset $NA/deliver/delivery/asset.mslb --stage2 $S2 --out $G/check-delivered.json | tee -a $TAB \
    || { echo "CHECK FAILED (delivered)"; exit 1; }
  python - <<PYROW | tee -a $TAB
import json, os, glob
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']
r=json.load(open('$G/original/rebuild.json')); c=json.load(open('$G/check-delivered.json'))
early=[json.load(open(p)) for p in sorted(glob.glob('$G/original/rebuild-*.json')) if json.load(open(p))['stage']=='early']
e0=(early[0]['rounds'][0] if early and early[0]['rounds'] else {}); r0=(r['rounds'][0] if r['rounds'] else {})
est=c.get('est_MB_with_labels_H1')
print('$SCENE lam $LAM RD-V5-$MODE rebuilds $REBUILDS | vertices', b['vertices'], 'faces', b['faces'], '| val', round(f['validation_best']['psnr'],3), '| test', round(f['test_best']['psnr'],3),
      '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC',
      '| early: V', (early[0]['vertices_before'] if early else None), '->', (early[0]['vertices'] if early else None), 'faces', (early[0]['faces_before'] if early else None), '->', (early[0]['faces'] if early else None),
      'retained', e0.get('soup_faces_retained'), 'new', e0.get('output_faces_new'), 'post-clean -', (early[0].get('faces_removed_by_post_clean') if early else None),
      '| final (' + r['stage'] + '): V', r['vertices_before'], '->', r['vertices'], 'faces', r['faces_before'], '->', r['faces'], 'rounds', len(r['rounds']), 'dup', r['duplicates_merged'],
      'retained', r0.get('soup_faces_retained'), 'new', r0.get('output_faces_new'), 'unused kept', r.get('unused_kept', r.get('unused_final_event')), 'post-clean -', r.get('faces_removed_by_post_clean'),
      '| coverage %.2f %%' % (100*c['coverage']), '| labels H0/H1', c['H0_bits_per_vertex'], c['H1_bits_per_vertex'], 'bits/v | topology now', c.get('topology_bits_per_vertex'), 'bits/v',
      '| est', est, 'MB with labels (%.0fx of 742 MB)' % (742/est) if est else '')
PYROW
done
echo "===== $TAB ====="; cat $TAB
echo "references (Room lam 0.0062): v3 5.228 MB / test 27.382 (142x) | v4 (single rebuild, drifting soup) 4.56 MB / 25.61, est 3.53 MB with labels"
echo "v3/v1 Room RD curve near 3.5 MB: 3.01 MB / 26.51 and 5.70 MB / 27.37 (v1 lambda sweep) -> interpolated ~26.7 at 3.5 MB; v3 ~+0.1 above v1"
echo "pass rule: v5 delivered test >= 27.28 very viable | 26.98..27.28 compare est-MB point with the v3 curve (needs >= +0.15 dB above it) | < 26.98 dead; snap row = cost of quantise+freeze alone"
echo "rd-topo-v5 $SCENE exit 0  $(date)"
SBEOF_V5
bash -n $AN/rd-topo-v5.sbatch && echo "sbatch syntax OK" || echo "FAILED: sbatch syntax"
sbatch --export=ALL,SCENE=room,IMAGES=images_2,LAM=0.0062,MODES="v5 snap",XYZ=q14,REBUILDS="20000 26000",JIT=0.02 $AN/rd-topo-v5.sbatch
squeue -u $USER -o "%.10i %.12j %.8T %.10M %.6D %R"
```

只跑 v5 不跑对照：`MODES="v5"`；换 λ：`LAM=0.00185`（输出目录 /scratch/zr2245/room-rd5-<mode>-lam<λ>/）。

## §3 看结果

```bash
AN=/scratch/zr2245/analysis
tail -5 $AN/topo5-room-topo5-*.out; echo; cat $AN/rd-topo-v5-room.txt
grep -h "^\[RD rebuild" /scratch/zr2245/room-rd5-v5-lam0.0062/train.log
```
把 `.out` 的最后 ~60 行和 `rd-topo-v5-room.txt` 贴回来。

## 设计备注
- restriction 规则与上游 `rdel.run` 逐行一致（interior Delaunay 面，其两个 tet 外心连线命中 soup 则保留）；唯一区别是 soup 固定为事件开始时清理过的网格，BVH 只建一次。
- final 事件的 Delaunay 输入是 `delaunay_canon_v1` 的 canonical jittered 网格点（decoder 可逐位复现）；early 事件用浮点位置（不需要可复现，只是让几何提前适应 Delaunay 拓扑）。
- 顶点集在 final 之后不再变化（只删面），保存时断言无漂移；decoder 侧 `delaunay_check_v1.py` 量 coverage / H0 / H1。
- 下一步若 v5 通过：把 face-label 流 + decoder 端 Delaunay 接进 stage2 codec（需要你同意改 codec 包，新版本号）。
