# rd-topo-v4: 方向 A probe — decoder 再生拓扑（train_rd_v4 + delaunay_check_v1），2026-10-26

**目的**：在 Room 上验证"训练末尾在量化顶点上重做 restricted Delaunay 并冻结 XYZ"的代价与收益：test PSNR 相对 v3 的差、交付面在 Delaunay(交付顶点) 中的覆盖率（目标 100 %）、面标签的熵（bits/v）对比现在的 topology 流。
**文件**（全部新文件，不改上游和 v3）：`rdgrowth/delaunay_canon_v1.py`（两端一致的 canonical jittered Delaunay 定义）、`rdgrowth/rd_rebuild_v1.py`（v4 的 rebuild 事件）、`rdgrowth/train_rd_v4.py`（= v3 + `--rebuild_iter 26000 --xyz qN`）、`rdgrowth/delaunay_check_v1.py`（decoder 侧检查）、`rdgrowth/test_rd_rebuild_cpu.py`（CPU 测试）、`rd-topo-v4.sbatch`。
**GPU 成本**：smoke 3 min + 每个 λ 约 40 min（训练 26–34 min + protocol 10 min + 检查 2 min）；LAMS="0.0062 0.00185" 共 ≈1.4 GPU-h，walltime 2.5 h，48 GB，general。
**判据**：delivered 行 coverage = 100 %；test PSNR 相对 v3（λ6.2e-3 27.382 @ 5.228 MB；λ1.85e-3 28.004 @ 12.572 MB）损失 ≤ 0.1 dB；H1 bits/v ≪ 现在的 topology bits/v（Room ≈30）。

## §1 安装（任意 shell，整块粘贴；不会退出你的终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/delaunay_canon_v1.py <<'PYEOF_CANON'
"""delaunay_canon_v1.py (2026-10-26): canonical, decoder-reproducible Delaunay tetrahedralisation of a quantised vertex set.

Shared by train_rd_v4.py (encoder side: late restricted-Delaunay rebuild on the delivery grid) and delaunay_check_v1.py
(decoder side: verify that every delivered face is a face of Delaunay(delivered vertices) and estimate the cost of the
face-selection labels).  Pure numpy / scipy, no torch.

Definition (both sides must agree bit for bit):
  q          = delivery-grid integers of the vertices (same rule as stage2 codec/delivery.quantize: low = per-axis min,
               scale = span / (2^bits - 1), q = rint((x - low) / scale))
  perm       = lexsort of the rows of q (x major, then y, then z); canonical rank r = position in perm
  P[r]       = float64(q[perm[r]]) * float32(scale) + float32(low)  + jitter * scale * (hash(r, axis) - 0.5)
  P32        = float32(P)                      <- the array scipy.spatial.Delaunay receives, in canonical order
The jitter (default 2 % of a grid step, hash = splitmix64 of 3*r+axis) breaks the co-sphericity of grid points so the
Delaunay is unique and any correct implementation returns the same tetrahedra; it is far below the grid step, so it does
not change what the mesh looks like (rendering uses the unjittered dequantised positions).
"""
import numpy as np

VERSION = 'delaunay_canon_v1 2026-10-26'
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def quantize_xyz(x, bits):
    """Identical to stage2 codec/delivery.quantize for mode q<bits>: returns (q uint, low float64[3], scale float64[3])."""
    flat = np.asarray(x, np.float32).reshape(-1, 3).astype(np.float64)
    if not np.isfinite(flat).all():
        raise ValueError('non-finite positions')
    low = flat.min(0); span = flat.max(0) - low
    scale = span / ((1 << bits) - 1); safe = np.where(scale > 0, scale, 1.)
    q = np.rint((flat - low) / safe).clip(0, (1 << bits) - 1)
    return q.astype(np.uint16 if bits <= 16 else np.uint32), low, scale


def dequantize_xyz(q, low, scale):
    """Identical to delivery.dequantize (kind 2): float32 positions."""
    return (np.asarray(q, np.float64).reshape(-1, 3) * np.asarray(scale) + np.asarray(low)).astype(np.float32)


def _hash01(idx):
    """splitmix64 of uint64 idx -> uniform in [0,1); platform independent (pure uint64 arithmetic)."""
    with np.errstate(over='ignore'):
        x = (np.asarray(idx, np.uint64) * np.uint64(0x9E3779B97F4A7C15)) & _M64
        x ^= x >> np.uint64(30); x = (x * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        x ^= x >> np.uint64(27); x = (x * np.uint64(0x94D049BB133111EB)) & _M64
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def canonical_points(q, low, scale, jitter=0.02):
    """-> (perm, P32).  P32[r] is the Delaunay input point of original vertex perm[r]."""
    q = np.asarray(q).reshape(-1, 3)
    perm = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    qs = q[perm].astype(np.float64)
    if len(np.unique(qs, axis=0)) != len(qs):
        raise ValueError('duplicate grid points: merge them before the canonical Delaunay')
    low32 = np.asarray(low, np.float32).astype(np.float64); scale32 = np.asarray(scale, np.float32).astype(np.float64)
    P = qs * scale32 + low32
    if jitter > 0:
        r = np.arange(len(qs), dtype=np.uint64)[:, None] * np.uint64(3) + np.arange(3, dtype=np.uint64)[None, :]
        P = P + (jitter * scale32)[None, :] * (_hash01(r) - 0.5)
    return perm, P.astype(np.float32)


def delaunay_tets(P32):
    """Delaunay tetrahedra of the canonical points (scipy / Qhull, default options)."""
    from scipy.spatial import Delaunay
    return Delaunay(np.asarray(P32, np.float32)).simplices.astype(np.int64)


def interior_faces(tets):
    """Faces shared by exactly two tetrahedra -> (faces [M,3] sorted rows, tet_pair [M,2]).  Boundary (hull) faces are
    never surface candidates (effrdel keeps only faces with two incident tets)."""
    T = len(tets)
    f = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]])
    f = np.sort(f, axis=1)
    owner = np.tile(np.arange(T, dtype=np.int64), 4)
    key = np.lexsort((f[:, 2], f[:, 1], f[:, 0]))
    f, owner = f[key], owner[key]
    same = np.all(f[1:] == f[:-1], axis=1)
    first = np.flatnonzero(same)  # f[first] == f[first+1]
    if len(first) > 1 and np.any(np.diff(first) == 1):
        raise RuntimeError('a face owned by three tetrahedra (degenerate Delaunay)')
    return f[first], np.stack([owner[first], owner[first + 1]], 1)


def face_rows_to_ids(faces, table):
    """Index of each row of `faces` (sorted triples) in `table` (sorted-row, lexsorted unique table), -1 if absent."""
    faces = np.sort(np.asarray(faces, np.int64), axis=1)
    key = np.lexsort((table[:, 2], table[:, 1], table[:, 0]))
    ts = table[key]
    # encode rows as one int (vertex count < 2^21 -> 63 bits)
    def enc(a): return (a[:, 0].astype(np.int64) << 42) | (a[:, 1].astype(np.int64) << 21) | a[:, 2].astype(np.int64)
    te, fe = enc(ts), enc(faces)
    pos = np.searchsorted(te, fe)
    pos_c = np.minimum(pos, len(te) - 1)
    hit = (te[pos_c] == fe) if len(te) else np.zeros(len(fe), bool)
    out = np.full(len(fe), -1, np.int64); out[hit] = key[pos_c[hit]]
    return out


def label_entropy(cand_faces, tet_pair, selected_ids, n_vertices):
    """Bits per vertex to code the 'selected' bit of every candidate face.
    H0: order-0; H1: conditioned on how many of the 6 sibling faces (other faces of the two incident tets) are selected
    (non-causal, so a lower bound for a causal context coder)."""
    M = len(cand_faces); sel = np.zeros(M, bool); sel[selected_ids[selected_ids >= 0]] = True
    n1 = int(sel.sum()); p = n1 / max(M, 1)

    def H(p):
        return 0.0 if p <= 0 or p >= 1 else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    H0 = M * H(p) / n_vertices
    T = int(tet_pair.max()) + 1 if M else 0
    per_tet = np.bincount(tet_pair.reshape(-1), weights=np.repeat(sel, 2), minlength=T)  # selected faces per tet
    ctx = (per_tet[tet_pair[:, 0]] + per_tet[tet_pair[:, 1]] - 2 * sel).astype(np.int64)  # 0..6 siblings selected
    H1 = 0.0; table = {}
    for c in range(7):
        m = ctx == c; n = int(m.sum())
        if n:
            pc = float(sel[m].mean()); H1 += n * H(pc); table[c] = (n, round(pc, 4))
    return dict(candidates=M, selected=n1, candidates_per_vertex=round(M / n_vertices, 3), p_selected=round(p, 4),
                H0_bits_per_vertex=round(H0, 3), H1_bits_per_vertex=round(H1 / n_vertices, 3), context_table=table)


def selftest():
    rng = np.random.default_rng(0)
    x = (rng.random((3000, 3)) * 4).astype(np.float32); x[:, 2] *= 0.3  # flat-ish cloud
    q, low, scale = quantize_xyz(x, 10)            # coarse grid -> many co-spherical points and some duplicates
    _, idx = np.unique(q, axis=0, return_index=True); q = q[np.sort(idx)]   # merge duplicates for the test
    x2 = dequantize_xyz(q, low, scale); q2, low2, scale2 = quantize_xyz(x2, 10)
    assert np.array_equal(q2, q) and np.allclose(low2, low) and np.allclose(scale2, scale), 'quantiser not idempotent'
    perm, P = canonical_points(q, low, scale)
    perm_b, P_b = canonical_points(q[::-1], low, scale)         # different input order -> same canonical points
    assert np.array_equal(P, P_b), 'canonical points depend on input order'
    t1 = delaunay_tets(P); t2 = delaunay_tets(P_b)
    f1, _ = interior_faces(t1); f2, _ = interior_faces(t2)
    assert np.array_equal(f1, f2), 'Delaunay differs for identical canonical input'
    # no-jitter grid points are degenerate: Qhull may still succeed but the jittered result must be a proper Delaunay
    ids = face_rows_to_ids(f1[::7], f1); assert np.array_equal(ids, np.arange(0, len(f1), 7))
    assert face_rows_to_ids(np.array([[0, 1, 2]]), f1)[0] in (-1, int(np.flatnonzero((f1 == [0, 1, 2]).all(1))[0]) if ((f1 == [0, 1, 2]).all(1)).any() else -1)
    pick = f1[::9]; ent = label_entropy(f1, _, face_rows_to_ids(pick, f1), len(q))
    assert ent['selected'] == len(pick) and ent['H1_bits_per_vertex'] <= ent['H0_bits_per_vertex'] + 1e-9
    print('delaunay_canon_v1 selftest PASS |', len(q), 'points', len(t1), 'tets', len(f1), 'interior faces |', ent)


if __name__ == '__main__':
    selftest()
PYEOF_CANON
cat > $AN/rdgrowth/rd_rebuild_v1.py <<'PYEOF_REBUILD'
"""rd_rebuild_v1.py (2026-10-26): the v4 late rebuild used by train_rd_v4.py, kept importable without the upstream
(CPU unit test: python test_rd_rebuild_cpu.py).  See train_rd_v4.py for the protocol."""
import time
import numpy as np
import torch
from delaunay_canon_v1 import quantize_xyz, dequantize_xyz, canonical_points, VERSION as CANON_VERSION

VERSION = 'rd_rebuild_v1 2026-10-26'

def _reset_face_trackers(triangles):
    T = triangles._triangle_indices.shape[0]
    triangles.image_size = torch.zeros((T,), dtype=torch.float, device=triangles.vertices.device)
    triangles.importance_score = torch.zeros((T,), dtype=torch.float, device=triangles.vertices.device)
    triangles.pixel_count = torch.zeros((T,), dtype=torch.int, device=triangles.vertices.device)


def _set_faces(triangles, faces_np):
    triangles._triangle_indices = torch.as_tensor(np.asarray(faces_np, dtype=np.int64), device=triangles.vertices.device).contiguous().to(torch.int32)
    _reset_face_trackers(triangles)


def late_rebuild(triangles, rd, log, t0, rdel_run=None, clean_fn=None):
    """v4 event: clean -> snap XYZ to the delivery grid -> merge duplicate grid points -> canonical restricted Delaunay
    until every vertex is used -> freeze XYZ (the caller sets the lr).  Returns the stats dict (also logged)."""
    if rdel_run is None:
        import rdel  # effrdel, the module upstream uses for run_restricted_delaunay
        rdel_run = lambda P, f: rdel.run(P, f, verbose=False, orient=False)  # noqa: E731
    bits = int(str(rd['xyz'])[1:]); jitter = float(rd['rebuild_jitter']); max_rounds = int(rd['rebuild_max_rounds'])
    dev = triangles.vertices.device
    V0, T0 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    if clean_fn is not None:
        clean_fn()
    V1, T1 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    rounds = []
    q = low = scale = None
    for r in range(max_rounds + 1):
        x = triangles.vertices.detach().cpu().numpy().astype(np.float32)
        q, low, scale = quantize_xyz(x, bits)
        uq, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        n_dup = int(len(q) - len(uq))
        if n_dup:
            rep = first[inv.reshape(-1)]                      # representative (first occurrence) of each vertex's grid cell
            f = rep[triangles._triangle_indices.long().cpu().numpy()]
            degenerate = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
            f = f[~degenerate]
            f = np.unique(np.sort(f, axis=1), axis=0) if len(f) else f   # drop duplicate faces (orientation is not used)
            _set_faces(triangles, f)
            keep = np.zeros(len(q), bool); keep[first] = True
            triangles._prune_vertices(torch.from_numpy(keep).to(dev))    # also drops now-unused vertices
            x = triangles.vertices.detach().cpu().numpy().astype(np.float32)
            q, low, scale = quantize_xyz(x, bits)
        xq = dequantize_xyz(q, low, scale)
        q2, low2, scale2 = quantize_xyz(xq, bits)
        if not np.array_equal(q2, q):
            raise RuntimeError(f'delivery quantiser is not idempotent at {bits} bits ({int((q2 != q).any(1).sum())} vertices move)')
        q, low, scale = q2, low2, scale2                                  # exactly what the decoder will see
        triangles.vertices.data = torch.from_numpy(xq).to(dev)
        perm, P32 = canonical_points(q, low, scale, jitter)
        inv_perm = np.empty_like(perm); inv_perm[perm] = np.arange(len(perm))
        faces = triangles._triangle_indices.long().cpu().numpy()
        hit = np.asarray(rdel_run(P32, inv_perm[faces].astype(np.int32)), dtype=np.int64).reshape(-1, 3)
        new_faces = perm[hit]
        _set_faces(triangles, new_faces)
        V = int(triangles.vertices.shape[0])
        used = np.zeros(V, bool); used[new_faces.reshape(-1)] = True
        n_unused = int((~used).sum())
        rounds.append(dict(round=r, vertices=V, faces_in=int(len(faces)), faces_out=int(len(new_faces)), duplicates_merged=n_dup, unused=n_unused))
        print(f'\n[RD rebuild round {r}] V {V} faces {len(faces)} -> {len(new_faces)} | duplicates merged {n_dup} | unused vertices {n_unused}')
        if n_unused == 0 or r == max_rounds:
            break
        triangles._prune_vertices(torch.from_numpy(used).to(dev))
    stats = dict(event='rebuild', version='v4', canon=CANON_VERSION, xyz=f'q{bits}', jitter=jitter, vertices_before=V0, faces_before=T0,
                 vertices_after_clean=V1, faces_after_clean=T1, rounds=rounds, vertices=int(triangles.vertices.shape[0]),
                 faces=int(triangles._triangle_indices.shape[0]), unused_kept=rounds[-1]['unused'], elapsed_s=round(time.time() - t0))
    triangles._v4_grid = (q.copy(), np.asarray(low).copy(), np.asarray(scale).copy())
    if log is not None:
        log.write(**stats)
    return stats
PYEOF_REBUILD
cat > $AN/rdgrowth/train_rd_v4.py <<'PYEOF_V4'
"""train_rd_v4.py (v4, 2026-10-26 = v3 + decoder-reproducible topology; v3 kept as train_rd_v3.py).

v4 adds ONE event at --rebuild_iter (default 26000, after the last prune round, inside the 4x supersampling phase):
  1. upstream importance cleaning (faces with max_blending <= 0.5 removed, unused vertices removed) -- moved here from the
     end of training so that the Delaunay below sees the delivered vertex set;
  2. XYZ snapped to the delivery grid (--xyz qN, same quantiser as stage2 codec/delivery.quantize); duplicate grid points
     merged (faces remapped, degenerate faces dropped);
  3. restricted Delaunay (upstream effrdel rdel.run) recomputed on the CANONICAL jittered grid points
     (delaunay_canon_v1.py) with the current faces as the restriction soup; repeated until every vertex is used
     (--rebuild_max_rounds), so that every delivered face is an interior face of Delaunay(delivered vertices);
  4. XYZ frozen (vertex lr = 0) for the rest of training; appearance (DC, SH, weight) keeps training;
  5. final cleaning prunes faces only (no vertex removal), and the save asserts the positions are still on the grid.
The delivered connectivity can then be regenerated by the decoder from the quantised vertices plus one selection bit per
candidate Delaunay face (delaunay_check_v1.py measures coverage and the label entropy).  rebuild.json records the stats.
--rebuild_iter 0 disables the event (then v4 == v3).

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
from rd_rebuild_v1 import late_rebuild  # noqa: E402


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
    rebuild_iter = int(rd['rebuild_iter'])
    rebuilt = False
    if rebuild_iter:
        if not (rebuild_iter > max(last_prune, run_restricted_delaunay) and rebuild_iter <= opt.iterations) and not rd['allow_any_rebuild_iter']:
            raise SystemExit(f'rebuild_iter {rebuild_iter} must lie after the last prune round ({last_prune}) / Delaunay ({run_restricted_delaunay}) and <= iterations; --allow_any_rebuild_iter overrides (smoke only)')
        print('[RD] v4 late rebuild at', rebuild_iter, '| grid', rd['xyz'], '| jitter', rd['rebuild_jitter'], '| max rounds', rd['rebuild_max_rounds'], '|', CANON_VERSION)
    log.write(event='config', version='v4', rebuild_iter=rebuild_iter, xyz=rd['xyz'], rebuild_jitter=rd['rebuild_jitter'], rebuild_max_rounds=rd['rebuild_max_rounds'], lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'],
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
        if rebuild_iter and iteration == rebuild_iter and not rebuilt:
            with torch.no_grad():
                rb = late_rebuild(triangles, rd, log, t0, clean_fn=lambda: importance_clean(triangles, scene, pipe, background))
            rebuilt = True
            acc.reset(triangles._triangle_indices.shape[0])
            with open(os.path.join(dataset.model_path, 'rebuild.json'), 'w') as fh:
                json.dump(rb, fh, indent=1)
            print(f"[RD rebuild done] V {rb['vertices_before']} -> {rb['vertices']} faces {rb['faces_before']} -> {rb['faces']} | rounds {len(rb['rounds'])} | unused kept {rb['unused_kept']} | XYZ frozen from here")
        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)
        if rebuilt:
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
    if rebuilt:
        q, low, scale = triangles._v4_grid
        xq = dequantize_xyz(q, low, scale)
        drift = float(np.abs(triangles.vertices.detach().cpu().numpy().astype(np.float32) - xq).max()) if len(xq) == triangles.vertices.shape[0] else float('inf')
        if drift != 0.0:
            raise RuntimeError(f'vertices moved after the rebuild (max drift {drift}); the Delaunay guarantee is broken')
        print(f'[RD v4] final cleaning: faces pruned, {unused_final} unused vertices KEPT (Delaunay vertex set unchanged), drift 0')
    else:
        triangles._prune_vertices(used_vertex_mask)
    scene.save(iteration)
    log.write(event='final', vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0]), unused_vertices=unused_final, rebuilt=rebuilt,
              elapsed_s=round(time.time() - t0))
    with open(os.path.join(dataset.model_path, 'rd_summary.json'), 'w') as f:
        json.dump(dict(version='v4', rebuild_iter=rebuild_iter, xyz=rd['xyz'], rebuilt=rebuilt, unused_vertices_final=unused_final, lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'], split_schedule=rd['split_schedule'],
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
    parser.add_argument('--rebuild_iter', type=int, default=26000, help='v4: iteration of the late quantised Delaunay rebuild + XYZ freeze (0 = off, then v4 == v3)')
    parser.add_argument('--xyz', type=str, default='q14', help='v4: delivery XYZ grid (q13..q18, from xyz_rule_v1.py); the rebuild snaps vertices to it')
    parser.add_argument('--rebuild_jitter', type=float, default=0.02, help='v4: symbolic jitter of the Delaunay input, in grid steps (delaunay_canon_v1)')
    parser.add_argument('--rebuild_max_rounds', type=int, default=6, help='v4: max Delaunay/unused-vertex-removal rounds')
    parser.add_argument('--allow_any_rebuild_iter', action='store_true', help='smoke only')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path, '| RD growth+prune+rebuild v4 lam', args.lam, 'mode', args.dd_mode, 'norm', args.dd_norm, 'B/v', args.bytes_per_vertex,
          'schedule', args.split_schedule, '| prune', args.prune_iters, 'mu', args.prune_mu, 'max_frac', args.prune_max_frac, 'B/face', args.bytes_per_face, '| v4 rebuild', args.rebuild_iter, args.xyz)
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
                     allow_any_prune_iter=args.allow_any_prune_iter, rebuild_iter=args.rebuild_iter, xyz=args.xyz, rebuild_jitter=args.rebuild_jitter,
                     rebuild_max_rounds=args.rebuild_max_rounds, allow_any_rebuild_iter=args.allow_any_rebuild_iter))
PYEOF_V4
cat > $AN/rdgrowth/delaunay_check_v1.py <<'PYEOF_CHECK'
"""delaunay_check_v1.py (2026-10-26, decoder-side check for direction A):
does every delivered face belong to Delaunay(delivered vertices), and what would the face-selection labels cost?

  python delaunay_check_v1.py --state <point_cloud_state_dict.pt | shared-final.pt | decoded.pt> --xyz q14 [--label room-lam0.0062]
        [--asset deliver/delivery/asset.mslb --stage2 /scratch/zr2245/shared-collapse-stage2-v2]   (current topology bytes)
        [--jitter 0.02] [--out check.json]

Steps: quantise positions with the delivery rule (idempotent on v4 outputs; on v3 outputs this is what the codec would do),
merge duplicate grid points, canonical jittered points (delaunay_canon_v1), scipy Delaunay, interior faces = candidates,
coverage = share of delivered faces that are candidates, H0 / H1 = order-0 / sibling-context entropy of the selection bit
in bits per vertex.  Prints one TOPO-CHECK line and writes JSON.  CPU only; ~1 min and ~3 GB per million vertices."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from delaunay_canon_v1 import quantize_xyz, dequantize_xyz, canonical_points, delaunay_tets, interior_faces, face_rows_to_ids, label_entropy, VERSION as CANON_VERSION  # noqa: E402

VERSION = 'delaunay_check_v1 2026-10-26'


def arr(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True); p.add_argument('--xyz', required=True, help='q13..q18 (the delivery grid of this scene)')
    p.add_argument('--jitter', type=float, default=0.02); p.add_argument('--label', default='')
    p.add_argument('--asset', default=None); p.add_argument('--stage2', default='/scratch/zr2245/shared-collapse-stage2-v2')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    t0 = time.time(); bits = int(a.xyz[1:])
    w = torch.load(a.state, map_location='cpu', weights_only=False)
    x = arr(w['triangles_points']).astype(np.float32).reshape(-1, 3); f = arr(w['_triangle_indices']).astype(np.int64).reshape(-1, 3)
    V, F = len(x), len(f)
    q, low, scale = quantize_xyz(x, bits); xq = dequantize_xyz(q, low, scale)
    drift = float(np.abs(x - xq).max()); q2, _, _ = quantize_xyz(xq, bits); idem = bool(np.array_equal(q2, q))
    uq, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True); n_dup = int(V - len(uq))
    rep = first[inv.reshape(-1)]; fr = rep[f]
    degenerate = (fr[:, 0] == fr[:, 1]) | (fr[:, 1] == fr[:, 2]) | (fr[:, 0] == fr[:, 2]); fr = fr[~degenerate]
    keep = np.zeros(V, bool); keep[first] = True; new_id = np.full(V, -1, np.int64); new_id[first] = np.arange(len(first)); fr = new_id[fr]
    qu = q[first]; xu = xq[first]
    perm, P32 = canonical_points(quantize_xyz(xu, bits)[0], *quantize_xyz(xu, bits)[1:], a.jitter)   # exactly the decoder's input
    inv_perm = np.empty_like(perm); inv_perm[perm] = np.arange(len(perm))
    t1 = time.time(); tets = delaunay_tets(P32); t_del = time.time() - t1
    cand, tet_pair = interior_faces(tets)
    ids = face_rows_to_ids(inv_perm[fr], cand)
    covered = int((ids >= 0).sum()); coverage = covered / max(len(fr), 1)
    ent = label_entropy(cand, tet_pair, ids, len(qu))
    res = dict(version=VERSION, canon=CANON_VERSION, state=str(a.state), label=a.label, xyz=a.xyz, jitter=a.jitter, vertices=V, faces=F,
               duplicates_merged=n_dup, degenerate_faces_dropped=int(degenerate.sum()), on_grid=(drift == 0.0), max_drift=drift, quantiser_idempotent=idem,
               tets=int(len(tets)), delaunay_s=round(t_del, 2), coverage=round(coverage, 6), faces_not_delaunay=int(len(fr) - covered), **ent)
    if a.asset:
        try:
            sys.path.insert(0, str(Path(a.stage2) / 'codec')); sys.path.insert(0, str(Path(a.stage2)))
            from codec_benchmark import unpack
            wire = unpack(a.asset); top = int(np.asarray(wire['top_stream']).size); fb = Path(a.asset).stat().st_size
            res.update(asset_bytes=fb, topology_bytes=top, topology_bits_per_vertex=round(top * 8 / V, 2),
                       est_bytes_with_labels_H1=int(fb - top + ent['H1_bits_per_vertex'] * V / 8),
                       est_MB_with_labels_H1=round((fb - top + ent['H1_bits_per_vertex'] * V / 8) / 1e6, 3), asset_MB=round(fb / 1e6, 3))
        except Exception as e:  # noqa: BLE001
            res['asset_error'] = repr(e)
    res['elapsed_s'] = round(time.time() - t0, 1)
    print(f"TOPO-CHECK {a.label} | V {V} F {F} | grid {a.xyz} on_grid {res['on_grid']} drift {drift:.2e} dup {n_dup} | tets {res['tets']} cand/V {ent['candidates_per_vertex']} "
          f"| coverage {100 * coverage:.2f} % ({res['faces_not_delaunay']} faces outside) | sel/cand {ent['p_selected']} | H0 {ent['H0_bits_per_vertex']} H1 {ent['H1_bits_per_vertex']} bits/v"
          + (f" | topology now {res['topology_bits_per_vertex']} bits/v, asset {res['asset_MB']} MB -> est {res['est_MB_with_labels_H1']} MB with labels" if 'topology_bytes' in res else '')
          + f" | {res['elapsed_s']} s")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
PYEOF_CHECK
cat > $AN/rdgrowth/test_rd_rebuild_cpu.py <<'PYEOF_TEST'
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
PYEOF_TEST
cd $AN/rdgrowth && python -m py_compile delaunay_canon_v1.py rd_rebuild_v1.py train_rd_v4.py delaunay_check_v1.py && echo "compiled OK" || echo "FAILED: py_compile"
cd $AN/rdgrowth && python delaunay_canon_v1.py || echo "FAILED: delaunay_canon selftest"
cd $AN/rdgrowth && python test_rd_rebuild_cpu.py | tail -2 || echo "FAILED: rebuild cpu test"
grep -c "rebuild" $AN/rdgrowth/train_rd_v4.py; head -1 $AN/rdgrowth/train_rd_v4.py | cut -c1-80
```

## §2 提交 Room 两个 λ（整块粘贴）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/rd-topo-v4.sbatch <<'SBEOF_V4'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=room-topo4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/topo4-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/topo4-%x-%j.err
# rd-topo-v4.sbatch (2026-10-26): direction A probe.  For SCENE and each L in LAMS: train_rd_v4 (= v3 + late quantised
# Delaunay rebuild at REBUILD, XYZ frozen) -> decoder-side check on the checkpoint -> level-1.0 protocol (unchanged) ->
# check on shared-final.pt + delivered asset (coverage %, label entropy, topology bytes now vs estimated).
# Also runs the check on the existing v3 checkpoint as the baseline (expected coverage ~1-2 %).
# -> /scratch/zr2245/$SCENE-rd4-lam$L/, rows in /scratch/zr2245/analysis/rd-topo-v4-$SCENE.txt.  Restartable.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0.0062 0.00185" via --export}
XYZ=${XYZ:?set XYZ=q14 via --export (xyz_rule_v1 value of this scene)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}; REBUILD=${REBUILD:-26000}; JIT=${JIT:-0.02}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; S2=/scratch/zr2245/shared-collapse-stage2-v2
DATA=/scratch/zr2245/datasets/m360/$SCENE; TAB=$AN/rd-topo-v4-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v4.py; CK=$AN/rdgrowth/delaunay_check_v1.py
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | rebuild $REBUILD jitter $JIT | $XYZ/$DC K$K R$R | prune $PRUNE mu $MU | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $CK $AN/rdgrowth/rd_rebuild_v1.py $AN/rdgrowth/delaunay_canon_v1.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py $S2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
(cd $AN/rdgrowth && python delaunay_canon_v1.py && python test_rd_rebuild_cpu.py | tail -1) || { echo "CPU SELFTEST FAILED"; exit 1; }
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
# --- baseline: the existing v3 checkpoint of this scene (first lambda), decoder-side coverage ---
L0=$(echo $LAMS | awk '{print $1}'); V3CK=/scratch/zr2245/$SCENE-rd3-lam$L0-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
if [ -f $V3CK ]; then
  echo "=== baseline check on v3 checkpoint ($(date)) ==="
  python $CK --state $V3CK --xyz $XYZ --jitter $JIT --label "$SCENE v3 lam$L0 (baseline)" --out $AN/topo4-baseline-$SCENE-lam$L0.json | tee -a $TAB || echo "baseline check FAILED (non-fatal)"
else echo "no v3 checkpoint at $V3CK (baseline skipped)"; fi
# --- smoke: 1300 iterations, prune at 1200 (forced), rebuild at 1250, then 50 iterations on the rebuilt mesh + save; check must give 100 % ---
SM=$AN/topo4-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
echo "=== smoke ($(date)) ==="
python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam 0.0062 --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
  --prune_iters 1200 --allow_any_prune_iter --prune_mu 1e6 --prune_max_frac 0.1 --rebuild_iter 1250 --allow_any_rebuild_iter --xyz $XYZ --rebuild_jitter $JIT > $SM/smoke.log 2>&1 \
  || { echo "SMOKE FAILED"; tail -40 $SM/smoke.log; exit 1; }
grep -h "^\[RD rebuild\|^\[RD v4\|^\[RD prune" $SM/smoke.log | tail -8
python $CK --state $SM/original/point_cloud/iteration_1300/point_cloud_state_dict.pt --xyz $XYZ --jitter $JIT --label "smoke" --out $SM/check.json || { echo "SMOKE CHECK FAILED"; exit 1; }
python - <<PY || { echo "SMOKE CHECK FAILED (coverage)"; exit 1; }
import json; c=json.load(open('$SM/check.json')); r=json.load(open('$SM/original/rebuild.json'))
print('smoke rebuild:', {k:r[k] for k in ('vertices_before','vertices_after_clean','vertices','faces','unused_kept')}, '| rounds', len(r['rounds']), '| check coverage', c['coverage'], 'on_grid', c['on_grid'], 'H1', c['H1_bits_per_vertex'])
assert c['on_grid'] and c['quantiser_idempotent'], 'saved vertices are not on the grid'
assert c['coverage'] == 1.0, 'smoke coverage %s: delivered faces are not all Delaunay faces' % c['coverage']
PY
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd4-lam$L; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L: train v4 (rebuild $REBUILD) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --rebuild_iter $REBUILD --xyz $XYZ --rebuild_jitter $JIT > $G/train.log 2>&1 \
      || { echo "TRAIN FAILED $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "lam $L -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD prune\|^\[RD rebuild\|^\[RD v4" $G/train.log; grep -h "ITER 30000\] Evaluating test" $G/train.log | tail -1
  python $CK --state $CB --xyz $XYZ --jitter $JIT --label "$SCENE v4 lam$L checkpoint" --out $G/check-ckpt.json | tee -a $TAB || { echo "CHECK FAILED (checkpoint)"; exit 1; }
  echo "=== $SCENE lam $L: level-1.0 protocol ($(date)) ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python $CK --state $NA/deliver/shared-final.pt --xyz $XYZ --jitter $JIT --label "$SCENE v4 lam$L delivered" --asset $NA/deliver/delivery/asset.mslb --stage2 $S2 --out $G/check-delivered.json | tee -a $TAB \
    || { echo "CHECK FAILED (delivered)"; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']
r=json.load(open('$G/original/rebuild.json')); c=json.load(open('$G/check-delivered.json'))
est=c.get('est_MB_with_labels_H1'); print('$SCENE lam $L RD-V4-REBUILD$REBUILD | vertices', b['vertices'], 'faces', b['faces'], '| val', round(f['validation_best']['psnr'],3), '| test', round(f['test_best']['psnr'],3),
      '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC',
      '| rebuild: V', r['vertices_before'], '->', r['vertices'], 'faces', r['faces_before'], '->', r['faces'], 'rounds', len(r['rounds']), 'dup', r['rounds'][0]['duplicates_merged'], 'unused kept', r['unused_kept'],
      '| coverage %.2f %%' % (100*c['coverage']), '| labels H0/H1', c['H0_bits_per_vertex'], c['H1_bits_per_vertex'], 'bits/v | topology now', c.get('topology_bits_per_vertex'), 'bits/v',
      '| est', est, 'MB with labels (%.0fx of 742 MB)' % (742/est) if est else '')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "v3 reference (rd-prune3-room): lam 0.0062 5.228 MB / test 27.382 (142x); lam 0.00185 12.572 MB / 28.004 (59x)"
echo "rd-topo-v4 $SCENE exit 0  $(date)"
SBEOF_V4
chmod +x $AN/rd-topo-v4.sbatch
j=$(sbatch --parsable --export=ALL,SCENE=room,IMAGES=images_2,LAMS="0.0062 0.00185",XYZ=q14,REBUILD=26000,JIT=0.02 --job-name=room-topo4 --account=torch_pr_38_general $AN/rd-topo-v4.sbatch) && echo "room-topo4 (general) $j" || echo "SUBMIT FAILED"
squeue -u $USER -o "%.10i %.12j %.8T %.10M %.6D %R" | head
```

## §3 看结果

```bash
AN=/scratch/zr2245/analysis; tail -5 $AN/topo4-room-topo4-*.out; echo; cat $AN/rd-topo-v4-room.txt
```

每个 λ 三行：`TOPO-CHECK ... checkpoint`（训练输出）、`TOPO-CHECK ... delivered`（protocol 之后，含 topology 流现状和估计）、`room lam L RD-V4-REBUILD26000 ...`（PSNR/MB 汇总）；最前面一行是 v3 checkpoint 的 baseline coverage。把 `.txt` 全文贴回来。
其它场景：`--export=ALL,SCENE=garden,IMAGES=images_4,LAMS="0.0062",XYZ=q15`（XYZ 用 batch B 的值：garden/bonsai/counter q15，kitchen/bicycle/stump q16，flowers q15，treehill q16，truck/train q18 + IMAGES=images）。

## 设计备注

* rebuild 在 26000 步：最后一轮 prune 18000 之后、4× supersampling 期间；之后 4000 步只训 appearance（vertices lr=0，Adam 步长严格为 0；保存前断言位置未漂移）。可用 `REBUILD=24000` 试更早。
* 事件顺序：上游 importance cleaning（提前到此处，去掉 max_blending ≤ 0.5 的面和无用顶点）→ XYZ 量化到交付网格（与 stage2 delivery.quantize 同一规则，断言幂等）→ 合并落在同一格点的顶点 → 在 canonical 顺序（按格点 lexsort）+ 确定性 jitter（splitmix64，2 % 格距）的点上调用上游 effrdel `rdel.run`（restriction soup = 当前面）→ 若有无用顶点则删除并重做，最多 6 轮 → 冻结。
* 最终 cleaning 只删面不删顶点（删顶点会改变 Delaunay）；`rebuild.json` 记录每轮 V/F/重复点/无用顶点。
* decoder 侧 `delaunay_check_v1.py` 只用 (量化整数, low, scale) 重建同样的点并跑 scipy Delaunay；`coverage` 必须 100 %，`H1` 是按两侧四面体中已选面数做上下文的熵（非因果下界），`est MB with labels` = 资产字节 − 现 topology 流 + H1·V/8。
* 还没做的（probe 通过后）：codec 里的面标签流 + decoder 端 Delaunay（改 stage2，需要你同意）。
