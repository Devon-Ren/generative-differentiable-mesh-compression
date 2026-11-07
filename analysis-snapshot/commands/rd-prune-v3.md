# rd-prune-v3（2026-10-19）：B = phase-B 周期 λ-prune，Room 两个 λ，1 个 job（约 2 GPU 小时）

**内容**：`train_rd_v3.py` = v2m + 在 12000/14000/16000/18000 步各做一轮 edge collapse（`rd_prune.py`）：
边 (u,v) 被合并 iff（触及 u 或 v 的面的残差质量）/（省下的字节）< λ·μ，残差仍然经 `rend_ids` 归属，单位和 split 端相同（每视角、每 megapixel 的 L1 质量），λ 共用，μ 默认 1。
省下的字节 ΔR = (BPV − 2·B_face) + n_faces(uv)·B_face：内部边删 2 面 = BPV，边界边删 1 面 = BPV−2，非流形边 3+ 面更贵——顶点的价格随拓扑变化。
每轮最多删 15 % 顶点（`--prune_max_frac`，剪枝端的限速器），翻面的 collapse 被拒绝，合并顶点取中点、属性和 Adam 动量取平均，v 槽走 upstream 的 `_prune_vertices` 删除。

**对照**：同 λ 不剪枝的行已经有（rd-sweep-room：`lam 0.01` = 6.2e-3 v2 → 493772 顶点 5.70 MB / 27.370；`lam 0.003` = 1.85e-3 v2 → 1323185 顶点 13.70 MB / 27.985）。

一次粘贴：写 `rd_prune.py`、`train_rd_v3.py`、`rd-prune-v3.sbatch`，提交 1 个 job（general；C 在 tandon）。job 开头 1300 步 smoke（强制在 1250 步剪一轮、μ=1e6 让 cap 生效、再训 50 步验证剪完还能渲染），然后每个 λ：train+prune → level-1.0 protocol → 一行到 `/scratch/zr2245/analysis/rd-prune3-room.txt`。
只想先跑一个 λ：把下面 `LAMS="0.0062 0.00185"` 改成 `LAMS="0.0062"`（约 1 GPU 小时）。

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/rd_prune.py <<'PYEOF'
"""Rate-distortion controlled edge-collapse pruning for MeshSplatting (v3, 2026-10-19).

Pure torch, no rasterizer import, so it is unit-testable on CPU with a mock TriangleModel.

Idea.  After the restricted Delaunay rebuild upstream never removes a vertex again (only the final importance<=0.5
cleaning of triangles).  Here, at a few iterations of phase B (before supersampling), we run the split rule backwards:
collapse edge (u,v) into its midpoint iff

        dD_uv / dR_uv  <  lam * mu,        dD_uv = residual mass of the faces touching u or v   (same units as the split
                                            side: per-view averaged, per-megapixel pixel-L1 attributed through rend_ids)
                                            dR_uv = B_attr + n_faces(uv) * B_face   (bytes the collapse saves in the file:
                                            one vertex's attributes + the faces that disappear; interior edge 2 faces,
                                            boundary 1, non-manifold 3+ -> the price of a vertex varies with its topology)

Collapses are chosen as an independent set (no vertex takes part in two collapses in one round) by repeated locally-minimal
edge matching, cheapest score first, capped at a fraction of the vertices per round (prune_max_frac: the pruning twin of the
upstream growth limiter: lam decides which and when to stop, the cap only limits speed).  Collapses that would flip the
orientation of a surviving face are rejected (foldover guard).  The merged vertex keeps u's slot: parameters, vertex
opacity (averaged in opacity space, like the split) and Adam moments are averaged into u, v is deleted with the upstream
_prune_vertices path, degenerate and duplicate faces are dropped.
"""
import torch


def _face_normals(pos, faces):
    t = pos[faces]
    return torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=1)


@torch.no_grad()
def edge_table(faces):
    """Unique undirected edges of an int face tensor [T,3] -> (edges [E,2] long sorted, inverse [3T] long, count [E] long)."""
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], dim=0)
    e, _ = torch.sort(e, dim=1)
    edges, inv, cnt = torch.unique(e, dim=0, return_inverse=True, return_counts=True)
    return edges, inv, cnt


@torch.no_grad()
def foldover_filter(pos, f, edges, sel):
    """Reject selected edges whose midpoint collapse flips the normal of a surviving face.  Returns (sel kept, n rejected)."""
    if sel.numel() == 0:
        return sel, 0
    dev = pos.device
    V = pos.shape[0]
    u, v = edges[sel, 0], edges[sel, 1]
    pos2 = pos.clone()
    mid = 0.5 * (pos2[u] + pos2[v])
    pos2[u] = mid
    pos2[v] = mid
    new_id = torch.arange(V, device=dev)
    new_id[v] = u
    f2 = new_id[f]
    degenerate = (f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2])
    n0 = _face_normals(pos, f)
    n1 = _face_normals(pos2, f)
    flipped = ((n0 * n1).sum(1) < 0) & ~degenerate
    if not bool(flipped.any()):
        return sel, 0
    pair_of_v = torch.full((V,), -1, dtype=torch.long, device=dev)
    k = torch.arange(sel.numel(), device=dev)
    pair_of_v[u] = k
    pair_of_v[v] = k
    bad = pair_of_v[f[flipped].reshape(-1)]
    bad = torch.unique(bad[bad >= 0])
    keep = torch.ones(sel.numel(), dtype=torch.bool, device=dev)
    keep[bad] = False
    return sel[keep], int((~keep).sum())


@torch.no_grad()
def select_collapses(vertices, faces, face_err, lam, mu, bytes_per_vertex, bytes_per_face, max_frac, max_passes=64):
    """Pick an independent set of edges to collapse.  Returns (pairs [P,2] long (u,v), stats dict).

    face_err: [T] distortion per face over the window (acc.distortion()).  Score per edge (float64):
      s_uv = (D_u + D_v - D_uv) / dR_uv, D_v = sum of face_err over faces touching v, D_uv = over faces containing the edge.
    Candidates: s_uv < lam * mu.  Cap: at most max_frac * V pairs (cheapest first).
    """
    dev = vertices.device
    V = vertices.shape[0]
    f = faces.long()
    T = f.shape[0]
    edges, inv, cnt = edge_table(f)
    E = edges.shape[0]
    err = face_err.double()
    D_v = torch.zeros(V, dtype=torch.float64, device=dev)
    for k in range(3):
        D_v.index_add_(0, f[:, k], err)
    D_uv = torch.zeros(E, dtype=torch.float64, device=dev)
    D_uv.index_add_(0, inv, err.repeat(3))
    b_attr = float(bytes_per_vertex) - 2.0 * float(bytes_per_face)  # attributes of one vertex (XYZ + DC + index), topology priced per face
    if b_attr <= 0:
        raise ValueError(f'bytes_per_vertex {bytes_per_vertex} must exceed 2*bytes_per_face {bytes_per_face}')
    dR = b_attr + cnt.double() * float(bytes_per_face)
    dD = D_v[edges[:, 0]] + D_v[edges[:, 1]] - D_uv
    score = dD / dR
    thr = float(lam) * float(mu)
    cand = score < thr
    n_cand = int(cand.sum())
    max_pairs = int(max_frac * V)
    # tie-break (many exact zeros where nothing was rendered): tiny seeded random jitter, float64 so it is not lost.
    # Random rather than index-ordered: index order makes the locally-minimal matching crawl one row per pass on grids.
    gen = torch.Generator(device=dev); gen.manual_seed(1234 + E)
    jit = torch.rand(E, dtype=torch.float64, device=dev, generator=gen) * 1e-9 * max(thr, 1e-30)
    s = torch.where(cand, score + jit, torch.full_like(score, float('inf')))
    avail = torch.ones(V, dtype=torch.bool, device=dev)
    picked = []
    n_picked = 0
    passes = 0
    for _ in range(max_passes):
        passes += 1
        live = cand & avail[edges[:, 0]] & avail[edges[:, 1]]
        if not bool(live.any()) or n_picked >= max_pairs:
            break
        sl = torch.where(live, s, torch.full_like(s, float('inf')))
        vmin = torch.full((V,), float('inf'), dtype=torch.float64, device=dev)
        vmin.scatter_reduce_(0, edges[:, 0], sl, reduce='amin')
        vmin.scatter_reduce_(0, edges[:, 1], sl, reduce='amin')
        pick = live & (sl == vmin[edges[:, 0]]) & (sl == vmin[edges[:, 1]])
        idx = torch.nonzero(pick, as_tuple=True)[0]
        if idx.numel() == 0:
            break
        use = torch.zeros(V, dtype=torch.long, device=dev)
        use.index_add_(0, edges[idx, 0], torch.ones_like(idx))
        use.index_add_(0, edges[idx, 1], torch.ones_like(idx))
        ok = (use[edges[idx, 0]] == 1) & (use[edges[idx, 1]] == 1)
        idx = idx[ok]
        if idx.numel() == 0:
            break
        picked.append(idx)
        n_picked += int(idx.numel())
        avail[edges[idx].reshape(-1)] = False
    sel = torch.cat(picked) if picked else torch.zeros(0, dtype=torch.long, device=dev)
    if sel.numel() > max_pairs:
        order = torch.argsort(s[sel])
        sel = sel[order[:max_pairs]]
    capped = bool(n_picked > max_pairs)
    sel, n_flip = foldover_filter(vertices.detach(), f, edges, sel)
    pairs = edges[sel]
    def share(idx_mask_or_idx):
        c = cnt[idx_mask_or_idx]
        n = max(int(c.numel()), 1)
        return dict(f1=round(float((c == 1).sum()) / n, 4), f2=round(float((c == 2).sum()) / n, 4), f3p=round(float((c >= 3).sum()) / n, 4))
    fin = score[torch.isfinite(score)]
    if fin.numel() > 8_000_000:  # torch.quantile size limit
        fin = fin[torch.randperm(fin.numel(), device=dev)[:8_000_000]]
    q = torch.quantile(fin, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=dev)).tolist() if fin.numel() > 1000 else [float('nan')] * 3
    stats = dict(vertices=V, faces=T, edges=E, n_cand=n_cand, cand_frac=round(n_cand / max(E, 1), 4), n_pass_pairs=n_picked,
                 n_pairs=int(pairs.shape[0]), max_pairs=max_pairs, capped=capped, n_flip_rejected=n_flip, passes=passes,
                 thr=thr, score_q10=q[0], score_q50=q[1], score_q90=q[2], dR_mean_sel=(float(dR[sel].mean()) if sel.numel() else None),
                 faces_removed_share_sel=share(sel), faces_removed_share_cand=share(cand), faces_removed_share_all=share(torch.ones(E, dtype=torch.bool, device=dev)))
    return pairs, stats


@torch.no_grad()
def collapse_edges(triangles, pairs):
    """Collapse each (u,v) pair into u (midpoint position, averaged attributes and Adam moments), delete v.

    Pairs must be vertex-disjoint (select_collapses guarantees it).  Uses only upstream members: vertices, vertex_weight,
    _features_dc, _features_rest, optimizer, opacity_activation / inverse_opacity_activation, opacity_floor, eps,
    _triangle_indices, image_size, importance_score, pixel_count, _prune_vertices.  Returns stats dict.
    """
    V = triangles.vertices.shape[0]
    T0 = triangles._triangle_indices.shape[0]
    if pairs.numel() == 0:
        return dict(vertices_before=V, vertices_after=V, faces_before=T0, faces_after=T0, faces_degenerate=0, faces_duplicate=0)
    dev = triangles.vertices.device
    u = pairs[:, 0].to(dev)
    v = pairs[:, 1].to(dev)
    if int(torch.unique(pairs.reshape(-1)).numel()) != 2 * pairs.shape[0]:
        raise RuntimeError('collapse pairs are not vertex-disjoint')
    # 1) merge parameters into slot u
    for name in ('vertices', '_features_dc', '_features_rest'):
        p = getattr(triangles, name)
        p.data[u] = 0.5 * (p.data[u] + p.data[v])
    w = triangles.vertex_weight
    op = triangles.opacity_activation(w.data)
    avg = torch.clamp(0.5 * (op[u] + op[v]), triangles.opacity_floor + triangles.eps, 1 - triangles.eps)
    w.data[u] = triangles.inverse_opacity_activation(avg)
    # 2) Adam moments of the merged vertex = average of the two
    for group in triangles.optimizer.param_groups:
        if group['name'] in ('vertices', 'vertex_weight', 'f_dc', 'f_rest'):
            st = triangles.optimizer.state.get(group['params'][0], None)
            if st is not None:
                for key in ('exp_avg', 'exp_avg_sq'):
                    m = st[key]
                    m[u] = 0.5 * (m[u] + m[v])
    # 3) faces: v -> u, drop degenerate and duplicate faces, keep per-face trackers aligned
    f = triangles._triangle_indices.long()
    new_id = torch.arange(V, device=dev)
    new_id[v] = u
    f2 = new_id[f]
    degenerate = (f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2])
    keep = ~degenerate
    key, _ = torch.sort(f2, dim=1)
    _, first = torch.unique(key, dim=0, return_inverse=True)
    # first occurrence of each sorted triple survives (orientation of that face kept)
    order = torch.arange(f2.shape[0], device=dev)
    firstpos = torch.full((int(first.max()) + 1,), f2.shape[0], dtype=torch.long, device=dev)
    firstpos.scatter_reduce_(0, first, order, reduce='amin')
    dup = firstpos[first] != order
    n_dup = int((dup & keep).sum())
    keep = keep & ~dup
    triangles._triangle_indices = f2[keep].to(torch.int32).contiguous()
    for name in ('image_size', 'importance_score', 'pixel_count'):
        t = getattr(triangles, name)
        if isinstance(t, torch.Tensor) and t.numel() == T0:
            setattr(triangles, name, t[keep])
    # 4) delete the v slots through the upstream path (prunes optimizer state and reindexes faces)
    vmask = torch.ones(V, dtype=torch.bool, device=dev)
    vmask[v] = False
    triangles._prune_vertices(vmask)
    V1 = int(triangles.vertices.shape[0])
    return dict(vertices_before=V, vertices_after=V1, pairs=int(pairs.shape[0]), vertices_isolated_removed=V - int(pairs.shape[0]) - V1,
                faces_before=T0, faces_after=int(triangles._triangle_indices.shape[0]), faces_degenerate=int(degenerate.sum()), faces_duplicate=n_dup)
PYEOF
cat > $AN/rdgrowth/train_rd_v3.py <<'PYEOF'
"""train_rd_v3.py (v3, 2026-10-19 = v2m + phase-B rate-controlled edge-collapse pruning; v2m kept as train_rd_v2m.py): MeshSplatting training with rate-distortion controlled growth AND pruning.

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
    log.write(event='config', version='v3', lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'],
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
        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)
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
    triangles._prune_vertices(used_vertex_mask)
    scene.save(iteration)
    log.write(event='final', vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0]),
              elapsed_s=round(time.time() - t0))
    with open(os.path.join(dataset.model_path, 'rd_summary.json'), 'w') as f:
        json.dump(dict(version='v3', lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'], split_schedule=rd['split_schedule'],
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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path, '| RD growth+prune v3 lam', args.lam, 'mode', args.dd_mode, 'norm', args.dd_norm, 'B/v', args.bytes_per_vertex,
          'schedule', args.split_schedule, '| prune', args.prune_iters, 'mu', args.prune_mu, 'max_frac', args.prune_max_frac, 'B/face', args.bytes_per_face)
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
                     allow_any_prune_iter=args.allow_any_prune_iter))
PYEOF
cat > $AN/rd-prune-v3.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=room-rd3
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/rd3-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3-%x-%j.err
# rd-prune-v3.sbatch (2026-10-19): for SCENE and each L in LAMS: train with rate-controlled growth AND phase-B lambda-prune
# (rdgrowth/train_rd_v3.py: --lam L --dd_mode error --dd_norm mpx --bytes_per_vertex BPV --prune_iters PRUNE --prune_mu MU
# --prune_max_frac MAXF --bytes_per_face BPF), then the level-1.0 protocol (run_garden.sh FRACS=1.0, K4096 R256 XYZ/DC stage2-v2).
# -> /scratch/zr2245/$SCENE-rd3-lam$L-mu$MU/, one row per lambda in /scratch/zr2245/analysis/rd-prune3-$SCENE.txt.  Restartable.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0.0062 0.00185" via --export}
XYZ=${XYZ:-q14}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/$SCENE; TAB=$AN/rd-prune3-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
# --- smoke: 1300 iterations, one forced prune round at 1250 (mu 1e6 -> every edge is a candidate, the per-round cap binds),
#     then 50 more iterations of rendering/training on the collapsed mesh and the final save ---
SM=$AN/rd3-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
echo "=== smoke ($(date)) ==="
python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam 0.0062 --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
  --prune_iters 1250 --allow_any_prune_iter --prune_mu 1e6 --prune_max_frac 0.1 > $SM/smoke.log 2>&1 \
  || { echo "SMOKE FAILED"; tail -40 $SM/smoke.log; exit 1; }
python - <<PY || { echo "SMOKE CHECK FAILED"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
assert p, 'no prune event in smoke'; e=p[0]
print('smoke prune:', {k:e[k] for k in ('iteration','views','vertices_before','vertices_after','pairs','vertices_isolated_removed','faces_before','faces_after','faces_degenerate','faces_duplicate','n_cand','n_pairs','max_pairs','n_flip_rejected','passes','dR_mean_sel')})
assert e['n_pairs'] > 0 and e['vertices_after'] < e['vertices_before'], 'collapse did nothing'
assert e['faces_after'] == e['faces_before'] - e['faces_degenerate'] - e['faces_duplicate'], 'face bookkeeping'
assert fin, 'training did not finish after the prune (render on collapsed mesh failed?)'
print('smoke final:', fin[0])
PY
grep -h "^\[RD" $SM/smoke.log | tail -3
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3-lam$L-mu$MU; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU: train+prune ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF > $G/train.log 2>&1 \
      || { echo "TRAIN FAILED $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "lam $L mu $MU -> trained vertices $VB ($(date))"; grep -h "^\[RD prune" $G/train.log
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
print('$SCENE lam $L RD-GROWTH+PRUNE-mu$MU | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'cand%.3f' % e['cand_frac'], 'pairs', e['n_pairs'], 'cap', e['max_pairs'], 'flip', e['n_flip_rejected'],
                          'q50/thr %.2f' % (e['score_q50'] / e['thr'] if e['thr'] else float('nan')), 'sel', e['faces_removed_share_sel'], 'all', e['faces_removed_share_all']) for e in pr])
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune3 $SCENE exit 0  $(date)"
SBEOF
chmod +x $AN/rd-prune-v3.sbatch
python -m py_compile $AN/rdgrowth/rd_prune.py $AN/rdgrowth/train_rd_v3.py && echo "rd_prune + train_rd_v3 compiled"
[ -f $AN/rdgrowth/rd_growth_v2.py ] && echo "rd_growth_v2.py present" || echo "MISSING $AN/rdgrowth/rd_growth_v2.py (paste rd-sweep-v2-min first)"
j=$(sbatch --parsable --export=ALL,SCENE=room,IMAGES=images_2,LAMS="0.0062 0.00185",XYZ=q14,BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=room-rd3 --account=torch_pr_38_general $AN/rd-prune-v3.sbatch) && echo "room-rd3 (general) $j" || echo "SUBMIT FAILED"
squeue -u $USER -o "%.10i %.14j %.24a %.2t %.10M %.6D %R"
```

跑完给我：`cat /scratch/zr2245/analysis/rd-prune3-room.txt`（2 行，每行末尾带 4 轮剪枝的 V 前后、候选比例 cand、q50/thr、删面份额 sel vs all）。失败就给 `rd3-room-rd3-<jobid>.out` 最后 40 行；smoke 失败也在那里。

读法：
- 同字节下 test PSNR ≥ 不剪枝的 λ 行 → B 成立；差距应在 6.2e-3 比 1.85e-3 大。
- `cand` 是通过 λ·μ 门槛的边比例，`q50/thr` 是中位分数与门槛之比：q50/thr ≫ 1 且 cand ≈ 0 说明 μ=1 太严（剪枝没发生，run 退化成 λ-only，结果仍可用作"无事可剪"的结论）；cand 很大且每轮都顶到 cap 说明 μ=1 太松。
- `sel` 里 f2（内部边）/f3p 的份额高于 `all` → ΔR 变价确实优先删省字节多的顶点。
