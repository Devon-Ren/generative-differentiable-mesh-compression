# rd-sweep-v1（2026-10-18）：Room λ sweep，rate-controlled growth v1

一次粘贴：写 `$AN/rdgrowth/rd_growth.py`、`$AN/rdgrowth/train_rd.py`、`$AN/rd-sweep-v1.sbatch`，然后提交两个 job（各 3 个 λ，两个账户各一个）。
每个 λ：train_rd.py（30000 步，和 run_garden 同一套 flags，不加 --indoor，与现有 cap/collapse 结果同口径）→ level-1.0 protocol（run_garden.sh FRACS=1.0，K4096 R256 q14/q7）→ 一行追加到 `/scratch/zr2245/analysis/rd-sweep-room.txt`。
job 开头先跑 1200 步 smoke（λ=0），检查 `rend_ids` 归属有效、有 densify 事件，否则直接失败。
λ 的单位是"每字节的 per-view 像素 L1 残差"，估计有效区间 3e-4 … 3e-2；λ=0 是 error-ranked 无价格的对照（会长到接近 dense）。

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/rd_growth.py <<'PYEOF'
"""Rate-distortion controlled mesh growth for MeshSplatting (v1, 2026-10-18).

Pure-torch helpers, no rasterizer import, so this file is unit-testable on CPU.

Idea.  Upstream `TriangleModel.add_new_gs` grows the soup by a fixed ratio (add_percentage) up to a hand-set cap
(`--max_points`) and picks the triangles to split by importance-weighted sampling.  Here the triangle count is an
OUTPUT of a Lagrangian rule: split triangle t only if   dD_t > lam * dR_t,  where
  dD_t = distortion mass attributable to t over the densification interval
         ('error' mode: sum over pixels whose dominant triangle is t of the per-pixel L1 residual, averaged per view;
          'grad'  mode: accumulated gradient norm of its three vertices, the 3DGS-style proxy)
  dR_t = bytes the split adds in the delivered file (3 midpoints x bytes/vertex, bytes/vertex measured per scene by
         our codec: topology ~4 B + XYZ 2-2.6 B + DC ~1.5 B + index ~2.4 B => 9-11 B).
The per-interval growth limiter of upstream (add_percentage) is kept as a schedule safety, so lam decides WHICH
triangles split and WHEN growth stops; the limiter only caps how fast.  lam = 0 reduces to error-ranked growth
without a price (ablation), 'importance' mode reproduces upstream sampling through the same code path (control).
"""
import json
import torch
import torch.nn.functional as F


class RDAccumulator:
    """Accumulates per-triangle distortion over one densification interval."""

    def __init__(self, n_tri, mode='error', device='cuda'):
        assert mode in ('error', 'grad', 'importance'), mode
        self.mode = mode
        self.device = device
        self.reset(n_tri)

    def reset(self, n_tri):
        self.err = torch.zeros(n_tri, dtype=torch.float32, device=self.device)
        self.views = 0

    def apply_mask(self, keep_mask):
        """Keep the accumulator aligned with triangle pruning (keep_mask over current triangles)."""
        if self.err.numel() != keep_mask.numel():
            raise RuntimeError(f'accumulator/triangle mismatch: {self.err.numel()} vs {keep_mask.numel()}')
        self.err = self.err[keep_mask]

    @torch.no_grad()
    def accumulate(self, render_pkg, image, gt_image, triangle_indices, vertex_grad=None):
        n = self.err.numel()
        if self.mode == 'error':
            ids = render_pkg['rend_ids']
            ids = ids.reshape(ids.shape[-2], ids.shape[-1])
            res = (image.detach() - gt_image).abs().mean(0)  # [H0, W0]
            if ids.shape != res.shape:  # supersampled id map -> nearest-neighbour to the loss resolution
                ids = F.interpolate(ids[None, None].float(), size=res.shape, mode='nearest')[0, 0]
            ids = ids.reshape(-1).round().long()
            res = res.reshape(-1)
            valid = (ids >= 0) & (ids < n)
            self.err.index_add_(0, ids[valid], res[valid])
            self.valid_frac = float(valid.float().mean())
        elif self.mode == 'grad':
            if vertex_grad is None:
                raise RuntimeError('grad mode needs vertex_grad (triangles.vertices.grad before zero_grad)')
            g = vertex_grad.norm(dim=1)  # [V]
            self.err += g[triangle_indices.long()].sum(1)
        else:  # importance: nothing accumulated, selection samples like upstream
            pass
        self.views += 1

    def distortion(self):
        return self.err / max(self.views, 1)


def rd_select(acc, triangles, lam, bytes_per_split, max_split, size_probs_zero, size_probs_zero_image_space):
    """Return (indices of triangles to split, stats dict).

    Candidates: dD_t / dR_t > lam, after the same area / image-size filters upstream applies to its sampling
    probabilities.  If more than max_split candidates pass, keep the ones with the largest ratio (growth limiter).
    """
    dD = acc.distortion()
    areas = triangles.triangle_areas().squeeze()
    ratio = dD / float(bytes_per_split)
    allowed = (areas >= size_probs_zero) & (triangles.image_size >= size_probs_zero_image_space)
    ratio = torch.where(allowed, ratio, torch.zeros_like(ratio))
    passing = ratio > lam
    n_pass = int(passing.sum())
    if acc.mode == 'importance':  # control: upstream multinomial sampling on importance_score
        probs = triangles.importance_score.squeeze().clone()
        probs = torch.where(allowed, probs, torch.zeros_like(probs))
        k = min(max_split, int((probs > 0).sum()))
        idx = triangles._sample_alives(probs=probs, num=k) if k > 0 else torch.zeros(0, dtype=torch.long, device=probs.device)
        n_pass = k
    elif n_pass == 0 or max_split <= 0:
        idx = torch.zeros(0, dtype=torch.long, device=ratio.device)
    elif n_pass <= max_split:
        idx = torch.nonzero(passing, as_tuple=True)[0]
    else:
        idx = torch.topk(ratio, max_split, largest=True, sorted=False).indices
    nz = ratio[allowed]
    q = torch.quantile(nz, torch.tensor([0.5, 0.9, 0.99], device=nz.device)).tolist() if nz.numel() > 1000 else [float('nan')] * 3
    stats = dict(n_tri=int(ratio.numel()), n_pass=n_pass, n_split=int(idx.numel()), max_split=int(max_split),
                 ratio_q50=q[0], ratio_q90=q[1], ratio_q99=q[2], views=acc.views, dD_sum=float(dD.sum()),
                 valid_frac=getattr(acc, 'valid_frac', None))
    return idx, stats


def rd_split(triangles, add_idx, iteration, splitt_large_triangles):
    """Same body as upstream add_new_gs after index selection: also split the largest triangles, subdivide, drop parents."""
    areas = triangles.triangle_areas().squeeze()
    k = min(int(splitt_large_triangles), areas.numel())
    top_idx = torch.topk(areas, k, largest=True, sorted=False).indices if k > 0 else add_idx[:0]
    add_idx = torch.unique(torch.cat([add_idx.to(top_idx.device), top_idx]), sorted=False)
    if add_idx.numel() == 0:
        return 0
    new = triangles._update_params_fast(add_idx, iteration)
    triangles.densification_postfix(*new)
    mask = torch.ones(triangles._triangle_indices.shape[0], dtype=torch.bool)
    mask[add_idx.cpu()] = False
    triangles.prune_triangles(mask)
    return int(add_idx.numel())


class RDLog:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'a')

    def write(self, **kw):
        self.f.write(json.dumps(kw) + '\n')
        self.f.flush()
PYEOF
cat > $AN/rdgrowth/train_rd.py <<'PYEOF'
"""train_rd.py (v1, 2026-10-18): MeshSplatting training with rate-distortion controlled growth.

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
from rd_growth import RDAccumulator, RDLog, rd_select, rd_split  # noqa: E402


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
    acc = RDAccumulator(triangles._triangle_indices.shape[0], mode=rd['dd_mode'])
    log = RDLog(os.path.join(dataset.model_path, 'rd_log.jsonl'))
    bytes_per_split = 3.0 * rd['bytes_per_vertex']  # three edge midpoints per split (upper bound; shared midpoints cost less)
    log.write(event='config', lam=rd['lam'], dd_mode=rd['dd_mode'], bytes_per_vertex=rd['bytes_per_vertex'],
              bytes_per_split=bytes_per_split, max_points=opt.max_points, add_percentage=opt.add_percentage,
              densify_from=opt.densify_from_iter, densify_until=opt.densify_until_iter, interval=opt.densification_interval)
    t0 = time.time()

    for iteration in range(first_iter, opt.iterations + 1):
        if need_delaunay:
            with torch.no_grad():
                triangles.run_restricted_delaunay()
            need_delaunay = False
            log.write(event='delaunay', iteration=iteration, vertices=int(triangles.vertices.shape[0]),
                      triangles=int(triangles._triangle_indices.shape[0]), elapsed_s=round(time.time() - t0))
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
            # RD: attribute this view's residual to triangles (phase A only: before the Delaunay rebuild)
            if iteration < run_restricted_delaunay:
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
                    idx, st = rd_select(acc, triangles, rd['lam'], bytes_per_split, max_split,
                                        triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    n_split = rd_split(triangles, idx, iteration, splitt_large_triangles)
                    st.update(event='densify', iteration=iteration, vertices_before=V, vertices_after=int(triangles.vertices.shape[0]),
                              triangles_after=int(triangles._triangle_indices.shape[0]), n_split_total=n_split, elapsed_s=round(time.time() - t0))
                    log.write(**st)
                    print(f"\n[RD {iteration}] V {V}->{st['vertices_after']} pass {st['n_pass']}/{st['n_tri']} split {n_split} "
                          f"(limit {max_split}) ratio q50/q90/q99 {st['ratio_q50']:.2e}/{st['ratio_q90']:.2e}/{st['ratio_q99']:.2e} "
                          f"lam {rd['lam']:.2e} valid {st['valid_frac']}")
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
        json.dump(dict(lam=rd['lam'], dd_mode=rd['dd_mode'], bytes_per_vertex=rd['bytes_per_vertex'],
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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path, '| RD growth lam', args.lam, 'mode', args.dd_mode, 'B/v', args.bytes_per_vertex)
    up.lpips_fn = lpips.LPIPS(net='vgg').to(device='cuda')  # training_report reads this module global
    safe_state(args.quiet)
    lps, ops, pps = lp.extract(args), op.extract(args), pp.extract(args)
    if args.indoor:
        ops = update_indoor(ops)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps, ops, pps, args.test_iterations, args.start_checkpoint, args.debug_from, args.scene_name,
             rd=dict(lam=args.lam, bytes_per_vertex=args.bytes_per_vertex, dd_mode=args.dd_mode))
PYEOF
cat > $AN/rd-sweep-v1.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd-sweep
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zr2245/analysis/rdsweep-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rdsweep-%x-%j.err
# rd-sweep-v1.sbatch (2026-10-18): for SCENE and each L in LAMS: train with rate-controlled growth (rdgrowth/train_rd.py, --lam L,
# --dd_mode MODE, --bytes_per_vertex BPV), then the level-1.0 protocol (run_garden.sh FRACS=1.0, K4096 R256 XYZ/DC stage2-v2).
# -> /scratch/zr2245/$SCENE-rd-$MODE-lam$L/, one row per lambda in /scratch/zr2245/analysis/rd-sweep-$SCENE.txt.  Restartable.
# Submit: sbatch --export=ALL,SCENE=room,IMAGES=images_2,LAMS="0 0.001 0.01",XYZ=q14,BPV=10.0,MODE=error --job-name=room-rd-a /scratch/zr2245/analysis/rd-sweep-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0 0.001" via --export}
XYZ=${XYZ:-q14}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; MODE=${MODE:-error}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/$SCENE; TAB=$AN/rd-sweep-$SCENE.txt; TR=$AN/rdgrowth/train_rd.py
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | mode $MODE B/v $BPV | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
# --- smoke: 1200 iterations, lam 0 -> must log a densify event with valid id attribution ---
SM=$AN/rd-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
echo "=== smoke ($(date)) ==="
python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1200 --test_iterations 1200 --lam 0 --bytes_per_vertex $BPV --dd_mode $MODE > $SM/smoke.log 2>&1 \
  || { echo "SMOKE FAILED"; tail -30 $SM/smoke.log; exit 1; }
python - <<PY || { echo "SMOKE CHECK FAILED"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; d=[e for e in ev if e['event']=='densify']
assert d, 'no densify event in smoke'
print('smoke densify events', len(d), 'last:', {k:d[-1][k] for k in ('iteration','vertices_before','vertices_after','n_pass','n_split','ratio_q50','ratio_q90','ratio_q99','valid_frac')})
vf=d[-1]['valid_frac']
assert '$MODE'!='error' or (vf is not None and vf>0.5), f'rend_ids attribution invalid: valid_frac={vf}'
PY
grep -h "^\[RD" $SM/smoke.log | tail -3
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd-$MODE-lam$L; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L: train ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --dd_mode $MODE > $G/train.log 2>&1 \
      || { echo "TRAIN FAILED $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "lam $L -> trained vertices $VB ($(date))"; grep -h "^\[RD" $G/train.log | tail -4
  echo "=== $SCENE lam $L: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; dl=[e for e in ev if e['event']=='delaunay']
print('$SCENE lam $L RD-GROWTH-$MODE | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: last split iter', max([e['iteration'] for e in dn if e['n_split_total']>0] or [0]),
      'V@delaunay', dl[0]['vertices'] if dl else None, 'splits/interval', [e['n_split_total'] for e in dn])
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-sweep $SCENE exit 0  $(date)"
SBEOF
chmod +x $AN/rd-sweep-v1.sbatch
python -m py_compile $AN/rdgrowth/rd_growth.py $AN/rdgrowth/train_rd.py && echo "rdgrowth compiled"
S=$AN/rd-sweep-v1.sbatch
ja=$(sbatch --parsable --export=ALL,SCENE=room,IMAGES=images_2,LAMS="0 0.001 0.01",XYZ=q14,BPV=10.0,MODE=error --job-name=room-rd-a --account=torch_pr_38_general $S); echo "room-rd-a (general) $ja"
jb=$(sbatch --parsable --export=ALL,SCENE=room,IMAGES=images_2,LAMS="0.0003 0.003 0.03",XYZ=q14,BPV=10.0,MODE=error --job-name=room-rd-b --account=torch_pr_38_tandon_advanced $S); echo "room-rd-b (tandon) $jb"
squeue -u $USER -o "%.10i %.12j %.24a %.2t %.10M %.6D %R"
```

跑完要给我的：`cat /scratch/zr2245/analysis/rd-sweep-room.txt`（6 行），以及任一 `train.log` 里 `grep "^\[RD"` 的全部行（看 λ 什么时候让增长停下来）。失败时给 `rdsweep-room-rd-a-<jobid>.out` 的最后 40 行。
