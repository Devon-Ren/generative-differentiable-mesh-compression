# node-batch2-v1：过夜包（2026-11-05 16:40）— end-to-end adaptive λ（主）+ B_a 敏感性 + 8 场景 fan，16 个 job，4 × L40S 各 ≈8 h

**adaptive λ 是什么**（`rdgrowth/train_rd_adapt_v1.py` = train_rd_v3 + 一个闭环码率控制器，v3 不动）：λ 不再是常数，而是训练中在线搜索的拉格朗日乘子，使交付体积命中 `--target_mb`。做法和视频编码器的 rate control 同构：
- phase A（增长，每个 densify 间隔）：投影体积 R̂ = (V·B_v + F·B_f)/1e6 + 0.6；把剩余预算 target − R̂ 均摊到剩余间隔得到本间隔允许的 split 数 desired；用当前 λ 跑 split 规则得到 n_pass（dD > λ·dR 的三角形数），然后 λ ← λ·(n_pass/desired)^0.5（±10 % 死区，夹在 [1e-4, 1e-1]）：规则放过的比预算多 → 涨价，少 → 降价；选择仍按 dD/dR 排序，数量封顶 desired，所以 phase A 在 densify_until 正好落在 target，λ 收敛到该预算的影子价格。
- phase B（4 轮 prune）：先把 λ 换成 phase A 最后 20 个间隔的几何均值（影子价格的稳健估计），每轮再按 (R̂/target)^2 修正，用它做 collapse 判据。
- 产物：每个 target 一行（lam0 → λ 轨迹分位数 → lam_final、投影 vs 实际交付 MB、命中比、test 指标），rd_log.jsonl 里有逐间隔的 λ / R̂ / desired / n_pass。回答 Yao 的"λ 怎么定给定码率"和"不是 end-to-end"：λ 由闭环给出，价格常数的误差通过命中比直接可见（配合 §2 的 B_a 敏感性）。

实验：room / kitchen（--indoor）/ garden / bicycle × target {5, 10, 20 MB}，每个 job 先 4 min smoke（1300 it + 强制 prune，控制器字段齐全才继续），再 3 个 target 各训练 + level-1.0 交付（≈3 h/job）。读法：命中比 0.85–1.15 = 控制器可用；同字节下 test PSNR 和该场景固定 λ 的 hull 比，差 <0.1 dB = adaptive λ 不损质量。

| 块 | job | GPU-h | 表 |
|---|---|---|---|
| §1 adaptive λ | 4 | ≈12 | `$AN/adapt-v1.txt` |
| §2 B_a 敏感性（kitchen/garden × BPV {8,13} × λ {6.2e-3, 6.2e-4}） | 4 | ≈8 | `$AN/ba-sens-v1.txt` |
| §3 fan v2（另外 8 场景） | 8 | ≈12 | `$AN/lambda-k-fan-v1.txt` |

全部 `--reservation=yw523-l40s`，account general。按 §1 → §2 → §3 顺序粘贴（每块自检通过才提交；某块 CHECK FAILED 就把输出贴我，其余块照常）。16 个 job 排在 4 张卡上约 8–9 h。

## §1 adaptive λ：写 py + sbatch + 自检 + 提交 4 个 job（整块粘贴，任意节点）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/train_rd_adapt_v1.py <<'PYEOF_ADAPT1'
"""train_rd_adapt_v1.py (adapt-v1, 2026-11-05 = train_rd_v3 + a closed-loop rate controller on lambda; v3 kept untouched):
MeshSplatting training with rate-distortion controlled growth AND pruning where lambda is NOT a user constant but is
adapted during training so that the projected delivered size hits --target_mb.

Rate controller (the same idea as rate control in a video encoder: the Lagrange multiplier is searched online so that
the bit budget is met).  Projected delivered size  R_hat = (V * bytes_per_vertex + F * bytes_per_face) / 1e6 + overhead_mb.
Phase A (growth, every densification interval): the byte headroom  target - R_hat  is spread over the remaining
densification intervals, giving the number of splits the schedule allows now,  desired = headroom / (bytes_per_split *
remaining).  The split rule runs with the current lambda and reports n_pass (triangles with dD > lambda * dR); then
    lambda <- clip(lambda * (n_pass / desired)^gamma, lam_min, lam_max)   unless |n_pass/desired - 1| < deadband,
so the price rises when the rule lets more through than the budget affords and falls when it lets fewer.  The selection
keeps the RD ranking (largest dD/dR first) and the per-interval split count is capped at desired, so phase A lands on
the target at densify_until and lambda converges to the shadow price of that budget (the value phase B then prunes with).  Phase B (prune rounds): lambda <- clip(lambda * (R_hat/target)^gamma_prune),
so an asset above target (projection error) is pruned harder.  Everything else (split rule, prune rule, units, upstream
schedule) is train_rd_v3 verbatim.  rd_log.jsonl densify/prune events carry lam_now, rhat_mb, desired, n_pass;
rd_summary.json carries lam_final / target_mb / rhat_final.  Answers "how is lambda chosen for a target rate": by this loop.

train_rd_v3.py (v3, 2026-10-19 = v2m + phase-B rate-controlled edge-collapse pruning; v2m kept as train_rd_v2m.py): MeshSplatting training with rate-distortion controlled growth AND pruning.

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
import math
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


def ctrl_update(lam, actual, wanted, gamma, deadband, lo, hi):
    """one multiplicative step of the rate controller: lam *= (actual/wanted)^gamma, clipped; returns (new_lam, ratio)."""
    ratio = actual / max(wanted, 1e-9)
    if abs(ratio - 1.0) < deadband:
        return lam, ratio
    return min(hi, max(lo, lam * ratio ** gamma)), ratio


def projected_mb(V, F, rd):
    return (V * rd['bytes_per_vertex'] + F * rd['bytes_per_face']) / 1e6 + rd['overhead_mb']


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
    lam_now = float(rd['lam']); lam_hist = []; phase_b_started = False
    log.write(event='config', version='adapt-v1', target_mb=rd['target_mb'], ctrl_gamma=rd['ctrl_gamma'], ctrl_gamma_prune=rd['ctrl_gamma_prune'], ctrl_deadband=rd['ctrl_deadband'],
              lam_min=rd['lam_min'], lam_max=rd['lam_max'], overhead_mb=rd['overhead_mb'], lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'],
              bytes_per_split=bytes_per_split, max_points=opt.max_points, add_percentage=opt.add_percentage,
              densify_from=opt.densify_from_iter, densify_until=opt.densify_until_iter, interval=opt.densification_interval,
              split_schedule=rd['split_schedule'], schedule_total=(sum(sched.values()) if sched else None),
              prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'])
    print(f"[RD adapt] target {rd['target_mb']} MB | lam0 {lam_now:.3e} | gamma {rd['ctrl_gamma']} (prune {rd['ctrl_gamma_prune']}) deadband {rd['ctrl_deadband']} | lam in [{rd['lam_min']:.1e}, {rd['lam_max']:.1e}] | overhead {rd['overhead_mb']} MB")
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
                    rhat = projected_mb(V, int(triangles._triangle_indices.shape[0]), rd)
                    remaining = max(1, (opt.densify_until_iter - iteration) // opt.densification_interval + 1)
                    headroom = max(0.0, (rd['target_mb'] - rhat) * 1e6)
                    desired = headroom / (bytes_per_split * remaining); cap = int(math.ceil(desired))
                    max_split = min(max_split, cap)  # the budget schedule caps the count; lambda converges to the price at which n_pass ~ desired
                    lam_used = lam_now; lam_hist.append(lam_now)
                    idx, st = rd_select(acc, triangles, lam_now, bytes_per_split, max_split,
                                        triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    n_split = rd_split(triangles, idx, iteration, splitt_large_triangles)
                    lam_now, ratio = ctrl_update(lam_now, float(st['n_pass']), max(desired, 1.0), rd['ctrl_gamma'], rd['ctrl_deadband'], rd['lam_min'], rd['lam_max'])
                    st.update(event='densify', iteration=iteration, vertices_before=V, vertices_after=int(triangles.vertices.shape[0]),
                              triangles_after=int(triangles._triangle_indices.shape[0]), n_split_total=n_split, k_ref=k_ref, elapsed_s=round(time.time() - t0),
                              lam_used=lam_used, lam_next=lam_now, rhat_mb=round(rhat, 4), desired=round(desired, 1), remaining=remaining, cap=cap, pass_ratio=round(ratio, 4))
                    log.write(**st)
                    print(f"\n[RD {iteration}] V {V}->{st['vertices_after']} pass {st['n_pass']}/{st['n_tri']} split {n_split} "
                          f"(limit {max_split}) ratio q50/q90/q99 {st['ratio_q50']:.2e}/{st['ratio_q90']:.2e}/{st['ratio_q99']:.2e} "
                          f"lam {lam_used:.2e} -> {lam_now:.2e} (projected {rhat:.2f} / {rd['target_mb']} MB, desired {desired:.0f}, pass/desired {ratio:.2f}, {remaining} intervals left) valid {st['valid_frac']}")
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
                rhat = projected_mb(V0, T0, rd)
                if not phase_b_started:  # phase-B price = robust estimate of the phase-A shadow price (geometric mean of the last 20 intervals)
                    phase_b_started = True
                    if lam_hist:
                        tail = lam_hist[-20:]; lam_est = math.exp(sum(math.log(x) for x in tail) / len(tail))
                        print(f"\n[RD adapt {iteration}] phase B: lam_now {lam_now:.2e} -> shadow-price estimate {lam_est:.2e} (geo-mean of last {len(tail)} intervals)")
                        lam_now = lam_est
                lam_now, ratio = ctrl_update(lam_now, rhat, rd['target_mb'], rd['ctrl_gamma_prune'], rd['ctrl_deadband'], rd['lam_min'], rd['lam_max'])
                print(f"\n[RD adapt {iteration}] prune round: projected {rhat:.2f} MB = {ratio:.2f} x target -> lam {lam_now:.2e}")
                pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, lam_now, rd['prune_mu'],
                                             rd['bytes_per_vertex'], rd['bytes_per_face'], rd['prune_max_frac'])
                cs = collapse_edges(triangles, pairs)
                acc.reset(triangles._triangle_indices.shape[0])
                ev = dict(event='prune', iteration=iteration, views=n_views, elapsed_s=round(time.time() - t0), lam_now=lam_now, rhat_mb=round(rhat, 4), rhat_ratio=round(ratio, 4))
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
    rhat_final = projected_mb(int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0]), rd)
    log.write(event='final', vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0]),
              elapsed_s=round(time.time() - t0), lam_final=lam_now, rhat_mb=round(rhat_final, 4), target_mb=rd['target_mb'])
    with open(os.path.join(dataset.model_path, 'rd_summary.json'), 'w') as f:
        json.dump(dict(version='adapt-v1', target_mb=rd['target_mb'], lam_final=lam_now, rhat_final_mb=round(rhat_final, 4), lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'], split_schedule=rd['split_schedule'],
                       prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'],
                       vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0])), f, indent=1)
    print(f"Training is done; vertices {int(triangles.vertices.shape[0])} triangles {int(triangles._triangle_indices.shape[0])} | adapt: target {rd['target_mb']} MB projected {rhat_final:.2f} MB lam0 {rd['lam']:.2e} -> lam_final {lam_now:.2e}")


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
    parser.add_argument('--target_mb', type=float, required=True, help='target delivered size in MB; lambda is adapted so the projected size V*B_v + F*B_f + overhead hits it')
    parser.add_argument('--ctrl_gamma', type=float, default=0.5, help='phase-A gain: lambda *= (n_pass/desired)^gamma per densification interval')
    parser.add_argument('--ctrl_gamma_prune', type=float, default=2.0, help='phase-B gain: lambda *= (projected/target)^gamma_prune per prune round')
    parser.add_argument('--ctrl_deadband', type=float, default=0.1, help='no update while the controlled ratio is within 1 +- deadband')
    parser.add_argument('--lam_min', type=float, default=1e-4); parser.add_argument('--lam_max', type=float, default=1e-1)
    parser.add_argument('--overhead_mb', type=float, default=0.6, help='fixed bytes of the delivered file (dictionaries + header)')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.target_mb <= 0 or args.ctrl_gamma <= 0 or not (0 < args.lam_min < args.lam_max):
        sys.exit(f'bad controller settings: target_mb {args.target_mb} gamma {args.ctrl_gamma} lam range [{args.lam_min}, {args.lam_max}]')
    print('Optimizing ' + args.model_path, '| RD growth+prune adapt-v1 target_mb', args.target_mb, 'lam0', args.lam, 'mode', args.dd_mode, 'norm', args.dd_norm, 'B/v', args.bytes_per_vertex,
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
                     allow_any_prune_iter=args.allow_any_prune_iter, target_mb=args.target_mb, ctrl_gamma=args.ctrl_gamma, ctrl_deadband=args.ctrl_deadband,
                     ctrl_gamma_prune=args.ctrl_gamma_prune, lam_min=args.lam_min, lam_max=args.lam_max, overhead_mb=args.overhead_mb))
PYEOF_ADAPT1
cat > $AN/adapt-v1.sbatch <<'SBEOF_ADAPT1'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=adapt1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/zr2245/analysis/adapt1-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/adapt1-%x-%j.err
# adapt-v1.sbatch (2026-11-05, VERSION v1): closed-loop rate control -- lambda is adapted during training so the delivered size hits TARGETS (MB).
# For SCENE and each T in TARGETS: train with rdgrowth/train_rd_adapt_v1.py (= train_rd_v3 + rate controller, see its docstring), then the usual
# level-1.0 protocol (K4096 R256 XYZ/DC stage2-v2).  Smoke stage first (1300 iterations incl. one forced prune round): a crash shows in minutes.
# Output /scratch/zr2245/$SCENE-adapt1-t${T}mb/ ; one row per target in $AN/adapt-v1.txt: target | lam0 -> lam_final | projected | delivered MB (hit ratio) | test metrics.
# Restartable (existing checkpoints / deliveries skipped).  INDOOR=1 passes --indoor (room/kitchen/counter/bonsai).
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; TARGETS=${TARGETS:-"5 10 20"}; INDOOR=${INDOOR:?set INDOOR=0 or 1 via --export}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
GAMMA=${GAMMA:-0.5}; GAMMAP=${GAMMAP:-2.0}; DEAD=${DEAD:-0.1}; SMOKE=${SMOKE:-1}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/adapt-v1.txt; TR=$AN/rdgrowth/train_rd_adapt_v1.py
echo "adapt-v1 | scene $SCENE | IMAGES $IMAGES | targets $TARGETS MB | indoor '$FLAG' | B/v $BPV B/face $BPF | gamma $GAMMA / prune $GAMMAP dead $DEAD | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python -m py_compile $TR || { echo "FAILED: $TR does not compile"; exit 1; }
grep -q "'--target_mb'" $TR || { echo "FAILED: $TR has no --target_mb (wrong file pasted?)"; exit 1; }
lam0_of () { python -c "import sys; t=float(sys.argv[1]); print('%.4e' % min(1e-1, max(1e-4, 6.2e-3 * (9.0 / t) ** 1.67)))" $1; }   # initial guess from log MB ~ -0.6 log lambda
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
if [ "$SMOKE" = "1" ]; then
  SM=$AN/adapt1-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM; echo "=== smoke 1300 it, target 5 MB ($(date)) ==="
  python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam $(lam0_of 5) --target_mb 5 \
    --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx --ctrl_gamma $GAMMA --ctrl_gamma_prune $GAMMAP --ctrl_deadband $DEAD \
    --prune_iters 1250 --allow_any_prune_iter --prune_mu 1e6 --prune_max_frac 0.1 $FLAG > $SM/smoke.log 2>&1 || { echo "FAILED: SMOKE"; tail -40 $SM/smoke.log; exit 1; }
  python - <<PY || { echo "FAILED: SMOKE CHECK"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
assert dn, 'no densify event'; assert all('lam_used' in e and 'desired' in e for e in dn), 'densify events lack controller fields'
assert p and 'lam_now' in p[0], 'no prune event with lam_now'; assert fin and 'lam_final' in fin[0], 'no final event with lam_final'
print('smoke densify:', [(e['iteration'], e['lam_used'], e['lam_next'], e['rhat_mb'], e['desired'], e['n_pass'], e['n_split']) for e in dn[:3]], '...', len(dn), 'events')
print('smoke prune:', {k: p[0][k] for k in ('iteration','lam_now','rhat_mb','vertices_before','vertices_after','n_pairs')}); print('smoke final:', fin[0])
PY
  grep -h "^\[RD adapt\|^\[RD 1[0-3]00\]" $SM/smoke.log | tail -4
else echo "smoke skipped (SMOKE=0)"; fi
for T in $TARGETS; do
  G=/scratch/zr2245/$SCENE-adapt1-t${T}mb; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G; L0=$(lam0_of $T)
  echo "=== $SCENE target $T MB (lam0 $L0) $FLAG: train with rate control ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L0 --target_mb $T \
      --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx --ctrl_gamma $GAMMA --ctrl_gamma_prune $GAMMAP --ctrl_deadband $DEAD \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF $FLAG > $G/train.log 2>&1 || { echo "FAILED: TRAIN $SCENE target $T"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE target $T"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "target $T -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD adapt\|^\[RD prune\|^Training is done" $G/train.log | tail -8
  echo "=== $SCENE target $T: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  python - "$SCENE" "$T" "$G" "$NA" "$L0" "$XYZ" "$DC" <<'PY' | tee -a $TAB
import json, sys
S, T, G, NA, L0, xyz, dc = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
rs = json.load(open(G + '/original/rd_summary.json')); b = json.load(open(NA + '/report.json')); f = json.load(open(NA + '/finetune/summary.json')); d = json.load(open(NA + '/deliver/summary.json')); t = d['delivery_test']
ev = [json.loads(l) for l in open(G + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']; pr = [e for e in ev if e['event'] == 'prune']
lams = [e['lam_used'] for e in dn]; q = lambda a, p: sorted(a)[int(p * (len(a) - 1))] if a else float('nan')
print(f"{S} | target {T:g} MB | lam0 {L0} -> lamA q10/q50/q90 {q(lams,.1):.2e}/{q(lams,.5):.2e}/{q(lams,.9):.2e} -> lam_final {rs['lam_final']:.2e} | projected {rs['rhat_final_mb']:.2f} MB"
      f" | trained V {b['vertices']} F {b['faces']} | delivered {d['file_MB']:.3f} MB (hit {d['file_MB'] / T:.3f} x target) | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f}"
      f" | val {f['validation_best']['psnr']:.3f} | K4096 R256 {xyz}/{dc} | densify events {len(dn)}, prune rounds {[(e['iteration'], e['vertices_before'], e['vertices_after'], round(e['rhat_mb'], 2)) for e in pr]}")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
echo "adapt-v1 $SCENE exit 0  $(date)"
SBEOF_ADAPT1
cat > $AN/submit-adapt-v1.sh <<'SUB'
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; RES=yw523-l40s
n=$(wc -l < $AN/rdgrowth/train_rd_adapt_v1.py); [ "$n" = 441 ] || { echo "CHECK FAILED: train_rd_adapt_v1.py has $n lines, expected 441 (paste truncated?)"; exit 1; }
n=$(wc -l < $AN/adapt-v1.sbatch); [ "$n" = 77 ] || { echo "CHECK FAILED: adapt-v1.sbatch has $n lines, expected 77"; exit 1; }
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v1.py || { echo "CHECK FAILED: train_rd_adapt_v1.py does not compile"; exit 1; }
grep -q "'--target_mb'" $AN/rdgrowth/train_rd_adapt_v1.py || { echo "CHECK FAILED: no --target_mb in the py"; exit 1; }
for f in $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $AN/vis_collapse-v12/scripts/run_garden.sh $D/m360/room/images_2 $D/m360/kitchen/images_2 $D/m360/garden/images_4 $D/m360/bicycle/images_4; do [ -e $f ] || { echo "CHECK FAILED: MISSING $f"; exit 1; }; done
scontrol show res $RES > /dev/null 2>&1 || { echo "CHECK FAILED: reservation $RES not visible"; exit 1; }
echo "CHECK OK"; k=0
sub () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS="5 10 20" --job-name=$1-adapt1 $AN/adapt-v1.sbatch) && { echo "submitted $1-adapt1 $j"; k=$((k+1)); } || echo "FAILED: sbatch $1"; }
sub room 1 images_2 q14; sub kitchen 1 images_2 q16; sub garden 0 images_4 q15; sub bicycle 0 images_4 q16
echo "submitted $k of 4"
SUB
bash $AN/submit-adapt-v1.sh
squeue -u $USER -o "%.10i %.14j %.3t %.10M %.6D %R" | grep -E "adapt1|JOBID"
```

## §2 ba-sens v1：写文件 + 自检 + 提交 4 个 job（整块粘贴，任意节点）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/ba-sens-v1.sbatch <<'SBEOF_BAS1'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=basens
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/basens-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/basens-%x-%j.err
# ba-sens-v1.sbatch (2026-11-05, VERSION v1): price-constant sensitivity = rd-prune-v3c.sbatch with BPV from --export (B_a = BPV - 2 B_f: BPV 8 / 13 = B_a 4 / 9 vs the default 10 / 6),
#   INDOOR explicit (0 outdoor params, 1 --indoor), output dir /scratch/zr2245/$SCENE-rd3<i>-lam$L-mu$MU-bpv$BPV (never the default dirs), table $AN/ba-sens-v1.txt.
#   Reading: a price error c is identical to lambda -> c*lambda, so the bpv8 / bpv13 points should fall ON the scene's lambda hull, not off it.
# rd-prune-v3c.sbatch (2026-11-01, VERSION v3c): = rd-prune-v3b.sbatch + INDOOR=1 passes --indoor to train_rd_v3.py (upstream update_indoor:
# feature_lr 0.004, add_percentage 1.27, iteration_mesh 12000, start_pruning 3000, prune_size 1300, depth priors off, ...), output dir
# /scratch/zr2245/$SCENE-rd3i-lam$L-mu$MU (outdoor-parameter runs in $SCENE-rd3-... are kept), table rd-prune3i-$SCENE.txt, smoke stage removed,
# XYZ must be explicit (qN).  Same lambda rule, same prune schedule (12000 14000 16000 18000: after the Delaunay rebuild at 11000), same codec.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0.00062 0.0185" via --export}; INDOOR=${INDOOR:?set INDOOR=0 (outdoor params) or 1 (--indoor) via --export}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/ba-sens-v1.txt; SUF=""; [ "$INDOOR" = "1" ] && SUF=i; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | indoor flag '$FLAG' | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3$SUF-lam$L-mu$MU-bpv$BPV; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU $FLAG: train+prune ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "lam $L mu $MU -> trained vertices $VB ($(date))"; grep -h "^\[RD prune" $G/train.log
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "FAILED: LEVEL-1.0 $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
print('$SCENE lam $L bpv $BPV bpf $BPF RD-GROWTH+PRUNE-mu$MU indoor=$INDOOR | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| $FLAG')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "ba-sens-v1 $SCENE bpv $BPV exit 0  $(date)"
SBEOF_BAS1
cat > $AN/submit-ba-sens-v1.sh <<'SUB'
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; RES=yw523-l40s; LAMS="0.0062 0.00062"
n=$(wc -l < $AN/ba-sens-v1.sbatch); [ "$n" = 61 ] || { echo "CHECK FAILED: ba-sens-v1.sbatch has $n lines, expected 61 (paste truncated?)"; exit 1; }
grep -q -- '-bpv$BPV' $AN/ba-sens-v1.sbatch || { echo "CHECK FAILED: output dir has no -bpv suffix"; exit 1; }
grep -q "'--indoor'" $AN/rdgrowth/train_rd_v3.py || { echo "CHECK FAILED: train_rd_v3.py has no --indoor flag"; exit 1; }
for f in $D/m360/kitchen/images_2 $D/m360/garden/images_4 /scratch/zr2245/kitchen-rd3i-lam0.0062-mu1.0/axisB/v1.0/deliver/summary.json /scratch/zr2245/garden-rd3-lam0.0062-mu1.0/axisB/v1.0/deliver/summary.json; do [ -e $f ] || { echo "CHECK FAILED: MISSING $f"; exit 1; }; done
for d in /scratch/zr2245/kitchen-rd3i-lam0.0062-mu1.0-bpv8 /scratch/zr2245/garden-rd3-lam0.0062-mu1.0-bpv13; do [ -e $d ] && echo "note: $d exists, its training will be skipped"; done
echo "CHECK OK"; k=0
for BPV in 8 13; do
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=kitchen,INDOOR=1,IMAGES=images_2,XYZ=q16,LAMS="$LAMS",BPV=$BPV,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=kitchen-bpv$BPV $AN/ba-sens-v1.sbatch) && { echo "submitted kitchen-bpv$BPV $j"; k=$((k+1)); } || echo "FAILED: sbatch kitchen bpv $BPV"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,LAMS="$LAMS",BPV=$BPV,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=garden-bpv$BPV $AN/ba-sens-v1.sbatch) && { echo "submitted garden-bpv$BPV $j"; k=$((k+1)); } || echo "FAILED: sbatch garden bpv $BPV"
done
echo "submitted $k of 4"
SUB
bash $AN/submit-ba-sens-v1.sh
squeue -u $USER -o "%.10i %.14j %.3t %.10M %.6D %R" | grep -E "lkfan2|basens|bpv|JOBID"
```

## §3 fan v2：写文件 + 自检 + 提交 8 个 job（整块粘贴，任意节点）


```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/lambda-k-fan-v2.sbatch <<'SBEOF_LKF2'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=lkfan2
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/lkfan-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/lkfan-%x-%j.err
# lambda-k-fan-v2.sbatch (2026-11-05, VERSION v2): = v1 + DATA override via --export (T&T: DATA=/scratch/zr2245/datasets/tandt/$SCENE, IMAGES=images)
#   + the delivered asset's XYZ (config.json) must equal xyz_of SCENE, else FAIL before any GPU work.  Same table file as v1 ($AN/lambda-k-fan-v1.txt).
# lambda-k-fan-v1.sbatch (2026-11-05, VERSION v1): Yao Q1/Q2 -- the joint lambda x dictionary-size RD fan, WITHOUT training.
# For one SCENE and every lambda asset $G = /scratch/zr2245/$SCENE-$TAG-lam$L-mu1.0 (level-1.0 protocol already run: axisB/v1.0/{finetune/best.pt, deliver/}):
#   (a) DC q8 instead of q7 on the delivered K4096 R256 asset      codec_variants const/ctx --dc q8   (seconds + test render)
#   (d) dictionary K x R in KRS instead of K4096 R256                deliver_coarse.sh on the same finetuned mesh (k-means 4 rounds + 400-step fit, 3-15 min each)
# Rows -> $AN/lambda-k-fan-v1.txt: scene | lam | lever | MB | test PSNR SSIM LPIPS | dPSNR / dMB vs that lambda's baseline (const/ctx q7 K4096R256).
# Restartable: existing sweep rows / summary.json are skipped.  48G: the dictionary fit holds a 2-3 M-vertex model.
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; TAG=${TAG:?set TAG=rd3i (indoor rerun) or rd3 (outdoor) via --export}; IMAGES=${IMAGES:-images_2}
LAMS=${LAMS:-"0.00062 0.00185 0.0062 0.0185"}; KRS=${KRS:-"4096 1|1024 256|4096 1024|16384 1024"}
AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/lambda-k-fan-v1.txt
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}
xyz_of () { case $1 in room) echo q14;; garden|bonsai|counter|flowers) echo q15;; kitchen|bicycle|stump|treehill) echo q16;; truck|train) echo q18;; *) echo "FAILED: no XYZ recorded for $1" >&2; return 1;; esac; }
XYZ=$(xyz_of $SCENE) || exit 1
echo "lambda-k-fan-v2 | scene $SCENE tag $TAG | DATA $DATA/$IMAGES | lams $LAMS | K x R $KRS | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/codec_variants.py $PKG/deliver_coarse.sh $PKG/stage2_run.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $DATA/$IMAGES/.; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
nf=0
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-$TAG-lam$L-mu1.0; d=$G/axisB/v1.0/deliver; MESH=$G/axisB/v1.0/finetune/best.pt; C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  miss=0; for f in $d/shared-final.pt $d/config.json $d/summary.json $MESH $C; do [ -e $f ] || { echo "FAILED: MISSING $f"; miss=1; }; done
  [ $miss = 0 ] || { nf=$((nf+1)); continue; }
  python -c "import json; c=json.load(open('$d/config.json')); k=c.get('k',4096); r=c.get('r',256); assert (k,r)==(4096,256), ('baseline delivery is not K4096 R256', k, r); assert c.get('xyz')=='$XYZ', ('delivered xyz', c.get('xyz'), 'differs from xyz_of', '$XYZ')" || { echo "FAILED: $SCENE lam $L baseline is not K4096 R256 at XYZ $XYZ (see assert above)"; nf=$((nf+1)); continue; }
  echo "=== $SCENE lam $L: baseline + (a) DC q8 on $d (XYZ $XYZ) ($(date +%H:%M)) ==="
  for V in "const/ctx q7" "const/ctx q8"; do set -- $V; W=$1; DC=$2
    if python -c "import json,sys; s=json.load(open('$d/codec-variants/sweep.json')); sys.exit(0 if any((r['weight']+'/'+r['index_codec'],r['xyz'],r['dc'])==('$W','$XYZ','$DC') and r.get('test') for r in s['rows']) else 1)" 2>/dev/null; then echo "exists: $W $XYZ/$DC"; continue; fi
    bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $d --xyz $XYZ --dc $DC --variants $W > $d/codec-variants-fan-${W/\//-}-$DC.log 2>&1 || { echo "FAILED: codec_variants $SCENE lam $L $W $XYZ/$DC"; tail -5 $d/codec-variants-fan-${W/\//-}-$DC.log; nf=$((nf+1)); }
  done
  echo "=== $SCENE lam $L: (d) dictionary re-fits on $MESH ($(date +%H:%M)) ==="
  OUT=$G/kfan-v1; mkdir -p $OUT
  echo "$KRS" | tr '|' '\n' | while read K R; do name=$OUT/k${K}_r${R}
    if [ -f $name/summary.json ]; then echo "exists: k${K}_r${R}"; continue; fi
    rm -rf $name; echo "--- K $K R $R $(date +%H:%M:%S)"
    bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --source $C --dataset $DATA --lab $LAB --images $IMAGES --xyz $XYZ --dc q7 \
      --historical-config none --k $K --r $R --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED: deliver_coarse $SCENE lam $L k${K}_r${R}"; tail -8 $name.log; }
    [ -f $name/summary.json ] || echo "FAILED: no summary for $SCENE lam $L k${K}_r${R}"
  done
  python - "$SCENE" "$L" "$d" "$XYZ" "$OUT" "$KRS" <<'PY' | tee -a $TAB
import json, sys, pathlib
S, L, d, xyz, out, krs = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], pathlib.Path(sys.argv[5]), sys.argv[6]
s = json.load(open(d / 'codec-variants' / 'sweep.json')); rows = {}
for r in s['rows']:
    if r['xyz'] == xyz and r.get('test'): rows[(r['weight'] + '/' + r['index_codec'], r['dc'])] = r
base = rows.get(('const/ctx', 'q7'))
if base is None: print(f"{S} | lam {L} | baseline MISSING"); sys.exit(0)
bp = base['test']['psnr']; bm = base['file_MB']
def line(name, mb, t): print(f"{S} | lam {L} | {name:22s} | {mb:7.3f} MB | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | dPSNR {t['psnr'] - bp:+.3f} | dMB {mb - bm:+.3f}")
line('K4096 R256 q7 (base)', bm, base['test'])
r = rows.get(('const/ctx', 'q8')); line('K4096 R256 q8', r['file_MB'], r['test']) if r else print(f"{S} | lam {L} | K4096 R256 q8: MISSING")
for kr in krs.split('|'):
    K, R = kr.split(); p = out / f'k{K}_r{R}' / 'summary.json'
    if p.exists(): j = json.load(open(p)); line(f'K{K} R{R} q7', j['file_MB'], j['delivery_test'])
    else: print(f"{S} | lam {L} | K{K} R{R} q7: MISSING")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
[ $nf = 0 ] && echo "lambda-k-fan-v2 $SCENE exit 0  $(date)" || { echo "lambda-k-fan-v2 $SCENE FINISHED WITH $nf FAILED items  $(date)"; exit 1; }
SBEOF_LKF2
cat > $AN/submit-fan-v2.sh <<'SUB'
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; RES=yw523-l40s
n=$(wc -l < $AN/lambda-k-fan-v2.sbatch); [ "$n" = 74 ] || { echo "CHECK FAILED: lambda-k-fan-v2.sbatch has $n lines, expected 74 (paste truncated?)"; exit 1; }
grep -q "<<'PY' | tee -a" $AN/lambda-k-fan-v2.sbatch || { echo "CHECK FAILED: heredoc/tee form"; exit 1; }
grep -q 'DATA=${DATA:-' $AN/lambda-k-fan-v2.sbatch || { echo "CHECK FAILED: no DATA override (v1 pasted instead of v2?)"; exit 1; }
scontrol show res $RES > /dev/null 2>&1 || { echo "CHECK FAILED: reservation $RES not visible"; exit 1; }
miss=0
for S in counter bonsai bicycle stump flowers treehill truck train; do
  case $S in counter|bonsai) TAG=rd3i; IM=images_2; DX="" ;; truck|train) TAG=rd3; IM=images; DX=",DATA=$D/tandt/$S" ;; *) TAG=rd3; IM=images_4; DX="" ;; esac
  for L in 0.00062 0.00185 0.0062 0.0185; do f=/scratch/zr2245/$S-$TAG-lam$L-mu1.0/axisB/v1.0/deliver/summary.json; [ -f $f ] || { echo "MISSING asset: $f"; miss=$((miss+1)); }; done
  case $S in truck|train) DD=$D/tandt/$S/$IM ;; *) DD=$D/m360/$S/$IM ;; esac; [ -d $DD ] || { echo "MISSING dataset: $DD"; miss=$((miss+1)); }
done
[ $miss = 0 ] || { echo "CHECK FAILED: $miss missing items above -- nothing submitted"; exit 1; }
echo "CHECK OK"; k=0
for S in counter bonsai bicycle stump flowers treehill truck train; do
  case $S in counter|bonsai) TAG=rd3i; IM=images_2; DX="" ;; truck|train) TAG=rd3; IM=images; DX=",DATA=$D/tandt/$S" ;; *) TAG=rd3; IM=images_4; DX="" ;; esac
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=$S,TAG=$TAG,IMAGES=$IM$DX --job-name=$S-lkfan2 $AN/lambda-k-fan-v2.sbatch) \
    && { echo "submitted $S-lkfan2 $j ($TAG $IM$DX)"; k=$((k+1)); } || echo "FAILED: sbatch $S"
done
echo "submitted $k of 8"
SUB
bash $AN/submit-fan-v2.sh
squeue -u $USER -o "%.10i %.14j %.3t %.10M %.6D %R" | grep -E "lkfan2|basens|JOBID"
```

## §4 看进度 / 结果（任意节点；早上贴我整个输出）

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.14j %.3t %.10M %.6D %R" | grep -E "adapt1|bpv|lkfan2|JOBID"; echo
for f in $AN/adapt1-*.out $AN/basens-*.out $AN/lkfan-*-lkfan2-*.out; do [ -f $f ] && { echo "== $f"; grep -E "^===|FAILED|exit 0|FINISHED|smoke" $f | tail -n 5; }; done 2>/dev/null
echo "===== adapt-v1"; cat $AN/adapt-v1.txt 2>/dev/null
echo "===== ba-sens"; cat $AN/ba-sens-v1.txt 2>/dev/null
echo "===== fan rows (new scenes)"; wc -l $AN/lambda-k-fan-v1.txt; grep -E "^(counter|bonsai|bicycle|stump|flowers|treehill|truck|train) " $AN/lambda-k-fan-v1.txt
```

早上贴 §4 输出：adapt 12 行 → 控制器命中率表 + λ 轨迹图 + 和固定 λ hull 的对比；ba-sens 8 行 → bpv8/13 点画到 hull 上；fan 48 行 → Table 6 / Fig 7 扩到 11 场景。
