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
