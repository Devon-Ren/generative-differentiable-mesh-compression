# speed-v4-v1: 训练提速（不改优化内容）—— profile 结论 + train_rd_v4 + Room 验证 job，2026-10-31

## §0 profile 结论（Room λ1.85e-3 最终 mesh，1.19 M 顶点 / 2.62 M 面，272 训练视角，1038×1557）

每步耗时（ms）：A 第 5000 步 34.6 ｜ B 第 15000 步 38.5 ｜ C 2x 65.1 ｜ D 4x 171.3 → 按步数加权 ≈ 31.8 min（用最终 mesh 算，前期偏高，是上界）。

| 项目 | 每步 | 整次训练 | 能否不改结果地去掉 |
|---|---|---|---|
| rasterizer backward kernel（renderCUDA 反向） | 8.6 / 8.6 / 26.7 / 102 ms | ≈ 13.6 min（43 %） | 不能去；只能重写 kernel（见 §4） |
| rasterizer forward（renderCUDA + preprocess + sort） | 7.7 / 7.7 / 28 / 112 ms | ≈ 10 min | 不能 |
| Adam（foreach，4 组 62 M 参数） | 6.6 ms 恒定 | 3.3 min（10 %） | fused Adam ≈ 2–2.5 ms，同一公式、只差浮点顺序 |
| 中途评估 training_report（39 test + 5 train 视角 × LPIPS-VGG，每 1000 步） | — | 未量到（我的计时漏了 no_grad 才 OOM）；按每视角 ≈ 0.1 s 估 3–5 min | 纯日志，可完全去掉 |
| 5 次 area 下采样（scale 1 时是恒等变换） | 2.3 + 1.8 ms（前+反向） | 1.4 min（前 20k 步） | 恒等池化跳过是逐位相同 |
| GT 图 H2D 拷贝（data_device cpu） | 1.8 ms | 0.9 min | `--data_device cuda`，4x 阶段峰值也只 10.8 GB |
| 66 次 zero-fill（rasterizer 输出/梯度缓冲） | 6 ms @2x，16 ms @4x | ≈ 1.8 min | 要改 rasterize_points.cu 的 zeros→empty，需逐个缓冲证明被完全覆盖，暂不动 |
| `get_features` 的 cat（dc+rest 228 MB）| 1.9 ms | 1 min | 要改 TriangleModel 的参数布局，暂不动 |
| SSIM：fused_ssim **没装**（import failed），在用 Python 版 | loss 段共 2.2–2.9 ms | < 1 min 可省 | 装 fused-ssim 会有微小浮点差；不值当 |
| 编译 | .so 含 sm_89 SASS（fat binary），无 JIT 问题 | — | 不用重编 |

**结论**：不动 CUDA 的话，能省的 ≈ 7–9 min / 约 40 min（中途评估 3–5 + Adam 2 + 池化 1.4 + H2D 0.9），约 20 %。超采样阶段（C+D）占 62 %，其中反向 kernel 一家 43 %，这是唯一的大头，见 §4。

**更正**：v1 的 OOM 是我在 report 计时里没套 `no_grad`（训练里 training_report 是在 no_grad 下调的），39 个视角的渲染图 + LPIPS 图都留着才爆。图驻留显存在 4x 阶段完全放得下（峰值 10.8 GB）。

## §1 v4 文件（任意节点，整块粘贴；v3 文件不动）

`rdgrowth/render_v4.py` = upstream render() 原文 + 两处：scale 1 时跳过 5 次恒等 area 池化（逐位相同）；没有单目深度图的相机不算 expected_depth 的池化（Mip-360/T&T 都没有）。
`rdgrowth/train_rd_v4.py` = v3 + 三处：`--eval_every N`（默认 0：只在 `--test_iterations` 7000/30000 评估）；用 render_v4；fused Adam（`--no_fused_adam` 还原）。生长/剪枝规则、损失、学习率、σ/opacity 日程全部不变。
`rd-prune-v4.sbatch` = rd-prune-v3b.sbatch 换 v4 trainer + `--data_device cuda`，输出到 `/scratch/zr2245/$SCENE-rd4-lam$L-mu$MU`，行写到 `rd-prune4-$SCENE.txt`，并打印 v4 和（若存在）v3 同 λ 的训练墙钟 `TRAINTIME`。

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/render_v4.py <<'PYEOF'
"""render_v4.py (VERSION render_v4 2026-10-31): upstream triangle_renderer.render with two cost-only changes.

Same outputs as upstream `render` (same tensors, same values) except:
  1. When the model is not supersampled (pc.scaling == 1) the five `F.interpolate(mode="area")` calls are skipped.
     An area/adaptive-average pool whose output size equals its input size averages exactly one element per output,
     so skipping it is bit-identical in forward and backward; it only removes 5 copies + 5 backward kernels per iteration.
  2. `expected_depth` (used only by the monocular-depth loss) is downsampled only when the camera has an
     `invdepthmap`; otherwise the key holds None.  Mip-360 / T&T cameras have none.
Everything else is the upstream code, verbatim, with its helpers imported from triangle_renderer.
"""
import math
import torch
import torch.nn.functional as F
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from scene.triangle_model import TriangleModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from triangle_renderer import transform_point_4x3, compute_image_2d_pytorch_exact

VERSION = 'render_v4 2026-10-31'


def _down(x, H_init, W_init, upsample):
    """area-downsample [C,H,W] -> [C,H_init,W_init]; identity when not supersampled (bit-identical, see module doc)."""
    if upsample == 1:
        return x
    return F.interpolate(x.unsqueeze(0), size=(H_init, W_init), mode="area").squeeze(0)


def render(viewpoint_camera, pc: TriangleModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    triangles_indices = pc.get_triangle_indices
    vertices = pc.get_vertices
    vertex_weights = pc.get_vertex_weight
    scaling = torch.zeros_like(triangles_indices[:, 0], dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda").detach()

    vertex_index = pc._triangle_indices.shape[0]

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H_init = int(viewpoint_camera.image_height)
    W_init = int(viewpoint_camera.image_width)

    upsample = pc.scaling

    H = upsample * H_init
    W = upsample * W_init

    raster_settings = TriangleRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    sigma = pc.get_sigma

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, scaling, allmap, max_blending, was_rendered = rasterizer(
        vertices=vertices,
        triangles_indices=triangles_indices,
        vertex_weights=vertex_weights.squeeze(),
        sigma=sigma,
        shs=shs,
        colors_precomp=colors_precomp,
        scaling=scaling,
    )

    radii = radii[:vertex_index]
    scaling = scaling[:vertex_index]
    max_blending = max_blending[:vertex_index]

    rendered_image_small = _down(rendered_image, H_init, W_init, upsample)

    V = vertices.shape[0]
    idx = triangles_indices[was_rendered > 0].reshape(-1).long()
    vertex_rendered = torch.zeros(V, device=triangles_indices.device, dtype=was_rendered.dtype)
    vertex_rendered.index_fill_(0, idx, 1)

    vertices_cam = transform_point_4x3(vertices, viewpoint_camera.world_view_transform,)
    vertex_depths_pytorch = vertices_cam[:, 2]
    image_2D_pytorch = compute_image_2d_pytorch_exact(vertices, viewpoint_camera.full_proj_transform, W_init, H_init)

    rets = {"render": rendered_image_small,
            "visibility_filter": radii > 0,
            "radii": radii,
            "scaling": scaling,
            "max_blending": max_blending,
            "vertex_depth_out": vertex_depths_pytorch,
            "image_2D": image_2D_pytorch,
            "vertex_rendered": vertex_rendered,
            "full_image": rendered_image,
            "render_normal_full": allmap[2:5],
            "triangle_was_rendered": was_rendered
            }

    render_alpha = _down(allmap[1:2], H_init, W_init, upsample)

    render_normal = _down(allmap[2:5], H_init, W_init, upsample)
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    render_depth_median = _down(allmap[5:6], H_init, W_init, upsample)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    if getattr(viewpoint_camera, "invdepthmap", None) is not None:
        render_expected_depth = _down(allmap[0:1], H_init, W_init, upsample)
        render_expected_depth = torch.nan_to_num(render_expected_depth, 0, 0)
    else:
        render_expected_depth = None

    render_id = allmap[6:7]

    surf_depth = render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_ids': render_id,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            "expected_depth": render_expected_depth,
            "depth_full": allmap[5:6]
    })

    return rets
PYEOF
cat > $AN/rdgrowth/train_rd_v4.py <<'PYEOF'
"""train_rd_v4.py (v4, 2026-10-31 = v3 + three speed-only changes; v3 kept as train_rd_v3.py; same growth/prune rules, same losses).

v4 changes nothing in what is optimised.  It removes work that does not feed the gradient:
  1. --eval_every N (default 0): upstream `training_report` evaluates every test view + 5 train views with LPIPS-VGG at
     every iteration % 1000 == 0 (30 times per run, during 2x/4x supersampling too).  v4 calls it only at --test_iterations
     (default 7000 30000) and, if N > 0, every N iterations.  Pure logging; results unchanged.
  2. render from rdgrowth/render_v4.py: skips the five identity area-downsamples while pc.scaling == 1 (bit-identical) and
     the unused expected-depth downsample when the camera has no monocular depth.
  3. fused Adam (torch.optim.Adam(..., fused=True), one kernel instead of ~6 foreach passes per step; --no_fused_adam
     restores the upstream optimizer).  Same update rule, eps 1e-15, same lrs; only the floating-point evaluation order
     differs, which is below the run-to-run noise the rasterizer's atomicAdd backward already has.
Use --data_device cuda in the launch command (training images + normal maps on the GPU; Room 272 views = 4.9 GB).


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
from render_v4 import render, VERSION as RENDER_VERSION  # noqa: E402  (upstream render minus identity downsamples)
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
    if rd['fused_adam']:  # same groups/lrs/eps, fused kernel (fresh group dicts: the old ones carry fused=None)
        groups = [{'params': g['params'], 'lr': g['lr'], 'name': g['name']} for g in triangles.optimizer.param_groups]
        triangles.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15, fused=True)
    print('[v4] render', RENDER_VERSION, '| fused_adam', rd['fused_adam'], '| eval_every', rd['eval_every'], '| test_iterations', testing_iterations)
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
    log.write(event='config', version='v4', lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'],
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
            if iteration in testing_iterations or (rd['eval_every'] > 0 and iteration % rd['eval_every'] == 0):
                # upstream evaluates inside only when iteration % 1000 == 0; test_iterations must be multiples of 1000
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
        json.dump(dict(version='v4', lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'], split_schedule=rd['split_schedule'],
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
    parser.add_argument('--eval_every', type=int, default=0, help='v4: also run the upstream test/train evaluation every N iterations (0 = only at --test_iterations)')
    parser.add_argument('--no_fused_adam', action='store_true', help='v4: use the upstream (foreach) Adam instead of fused Adam')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print('Optimizing ' + args.model_path, '| RD growth+prune v4 lam', args.lam, 'mode', args.dd_mode, 'norm', args.dd_norm, 'B/v', args.bytes_per_vertex,
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
                     eval_every=args.eval_every, fused_adam=not args.no_fused_adam,
                     allow_any_prune_iter=args.allow_any_prune_iter))
PYEOF
cat > $AN/rd-prune-v4.sbatch <<'SBEOF'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/rd4-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd4-%x-%j.err
# rd-prune-v4.sbatch (2026-10-31): = rd-prune-v3b.sbatch with the v4 trainer (rdgrowth/train_rd_v4.py: no per-1000 evaluation, render_v4, fused Adam)
# and --data_device cuda; outputs under /scratch/zr2245/$SCENE-rd4-lam$L-mu$MU, rows in analysis/rd-prune4-$SCENE.txt, train wall time printed.  XYZ must be explicit (qN).
# rd-prune-v3b.sbatch (2026-10-20, batch B): = rd-prune-v3.sbatch + DATA override (T&T), SMOKE=0 to skip the smoke stage.
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
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/rd-prune4-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v4.py
SMOKE=${SMOKE:-0}; EXTRA=${EXTRA:-}; DEV=${DEV:-cuda}
[ "$XYZ" = "auto" ] && { echo "XYZ=auto is not supported here: pass XYZ=qN (xyz_rule_v1.py)"; exit 1; }
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/render_v4.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
# --- smoke: 1300 iterations, one forced prune round at 1250 (mu 1e6 -> every edge is a candidate, the per-round cap binds),
#     then 50 more iterations of rendering/training on the collapsed mesh and the final save ---
if [ "$SMOKE" = "1" ]; then
SM=$AN/rd4-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
echo "=== smoke ($(date)) ==="
python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device $DEV -m $SM/original --iterations 1300 --test_iterations 1300 --lam 0.0062 --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
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
else echo "smoke skipped (SMOKE=0)"; fi
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd4-lam$L-mu$MU; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU: train+prune ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device $DEV -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF $EXTRA > $G/train.log 2>&1 \
      || { echo "TRAIN FAILED $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "lam $L mu $MU -> trained vertices $VB ($(date))"; grep -h "^\[RD prune\|^\[v4\]" $G/train.log
  python - <<PY
import json,os
for tag,p in (('v4','$G/original/rd_log.jsonl'),('v3','/scratch/zr2245/$SCENE-rd3-lam$L-mu$MU/original/rd_log.jsonl')):
    if os.path.exists(p):
        fin=[json.loads(l) for l in open(p) if '"final"' in l]
        print('TRAINTIME', tag, '$SCENE lam $L', 'train_wall_s', fin[-1]['elapsed_s'] if fin else None, 'min', round(fin[-1]['elapsed_s']/60,1) if fin else None)
PY
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
print('$SCENE lam $L RD-GROWTH+PRUNE-v4-mu$MU | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'cand%.3f' % e['cand_frac'], 'pairs', e['n_pairs'], 'cap', e['max_pairs'], 'flip', e['n_flip_rejected'],
                          'q50/thr %.2f' % (e['score_q50'] / e['thr'] if e['thr'] else float('nan')), 'sel', e['faces_removed_share_sel'], 'all', e['faces_removed_share_all']) for e in pr])
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune4 $SCENE exit 0  $(date)"
SBEOF
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
python -m py_compile $AN/rdgrowth/render_v4.py $AN/rdgrowth/train_rd_v4.py && bash -n $AN/rd-prune-v4.sbatch && grep -c "" $AN/rdgrowth/train_rd_v4.py && echo "v4 files written"
```

## §2 Room 验证 job（1 个 job ≈ 1 GPU-h：smoke 1300 步含强制剪枝一轮 + λ1.85e-3 完整训练 + level-1.0 protocol）

```bash
AN=/scratch/zr2245/analysis
sbatch --parsable --time=02:00:00 --export=ALL,SCENE=room,IMAGES=images_2,XYZ=q14,LAMS="0.00185",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15,SMOKE=1 --job-name=room-v4 $AN/rd-prune-v4.sbatch
```

跑完看（v3 行在 rd-prune3-room.txt，v4 行在 rd-prune4-room.txt，墙钟在 .out）：

```bash
AN=/scratch/zr2245/analysis
grep -h "lam 0.00185" $AN/rd-prune3-room.txt $AN/rd-prune4-room.txt; grep -h "TRAINTIME\|FAILED\|exit 0" $AN/rd4-room-v4-*.out
```

## §3 判据

- v4 行的 test PSNR 与 v3 行相差 ≤ 0.05 dB、delivered MB 相差 ≤ 2 %、顶点数同量级：通过（rasterizer 的 atomicAdd 反向本来就让 v3 两次运行逐位不同，这是本来就有的噪声级别）。通过后 hull-6 的 11 个 job 可以直接换 `rd-prune-v4.sbatch` 提（我再给 rd-hull6-v2 的提交块；输出目录 rd4-，画图脚本同时读 rd-prune3/4 两张表）。
- `TRAINTIME v4` 对 `TRAINTIME v3` 的差就是实际省下的分钟数（预期 7–9 min）。
- 若 PSNR 差 > 0.05 dB：先 `EXTRA=--no_fused_adam` 重跑一次（只剩逐位相同的三项），定位是不是 fused Adam。

## §4 唯一的大头：反向 kernel（要改 CUDA，先问你）

renderCUDA 反向每步 102 ms @4x / 27 ms @2x / 8.6 ms @1x ≈ 13.6 min/次（43 %）。它对每个像素-三角形对直接 `atomicAdd` 到全局梯度（backward.cu 里 21 处：颜色 3、深度 3+3、2D 顶点 6、法线/偏移 3、opacity 1……）。tile 内 256 个线程是按同一个三角形顺序遍历的，所以可以先在 warp 内用 shuffle 归约再做 1 次 atomic（gsplat / Taming-3DGS 的做法，backward 通常 1.5–2.5x）。代价：改 upstream 的 backward.cu（新建 submodule 副本 `diff-triangle-mesh-rasterization-v2`，不覆盖原 .so，trainer 加 `--rasterizer v2` 切换）+ 一次 Room 对照训练；浮点求和顺序会变（在你说的"小浮点差"范围内）。预期再省 5–7 min/次。工作量 1–2 天。要不要做，你定。
