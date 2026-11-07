"""profile_train_v1.py  (VERSION profile_train_v1 2026-10-31)

Where does a MeshSplatting RD-growth training iteration spend its time on the L40S?

Loads a trained mesh (so the per-iteration cost is that of the late, expensive phase) and replays the
exact per-iteration work of analysis/rdgrowth/train_rd_v3.py (render -> losses -> backward -> RD
accumulate -> Adam step) for a few dozen iterations in each training phase:

    A  it 5000   scaling 1   (growth phase: weight reg on, RD accumulate on, vertex-depth loss off)
    B  it 15000  scaling 1   (after Delaunay: vertex-depth loss on, RD accumulate on until the last prune)
    C  it 21000  scaling 2   (2x supersampling)
    D  it 26000  scaling 4   (4x supersampling)

For each phase it reports (1) the true per-iteration wall time without extra synchronisation and
(2) a synchronised breakdown: host->device copy of GT image + normal map, normal-map resize, render
forward, losses, backward, per-triangle stat updates, RD accumulate, loss.item() sync, optimizer step.
Then it repeats the phases with the training images resident on the GPU (what `--data_device cuda`
would do), times one `training_report` call (upstream evaluates ALL test views + LPIPS-VGG every
1000 iterations, 30 times per run), and prints the top CUDA kernels of one iteration per phase.

Nothing is written to the model directory; the model is not modified (no optimizer step is kept:
parameters are restored after each phase).  Output: PROFILE lines + a json.

  python profile_train_v1.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 \
      --model /scratch/zr2245/room-rd3-lam0.00185-mu1.0/original --dataset /scratch/zr2245/datasets/m360/room \
      --images images_2 --out /scratch/zr2245/analysis/profile-v1/room.json
"""
import argparse, json, os, sys, time
from pathlib import Path

VERSION = 'profile_train_v1 2026-10-31'


def sync():
    torch.cuda.synchronize()


class Seg:
    """synchronised segment timer: with Seg(d, 'name'): ...  adds wall ms to d['name']"""
    def __init__(self, d, name, on=True):
        self.d, self.name, self.on = d, name, on
    def __enter__(self):
        if self.on:
            sync(); self.t = time.perf_counter()
    def __exit__(self, *a):
        if self.on:
            sync(); self.d[self.name] = self.d.get(self.name, 0.0) + (time.perf_counter() - self.t) * 1e3


def iteration_step(it, cam, tri, pipe, bg, opt, acc, last_prune, run_delaunay, d=None, timed=False):
    """one training iteration exactly as train_rd_v3.py does it (minus logging / densify / prune)."""
    S = lambda name: Seg(d, name, timed)
    with S('render'):
        pkg = render(cam, tri, pipe, bg)
        image = pkg['render']
    with S('h2d'):
        gt = cam.original_image.cuda()
        gt_normal = cam.normal_map.cuda() if getattr(cam, 'normal_map', None) is not None else None
    with S('normal_resize'):
        if gt_normal is not None:
            gt_normal = F.interpolate(gt_normal.unsqueeze(0), size=(gt.shape[1], gt.shape[2]), mode='area').squeeze(0)
    with S('loss'):
        pixel_loss = l1_loss(image, gt)
        if up.FUSED_SSIM_AVAILABLE:
            ssim_value = up.fused_ssim(image.unsqueeze(0), gt.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt)
        loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)
        if it < opt.start_opacity_floor and opt.lambda_weight > 0:
            vw = tri.get_vertex_weight[:tri.vertices.shape[0]][tri._triangle_indices]
            loss = loss + opt.lambda_weight * vw.mean()
        if it > opt.start_vertex_opt and opt.lambda_vertex > 0:
            loss = loss + opt.lambda_vertex * vertex_depth_loss_hr(pkg['vertex_depth_out'], pkg['image_2D'], pkg['vertex_rendered'],
                                                                    pkg['surf_depth'], max_diff_threshold=opt.max_diff_threshold)
        if it > opt.iteration_mesh:
            rn, sn = pkg['rend_normal'], pkg['surf_normal']
            loss = loss + opt.lambda_normals * (1 - (rn * sn).sum(dim=0))[None].mean()
            if gt_normal is not None:
                loss = loss + opt.lambda_normals_super * (1 - (rn * gt_normal).sum(dim=0))[None].mean()
    with S('stats'):
        for key, buf in (('scaling', 'image_size'), ('max_blending', 'importance_score'), ('triangle_was_rendered', 'pixel_count')):
            v = pkg[key].detach(); b = getattr(tri, buf); m = v > b; b[m] = v[m]
    with S('backward'):
        loss.backward()
    with S('rd_accumulate'):
        if acc is not None and (it < run_delaunay or it <= last_prune):
            with torch.no_grad():
                acc.accumulate(pkg, image, gt, tri._triangle_indices)
    with S('loss_item'):
        float(loss.item())
    with S('optimizer'):
        tri.optimizer.step()
        tri.optimizer.zero_grad(set_to_none=True)
    return pkg


def snapshot(tri):
    return {k: getattr(tri, k).detach().clone() for k in ('vertices', 'vertex_weight', '_features_dc', '_features_rest')}


def restore(tri, snap):
    with torch.no_grad():
        for k, v in snap.items():
            getattr(tri, k).copy_(v)
    tri.optimizer.state.clear()


def run_phase(name, it, scaling, cams, tri, pipe, bg, opt, acc, last_prune, run_delaunay, n_warm, n_iters, dev_label):
    tri.scaling = scaling
    tri.active_sh_degree = tri.max_sh_degree
    tri.update_learning_rate(it)
    snap = snapshot(tri)
    cams = list(cams)
    pick = lambda k: cams[k % len(cams)]
    # warm-up (allocator, cub temp buffers, kernels)
    for k in range(n_warm):
        iteration_step(it, pick(k), tri, pipe, bg, opt, acc, last_prune, run_delaunay)
    sync()
    # (1) true throughput: no extra syncs
    t0 = time.perf_counter()
    for k in range(n_iters):
        iteration_step(it, pick(k), tri, pipe, bg, opt, acc, last_prune, run_delaunay)
    sync(); wall = (time.perf_counter() - t0) * 1e3 / n_iters
    # (2) synchronised breakdown
    d = {}
    for k in range(n_iters):
        iteration_step(it, pick(k), tri, pipe, bg, opt, acc, last_prune, run_delaunay, d=d, timed=True)
    d = {k: v / n_iters for k, v in d.items()}
    d_sum = sum(d.values())
    peak = torch.cuda.max_memory_allocated() / 2**30
    restore(tri, snap)
    if acc is not None:
        acc.reset(tri._triangle_indices.shape[0])
    row = dict(phase=name, iteration=it, scaling=scaling, data=dev_label, wall_ms=round(wall, 2), synced_sum_ms=round(d_sum, 2),
               peak_gb=round(peak, 2), **{k: round(v, 2) for k, v in d.items()})
    order = ['render', 'h2d', 'normal_resize', 'loss', 'stats', 'backward', 'rd_accumulate', 'loss_item', 'optimizer']
    print(f"PROFILE phase {name} it={it} scale={scaling} data={dev_label} wall_ms={wall:.1f} | " +
          ' '.join(f"{k}={d.get(k, 0):.1f}" for k in order) + f" | synced_sum={d_sum:.1f} peak_gb={peak:.2f}", flush=True)
    return row


def kernel_table(name, it, scaling, cams, tri, pipe, bg, opt, acc, last_prune, run_delaunay, rows=14):
    from torch.profiler import profile, ProfilerActivity
    tri.scaling = scaling; tri.update_learning_rate(it)
    snap = snapshot(tri)
    iteration_step(it, cams[0], tri, pipe, bg, opt, acc, last_prune, run_delaunay); sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for k in range(3):
            iteration_step(it, cams[k % len(cams)], tri, pipe, bg, opt, acc, last_prune, run_delaunay)
        sync()
    restore(tri, snap)
    if acc is not None:
        acc.reset(tri._triangle_indices.shape[0])
    print(f"\n=== top CUDA kernels, phase {name} (3 iterations, self CUDA time) ===")
    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=rows, max_name_column_width=70))
    out = []
    for e in sorted(prof.key_averages(), key=lambda e: -getattr(e, 'self_device_time_total', getattr(e, 'self_cuda_time_total', 0)))[:rows]:
        t = getattr(e, 'self_device_time_total', getattr(e, 'self_cuda_time_total', 0))
        out.append(dict(name=e.key[:90], self_cuda_ms_per_iter=round(t / 1e3 / 3, 3), calls=e.count))
    return out


def time_report(scene, tri, pipe, bg, scaling, it):
    tri.scaling = scaling
    cams = scene.getTrainCameras()
    pkg = render(cams[0], tri, pipe, bg)
    gt = cams[0].original_image.cuda()
    pl = l1_loss(pkg['render'], gt); loss = pl
    sync(); t0 = time.perf_counter()
    up.training_report(None, 'profile', it, pl, loss, l1_loss, 0.0, [it], scene, render, (pipe, bg))
    sync(); s = time.perf_counter() - t0
    n_test = len(scene.getTestCameras())
    print(f"PROFILE report scale={scaling} seconds={s:.1f} test_views={n_test} (+5 train views, LPIPS-VGG + SSIM + PSNR each)", flush=True)
    return dict(scaling=scaling, seconds=round(s, 2), test_views=n_test)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lab', required=True, help='root containing upstream/')
    p.add_argument('--model', required=True, help='model dir holding point_cloud/iteration_<it>/point_cloud_state_dict.pt')
    p.add_argument('--dataset', required=True); p.add_argument('--images', default='images_2')
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--rd_dir', default=None, help='dir with rd_growth_v2.py (default: <analysis>/rdgrowth next to this file, or $AN/rdgrowth)')
    p.add_argument('--phases', nargs='+', default=['A:5000:1', 'B:15000:1', 'C:21000:2', 'D:26000:4'], help='name:iteration:scaling')
    p.add_argument('--iters', type=int, default=30); p.add_argument('--warm', type=int, default=5)
    p.add_argument('--modes', nargs='+', default=['cpu', 'cuda'], help='training images on: cpu (current runs) and/or cuda')
    p.add_argument('--no_report', action='store_true'); p.add_argument('--no_kernels', action='store_true')
    p.add_argument('--prune_last', type=int, default=18000); p.add_argument('--out', required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        sys.exit('FAILED: CUDA required')
    print('VERSION', VERSION)

    upstream = Path(a.lab).resolve() / 'upstream'
    if not (upstream / 'train.py').exists():
        sys.exit(f'FAILED: no upstream/train.py under {a.lab}')
    rd_dir = a.rd_dir
    for cand in ([rd_dir] if rd_dir else []) + [str(Path(__file__).resolve().parent.parent / 'rdgrowth'),
                                                 os.path.join(os.environ.get('AN', '/scratch/zr2245/analysis'), 'rdgrowth')]:
        if cand and os.path.exists(os.path.join(cand, 'rd_growth_v2.py')):
            rd_dir = cand; break
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    if rd_dir:
        sys.path.insert(0, rd_dir)
    cwd = os.getcwd(); os.chdir(str(upstream))
    global up, render, l1_loss, ssim, vertex_depth_loss_hr, F
    import torch.nn.functional as F
    import train as up
    from arguments import ModelParams, PipelineParams, OptimizationParams
    from scene import Scene, TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import l1_loss, ssim, vertex_depth_loss_hr
    import lpips
    acc_cls = None
    if rd_dir:
        from rd_growth_v2 import RDAccumulator as acc_cls
    else:
        print('note: rd_growth_v2.py not found, RD accumulate segment skipped')

    ap = argparse.ArgumentParser()
    lp, op, pp = ModelParams(ap), OptimizationParams(ap), PipelineParams(ap)
    args = ap.parse_args(['-s', a.dataset, '-i', a.images, '-r', '1', '--eval', '--data_device', 'cpu', '-m', a.model])
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    tri = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, tri, opt.set_weight, opt.set_sigma, load_iteration=a.iteration, shuffle=False)
    tri.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    tri.set_sigma(0.0001 + (opt.set_sigma - 0.0001) * 0.5)  # mid-schedule sigma; cost does not depend on it
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device='cuda')
    up.lpips_fn = lpips.LPIPS(net='vgg').to(device='cuda')
    cams = scene.getTrainCameras()
    V, T = int(tri.vertices.shape[0]), int(tri._triangle_indices.shape[0])
    H, W = cams[0].image_height, cams[0].image_width
    dev = torch.cuda.get_device_properties(0)
    env = dict(version=VERSION, gpu=dev.name, sm=f'{dev.major}.{dev.minor}', torch=torch.__version__, cuda=torch.version.cuda,
               fused_ssim=bool(up.FUSED_SSIM_AVAILABLE), vertices=V, triangles=T, train_views=len(cams), test_views=len(scene.getTestCameras()),
               image=[int(H), int(W)], has_normal_map=bool(getattr(cams[0], 'normal_map', None) is not None),
               tf32_matmul=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)
    import diff_triangle_rasterization
    env['rasterizer_so'] = getattr(diff_triangle_rasterization._C, '__file__', '?')
    try:
        import fused_ssim_cuda as fs; env['fused_ssim_so'] = getattr(fs, '__file__', '?')
    except Exception:
        pass
    print('PROFILE env ' + ' '.join(f'{k}={v}' for k, v in env.items()), flush=True)
    run_delaunay = opt.densify_until_iter + 1000
    acc = acc_cls(T, mode='error', norm='mpx') if acc_cls else None
    phases = []
    for s in a.phases:
        n, it, sc = s.split(':'); phases.append((n, int(it), int(sc)))

    res = dict(env=env, phases=[], report=[], kernels={})
    for mode in a.modes:
        if mode == 'cuda':
            free, total = torch.cuda.mem_get_info()
            need = sum(c.original_image.numel() * 4 + (c.normal_map.numel() * 4 if getattr(c, 'normal_map', None) is not None else 0) for c in cams)
            print(f'moving {len(cams)} training views to the GPU: {need / 2**30:.1f} GB (free {free / 2**30:.1f} GB)', flush=True)
            if need > 0.8 * free:
                print('PROFILE note: not enough free GPU memory for data_device cuda, mode skipped'); continue
            for c in cams:
                c.original_image = c.original_image.cuda()
                if getattr(c, 'normal_map', None) is not None:
                    c.normal_map = c.normal_map.cuda()
        for n, it, sc in phases:
            torch.cuda.reset_peak_memory_stats()
            res['phases'].append(run_phase(n, it, sc, cams, tri, pipe, bg, opt, acc, a.prune_last, run_delaunay, a.warm, a.iters, mode))
    if not a.no_report:
        for sc in (1, 4):
            res['report'].append(time_report(scene, tri, pipe, bg, sc, 1000))
    if not a.no_kernels:
        for n, it, sc in phases:
            res['kernels'][n] = kernel_table(n, it, sc, cams, tri, pipe, bg, opt, acc, a.prune_last, run_delaunay)

    # whole-run estimate from the measured per-iteration costs (final mesh size => upper bound for the early phases)
    def est(mode):
        ph = {r['phase']: r for r in res['phases'] if r['data'] == mode}
        if not all(k in ph for k in 'ABCD'):
            return None
        counts = dict(A=11000, B=9000, C=5000, D=5000)
        iters_s = sum(ph[k]['wall_ms'] * counts[k] for k in counts) / 1e3
        rep = {r['scaling']: r['seconds'] for r in res['report']}
        rep_s = (20 * rep.get(1, 0) + 5 * 0.5 * (rep.get(1, 0) + rep.get(4, 0)) + 5 * rep.get(4, 0)) if rep else 0.0
        h2d_s = sum(ph[k].get('h2d', 0) * counts[k] for k in counts) / 1e3
        return dict(data=mode, iterations_min=round(iters_s / 60, 1), reports_min=round(rep_s / 60, 1), h2d_min=round(h2d_s / 60, 1),
                    total_min=round((iters_s + rep_s) / 60, 1))
    res['estimate'] = [e for e in (est(m) for m in a.modes) if e]
    for e in res['estimate']:
        print(f"PROFILE estimate data={e['data']} iterations={e['iterations_min']} min + 30 reports={e['reports_min']} min "
              f"= {e['total_min']} min (h2d copies inside iterations: {e['h2d_min']} min; densify/prune/Delaunay not included; "
              f"final-mesh cost used for every phase)", flush=True)
    res['complete'] = True
    os.chdir(cwd)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    print('PROFILE done ->', a.out)


if __name__ == '__main__':
    import torch
    main()
