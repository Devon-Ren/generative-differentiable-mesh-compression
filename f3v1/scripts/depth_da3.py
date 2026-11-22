"""depth_da3.py (f3v1): pose-conditioned Depth Anything 3 depth for every TRAIN view -> npz {names, depth [N,H,W] float16, conf}.
Runs in the separate DA3 environment (do not install DA3 into the MeshSplatting env).
The poses are given to DA3 (pose-conditioned depth, align_to_input_ext_scale=True), so depths are in the poses' units.
Scale check: compare with sparse points (COLMAP points3D for real scenes; for the synthetic set an SfM stand-in = GT surface
points at 2000 random pixels per view). A global median correction is applied and printed (never per-view tuning).
usage: python scripts/depth_da3.py --kind colmap --data <scene> --images images_2 --out <npz> [--model depth-anything/DA3-LARGE-1.1]"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from f3 import cams

ap = argparse.ArgumentParser()
ap.add_argument('--kind', choices=['synth', 'colmap'], required=True); ap.add_argument('--data', required=True)
ap.add_argument('--images', default='images_2'); ap.add_argument('--out', required=True)
ap.add_argument('--model', default='depth-anything/DA3-LARGE-1.1'); ap.add_argument('--chunk', type=int, default=24)
ap.add_argument('--res', type=int, default=504); ap.add_argument('--limit', type=int, default=0, help='only first N views (smoke)')
a = ap.parse_args()
import torch, torch.nn.functional as Fn
from depth_anything_3.api import DepthAnything3
assert torch.cuda.is_available(), 'CUDA not available'
if a.kind == 'synth':
    views, gtd, _ = cams.load_synth(a.data); tr = [v for v in views if v.split == 'train']
    rng = np.random.default_rng(0); sfm = []
    for i, v in enumerate(tr):                       # SfM stand-in: surface points seen by this view
        D = gtd['train'][i]; ok = np.flatnonzero(D.reshape(-1) > 0); sel = rng.choice(ok, min(2000, len(ok)), replace=False)
        d, zs = v.rays(); sfm.append(v.center + d[sel] * (D.reshape(-1)[sel] / zs[sel])[:, None])
else:
    views, P, obs = cams.load_colmap(a.data, a.images); tr = [v for v in views if v.split == 'train']
    oi = {v.name: o for v, o in zip(views, obs)}; sfm = [P[oi[v.name]] for v in tr]
if a.limit: tr = tr[:a.limit]; sfm = sfm[:a.limit]
print(f'[da3] {len(tr)} train views {tr[0].W}x{tr[0].H}; model {a.model}', flush=True)
t0 = time.time()
model = DepthAnything3.from_pretrained(a.model).to('cuda').eval()
print(f'[da3] model loaded ({time.time() - t0:.0f}s)', flush=True)
depth = np.zeros((len(tr), tr[0].H, tr[0].W), np.float32); conf = np.zeros_like(depth)
for s in range(0, len(tr), a.chunk):
    ch = tr[s:s + a.chunk]
    with torch.no_grad():
        pred = model.inference([v.path for v in ch], extrinsics=np.stack([v.w2c for v in ch]).astype(np.float32),
                               intrinsics=np.stack([v.K for v in ch]).astype(np.float32), align_to_input_ext_scale=True,
                               process_res=a.res, process_res_method='upper_bound_resize')
    d = torch.tensor(np.asarray(pred.depth))[:, None]; c = torch.tensor(np.asarray(pred.conf))[:, None]
    depth[s:s + len(ch)] = Fn.interpolate(d, size=(ch[0].H, ch[0].W), mode='bilinear', align_corners=False)[:, 0].numpy()
    conf[s:s + len(ch)] = Fn.interpolate(c, size=(ch[0].H, ch[0].W), mode='bilinear', align_corners=False)[:, 0].numpy()
    print(f'[da3] views {s + len(ch)}/{len(tr)}  pred depth {tuple(d.shape[2:])}  ({time.time() - t0:.0f}s)', flush=True)
# scale check against sparse points
ratios = []
for i, (v, Pv) in enumerate(zip(tr, sfm)):
    if len(Pv) == 0: continue
    u, vv, z = v.project(Pv); ins = (z > 0) & (u >= 0) & (u < v.W) & (vv >= 0) & (vv < v.H)
    d = depth[i][vv[ins].astype(int), u[ins].astype(int)]; ok = d > 0
    if ok.sum() > 10: ratios.append(np.median(d[ok] / z[ins][ok]))
ratios = np.array(ratios); med = float(np.median(ratios))
print(f'[da3] scale vs sparse points: median {med:.4f}  p10 {np.percentile(ratios, 10):.4f}  p90 {np.percentile(ratios, 90):.4f}  (views {len(ratios)})')
if not (0.5 < med < 2.0): print('WARNING: DA3 depth scale far from the poses: check that extrinsics are w2c OpenCV')
depth /= med
np.savez_compressed(a.out, names=np.array([v.name for v in tr]), depth=depth.astype(np.float16), conf=conf.astype(np.float16), scale_fix=med, ratios=ratios)
print(f'[da3] DONE -> {a.out}  (global scale correction 1/{med:.4f}, {time.time() - t0:.0f}s)')
