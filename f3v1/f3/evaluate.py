"""Evaluation: image metrics on views with photos, depth error vs GT (synthetic), holes (uncovered pixels) anywhere."""
import numpy as np
import torch
from .optimize import ssim
from .cams import View

def get_lpips():
    try:
        import lpips
        fn = lpips.LPIPS(net='vgg', verbose=False).cuda().eval()
        return lambda a, b: float(fn(a.permute(2, 0, 1)[None] * 2 - 1, b.permute(2, 0, 1)[None] * 2 - 1))
    except Exception as e:
        print(f'  LPIPS unavailable ({type(e).__name__}: {e}); skipping LPIPS')
        return None

@torch.no_grad()
def eval_views(R, X, dc, rest, views, images=None, gt_depths=None, ss=4, lpips_fn=None, far_sky=None):
    Xt = torch.tensor(X, dtype=torch.float32, device='cuda'); rows = []
    for i, v in enumerate(views):
        o = R.render(Xt, dc, rest, v, ss=ss, depth=True)
        r = dict(hole=float(1 - o['cover_ss'].float().mean()))
        if images is not None:
            gt = torch.tensor(images[i], device='cuda').float() / 255.0; im = o['img'].clamp(0, 1)
            r['psnr'] = float(-10 * torch.log10(((im - gt) ** 2).mean())); r['ssim'] = float(ssim(im, gt))
            if lpips_fn is not None: r['lpips'] = lpips_fn(im, gt)
        if gt_depths is not None:
            g = torch.tensor(gt_depths[i], device='cuda'); d = o['depth']
            bad = torch.where(g > 0, (d - g).abs() / d.clamp_min(1e-6) > 0.05, d < 0.5 * far_sky)
            r['bad_depth'] = float(bad.float().mean())
        rows.append(r)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, rows

def random_poses(train_views, n, rad_frac=0.8, pitch=(-0.6, 0.6), seed=0):
    """random cameras inside the region spanned by the training cameras, looking anywhere (no photos exist there)."""
    rng = np.random.default_rng(seed)
    C = np.array([v.center for v in train_views]); c0 = C.mean(0)
    up = -np.mean([v.R[1] for v in train_views], 0); up /= np.linalg.norm(up)
    rad = np.median(np.linalg.norm(C - c0, axis=1)) * rad_frac
    h = (C - c0) @ up; hlo, hhi = np.percentile(h, 10), np.percentile(h, 90)
    e1 = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    v0 = train_views[0]; out = []
    for i in range(n):
        r = rad * np.sqrt(rng.uniform()); a = rng.uniform(0, 2 * np.pi)
        pos = c0 + r * (np.cos(a) * e1 + np.sin(a) * e2) + rng.uniform(hlo, hhi) * up
        yaw = rng.uniform(0, 2 * np.pi); pt = rng.uniform(*pitch)
        fwd = np.cos(pt) * (np.cos(yaw) * e1 + np.sin(yaw) * e2) + np.sin(pt) * up
        right = np.cross(fwd, up); right /= np.linalg.norm(right); dn = np.cross(fwd, right)
        Rm = np.stack([right, dn, fwd]); w2c = np.eye(4); w2c[:3, :3] = Rm; w2c[:3, 3] = -Rm @ pos
        out.append(View(f'rand_{i:03d}', 'rand', None, w2c, v0.K, v0.W, v0.H))
    return out
