"""nvdiffrast renderer for the closed surface: opaque z-buffer, per-vertex colour (SH degree 1), dr.antialias, ss x ss
supersampling. Clip transform maps OpenCV pixel coordinates exactly: ndc_x = 2u/W - 1, ndc_y = 2v/H - 1, so output row 0
is the top image row (nvdiffrast row 0 = ndc y = -1)."""
import torch
import torch.nn.functional as Fn
import nvdiffrast.torch as dr

C0 = 0.28209479177387814
C1 = 0.4886025119029199

class Renderer:
    def __init__(self, F, near, far, device='cuda'):
        self.ctx = dr.RasterizeCudaContext(device=device)
        self.F = F.to(device=device, dtype=torch.int32).contiguous()
        self.near, self.far, self.device = float(near), float(far), device
        self.topo = dr.antialias_construct_topology_hash(self.F)

    def cam(self, view):
        R = torch.tensor(view.R, dtype=torch.float32, device=self.device)
        t = torch.tensor(view.t, dtype=torch.float32, device=self.device)
        c = torch.tensor(view.center, dtype=torch.float32, device=self.device)
        return R, t, c

    def clip(self, X, view, ss):
        R, t, _ = self.cam(view)
        Xc = X @ R.T + t
        W, H = view.W * ss, view.H * ss
        fx, fy, cx, cy = view.K[0, 0] * ss, view.K[1, 1] * ss, view.K[0, 2] * ss, view.K[1, 2] * ss
        n, f = self.near, self.far
        z = Xc[:, 2]
        clip = torch.stack([(2 * fx / W) * Xc[:, 0] + (2 * cx / W - 1) * z,
                            (2 * fy / H) * Xc[:, 1] + (2 * cy / H - 1) * z,
                            (f + n) / (f - n) * z - 2 * f * n / (f - n),
                            z], 1)
        return clip, Xc

    @staticmethod
    def shade(X, cam_c, dc, rest):
        d = X - cam_c; d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        rgb = C0 * dc + 0.5
        if rest is not None:
            rgb = rgb - C1 * y * rest[:, 0] + C1 * z * rest[:, 1] - C1 * x * rest[:, 2]
        return rgb.clamp_min(0.0)

    def render(self, X, dc, rest, view, ss=2, depth=True, aa=True):
        clip, Xc = self.clip(X, view, ss)
        H, W = view.H * ss, view.W * ss
        rast, _ = dr.rasterize(self.ctx, clip[None].contiguous(), self.F, resolution=[H, W])
        _, _, c = self.cam(view)
        col = self.shade(X, c, dc, rest)
        img, _ = dr.interpolate(col[None].contiguous(), rast, self.F)
        if aa: img = dr.antialias(img, rast, clip[None].contiguous(), self.F, topology_hash=self.topo)
        out = {'cover_ss': rast[0, ..., 3] > 0}
        img = img[0].permute(2, 0, 1)[None]
        out['img'] = Fn.avg_pool2d(img, ss)[0].permute(1, 2, 0) if ss > 1 else img[0].permute(1, 2, 0)
        if depth:
            dep, _ = dr.interpolate(Xc[None, :, 2:3].contiguous(), rast, self.F)
            cov = (rast[..., 3:4] > 0).float()
            dep = dep * cov
            if ss > 1:
                num = Fn.avg_pool2d(dep[0].permute(2, 0, 1)[None], ss); den = Fn.avg_pool2d(cov[0].permute(2, 0, 1)[None], ss)
                out['depth'] = (num / den.clamp_min(1e-6))[0, 0]; out['cover'] = den[0, 0]
            else:
                out['depth'] = dep[0, ..., 0]; out['cover'] = cov[0, ..., 0]
        return out
