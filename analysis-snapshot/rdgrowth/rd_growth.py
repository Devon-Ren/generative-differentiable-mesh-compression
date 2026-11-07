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
