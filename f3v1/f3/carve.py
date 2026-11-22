"""Free-space carving from per-view z-depth maps (numpy, no GPU needed) and closed-boundary extraction.
A voxel is free if at least m_votes views see through it (voxel in front of the observed surface by a margin).
Omega = the free component(s) that contain the training camera centres (definition of the medium)."""
import numpy as np
from scipy.ndimage import label, gaussian_filter
from skimage import measure

def scene_box(views, depths, q=(0.5, 99.5), pad_frac=0.03, max_px=20000, seed=0):
    rng = np.random.default_rng(seed); pts = [np.array([v.center for v in views])]
    for v, D in zip(views, depths):
        ok = np.flatnonzero(D.reshape(-1) > 0)
        if len(ok) == 0: continue
        sel = rng.choice(ok, min(max_px, len(ok)), replace=False)
        d, zstep = v.rays(); pts.append(v.center + d[sel] * (D.reshape(-1)[sel] / zstep[sel])[:, None])
    P = np.concatenate(pts)
    lo, hi = np.percentile(P, q[0], axis=0), np.percentile(P, q[1], axis=0)
    pad = pad_frac * (hi - lo).max()
    return lo - pad, hi + pad

def carve(views, depths, lo, hi, vox, m_votes=2, rel=0.02, sky_free=True, chunk=4_000_000, log=print):
    shape = np.maximum(np.ceil((hi - lo) / vox).astype(int), 2)
    n = int(np.prod(shape)); log(f'  carve grid {tuple(shape)} = {n / 1e6:.1f} M voxels of {vox:.4f}')
    count = np.zeros(n, np.uint8)
    for vi, (v, D) in enumerate(zip(views, depths)):
        for a in range(0, n, chunk):
            idx = np.arange(a, min(n, a + chunk))
            ijk = np.stack(np.unravel_index(idx, shape), 1)
            X = lo + (ijk + 0.5) * vox
            u, vv, z = v.project(X)
            ins = (z > 0.05) & (u >= 0) & (u < v.W) & (vv >= 0) & (vv < v.H)
            ii = np.flatnonzero(ins)
            d = D[vv[ii].astype(int), u[ii].astype(int)]
            free = np.where(d > 0, z[ii] < d * (1 - rel) - vox, sky_free)
            c = count[idx[ii[free]]]; count[idx[ii[free]]] = np.minimum(c.astype(np.int32) + 1, 255).astype(np.uint8)
        if vi % 20 == 0: log(f'    view {vi + 1}/{len(views)}')
    free = (count >= m_votes).reshape(shape)
    lab, nl = label(free)
    keep = set()
    for v in views:
        ijk = np.floor((v.center - lo) / vox).astype(int)
        if np.all(ijk >= 0) and np.all(ijk < shape) and lab[tuple(ijk)] > 0: keep.add(lab[tuple(ijk)])
    if not keep: raise RuntimeError('no camera centre lies in carved free space: check depths / poses')
    omega = np.isin(lab, list(keep))
    log(f'  free voxels {free.sum()}  components {nl}  kept (contain cameras) {len(keep)}  Omega voxels {omega.sum()}')
    return omega

def extract(omega, lo, vox, smooth=0.6):
    f = np.pad(omega, 1).astype(np.float32)            # solid border => closed surface
    if smooth > 0: f = gaussian_filter(f, smooth)
    V, F, _, _ = measure.marching_cubes(f, 0.5)
    V = (V - 1) * vox + lo
    # weld duplicate vertices (marching cubes output is already indexed; keep as is) and drop degenerate faces
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
    used = np.unique(F); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used].astype(np.float64), remap[F].astype(np.int64)

def on_box(V, lo, hi, vox, k=1.6):
    """vertices on the grid border (= where carving reached the end of the observed world: sky / infinity lid)"""
    return np.any((V < lo + k * vox) | (V > hi - k * vox), axis=1)
