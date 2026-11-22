"""Photometric (+ depth prior) optimisation of the closed surface. Connectivity is never changed (the surface stays
closed). Positions use the Large-Steps reparametrisation u = (I + lam L) X with Adam on u (Nicolet et al. 2021)."""
import math, time
import numpy as np
import torch
import torch.nn.functional as Fn
from .render import Renderer, C0

def laplacian(F, nV, device):
    E = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).long()
    E = torch.unique(torch.sort(E, dim=1)[0], dim=0)
    i = torch.cat([E[:, 0], E[:, 1]]); j = torch.cat([E[:, 1], E[:, 0]])
    deg = torch.zeros(nV, device=device).index_add_(0, i, torch.ones_like(i, dtype=torch.float32))
    idx = torch.cat([torch.stack([i, j]), torch.arange(nV, device=device).repeat(2, 1)], 1)
    val = torch.cat([-torch.ones(len(i), device=device), deg])
    return torch.sparse_coo_tensor(idx, val, (nV, nV)).coalesce().to_sparse_csr()

def cg(matvec, b, x0, iters):
    x = x0.clone(); r = b - matvec(x); p = r.clone(); rs = (r * r).sum(0)
    for _ in range(iters):
        Ap = matvec(p); alpha = rs / (p * Ap).sum(0).clamp_min(1e-30)
        x = x + alpha * p; r = r - alpha * Ap
        rs_new = (r * r).sum(0)
        if float(rs_new.max()) < 1e-20: break
        p = r + (rs_new / rs.clamp_min(1e-30)) * p; rs = rs_new
    return x

def ssim(a, b, win=11, sigma=1.5):
    g = torch.exp(-((torch.arange(win, device=a.device) - win // 2) ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    a = a.permute(2, 0, 1)[None]; b = b.permute(2, 0, 1)[None]
    mu_a = Fn.conv2d(a, k, groups=3); mu_b = Fn.conv2d(b, k, groups=3)
    s_aa = Fn.conv2d(a * a, k, groups=3) - mu_a ** 2; s_bb = Fn.conv2d(b * b, k, groups=3) - mu_b ** 2
    s_ab = Fn.conv2d(a * b, k, groups=3) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    return m.mean()

def train(V, F, frozen, colors, views, images, priors, cfg, log=print):
    """V (nV,3) np, F (nF,3) np, frozen (nV,) bool, colors (nV,3) np in [0,1]; images: list of uint8 HxWx3 torch (cuda);
    priors: list of float HxW torch depth (0 = invalid) or None. Returns X (np), dc, rest (torch) and a history list."""
    dev = 'cuda'
    Ft = torch.tensor(F, device=dev)
    R = Renderer(Ft, cfg['near'], cfg['far'], dev)
    X = torch.tensor(V, dtype=torch.float32, device=dev)
    fz = torch.tensor(frozen, device=dev); Xfix = X[fz].clone()
    L = laplacian(Ft, len(V), dev); lam = cfg['lam']
    A = lambda x: x + lam * (L @ x)
    u = torch.nn.Parameter(A(X).detach())
    dc = torch.nn.Parameter(torch.tensor((colors - 0.5) / C0, dtype=torch.float32, device=dev))
    rest = torch.nn.Parameter(torch.zeros(len(V), 3, 3, device=dev))
    opt_u = torch.optim.Adam([u], lr=cfg['lr_pos'], betas=(0.9, 0.999))
    opt_c = torch.optim.Adam([{'params': [dc], 'lr': cfg['lr_dc']}, {'params': [rest], 'lr': cfg['lr_rest']}])
    n_it = cfg['iters']; hist = []; rng = np.random.default_rng(cfg.get('seed', 0)); t0 = time.time()
    for it in range(n_it):
        frac = it / max(n_it - 1, 1)
        decay = math.exp(math.log(cfg['lr_final_frac']) * frac)
        for g in opt_u.param_groups: g['lr'] = cfg['lr_pos'] * decay
        opt_c.param_groups[0]['lr'] = cfg['lr_dc'] * decay; opt_c.param_groups[1]['lr'] = cfg['lr_rest'] * decay
        geom_on = it >= cfg['geom_from']
        Xg = X.detach().requires_grad_(geom_on)
        vi = int(rng.integers(len(views))); view = views[vi]
        out = R.render(Xg, dc, rest, view, ss=cfg['ss_train'], depth=priors is not None)
        gt = images[vi].float() / 255.0
        l1 = (out['img'] - gt).abs().mean()
        loss = (1 - cfg['w_ssim']) * l1 + cfg['w_ssim'] * (1 - ssim(out['img'], gt))
        if priors is not None:
            P = priors[vi]; ok = (P > 0) & (out['cover'] > 0.99)
            w_d = cfg['w_depth'] * (1 - frac) + cfg['w_depth_final'] * frac
            if ok.any(): loss = loss + w_d * ((out['depth'][ok] - P[ok]).abs() / P[ok]).mean()
        opt_c.zero_grad(); loss.backward(); opt_c.step()
        if geom_on:
            gX = Xg.grad; gX[fz] = 0
            gu = cg(A, gX, torch.zeros_like(gX), cfg['cg_iters'])
            u.grad = gu; opt_u.step(); opt_u.zero_grad()
            with torch.no_grad():
                X = cg(A, u.detach(), X, cfg['cg_iters_x']); X[fz] = Xfix
        if it % cfg['log_every'] == 0 or it == n_it - 1:
            with torch.no_grad(): ps = -10 * torch.log10(((out['img'].clamp(0, 1) - gt) ** 2).mean())
            hist.append(dict(it=it, loss=float(loss), psnr_train_view=float(ps), cover=float(out['cover_ss'].float().mean()), t=time.time() - t0))
            log(f'    it {it:5d} loss {float(loss):.4f} train-view PSNR {float(ps):.2f} cover {hist[-1]["cover"]:.4f} ({hist[-1]["t"]:.0f}s)')
    return X.detach().cpu().numpy(), dc.detach(), rest.detach(), hist
