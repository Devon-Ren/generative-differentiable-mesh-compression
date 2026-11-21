"""F2 stage 1: star-shaped sphere maps. inside (Sponza, radial from c) and outside (helmet, radial towards o)."""
import numpy as np, sys, time, json
sys.path.insert(0, '.')
from f2lib import *; from sph import *
rng = np.random.default_rng(0)
FAR = 1e4

def jitter_dirs(D, ang, k):
    out = []
    for _ in range(k):
        j = D + rng.normal(size=D.shape) * ang * 0.5
        out.append(j / np.linalg.norm(j, axis=1, keepdims=True))
    return out

def build_inside(S, c, L, k=4):
    lev = icosphere_levels(L); D = lev[-1][0]; F = lev[-1][1]
    ang = 1.1 / (2 ** L)  # ~ edge angle at level L
    rho = np.zeros(len(D)); col = np.zeros((len(D), 3))
    for Dj in jitter_dirs(D, ang, k):
        t, rgb, _ = S.cast(c, Dj)
        rho += np.where(np.isfinite(t), 1 / np.maximum(t, 1e-3), 0.0); col += rgb
    rho /= k; col /= k
    return lev, D, F, rho, col

def X_inside(c, D, rho):
    r = np.where(rho > 1 / FAR, 1 / np.maximum(rho, 1e-12), FAR)
    return c + D * r[:, None]

def build_outside(S, o, L, R, k=4):
    lev = icosphere_levels(L); D = lev[-1][0]; F = lev[-1][1]
    ang = 1.1 / (2 ** L)
    rad = np.zeros(len(D)); col = np.zeros((len(D), 3))
    for Dj in jitter_dirs(D, ang, k):
        t, rgb, _ = S.cast(o + Dj * R, -Dj)       # outermost hit along the radial line
        rr = np.where(np.isfinite(t), R - t, 0.0)
        rad += np.maximum(rr, 0); col += rgb
    return lev, D, F, rad / k, col / k

def eval_views(S, M, views, W=160, H=120, ss=2):
    rows = []
    for pos, yaw, pitch in views:
        d = cam_rays(pos, yaw, pitch, W, H, ss=ss)
        tg, cg, _ = S.cast(pos, d); tr, cr = M.cast(pos, d)
        ig = np.where(np.isfinite(tg), 1 / tg, 0); ir = np.where(np.isfinite(tr) & (tr < FAR * 0.5), 1 / tr, 0)
        hole = np.mean(np.isfinite(tg) & ~np.isfinite(tr))
        bad = np.where(ig > 0, np.abs(ir - ig) / np.maximum(ig, 1e-9) > 0.05, ir > 1e-3)
        rows.append(dict(psnr=psnr(down(cg.reshape(H * ss, W * ss, 3), W, H, ss), down(cr.reshape(H * ss, W * ss, 3), W, H, ss)),
                         bad=float(bad.mean()), hole=float(hole)))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

def inside_views(S, c, r, n):
    out = []
    while len(out) < n:
        u = rng.normal(size=3); u[1] *= 0.3; u /= np.linalg.norm(u)
        t, _, _ = S.cast(c, u[None])
        rr = min(r, max(0.0, t[0] - 0.4)) if np.isfinite(t[0]) else r
        if r > 0 and rr < 0.8 * r: continue
        out.append((c + u * rr, rng.uniform(-np.pi, np.pi), rng.uniform(-0.3, 0.4)))
    return out

def outside_views(o, Dist, n):
    out = []
    for _ in range(n):
        u = rng.normal(size=3); u[1] = abs(u[1]) * 0.6; u /= np.linalg.norm(u)
        pos = o + u * Dist; f = o - pos; f /= np.linalg.norm(f)
        yaw = np.arctan2(f[0], f[2]); pitch = np.arcsin(f[1])
        out.append((pos, yaw + rng.normal() * 0.05, pitch + rng.normal() * 0.05))
    return out

if __name__ == "__main__":
    res = {}
    # ---------- inside: Sponza ----------
    S = GTScene('../gt/Sponza.gltf'); c = np.array([0, 1.6, 0.])
    radii = [0, 0.25, 0.5, 1, 2, 4]
    V_by_r = {r: inside_views(S, c, r, 12) for r in radii}
    for L in [5, 6, 7, 8]:
        t0 = time.time()
        lev, D, F, rho, col = build_inside(S, c, L)
        M = SphereMap(X_inside(c, D, rho), F, col)
        for r in radii:
            e = eval_views(S, M, V_by_r[r]); res[f'sponza L{L} r{r}'] = e
            print(f'sponza L{L} V={len(D)} r={r}: psnr {e["psnr"]:.2f} bad-depth {e["bad"]*100:.1f}% holes {e["hole"]*100:.2f}%', flush=True)
        print('  time', round(time.time() - t0, 1), flush=True)
        if L == 8: np.savez('sponza_L8.npz', rho=rho, col=col)
    # ---------- outside: helmet ----------
    H_ = GTScene('../gt/DamagedHelmet.glb'); o = H_.mesh.bounds.mean(0); R = 4.0
    VW = outside_views(o, 3.0, 24)
    for L in [5, 6, 7, 8]:
        lev, D, F, rad, col = build_outside(H_, o, L, R)
        M = SphereMap(o + D * rad[:, None], F, col)
        e = eval_views(H_, M, VW); res[f'helmet L{L}'] = e
        print(f'helmet L{L} V={len(D)}: psnr {e["psnr"]:.2f} bad-depth {e["bad"]*100:.1f}% holes {e["hole"]*100:.2f}%', flush=True)
    json.dump(res, open('s1_results.json', 'w'), indent=1)
