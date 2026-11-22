"""make_sponza.py (f3v1): synthetic 'photo' dataset of Sponza with exact poses and GT z-depth.
Runs on CPU (trimesh + embree ray casting, textured). Output layout (read by f3/cams.py):
  <out>/images/<split>_<i>.png, <out>/depth.npz (<split>: float32 [N,H,W] z-depth, 0 = sky/infinite), <out>/cameras.json
cameras.json: {"meta": {...}, "views": [{"split","file","w2c" (4x4, OpenCV: x right, y down, z forward),"K" (3x3),"W","H"}]}"""
import numpy as np, sys, json, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'f2'))
from f2lib import GTScene, SKY
from PIL import Image

def look(pos, yaw, pitch):
    fwd = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    up = np.array([0, 1., 0]); right = np.cross(fwd, up); right /= np.linalg.norm(right); upv = np.cross(right, fwd)
    R = np.stack([right, -upv, fwd])          # rows: camera x (right), y (down), z (forward) in world
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = -R @ pos
    return w2c

def render(S, w2c, K, W, H, ss=2):
    R = w2c[:3, :3]; pos = -R.T @ w2c[:3, 3]
    xs = (np.arange(W * ss) + 0.5) / ss; ys = (np.arange(H * ss) + 0.5) / ss
    X, Y = np.meshgrid(xs, ys)
    dc = np.stack([(X - K[0, 2]) / K[0, 0], (Y - K[1, 2]) / K[1, 1], np.ones_like(X)], -1).reshape(-1, 3)   # camera rays, z = 1
    dw = dc @ R; n = np.linalg.norm(dw, axis=1, keepdims=True); d = dw / n
    t, rgb, _ = S.cast(pos, d)
    z = t / n[:, 0]                                       # z-depth = distance / |ray in camera units|
    img = rgb.reshape(H, ss, W, ss, 3).mean((1, 3))
    zz = z.reshape(H, ss, W, ss)
    fin = np.isfinite(zz)
    depth = np.where(fin.all((1, 3)), np.where(fin, zz, 0).mean((1, 3)), 0.0)  # pixels touching sky -> sky
    return img, depth.astype(np.float32)

def sample_views(S, c, region_r, n, rng, pitch_rng, height_jitter=0.3):
    out = []
    while len(out) < n:
        r = region_r * np.sqrt(rng.uniform()); a = rng.uniform(0, 2 * np.pi)
        pos = c + np.array([r * np.cos(a), rng.uniform(-height_jitter, height_jitter), r * np.sin(a)])
        # stay in free space: the segment c -> pos must be unobstructed (with 0.4 m clearance)
        d = pos - c; L = np.linalg.norm(d)
        if L > 1e-6:
            t, _, _ = S.cast(c, (d / L)[None])
            if np.isfinite(t[0]) and t[0] < L + 0.4: continue
        out.append((pos, rng.uniform(-np.pi, np.pi), rng.uniform(*pitch_rng)))
    return out

if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    S = GTScene(os.path.join(os.path.dirname(__file__), '..', '..', 'gt', 'Sponza.gltf'))
    c = np.array([0, 1.6, 0.]); region_r = 2.0
    W, H, fov = 640, 480, 70.0
    f = 0.5 * W / np.tan(np.radians(fov) / 2); K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.]])
    rng = np.random.default_rng(2026)
    splits = {'train': sample_views(S, c, region_r, 120, rng, (-0.35, 0.5)),
              'test': sample_views(S, c, region_r, 30, rng, (-0.35, 0.5)),
              'off': sample_views(S, c, region_r, 50, rng, (-1.0, 1.3))}   # off-trajectory: any direction in the region
    views = []; depths = {}
    t0 = time.time()
    for sp, vs in splits.items():
        D = []
        for i, (pos, yaw, pitch) in enumerate(vs):
            w2c = look(pos, yaw, pitch)
            img, dep = render(S, w2c, K, W, H)
            fn = f'images/{sp}_{i:03d}.png'
            Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(out, fn))
            D.append(dep); views.append(dict(split=sp, file=fn, w2c=w2c.tolist(), K=K.tolist(), W=W, H=H))
        depths[sp] = np.stack(D)
        print(f'{sp}: {len(vs)} views ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(os.path.join(out, 'depth.npz'), **depths)
    meta = dict(name='sponza-synth-v1', region_center=c.tolist(), region_radius=region_r, up=[0, 1, 0], sky_color=SKY.tolist(),
                scene_bounds=S.mesh.bounds.tolist(), note='GT z-depth; 0 = sky (infinite); images 2x2 supersampled')
    json.dump(dict(meta=meta, views=views), open(os.path.join(out, 'cameras.json'), 'w'), indent=1)
    print('done', out)
