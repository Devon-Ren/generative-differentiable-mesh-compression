"""Camera / dataset loading. Convention everywhere: w2c 4x4 OpenCV (x right, y down, z forward), K 3x3 in pixels of the
image actually used, pixel centre of column i at u = i + 0.5."""
import json, os
import numpy as np
from PIL import Image

class View:
    def __init__(self, name, split, path, w2c, K, W, H):
        self.name, self.split, self.path = name, split, path
        self.w2c = np.asarray(w2c, np.float64); self.K = np.asarray(K, np.float64); self.W, self.H = int(W), int(H)
        self.R = self.w2c[:3, :3]; self.t = self.w2c[:3, 3]
        self.center = -self.R.T @ self.t
    def image(self):
        im = Image.open(self.path).convert('RGB')
        assert im.size == (self.W, self.H), f'{self.path}: size {im.size} != ({self.W},{self.H})'
        return np.asarray(im, np.float32) / 255.0
    def project(self, X):
        """world points (N,3) -> pixel u, v and camera z"""
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2]; zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.K[0, 0] * Xc[:, 0] / zs + self.K[0, 2]; v = self.K[1, 1] * Xc[:, 1] / zs + self.K[1, 2]
        return u, v, z
    def rays(self):
        """unit world ray per pixel centre (H*W,3) and the camera-z of a unit-distance step along it"""
        xs = np.arange(self.W) + 0.5; ys = np.arange(self.H) + 0.5
        X, Y = np.meshgrid(xs, ys)
        dc = np.stack([(X - self.K[0, 2]) / self.K[0, 0], (Y - self.K[1, 2]) / self.K[1, 1], np.ones_like(X)], -1).reshape(-1, 3)
        n = np.linalg.norm(dc, axis=1, keepdims=True)
        return (dc / n) @ self.R, 1.0 / n[:, 0]

def load_synth(root):
    """dataset written by scripts/make_sponza.py -> views, gt depth dict {split: [N,H,W] float32, 0 = sky}, meta"""
    js = json.load(open(os.path.join(root, 'cameras.json')))
    views = [View(os.path.splitext(os.path.basename(v['file']))[0], v['split'], os.path.join(root, v['file']), v['w2c'], v['K'], v['W'], v['H'])
             for v in js['views']]
    dz = np.load(os.path.join(root, 'depth.npz'))
    depth = {k: dz[k].astype(np.float32) for k in dz.files}
    return views, depth, js['meta']

def load_colmap(root, images='images_2', llffhold=8):
    """Mip-NeRF 360 style scene. Split exactly as MeshSplatting/3DGS --eval: sort by image name, every llffhold-th = test.
    Returns views, sparse points (P,3), and for each view the ids of the sparse points it observes."""
    from .colmap_rw import read_cameras_binary, read_images_binary, read_points3D_binary
    sp = os.path.join(root, 'sparse', '0')
    cams = read_cameras_binary(os.path.join(sp, 'cameras.bin')); imgs = read_images_binary(os.path.join(sp, 'images.bin'))
    pts = read_points3D_binary(os.path.join(sp, 'points3D.bin'))
    ids = sorted(pts.keys()); idmap = {k: i for i, k in enumerate(ids)}
    P = np.array([pts[k].xyz for k in ids], np.float64)
    order = sorted(imgs.values(), key=lambda im: im.name)
    views, obs = [], []
    for idx, im in enumerate(order):
        cam = cams[im.camera_id]
        path = os.path.join(root, images, im.name)
        W, H = Image.open(path).size
        s = W / cam.width
        assert abs(H / cam.height - s) < 0.01, f'{im.name}: anisotropic resize {W}x{H} vs {cam.width}x{cam.height}'
        if cam.model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL'):
            fx = fy = cam.params[0]; cx, cy = cam.params[1], cam.params[2]
        elif cam.model in ('PINHOLE', 'OPENCV', 'RADIAL'):
            fx, fy, cx, cy = cam.params[:4]
        else:
            raise ValueError(f'camera model {cam.model} not supported')
        if cam.model not in ('SIMPLE_PINHOLE', 'PINHOLE'):
            print(f'WARNING: camera model {cam.model}: distortion ignored (Mip-360 images are undistorted)')
        K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]])
        R = im.qvec2rotmat(); w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = im.tvec
        split = 'test' if idx % llffhold == 0 else 'train'
        views.append(View(os.path.splitext(im.name)[0], split, path, w2c, K, W, H))
        obs.append(np.array([idmap[p] for p in im.point3D_ids if p >= 0 and p in idmap], int))
    return views, P, obs
