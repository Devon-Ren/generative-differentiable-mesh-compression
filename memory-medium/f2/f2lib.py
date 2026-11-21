"""F2 helper library (2026-11-21): load textured GT scenes, ray-cast renders, sphere maps."""
import numpy as np, trimesh
from PIL import Image
from trimesh.ray.ray_pyembree import RayMeshIntersector

SKY = np.array([0.62, 0.75, 0.92])

class GTScene:
    def __init__(self, path):
        sc = trimesh.load(path)
        Vs, Fs, UVs, Ms, mats = [], [], [], [], []
        off = 0
        for node in sc.graph.nodes_geometry:
            T, gname = sc.graph[node]
            g = sc.geometry[gname]
            v = trimesh.transform_points(g.vertices, T)
            uv = getattr(g.visual, 'uv', None)
            if uv is None: uv = np.zeros((len(v), 2))
            mat = g.visual.material
            img = getattr(mat, 'baseColorTexture', None)
            fac = np.asarray(getattr(mat, 'baseColorFactor', None) if getattr(mat, 'baseColorFactor', None) is not None else [255, 255, 255, 255], float)
            if fac.max() > 1.0: fac = fac / 255.0
            tex = np.asarray(img.convert('RGB'), np.float32) / 255.0 if img is not None else None
            mats.append((tex, fac[:3]))
            Vs.append(v); Fs.append(g.faces + off); UVs.append(uv)
            Ms.append(np.full(len(g.faces), len(mats) - 1))
            off += len(v)
        self.V = np.concatenate(Vs); self.F = np.concatenate(Fs); self.UV = np.concatenate(UVs)
        self.M = np.concatenate(Ms); self.mats = mats
        self.mesh = trimesh.Trimesh(self.V, self.F, process=False)
        self.rt = RayMeshIntersector(self.mesh)

    def cast(self, o, d):
        """first hit: t (inf if miss), rgb, normal (facing the ray)"""
        n = len(d); o = np.broadcast_to(o, d.shape).astype(np.float64)
        t = np.full(n, np.inf); rgb = np.tile(SKY, (n, 1)); nor = np.zeros((n, 3))
        loc, ir, it = self.rt.intersects_location(o, d, multiple_hits=False)
        if len(ir):
            t[ir] = np.einsum('ij,ij->i', loc - o[ir], d[ir])
            tri = self.V[self.F[it]]
            bc = trimesh.triangles.points_to_barycentric(tri, loc)
            uv = np.einsum('ij,ijk->ik', bc, self.UV[self.F[it]])
            col = np.zeros((len(ir), 3))
            for mi in np.unique(self.M[it]):
                s = self.M[it] == mi; tex, fac = self.mats[mi]
                if tex is None: col[s] = fac; continue
                h, w = tex.shape[:2]
                x = (np.mod(uv[s, 0], 1.0) * w).astype(int).clip(0, w - 1)
                y = (np.mod(uv[s, 1], 1.0) * h).astype(int).clip(0, h - 1)  # glTF: v down
                col[s] = tex[y, x] * fac
            rgb[ir] = col
            fn = self.mesh.face_normals[it]
            fn = fn * -np.sign(np.einsum('ij,ij->i', fn, d[ir]))[:, None]
            nor[ir] = fn
        return t, rgb, nor

class SphereMap:
    """closed genus-0 mesh (icosphere connectivity or refined) with vertex colours; far vertices = infinity."""
    def __init__(self, X, F, C):
        self.X, self.F, self.C = X, F, C
        self.mesh = trimesh.Trimesh(X, F, process=False)
        self.rt = RayMeshIntersector(self.mesh)
    def cast(self, o, d):
        n = len(d); o = np.broadcast_to(o, d.shape).astype(np.float64)
        t = np.full(n, np.inf); rgb = np.tile(SKY, (n, 1))
        loc, ir, it = self.rt.intersects_location(o, d, multiple_hits=False)
        if len(ir):
            t[ir] = np.einsum('ij,ij->i', loc - o[ir], d[ir])
            bc = trimesh.triangles.points_to_barycentric(self.X[self.F[it]], loc)
            rgb[ir] = np.einsum('ij,ijk->ik', bc, self.C[self.F[it]])
        return t, rgb

def cam_rays(pos, yaw, pitch, W=160, H=120, fov=70, ss=2, up=np.array([0, 1, 0.])):
    """pinhole rays, y-up world; ss x ss supersampling."""
    fwd = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    right = np.cross(fwd, up); right /= np.linalg.norm(right); upv = np.cross(right, fwd)
    f = 0.5 * W / np.tan(np.radians(fov) / 2)
    xs = (np.arange(W * ss) + 0.5) / ss - W / 2; ys = (np.arange(H * ss) + 0.5) / ss - H / 2
    X, Y = np.meshgrid(xs, ys)
    d = fwd[None] * f + right[None] * X.reshape(-1, 1) - upv[None] * Y.reshape(-1, 1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)

def down(img, W, H, ss):
    return img.reshape(H, ss, W, ss, -1).mean((1, 3))

def psnr(a, b):
    return -10 * np.log10(np.mean((a - b) ** 2) + 1e-12)
