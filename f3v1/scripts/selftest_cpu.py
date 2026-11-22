"""CPU self-test of f3v1 (no GPU): camera conventions, COLMAP round trip, geometry chain on the synthetic dataset.
usage: python scripts/selftest_cpu.py <synthetic dataset dir> [--full]"""
import sys, os, time, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from f3 import cams, carve, refine

root = sys.argv[1]; full = '--full' in sys.argv
views, gt, meta = cams.load_synth(root)
tr = [v for v in views if v.split == 'train']; Dtr = list(gt['train'])
print(f'[1] loaded {len(views)} views, train {len(tr)}')

# [2] convention: back-project depth of view 0 and re-project into view 0 -> same pixels, same z
v = tr[0]; d, zs = v.rays(); D = Dtr[0].reshape(-1); ok = D > 0
X = v.center + d[ok] * (D[ok] / zs[ok])[:, None]
u, vv, z = v.project(X)
ys, xs = np.divmod(np.flatnonzero(ok), v.W)
err_px = max(np.abs(u - (xs + 0.5)).max(), np.abs(vv - (ys + 0.5)).max()); err_z = np.abs(z - D[ok]).max()
print(f'[2] reprojection max pixel err {err_px:.2e}, max z err {err_z:.2e}'); assert err_px < 1e-3 and err_z < 1e-3

# [3] cross-view consistency with GT depth: view-0 points seen by the most-overlapping other view agree with its depth
best = None
for j in range(1, len(tr)):
    uj, vj, zj = tr[j].project(X); ins = (zj > 0.1) & (uj >= 0) & (uj < tr[j].W) & (vj >= 0) & (vj < tr[j].H)
    if best is None or ins.sum() > best[0]: best = (ins.sum(), j, uj, vj, zj, ins)
n_, j, uj, vj, zj, ins = best
dj = Dtr[j][vj[ins].astype(int), uj[ins].astype(int)]
agree = np.mean(np.abs(dj - zj[ins]) < 0.02 * zj[ins])
print(f'[3] view0 points visible in view{j}: {n_}, depth agreement (2%): {agree * 100:.1f}% (occlusion/sky edges make this < 100%)')
assert agree > 0.5

# [4] COLMAP round trip
from f3 import colmap_rw as rw
from PIL import Image
tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, 'sparse', '0')); os.makedirs(os.path.join(tmp, 'images_2'))
camsd, imgsd, ptsd = {}, {}, {}
for i, w in enumerate(views[:12]):
    K = w.K
    camsd[i + 1] = rw.Camera(id=i + 1, model='PINHOLE', width=w.W * 2, height=w.H * 2, params=np.array([K[0, 0] * 2, K[1, 1] * 2, K[0, 2] * 2, K[1, 2] * 2]))
    q = rw.rotmat2qvec(w.R)
    imgsd[i + 1] = rw.Image(id=i + 1, qvec=q, tvec=w.t, camera_id=i + 1, name=f'{i:03d}.png', xys=np.zeros((1, 2)), point3D_ids=np.array([1]))
    Image.fromarray(np.zeros((w.H, w.W, 3), np.uint8)).save(os.path.join(tmp, 'images_2', f'{i:03d}.png'))
ptsd[1] = rw.Point3D(id=1, xyz=np.zeros(3), rgb=np.zeros(3, int), error=0.0, image_ids=np.array([1]), point2D_idxs=np.array([0]))
rw.write_cameras_binary(camsd, os.path.join(tmp, 'sparse/0/cameras.bin')); rw.write_images_binary(imgsd, os.path.join(tmp, 'sparse/0/images.bin'))
rw.write_points3D_binary(ptsd, os.path.join(tmp, 'sparse/0/points3D.bin'))
cv, P, obs = cams.load_colmap(tmp, 'images_2')
e = max(max(np.abs(a.w2c - b.w2c).max(), np.abs(a.K - b.K).max()) for a, b in zip(cv, views[:12]))
print(f'[4] COLMAP round trip max err {e:.2e}; splits {[c.split for c in cv[:9]]}'); assert e < 1e-6 and cv[0].split == 'test' and cv[1].split == 'train'

if not full: print('SELFTEST (quick) OK'); sys.exit(0)

# [5] geometry chain with GT depth
t0 = time.time()
lo, hi = carve.scene_box(tr, Dtr); vox = float(max((hi - lo).max() / 400, 0.02))
print(f'[5] box {lo.round(2)} .. {hi.round(2)}  vox {vox:.3f}')
om = carve.carve(tr, Dtr, lo, hi, vox, m_votes=2)
V, F = carve.extract(om, lo, vox); fz = carve.on_box(V, lo, hi, vox)
print(f'    coarse surface {refine.topology(F, len(V))}  lid vertices {fz.sum()}  ({time.time() - t0:.0f}s)')
V, F, fz = refine.build_surface(V, F, tr, Dtr, vmax=int(sys.argv[sys.argv.index('--vmax') + 1]) if '--vmax' in sys.argv else 2_000_000, frozen=fz)
print(f'    final {refine.topology(F, len(V))}  ({time.time() - t0:.0f}s)')
np.savez(os.path.join(tempfile.gettempdir(), 'f3_selftest_surface.npz'), V=V, F=F, frozen=fz, lo=lo, hi=hi, vox=vox)
print('SELFTEST (full) OK  surface saved for evaluation')
