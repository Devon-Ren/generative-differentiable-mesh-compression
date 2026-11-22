"""run_f3.py (f3v1): photos (+ per-view depth) -> closed free-space boundary -> photometric optimisation -> evaluation.
Every stage prints a line; any failed check raises BEFORE the expensive part (GPU-time rule).
examples:
  python scripts/run_f3.py --kind synth  --data $D/sponza-synth-v1 --depth gt            --out $O/f3-1
  python scripts/run_f3.py --kind synth  --data $D/sponza-synth-v1 --depth $O/da3-sponza.npz --out $O/f3-2
  python scripts/run_f3.py --kind colmap --data $DS/m360/room --images images_2 --depth $O/da3-room.npz --out $O/f3-3 --ms_ckpt <pt>
  add --smoke for a 2-minute end-to-end check (8 views, coarse grid, 60 iterations)."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from f3 import cams, carve, refine

ap = argparse.ArgumentParser()
ap.add_argument('--kind', choices=['synth', 'colmap'], required=True)
ap.add_argument('--data', required=True); ap.add_argument('--images', default='images_2')
ap.add_argument('--depth', required=True, help="'gt' (synthetic only) or an npz from scripts/depth_da3.py")
ap.add_argument('--out', required=True)
ap.add_argument('--grid', type=int, default=400, help='carving voxels along the longest box side')
ap.add_argument('--votes', type=int, default=2); ap.add_argument('--k', type=float, default=0.012)
ap.add_argument('--vmax', type=int, default=3_000_000); ap.add_argument('--iters', type=int, default=3000)
ap.add_argument('--ss_train', type=int, default=2); ap.add_argument('--ss_eval', type=int, default=4)
ap.add_argument('--lam', type=float, default=10.0); ap.add_argument('--lr_pos_vox', type=float, default=0.02)
ap.add_argument('--w_depth', type=float, default=0.5); ap.add_argument('--w_depth_final', type=float, default=0.1)
ap.add_argument('--n_rand', type=int, default=50); ap.add_argument('--ms_ckpt', default='')
ap.add_argument('--smoke', action='store_true')
a = ap.parse_args()
T = {}; t_all = time.time()
os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, 'run.log'), 'a')
def log(s):
    print(s, flush=True); logf.write(s + '\n'); logf.flush()
log(f'== run_f3 f3v1 {time.strftime("%Y-%m-%d %H:%M:%S")}  args {vars(a)}')

# ---------- 0. environment checks (fail before anything expensive) ----------
try:
    import torch, nvdiffrast.torch as dr
    assert torch.cuda.is_available(), 'CUDA not available (are you on a GPU node?)'
    log(f'[0] torch {torch.__version__}  cuda {torch.version.cuda}  gpu {torch.cuda.get_device_name(0)}  nvdiffrast OK')
except Exception as e:
    log(f'FAILED [0] environment: {type(e).__name__}: {e}'); sys.exit(2)
from f3.optimize import train
from f3.render import Renderer
from f3.evaluate import eval_views, random_poses, get_lpips

# ---------- 1. data ----------
t0 = time.time()
if a.kind == 'synth':
    views, gtd, meta = cams.load_synth(a.data)
    tr = [v for v in views if v.split == 'train']; te = [v for v in views if v.split == 'test']; off = [v for v in views if v.split == 'off']
else:
    views, P, obs = cams.load_colmap(a.data, a.images); gtd = None; meta = {}
    tr = [v for v in views if v.split == 'train']; te = [v for v in views if v.split == 'test']; off = []
if a.smoke: tr = tr[::max(1, len(tr) // 8)][:8]; te = te[:2]; off = off[:2]
if a.depth == 'gt':
    assert a.kind == 'synth', '--depth gt only exists for the synthetic dataset'
    idx = {v.name: i for i, v in enumerate([w for w in views if w.split == 'train'])}
    Dtr = [gtd['train'][idx[v.name]] for v in tr]
else:
    z = np.load(a.depth, allow_pickle=True); names = list(z['names']); dd = z['depth']
    miss = [v.name for v in tr if v.name not in names]
    if miss: log(f'FAILED [1] depth file has no depth for {len(miss)} train views, e.g. {miss[:3]}'); sys.exit(2)
    Dtr = [dd[names.index(v.name)].astype(np.float32) for v in tr]
    bad = [v.name for v, D in zip(tr, Dtr) if D.shape != (v.H, v.W)]
    if bad: log(f'FAILED [1] depth shape != image shape for {bad[:3]}'); sys.exit(2)
imgs_tr = [(np.asarray(v.image()) * 255 + 0.5).astype(np.uint8) for v in tr]
log(f'[1] views train {len(tr)} test {len(te)} off {len(off)}  image {tr[0].W}x{tr[0].H}  depth valid {np.mean([np.mean(D > 0) for D in Dtr]) * 100:.1f}%  ({time.time() - t0:.0f}s)')
T['load'] = time.time() - t0

# ---------- 2. carve Omega, closed boundary, refine ----------
t0 = time.time()
lo, hi = carve.scene_box(tr, Dtr)
grid = 120 if a.smoke else a.grid
vox = float((hi - lo).max() / grid)
om = carve.carve(tr, Dtr, lo, hi, vox, m_votes=1 if a.smoke else a.votes, log=log)
V, F = carve.extract(om, lo, vox)
lid = carve.on_box(V, lo, hi, vox)
log(f'[2] coarse closed surface {refine.topology(F, len(V))}  lid {lid.sum()}  box {lo.round(3)}..{hi.round(3)} vox {vox:.4f}')
V, F, lid = refine.build_surface(V, F, tr, Dtr, k=a.k, vmax=200_000 if a.smoke else a.vmax, frozen=lid, log=log)
tp = refine.topology(F, len(V)); log(f'[2] surface {tp}  ({time.time() - t0:.0f}s)')
if tp['boundary_edges'] != 0: log('FAILED [2] surface is not closed'); sys.exit(2)
diag = float(np.linalg.norm(hi - lo))
sky = a.kind == 'synth' and any((D == 0).any() for D in Dtr)
if sky:   # open sky: the lid is the infinity layer -> push it far along rays from the region centre
    c0 = np.array([v.center for v in tr]).mean(0); dl = V[lid] - c0
    V[lid] = c0 + dl / np.linalg.norm(dl, axis=1, keepdims=True) * 50 * diag
near, far = 1e-3 * diag, (200 if sky else 10) * diag
sky_col = None
if sky:
    px = np.concatenate([im[D == 0] for im, D in zip(imgs_tr, Dtr)]); sky_col = px.mean(0) / 255.0 if len(px) else None
C, cf = refine.init_colors(V, tr, Dtr, imgs_tr, sky_color=sky_col, lid=lid if sky else None)
log(f'[2] colours initialised from photos for {cf * 100:.1f}% of vertices')
T['surface'] = time.time() - t0
np.savez(os.path.join(a.out, 'surface_init.npz'), V=V, F=F, lid=lid, C=C)

# ---------- 3. GPU convention check (abort before training if projection / orientation disagree) ----------
dev = 'cuda'
R = Renderer(torch.tensor(F), near, far, dev)
Xt = torch.tensor(V, dtype=torch.float32, device=dev)
from f3.render import C0
dc0 = torch.tensor((C - 0.5) / C0, dtype=torch.float32, device=dev)
agree, agree_flip, ps, ps_flip = [], [], [], []
with torch.no_grad():
    for i in range(min(3, len(tr))):
        o = R.render(Xt, dc0, None, tr[i], ss=1, depth=True)
        d = o['depth'].cpu().numpy(); P_ = Dtr[i]; ok = (P_ > 0) & (d > 0)
        agree.append(np.mean(np.abs(d[ok] - P_[ok]) < 0.10 * P_[ok]))
        df = d[::-1]; okf = (P_ > 0) & (df > 0); agree_flip.append(np.mean(np.abs(df[okf] - P_[okf]) < 0.10 * P_[okf]))
        im = o['img'].clamp(0, 1).cpu().numpy(); g = imgs_tr[i] / 255.0
        ps.append(-10 * np.log10(np.mean((im - g) ** 2))); ps_flip.append(-10 * np.log10(np.mean((im[::-1] - g) ** 2)))
log(f'[3] convention check: depth agreement {np.mean(agree) * 100:.1f}% (y-flipped {np.mean(agree_flip) * 100:.1f}%), '
    f'init PSNR {np.mean(ps):.2f} (y-flipped {np.mean(ps_flip):.2f})')
if not (np.mean(agree) > 0.5 and np.mean(agree) > np.mean(agree_flip) + 0.1 and np.mean(ps) > np.mean(ps_flip)):
    log('FAILED [3] rendered depth/colour disagree with the photos: projection or orientation convention is wrong'); sys.exit(2)

# ---------- 4. optimise ----------
t0 = time.time()
imgs_gpu = [torch.tensor(im, device=dev) for im in imgs_tr]
priors = [torch.tensor(D, device=dev) for D in Dtr]
cfg = dict(near=near, far=far, lam=a.lam, lr_pos=a.lr_pos_vox * vox, lr_dc=0.01, lr_rest=0.005, lr_final_frac=0.1,
           iters=60 if a.smoke else a.iters, geom_from=0 if a.smoke else 200, ss_train=a.ss_train, w_ssim=0.2,
           w_depth=a.w_depth, w_depth_final=a.w_depth_final, cg_iters=20, cg_iters_x=10, log_every=20 if a.smoke else 250)
log(f'[4] optimise {cfg["iters"]} iterations, V {len(V)} F {len(F)}')
X, dc, rest, hist = train(V, F, lid if sky else np.zeros(len(V), bool), C, tr, imgs_gpu, priors, cfg, log=log)
T['optimise'] = time.time() - t0
del imgs_gpu, priors; torch.cuda.empty_cache()

# ---------- 5. evaluate ----------
t0 = time.time(); res = dict(args=vars(a), topology=tp, vox=vox, times={}, history=hist)
lp = None if a.smoke else get_lpips()
ss = 2 if a.smoke else a.ss_eval
def imgs(vs): return [(np.asarray(v.image()) * 255 + 0.5).astype(np.uint8) for v in vs]
if a.kind == 'synth':
    gi = {s: {v.name: i for i, v in enumerate([w for w in views if w.split == s])} for s in ('test', 'off', 'train')}
    res['test'], _ = eval_views(R, X, dc, rest, te, imgs(te), [gtd['test'][gi['test'][v.name]] for v in te], ss, lp, far * 0.25)
    res['off'], _ = eval_views(R, X, dc, rest, off, imgs(off), [gtd['off'][gi['off'][v.name]] for v in off], ss, lp, far * 0.25)
    res['train_sub'], _ = eval_views(R, X, dc, rest, tr[:10], imgs_tr[:10], [gtd['train'][gi['train'][v.name]] for v in tr[:10]], ss, None, far * 0.25)
else:
    res['test'], _ = eval_views(R, X, dc, rest, te, imgs(te), None, ss, lp)
    rp = random_poses(tr, 4 if a.smoke else a.n_rand)
    res['rand_ours'], _ = eval_views(R, X, dc, rest, rp, None, None, 1)
    if a.ms_ckpt:
        st = torch.load(a.ms_ckpt, map_location='cpu', weights_only=False)
        Vm = st['triangles_points'].float().numpy(); Fm = st['_triangle_indices'].long().numpy()
        Rm = Renderer(torch.tensor(Fm), near, far, dev)
        z3 = torch.zeros(len(Vm), 3, device=dev)
        res['rand_meshsplatting'], _ = eval_views(Rm, Vm, z3, None, rp, None, None, 1)
        log(f'[5] MeshSplatting checkpoint {a.ms_ckpt}: V {len(Vm)} F {len(Fm)}')
T['eval'] = time.time() - t0; T['total_without_eval'] = T['load'] + T['surface'] + T['optimise']; res['times'] = T
for k in ('test', 'off', 'train_sub', 'rand_ours', 'rand_meshsplatting'):
    if k in res: log(f'[5] {k:20s} ' + '  '.join(f'{m} {v:.4f}' for m, v in res[k].items()))
log(f'[5] times (s): ' + '  '.join(f'{k} {v:.0f}' for k, v in T.items()))
json.dump(res, open(os.path.join(a.out, 'result.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(a.out, 'surface_final.npz'), V=X.astype(np.float32), F=F.astype(np.int32),
                    dc=dc.cpu().numpy().astype(np.float16), rest=rest.cpu().numpy().astype(np.float16))
log(f'== DONE {a.out}  ({time.time() - t_all:.0f}s)')
