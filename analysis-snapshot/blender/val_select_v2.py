"""val_select_v2.py (VERSION val_select_v2 2026-11-04): checkpoint selection by the OFFICIAL NeRF-Synthetic validation split.

v2: the chosen checkpoint is materialised as a REAL pruned state dict (not a symlink): faces with importance_score <= thresh
(importance_score.pt written by train_ckpt_v2) and the vertices they leave unused are dropped, reproducing the final prune the
training scripts apply before their last save. The pruned mesh is re-rendered on val and must stay within --prune_tol dB of
the unpruned one (self-check against an inverted mask).

Why: on NeRF-Synthetic (100 training views, no normal/depth priors in the Blender loader) MeshSplatting's test PSNR peaks
after the mesh stage and then drifts down while train PSNR keeps rising (lego: 32.9 @ 14k -> 28.8 @ 30k).  Early stopping
on the test split would be cheating; the dataset ships transforms_val.json (100 views), which upstream never reads.

What: builds a shadow dataset dir whose transforms_test.json is transforms_val.json (symlinks, nothing copied), loads each
saved checkpoint through upstream's own Scene(load_iteration=N) so no loading code is duplicated, renders the val views,
reports val PSNR/SSIM per iteration, picks the best, and (optionally) materialises <model>-best/original whose
point_cloud/iteration_30000 symlinks to the chosen checkpoint so the unchanged level-1.0 delivery chain can consume it.
Background is black (white_background=False), matching the delivery chain.

Usage (on a GPU node):
  python val_select_v2.py --lab $LAB --dataset $DATA --model $G/original --best $G-best/original [--delete_unselected]
"""
import argparse, json, os, re, shutil, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

VERSION = 'val_select_v2 2026-11-04'


def log(*a):
    print('[val_select]', *a, flush=True)


def make_shadow(dataset: Path, shadow: Path):
    """dataset dir with transforms_test.json -> transforms_val.json; everything else symlinked."""
    if shadow.exists():
        shutil.rmtree(shadow)
    shadow.mkdir(parents=True)
    for p in dataset.iterdir():
        if p.name == 'transforms_test.json':
            continue
        os.symlink(p.resolve(), shadow / p.name)
    os.symlink((dataset / 'transforms_val.json').resolve(), shadow / 'transforms_test.json')
    return shadow


def prune_state(sd, imp, thresh):
    """drop faces with importance <= thresh and vertices no kept face uses; every per-face / per-vertex tensor is filtered by shape."""
    faces = sd['_triangle_indices']; nf = int(faces.shape[0]); nv = int(sd['triangles_points'].shape[0])
    imp = imp.reshape(-1)
    if int(imp.shape[0]) != nf:
        sys.exit(f'FAILED: importance_score has {int(imp.shape[0])} entries, mesh has {nf} faces')
    if nf == nv:
        sys.exit('FAILED: faces == vertices; shape-based filtering is ambiguous')
    keep = imp > thresh
    used = torch.zeros(nv, dtype=torch.bool); used[faces[keep].reshape(-1).long()] = True
    remap = torch.full((nv,), -1, dtype=torch.long); remap[used] = torch.arange(int(used.sum()))
    out = {}
    for k, t in sd.items():
        if torch.is_tensor(t) and t.dim() >= 1 and k != '_triangle_indices':
            if int(t.shape[0]) == nf:
                out[k] = t[keep].clone()
            elif int(t.shape[0]) == nv:
                out[k] = t[used].clone()
            else:
                out[k] = t
        else:
            out[k] = t
    out['_triangle_indices'] = remap[faces[keep].long()].to(faces.dtype)
    assert int(out['_triangle_indices'].min()) >= 0
    return out, int(keep.sum()), int(used.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lab', required=True, help='root containing upstream/')
    ap.add_argument('--dataset', required=True, help='NeRF-Synthetic scene dir (transforms_{train,val,test}.json)')
    ap.add_argument('--model', required=True, help='model dir holding point_cloud/iteration_*/ (e.g. $G/original)')
    ap.add_argument('--iters', nargs='*', type=int, default=None, help='iterations to score (default: every saved iteration)')
    ap.add_argument('--min_iter', type=int, default=12000, help='ignore checkpoints before this (soft stage; not deliverable as an opaque mesh)')
    ap.add_argument('--supersample', type=str, default='20000:2,25000:4', help='render supersampling by iteration threshold, mirrors training (1 below the first threshold)')
    ap.add_argument('--best', type=str, default=None, help='write <best>/point_cloud/iteration_30000 -> chosen checkpoint (+ symlinks to the other model files)')
    ap.add_argument('--delete_unselected', action='store_true', help='after choosing, delete the other saved iterations except the chosen one and iteration_30000')
    ap.add_argument('--out', type=str, default=None, help='summary json (default <model>/val_select.json)')
    ap.add_argument('--importance_thresh', type=float, default=0.5, help='keep faces with importance_score > thresh (training scripts: <= 0.5 pruned)')
    ap.add_argument('--prune_tol', type=float, default=0.5, help='FAIL if the pruned mesh loses more than this on val (dB)')
    a = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error('CUDA required')
    lab = Path(a.lab).resolve(); upstream = lab / 'upstream'
    sys.path.insert(0, str(upstream / 'submodules/effrdel/src')); sys.path.insert(0, str(upstream))
    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from utils.loss_utils import ssim

    model = Path(a.model).resolve(); dataset = Path(a.dataset).resolve()
    pc = model / 'point_cloud'
    saved = sorted(int(m.group(1)) for d in pc.iterdir() for m in [re.fullmatch(r'iteration_(\d+)', d.name)] if m and (d / 'point_cloud_state_dict.pt').exists())
    iters = a.iters if a.iters else [i for i in saved if i >= a.min_iter]
    missing = [i for i in iters if i not in saved]
    if missing:
        sys.exit(f'FAILED: checkpoints missing for iterations {missing}; saved = {saved}')
    if not iters:
        sys.exit(f'FAILED: no checkpoint >= {a.min_iter}; saved = {saved}')
    sched = sorted((int(k), int(v)) for k, v in (kv.split(':') for kv in a.supersample.split(',') if kv))

    def ss_of(it):
        s = 1
        for thr, v in sched:
            if it >= thr:
                s = v
        return s

    shadow = make_shadow(dataset, model.parent / f'{model.name}-valshadow')
    data = SimpleNamespace(source_path=str(shadow), model_path=str(model), images='train', resolution=1, eval=True,
                           white_background=False, data_device='cpu', sh_degree=3)
    bg = torch.zeros(3, device='cuda'); pipe = SimpleNamespace(debug=False, convert_SHs_python=False, depth_ratio=1.)
    def eval_model(model_dir, it, ss):
        d = SimpleNamespace(**{**vars(data), 'model_path': str(model_dir)})
        mesh = TriangleModel(3)
        scene = Scene(d, mesh, None, None, load_iteration=it, shuffle=False)
        cams = sorted(scene.getTestCameras(), key=lambda c: (c.image_name, c.uid))
        mesh.scaling = ss
        ps, ss_ = [], []
        with torch.no_grad():
            for c in cams:
                im = render(c, mesh, pipe, bg)['render'].clamp(0, 1); gt = c.original_image[:3].cuda().clamp(0, 1)
                ps.append(float(-10 * torch.log10((im - gt).square().mean().clamp_min(1e-12)))); ss_.append(float(ssim(im, gt)))
        nv = int(mesh.vertices.shape[0]) if hasattr(mesh, 'vertices') else -1
        nf = int(mesh._triangle_indices.shape[0])
        del mesh, scene; torch.cuda.empty_cache()
        return dict(val_psnr=float(np.mean(ps)), val_ssim=float(np.mean(ss_)), worst_view_psnr=float(min(ps)), vertices=nv, faces=nf, views=len(cams))

    rows = []
    for it in iters:
        r = eval_model(model, it, ss_of(it))
        if it == iters[0]:
            log(f"val views: {r['views']} (from transforms_val.json via shadow dir {shadow})")
        row = dict(iteration=it, supersample=ss_of(it), **{k: r[k] for k in ('val_psnr', 'val_ssim', 'worst_view_psnr', 'vertices', 'faces')})
        rows.append(row)
        log(f"iter {it:6d}  ss {row['supersample']}  val PSNR {row['val_psnr']:.3f}  SSIM {row['val_ssim']:.4f}  worst {row['worst_view_psnr']:.2f}  V {row['vertices']} F {row['faces']}")
    best = max(rows, key=lambda r: r['val_psnr'])
    log(f"BEST iteration {best['iteration']}  val PSNR {best['val_psnr']:.3f}  (last saved {rows[-1]['iteration']}: {rows[-1]['val_psnr']:.3f})")
    summary = dict(version=VERSION, model=str(model), dataset=str(dataset), min_iter=a.min_iter, supersample=a.supersample, rows=rows,
                   best_iteration=best['iteration'], best_val_psnr=best['val_psnr'])
    out_path = Path(a.out) if a.out else model / 'val_select.json'
    out_path.write_text(json.dumps(summary, indent=1))
    log(f'wrote {out_path}')
    if a.best:
        b = Path(a.best).resolve()
        if b.exists():
            shutil.rmtree(b)
        b.mkdir(parents=True)
        for p in model.iterdir():
            if p.name in ('point_cloud', 'val_select.json'):
                continue
            os.symlink(p.resolve(), b / p.name)          # cfg_args, cameras.json, input.ply, train.log, rd_log.jsonl, ...
        src_dir = pc / f"iteration_{best['iteration']}"
        dst_dir = b / 'point_cloud' / 'iteration_30000'; dst_dir.mkdir(parents=True)
        imp_path = src_dir / 'importance_score.pt'
        if best['iteration'] == 30000 or not imp_path.exists():
            if best['iteration'] != 30000:
                sys.exit(f'FAILED: {imp_path} missing; train with train_ckpt_v2 so the final importance prune can be reproduced')
            for p in src_dir.iterdir():
                os.symlink(p.resolve(), dst_dir / p.name)
            log('best is the final checkpoint (already pruned by the training script): linked as is')
            pruned = None
        else:
            sd = torch.load(src_dir / 'point_cloud_state_dict.pt', map_location='cpu', weights_only=False)
            imp = torch.load(imp_path, map_location='cpu', weights_only=False)
            out, nf2, nv2 = prune_state(sd, imp, a.importance_thresh)
            torch.save(out, dst_dir / 'point_cloud_state_dict.pt')
            for p in src_dir.iterdir():
                if p.name not in ('point_cloud_state_dict.pt', 'importance_score.pt'):
                    os.symlink(p.resolve(), dst_dir / p.name)
            log(f"pruned best checkpoint: faces {best['faces']} -> {nf2}, vertices {best['vertices']} -> {nv2} (importance > {a.importance_thresh})")
            pruned = eval_model(b, 30000, ss_of(best['iteration']))
            log(f"pruned val PSNR {pruned['val_psnr']:.3f} (unpruned {best['val_psnr']:.3f})  V {pruned['vertices']} F {pruned['faces']}")
            if pruned['val_psnr'] < best['val_psnr'] - a.prune_tol:
                sys.exit(f"FAILED: pruned mesh lost {best['val_psnr'] - pruned['val_psnr']:.2f} dB on val (> {a.prune_tol}); check importance mask semantics")
        summary['pruned'] = pruned
        out_path.write_text(json.dumps(summary, indent=1))
        (b / 'VAL_SELECTED').write_text(f"{VERSION}\nsource {model}\niteration {best['iteration']}\nval_psnr {best['val_psnr']:.4f}\npruned {json.dumps(pruned)}\n")
        log(f"best model dir {b}: point_cloud/iteration_30000 = pruned iteration_{best['iteration']}")
    if a.delete_unselected:
        keep = {best['iteration'], 30000}
        for it in saved:
            if it not in keep:
                shutil.rmtree(pc / f'iteration_{it}')
        log(f'deleted unselected checkpoints; kept {sorted(keep)}')
    shutil.rmtree(shadow, ignore_errors=True)
    pr = summary.get('pruned')
    print(f"VAL_SELECT_RESULT {model.parent.name} best_iter {best['iteration']} val_psnr {best['val_psnr']:.3f} last_iter {rows[-1]['iteration']} last_val_psnr {rows[-1]['val_psnr']:.3f} pruned_val {pr['val_psnr']:.3f} pruned_faces {pr['faces']}" if pr else f"VAL_SELECT_RESULT {model.parent.name} best_iter {best['iteration']} val_psnr {best['val_psnr']:.3f} last_iter {rows[-1]['iteration']} last_val_psnr {rows[-1]['val_psnr']:.3f} pruned_val n/a pruned_faces {best['faces']}")


if __name__ == '__main__':
    main()
