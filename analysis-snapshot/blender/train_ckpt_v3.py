"""train_ckpt_v3.py (VERSION train_ckpt_v3 2026-11-04): run an unmodified training script and save checkpoints at chosen iterations.

v3: optional --normals DIR attaches monocular normal maps (DIR/<image_name>.png, same encoding as the COLMAP path:
RGB/255*2-1) to the Blender-loaded TRAIN cameras by wrapping sceneLoadTypeCallbacks["Blender"], so the authors' Metric3D
normal prior also applies on NeRF-Synthetic (upstream's Blender reader never fills normal_map). Fails before training if no
camera received a map.

v2: also saves the model's per-face importance_score next to each checkpoint (importance_score.pt), so val_select_v2 can apply the
same final importance prune the training scripts apply before their own last save (lego: 2.98 M -> 1.32 M faces).

Why: upstream train.py accepts --save_iterations but never uses it (its only scene.save() is the final one), and our
train_rd_v3.py inherited that. The NeRF-Synthetic probe needs checkpoints every 1k from 12k for validation-split selection
(val_select_v1.py). No upstream file is edited: this wrapper hooks TriangleModel.update_learning_rate, which both training
loops call once per iteration with the iteration index, and saves through Scene.save (so the on-disk format is exactly the
one the delivery chain loads: point_cloud/iteration_N/point_cloud_state_dict.pt).

Note: the hook fires at the START of iteration N (before that iteration's optimizer step), so iteration_N holds the state after
N-1 steps; the "[ITER N] Evaluating" line in the log is after step N. Intermediate checkpoints also skip the final
importance-based triangle prune that the scripts apply before their own last save (a few % more faces; rendering unchanged).

Usage:
  python train_ckpt_v1.py --saves 12000 13000 ... --script /path/upstream/train.py      -- -s DATA -i train -r 1 --eval ... -m MODEL ...
  python train_ckpt_v1.py --saves 12000 13000 ... --script /path/rdgrowth/train_rd_v3.py -- --lab LAB -s DATA ... -m MODEL --lam ...
"""
import argparse, inspect, os, runpy, sys

VERSION = 'train_ckpt_v3 2026-11-04'


def main():
    if '--' not in sys.argv:
        sys.exit('FAILED: usage: train_ckpt_v3.py --saves N... --script PATH [--normals DIR] -- <script args>')
    k = sys.argv.index('--')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saves', nargs='+', type=int, required=True)
    ap.add_argument('--script', required=True, help='upstream/train.py or analysis/rdgrowth/train_rd_v3.py')
    ap.add_argument('--normals', default=None, help='dir of <image_name>.png normal maps to attach to Blender train cameras')
    a = ap.parse_args(sys.argv[1:k]); rest = sys.argv[k + 1:]
    script = os.path.abspath(a.script)
    if not os.path.isfile(script):
        sys.exit(f'FAILED: script not found: {script}')
    # model path and upstream dir from the script's own arguments
    mp = None
    for i, t in enumerate(rest):
        if t in ('-m', '--model_path') and i + 1 < len(rest):
            mp = os.path.abspath(rest[i + 1])
    if mp is None:
        sys.exit('FAILED: no -m/--model_path in the script arguments')
    if '--lab' in rest:
        up = os.path.join(os.path.abspath(rest[rest.index('--lab') + 1]), 'upstream')
    else:
        up = os.path.dirname(script)
    if not os.path.isfile(os.path.join(up, 'train.py')):
        sys.exit(f'FAILED: upstream train.py not found under {up}')
    src = open(script).read()
    if 'update_learning_rate(iteration)' not in src:
        sys.exit(f'FAILED: {script} does not call update_learning_rate(iteration) once per iteration; hook would never fire')
    sys.path.insert(0, os.path.join(up, 'submodules/effrdel/src')); sys.path.insert(0, up); os.chdir(up)
    import scene.triangle_model as tm
    from scene import Scene
    if not hasattr(tm.TriangleModel, 'update_learning_rate'):
        sys.exit('FAILED: TriangleModel has no update_learning_rate')
    ssrc = inspect.getsource(Scene.save)
    print(f'[ckpt] {VERSION} | saves {sorted(a.saves)} | model_path {mp} | script {script}\n[ckpt] Scene.save source:\n{ssrc}', flush=True)
    if 'model_path' not in ssrc:
        sys.exit('FAILED: Scene.save does not use self.model_path; hook cannot reuse it')

    class _Fake:  # stands in for the Scene object inside Scene.save: model_path + whatever attribute holds the model
        def __init__(self, model_path, model):
            self.model_path = model_path; self._model = model
        def __getattr__(self, name):
            return self._model

    saves = set(a.saves); done = set(); orig = tm.TriangleModel.update_learning_rate

    def hooked(self, iteration):
        r = orig(self, iteration)
        if iteration in saves and iteration not in done:
            done.add(iteration)
            os.makedirs(os.path.join(mp, 'point_cloud', f'iteration_{iteration}'), exist_ok=True)
            out = Scene.save(_Fake(mp, self), iteration)
            nv = int(self.vertices.shape[0]) if hasattr(self, 'vertices') else -1
            imp = getattr(self, 'importance_score', None)
            if imp is None:
                sys.exit('FAILED: model has no importance_score attribute; cannot reproduce the final importance prune')
            import torch
            torch.save(imp.detach().cpu().clone(), os.path.join(mp, 'point_cloud', f'iteration_{iteration}', 'importance_score.pt'))
            nf = int(self._triangle_indices.shape[0]); nkeep = int((imp.detach().cpu().reshape(-1) > 0.5).sum())
            print(f'[ckpt] saved iteration {iteration} (state after {iteration - 1} steps) vertices {nv} faces {nf} (importance>0.5: {nkeep}) -> {out}', flush=True)
        return r

    tm.TriangleModel.update_learning_rate = hooked

    if a.normals:
        import numpy as np
        from PIL import Image
        import scene.dataset_readers as dr
        ndir = os.path.abspath(a.normals)
        if not os.path.isdir(ndir):
            sys.exit(f'FAILED: normals dir not found: {ndir}')
        if 'Blender' not in dr.sceneLoadTypeCallbacks:
            sys.exit('FAILED: upstream has no Blender loader callback')
        orig_blender = dr.sceneLoadTypeCallbacks['Blender']

        def _with(obj, **kw):  # NamedTuple or dataclass
            if hasattr(obj, '_replace'):
                return obj._replace(**kw)
            import dataclasses
            return dataclasses.replace(obj, **kw)

        def blender_with_normals(*args, **kwargs):
            info = orig_blender(*args, **kwargs)
            cams, hit, miss = [], 0, []
            for ci in info.train_cameras:
                p = os.path.join(ndir, ci.image_name + '.png')
                if os.path.isfile(p):
                    nm = (np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0) * 2.0 - 1.0
                    W, H = ci.image.size if hasattr(ci.image, 'size') else (ci.width, ci.height)
                    if nm.shape[0] != H or nm.shape[1] != W:
                        sys.exit(f'FAILED: normal map {p} is {nm.shape[1]}x{nm.shape[0]} but image is {W}x{H}')
                    cams.append(_with(ci, normal_map=nm)); hit += 1
                else:
                    cams.append(ci); miss.append(ci.image_name)
            if hit == 0:
                sys.exit(f'FAILED: no normal map matched any train camera in {ndir} (expected <image_name>.png, e.g. r_0.png)')
            print(f'[ckpt] normal maps attached to {hit}/{len(cams)} train cameras from {ndir}' + (f'; missing {miss[:5]}...' if miss else ''), flush=True)
            return _with(info, train_cameras=cams)

        dr.sceneLoadTypeCallbacks['Blender'] = blender_with_normals
        cu_src = open(os.path.join(up, 'utils', 'camera_utils.py')).read()
        if 'normal_map' not in cu_src:
            sys.exit('FAILED: utils/camera_utils.py never passes normal_map into Camera; the hook would be dropped')
        print(f'[ckpt] Blender loader wrapped: normal_map from {ndir}', flush=True)
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'point_cloud_state_dict.pt')))
    if missing:
        sys.exit(f'FAILED: checkpoints missing after training: {missing}')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'importance_score.pt')))
    if missing:
        sys.exit(f'FAILED: importance_score.pt missing after training: {missing}')
    print(f'[ckpt] all {len(saves)} checkpoints present', flush=True)


if __name__ == '__main__':
    main()
