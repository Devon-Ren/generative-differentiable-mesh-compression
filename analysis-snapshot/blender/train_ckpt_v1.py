"""train_ckpt_v1.py (VERSION train_ckpt_v1 2026-11-04): run an unmodified training script and save checkpoints at chosen iterations.

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

VERSION = 'train_ckpt_v1 2026-11-04'


def main():
    if '--' not in sys.argv:
        sys.exit('FAILED: usage: train_ckpt_v1.py --saves N... --script PATH -- <script args>')
    k = sys.argv.index('--')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saves', nargs='+', type=int, required=True)
    ap.add_argument('--script', required=True, help='upstream/train.py or analysis/rdgrowth/train_rd_v3.py')
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
            print(f'[ckpt] saved iteration {iteration} (state after {iteration - 1} steps) vertices {nv} -> {out}', flush=True)
        return r

    tm.TriangleModel.update_learning_rate = hooked
    sys.argv = [script] + rest
    runpy.run_path(script, run_name='__main__')
    missing = sorted(i for i in saves if not os.path.isfile(os.path.join(mp, 'point_cloud', f'iteration_{i}', 'point_cloud_state_dict.pt')))
    if missing:
        sys.exit(f'FAILED: checkpoints missing after training: {missing}')
    print(f'[ckpt] all {len(saves)} checkpoints present', flush=True)


if __name__ == '__main__':
    main()
