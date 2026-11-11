#!/usr/bin/env python3
"""make_batch24r2_files_v1.py (2026-11-11): round 2 of batch24 on the remaining reservation hours.
  python3 make_batch24r2_files_v1.py --abl_train <train_rd_adapt_v9b_abl.py> --abl_sbatch <ablation-b24.sbatch> --abl_sbatch_blender <ablation-b24-blender.sbatch> --out <dir>
Writes train_rd_adapt_v9b_abl2.py (= abl + an optional --seed that reseeds python/numpy/torch after upstream's safe_state; without the flag
nothing changes) and ablation-b24-r2.sbatch / ablation-b24-r2-blender.sbatch (= the batch24 sbatch files + the variant 'repeat': the full
method re-run with --seed SEED into $SCENE-abl24-repeat$SEED-t${T}mb, rows tagged abl24-repeat-s$SEED).  Inputs are md5-checked."""
import argparse, hashlib, pathlib, sys
IN = {'train': '2ae0a46b27aa135a02c1b56194e52e6d', 'sbatch': 'a823daddfa8374bffd8537e0ba896d75', 'blender': '741bd58b89a9e4b1e410b70a440e8896'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--abl_train', required=True); ap.add_argument('--abl_sbatch', required=True); ap.add_argument('--abl_sbatch_blender', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
t = pathlib.Path(a.abl_train).read_text()
if md5(t) != IN['train']: sys.exit(f'FAILED: {a.abl_train} md5 {md5(t)} != {IN["train"]} (not the batch24 file)')
t = rep(t, "    parser.add_argument('--hold_lambda', action='store_true',", "    parser.add_argument('--seed', type=int, default=None, help='batch24 round 2: reseed python / numpy / torch after upstream safe_state (which fixes seed 0); None = unchanged')\n    parser.add_argument('--hold_lambda', action='store_true',", 'seed arg')
t = rep(t, "    safe_state(args.quiet)\n", "    safe_state(args.quiet)\n    if args.seed is not None:\n        import random as _random, numpy as _np\n        _random.seed(args.seed); _np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); print('[RD] reseeded with', args.seed)\n", 'seed call')
(out / 'train_rd_adapt_v9b_abl2.py').write_text(t)
for key, src, name, dirtag in [('sbatch', a.abl_sbatch, 'ablation-b24-r2.sbatch', 'abl24'), ('blender', a.abl_sbatch_blender, 'ablation-b24-r2-blender.sbatch', 'abl24b')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]} (not the batch24 file)')
    s = rep(s, "ABL=${ABL:?set ABL=nocollapse|matched|constants|constlam via --export}", "ABL=${ABL:?set ABL=nocollapse|matched|constants|constlam|repeat via --export}", 'ABL prompt')
    s = rep(s, "  constants) PAE=0 ;;\n", "  constants) PAE=0 ;;\n  repeat) SEED=${SEED:?set SEED=<int> via --export}; XARGS=\"--seed $SEED\" ;;\n", 'repeat case')
    s = rep(s, "train_rd_adapt_v9b_abl.py;", "train_rd_adapt_v9b_abl2.py;", 'script path')
    s = rep(s, "elif [ \"$ABL\" = matched ]; then T=$RUN; L0=0; G=/scratch/zr2245/$SCENE-" + dirtag + "-matched-t${T}mb;", "elif [ \"$ABL\" = matched ]; then T=$RUN; L0=0; G=/scratch/zr2245/$SCENE-" + dirtag + "-matched-t${T}mb; elif [ \"$ABL\" = repeat ]; then T=$RUN; L0=$(lam0_of $T); G=/scratch/zr2245/$SCENE-" + dirtag + "-repeat$SEED-t${T}mb;", 'repeat dir')
    s = rep(s, "{('-lam' + L0) if os.environ['ABL'] == 'constlam' else ''}", "{('-lam' + L0) if os.environ['ABL'] == 'constlam' else ('-s' + os.environ.get('SEED', '')) if os.environ['ABL'] == 'repeat' else ''}", 'row tag')
    s = rep(s, "export ABL;", "export ABL; export SEED;", 'export seed')
    s = s.replace("VERSION abl24-r1", "VERSION abl24-r2").replace("VERSION abl24b-r1", "VERSION abl24b-r2")
    (out / name).write_text(s)
for f in ('train_rd_adapt_v9b_abl2.py', 'ablation-b24-r2.sbatch', 'ablation-b24-r2-blender.sbatch'):
    txt = (out / f).read_text(); print(f'{f}: {len(txt.splitlines())} lines md5 {md5(txt)}')
