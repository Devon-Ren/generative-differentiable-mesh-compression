#!/usr/bin/env python3
"""make_batch35_files_v1.py (2026-11-13): ablation of the two selection rules of the hold controller against MeshSplatting's importance score.
  --split_rank importance : the triangles to split are drawn by upstream's importance-weighted sampling (the code path of the batch24 'matched'
                            runs); how MANY are split still follows the same plan, and the residual is still accumulated for the collapse rounds.
  --trim_score importance : in the collapse rounds upstream's per-triangle importance (largest blending weight seen) takes the place of the
                            residual in the cost of an edge; the byte target, the fold-over test and the rounds are unchanged.
Outputs train_rd_adapt_v9b_abl8.py and ablation-b24-r9.sbatch with ABL = holdis (split by importance) and holdic (collapse cost by importance);
ABL = hold is unchanged, and with both options at their default 'rd' the training script behaves exactly as abl7.
  python3 make_batch35_files_v1.py --train <train_rd_adapt_v9b_abl7.py> --sbatch <ablation-b24-r8.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'train': '1eeb64ab5032e903f69ad1a21466d99f', 'sbatch': '82259b73557bfcdb760648f42fad08cc'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
s = pathlib.Path(a.train).read_text()
if md5(s) != IN['train']: sys.exit(f'FAILED: {a.train} md5 {md5(s)} != {IN["train"]}')
SEL = ("                    idx, st = rd_select(acc, triangles, lam_now, bytes_per_split, max_split,\n"
       "                                        triangles.size_probs_zero, triangles.size_probs_zero_image_space)\n")
s = rep(s, SEL, "                    acc_mode_keep = acc.mode\n"
        "                    if rd.get('split_rank') == 'importance': acc.mode = 'importance'  # batch35 ablation: upstream importance sampling picks the triangles; the planned count and the residual accumulation stay\n"
        + SEL + "                    acc.mode = acc_mode_keep\n", 'split rank')
DD = ("                dD = acc.distortion()\n"
      "                if dD.numel() != T0:\n"
      "                    raise RuntimeError(f'accumulator/face mismatch at prune {iteration}: {dD.numel()} vs {T0}')\n")
s = rep(s, DD, DD + "                if rd.get('trim_score') == 'importance':  # batch35 ablation: upstream's importance score takes the place of the residual in the collapse cost\n"
        "                    dD = triangles.importance_score.detach().reshape(-1).to(device=dD.device, dtype=dD.dtype)\n"
        "                    if dD.numel() != T0:\n"
        "                        raise RuntimeError(f'importance/face mismatch at prune {iteration}: {dD.numel()} vs {T0}')\n", 'trim score')
s = rep(s, "    parser.add_argument('--trim_margin', type=float, default=0.0,",
        "    parser.add_argument('--split_rank', choices=['rd', 'importance'], default='rd', help='batch35 ablation: importance = upstream importance sampling picks the triangles to split (count plan unchanged)')\n"
        "    parser.add_argument('--trim_score', choices=['rd', 'importance'], default='rd', help='batch35 ablation: importance = upstream importance score replaces the residual in the collapse cost')\n"
        "    parser.add_argument('--trim_margin', type=float, default=0.0,", 'arguments')
s = rep(s, "rd=dict(trim_hold=args.trim_hold,", "rd=dict(split_rank=args.split_rank, trim_score=args.trim_score, trim_hold=args.trim_hold,", 'rd dict')
(out / 'train_rd_adapt_v9b_abl8.py').write_text(s); print(f'train_rd_adapt_v9b_abl8.py: {len(s.splitlines())} lines md5 {md5(s)}')
HOLD = "  hold) XARGS=\"--trim_exact 5 --trim_hold --allow_any_prune_iter\"; OVER=1.35; PRUNE=\"$(seq -s \" \" 12500 500 29500)\"; EXTRA=\"\" ;;\n"
NEW = HOLD + HOLD.replace("  hold) ", "  holdis) ").replace("--allow_any_prune_iter\"", "--allow_any_prune_iter --split_rank importance\"") \
           + HOLD.replace("  hold) ", "  holdic) ").replace("--allow_any_prune_iter\"", "--allow_any_prune_iter --trim_score importance\"")
s = pathlib.Path(a.sbatch).read_text()
if md5(s) != IN['sbatch']: sys.exit(f'FAILED: {a.sbatch} md5 {md5(s)} != {IN["sbatch"]}')
s = rep(s, "|ratio|ratio25|ratio35|final|hold via --export}", "|ratio|ratio25|ratio35|final|hold|holdis|holdic via --export}", 'ABL prompt')
s = rep(s, HOLD, NEW, 'hold cases')
s = rep(s, "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl7.py;", "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl8.py;", 'TR')
s = s.replace("VERSION abl24-r8", "VERSION abl24-r9")
(out / 'ablation-b24-r9.sbatch').write_text(s); print(f'ablation-b24-r9.sbatch: {len(s.splitlines())} lines md5 {md5(s)}')
