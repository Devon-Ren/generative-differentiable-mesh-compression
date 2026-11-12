#!/usr/bin/env python3
"""make_batch29_files_v1.py (2026-11-12): exact trimming.
Finding (batch28 logs): a collapse round delivers only 20-40 % of the pairs it plans.  The plan asks for `wanted` pairs, the candidates are the
4 x wanted cheapest edges, and the independent-set matching (no two collapses share a vertex) keeps a small part of them because cheap edges
cluster around the same vertices.  The controller therefore sees the excess, plans its removal, and removes a fraction; the growth aim had to
absorb the rest.
Change (--trim_exact, default 0 = unchanged):
  1 : per-round plan wanted = min(excess / rounds_left, excess) pairs, and the candidate threshold is raised (margin 4, 8, 16, ...) until the
      matching finds that many pairs (still at most prune_max_frac of the vertices per round);
  2 : the original wanted (max(plan, edges under lambda), capped by the whole excess), delivered in the same exact way (front-loaded).
Outputs: train_rd_adapt_v9b_abl3.py and ablation-b24-r4{,-blender}.sbatch with the variants
  exact   (--trim_exact 1, default aim)      exact25 (--trim_exact 1, aim 1.25)      exact35 (--trim_exact 1, aim 1.35)      exactfl (--trim_exact 2, default aim)
  python3 make_batch29_files_v1.py --train <train_rd_adapt_v9b_abl2.py> --sbatch <ablation-b24-r3.sbatch> --sbatch_blender <ablation-b24-r3-blender.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'train': '75e92ffc397018fe472fc97801d0b9f9', 'sbatch': '949703fb20c0b3c5ba58a8d3e53eb21c', 'blender': '69614021a3aaf74cc7f999697ed8be54'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
# ---- training script
s = pathlib.Path(a.train).read_text()
if md5(s) != IN['train']: sys.exit(f'FAILED: {a.train} md5 {md5(s)} != {IN["train"]}')
s = rep(s, "                if rd.get('hold_lambda'): wanted = float(n_lam); count_driven = False; cap_bound = False; thr = thr_lam",
        "                te = 0 if rd.get('hold_lambda') else int(rd.get('trim_exact') or 0)  # batch29\n"
        "                if te == 1: wanted = (min(desired, excess_pairs) if excess > 0 else 0.0); cap_bound = False  # batch29: the per-round plan, nothing else\n"
        "                if rd.get('hold_lambda'): wanted = float(n_lam); count_driven = False; cap_bound = False; thr = thr_lam", 'wanted')
s = rep(s, "                    pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac)\n",
        "                    pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac)\n"
        "                    if te:  # batch29: deliver the wanted pairs (the matching and the fold-over filter both lose pairs)\n"
        "                        goal = int(frac * V0); margin = 4.0; frac_eff = frac; tries = 1; first_pairs = int(ps['n_pairs']); n_fin = int(torch.isfinite(score).sum())\n"
        "                        while ps['n_pairs'] < 0.97 * goal and tries < 14:\n"
        "                            tries += 1\n"
        "                            if ps['n_pass_pairs'] < int(frac_eff * V0):  # the matching ran out of candidates: admit more edges\n"
        "                                margin *= 2.0; thr_m, k_m = rank_threshold(score, wanted, margin=margin)\n"
        "                                if thr_m <= thr:\n"
        "                                    if k_m >= n_fin - 1: break\n"
        "                                    continue\n"
        "                                thr = thr_m; count_driven = True\n"
        "                            else:  # enough matched pairs, the fold-over filter removed too many: ask the matching for more\n"
        "                                if frac_eff >= 0.45: break\n"
        "                                frac_eff = min(0.45, frac_eff * max(1.25, goal / max(ps['n_pairs'], 1)))\n"
        "                            pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac_eff)\n"
        "                        if int(pairs.shape[0]) > goal:  # keep the cheapest `goal` pairs\n"
        "                            Vn = int(edges.max()) + 1; key = edges[:, 0] * Vn + edges[:, 1]; order_e = torch.argsort(key); skey = key[order_e]\n"
        "                            pu = torch.minimum(pairs[:, 0], pairs[:, 1]).long(); pv = torch.maximum(pairs[:, 0], pairs[:, 1]).long(); pk = pu * Vn + pv\n"
        "                            pos = torch.searchsorted(skey, pk).clamp_max(skey.numel() - 1); sc = score[order_e[pos]]\n"
        "                            pairs = pairs[torch.argsort(sc)[:goal]]; ps['n_pairs'] = int(pairs.shape[0])\n"
        "                        ev.update(trim_exact=te, trim_goal=goal, trim_margin=margin, trim_frac_eff=round(frac_eff, 4), trim_tries=tries, trim_first_pairs=first_pairs)\n"
        "                        print(f\"[RD prune {iteration}] exact trim (mode {te}): goal {goal} pairs, first pass {first_pairs}, after {tries} tries (margin {margin:g}, pair cap {frac_eff:.3f} V) {ps['n_pairs']} pairs (matching found {ps['n_pass_pairs']}, flip-rejected {ps['n_flip_rejected']})\")\n", 'select loop')
s = rep(s, "    parser.add_argument('--hold_lambda', action='store_true',", "    parser.add_argument('--trim_exact', type=int, default=0, choices=[0, 1, 2], help='batch29: 1 = collapse the per-round plan exactly (candidate threshold raised until the matching delivers it); 2 = the original wanted count, delivered exactly; 0 = unchanged')\n    parser.add_argument('--hold_lambda', action='store_true',", 'argument')
s = rep(s, "rd=dict(hold_lambda=args.hold_lambda, lam=args.lam,", "rd=dict(trim_exact=args.trim_exact, hold_lambda=args.hold_lambda, lam=args.lam,", 'rd dict')
(out / 'train_rd_adapt_v9b_abl3.py').write_text(s); print(f'train_rd_adapt_v9b_abl3.py: {len(s.splitlines())} lines md5 {md5(s)}')
# ---- sbatch files
for key, src, name in [('sbatch', a.sbatch, 'ablation-b24-r4.sbatch'), ('blender', a.sbatch_blender, 'ablation-b24-r4-blender.sbatch')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]}')
    s = rep(s, "constlam|repeat|adaptcap via --export}", "constlam|repeat|adaptcap|exact|exact25|exact35|exactfl via --export}", 'ABL prompt')
    s = rep(s, "  adaptcap) PRUNE=\"\"; EXTRA=\"\"; DDMODE=importance ;;\n",
            "  adaptcap) PRUNE=\"\"; EXTRA=\"\"; DDMODE=importance ;;\n  exact) XARGS=\"--trim_exact 1\" ;;\n  exact25) XARGS=\"--trim_exact 1\"; OVER=1.25 ;;\n  exact35) XARGS=\"--trim_exact 1\"; OVER=1.35 ;;\n  exactfl) XARGS=\"--trim_exact 2\" ;;\n", 'exact cases')
    s = rep(s, "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl2.py;", "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl3.py;", 'TR')
    s = s.replace("VERSION abl24-r3", "VERSION abl24-r4").replace("VERSION abl24b-r3", "VERSION abl24b-r4")
    (out / name).write_text(s); print(f'{name}: {len(s.splitlines())} lines md5 {md5(s)}')
