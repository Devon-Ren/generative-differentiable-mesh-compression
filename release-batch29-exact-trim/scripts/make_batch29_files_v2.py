#!/usr/bin/env python3
"""make_batch29_files_v2.py (2026-11-12): exact trimming, second version.
What the first version showed (kitchen, 10 MB): raising the candidate threshold helps in the first rounds (39k pairs instead of 31k) and
hurts in the last ones (6k instead of 14k).  The fold-over filter tests all selected collapses together and rejects every selected pair
that touches a flipped face, so the more pairs are asked for at once, the more are rejected.  Many of the cheapest edges are also in the
part of the mesh that is not delivered and free no bytes.
Change (--trim_exact 3 or 4): the round asks for BYTES, not pairs.  select_feasible() ranks the edges that free delivered bytes by error per
freed byte and builds the set in batches; each batch is tested for fold-overs together with everything accepted so far, the offending pairs
are banned and the rest accepted, until the wanted bytes are reached (at most prune_max_frac of the vertices per round).
  3 : bytes wanted = excess / rounds left          4 : bytes wanted = excess / min(rounds left, 3)   (reacts faster to a late rise of the estimate)
Outputs train_rd_adapt_v9b_abl4.py and ablation-b24-r5{,-blender}.sbatch with ABL = feas | feasq (default growth aim), feas25 | feasq25 (aim 1.25), feasq35 (aim 1.35).
  python3 make_batch29_files_v2.py --train <train_rd_adapt_v9b_abl3.py> --sbatch <ablation-b24-r4.sbatch> --sbatch_blender <ablation-b24-r4-blender.sbatch> --feasible <select_feasible_v1.py> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'train': 'a30a25faa3c204d0551604ce78255122', 'sbatch': 'd16aa0d2f6fd254fa0b680c1f0c9043a', 'blender': '79f9ede7b9b0207563c471b486fd965c', 'feasible': '76bcd82359d901c2a8ad0111c3dfe6da'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--feasible', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
fs = pathlib.Path(a.feasible).read_text()
if md5(fs) != IN['feasible']: sys.exit(f'FAILED: {a.feasible} md5 {md5(fs)} != {IN["feasible"]}')
block = fs[fs.index('# >>> select_feasible'):fs.index('# <<< select_feasible')] + '# <<< select_feasible\n\n\n'
s = pathlib.Path(a.train).read_text()
if md5(s) != IN['train']: sys.exit(f'FAILED: {a.train} md5 {md5(s)} != {IN["train"]}')
s = rep(s, "from rd_prune import select_collapses, collapse_edges, edge_table  # noqa: E402", "from rd_prune import select_collapses, collapse_edges, edge_table, _face_normals  # noqa: E402", 'import')
s = rep(s, "def rank_threshold(score, n_wanted, margin=4.0):", block + "def rank_threshold(score, n_wanted, margin=4.0):", 'select_feasible block')
s = rep(s, "                if te == 1: wanted = (min(desired, excess_pairs) if excess > 0 else 0.0); cap_bound = False",
        "                if te >= 3: wanted = (max(1.0, min(desired, excess_pairs)) if excess > 0 else 0.0); cap_bound = False  # batch29 v2: any excess opens the round\n"
        "                if te == 1: wanted = (min(desired, excess_pairs) if excess > 0 else 0.0); cap_bound = False", 'wanted v2')
s = rep(s, "                    if te:  # batch29: deliver the wanted pairs", "                    if te in (1, 2):  # batch29: deliver the wanted pairs", 'old loop gate')
s = rep(s, "                    lam_sp = shadow_price(edges, score, pairs, rd['prune_mu'])\n",
        "                    if te >= 3:  # batch29 v2: byte-targeted, fold-over-safe selection on the delivered set\n"
        "                        first_pairs = int(ps['n_pairs']); want_bytes = excess / (rounds_left if te == 3 else min(rounds_left, 3)); cap_pairs = int(rd['prune_max_frac'] * V0)\n"
        "                        edges_f, sel_f, st_f = select_feasible(triangles.vertices.detach(), triangles._triangle_indices, dD, kept_v, kept_f, size.b_v, size.b_f, want_bytes, cap_pairs)\n"
        "                        if sel_f.numel() > 0:\n"
        "                            pairs = edges_f[sel_f]; ps['n_pairs'] = int(pairs.shape[0]); ps['n_flip_rejected'] = st_f['n_banned']; ps['max_pairs'] = cap_pairs; count_driven = True\n"
        "                        ev.update(trim_exact=te, trim_want_mb=round(want_bytes / 1e6, 4), trim_planned_mb=round(st_f['planned_bytes'] / 1e6, 4), trim_batches=st_f['iters'], trim_banned=st_f['n_banned'],\n"
        "                                  trim_first_pairs=first_pairs, trim_goal=int(st_f['n_pairs']), trim_priced=st_f['n_priced'], trim_dropped_batches=st_f['dropped_batches'])\n"
        "                        print(f\"[RD prune {iteration}] feasible trim (mode {te}): wanted {want_bytes / 1e6:.3f} MB, selected {st_f['n_pairs']} pairs freeing {st_f['planned_bytes'] / 1e6:.3f} MB in {st_f['iters']} batches ({st_f['n_banned']} pairs banned by the fold-over test, {st_f['n_priced']} of {st_f['n_edges']} edges free bytes); the original rule gave {first_pairs} pairs\")\n"
        "                    lam_sp = shadow_price(edges, score, pairs, rd['prune_mu'])\n", 'feasible call')
s = rep(s, "parser.add_argument('--trim_exact', type=int, default=0, choices=[0, 1, 2],", "parser.add_argument('--trim_exact', type=int, default=0, choices=[0, 1, 2, 3, 4],", 'choices')
(out / 'train_rd_adapt_v9b_abl4.py').write_text(s); print(f'train_rd_adapt_v9b_abl4.py: {len(s.splitlines())} lines md5 {md5(s)}')
for key, src, name in [('sbatch', a.sbatch, 'ablation-b24-r5.sbatch'), ('blender', a.sbatch_blender, 'ablation-b24-r5-blender.sbatch')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]}')
    s = rep(s, "exact|exact25|exact35|exactfl via --export}", "exact|exact25|exact35|exactfl|feas|feasq|feas25|feasq25|feasq35 via --export}", 'ABL prompt')
    s = rep(s, "  exactfl) XARGS=\"--trim_exact 2\" ;;\n", "  exactfl) XARGS=\"--trim_exact 2\" ;;\n  feas) XARGS=\"--trim_exact 3\" ;;\n  feasq) XARGS=\"--trim_exact 4\" ;;\n  feas25) XARGS=\"--trim_exact 3\"; OVER=1.25 ;;\n  feasq25) XARGS=\"--trim_exact 4\"; OVER=1.25 ;;\n  feasq35) XARGS=\"--trim_exact 4\"; OVER=1.35 ;;\n", 'feas cases')
    s = rep(s, "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl3.py;", "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl4.py;", 'TR')
    s = s.replace("VERSION abl24-r4", "VERSION abl24-r5").replace("VERSION abl24b-r4", "VERSION abl24b-r5")
    (out / name).write_text(s); print(f'{name}: {len(s.splitlines())} lines md5 {md5(s)}')
