# release-adapt-v9d — adaptive λ v9d: phase-B collapse priced on the delivered set (2026-11-08)

## Why (from the v9c test, release-adapt-v9c/results/)
v9c's slope clamp worked: the landing estimate never read below the probe and rose with the un-stacking kept set (kitchen 10: 9.4 → 11.3 MB, final 11.15). Phase B ran 15–24 of 30 rounds. Yet the five over-target rows only moved 1.061 / 1.092 / 1.135 / 1.079 / 1.100 → 1.028 / 1.051 / 1.116 / 1.067 / 1.075. The per-round accounting shows why: 4–8k pairs collapsed per round freed 0.002–0.03 MB each (kitchen 10: 109 724 pairs for 0.5 MB of a 1.3 MB excess); the both-kept share was 0.35–0.57 on Mip-360 and the measured bytes per pair 0.6–2.4 against the 6.3 prior. The cheapest dD/dR edges are the zero-residual ones: the triangles nobody renders, which the final cleaning drops anyway and which the byte probe already excludes. **Phase B was trimming geometry that costs no bytes.**

## What changed vs v9c
1. **`code/rdgrowth/rd_prune_v2.py`** (new file; `rd_prune.py` untouched): `select_collapses(..., kept_v=None, kept_f=None)`. With the probe's kept-vertex / kept-face masks, dR_uv = b_attr · [u and v both kept] + b_face · (kept faces on the edge); edges with dR_uv = 0 score +inf and are not candidates. Without masks it is `rd_prune.select_collapses` verbatim (unit-tested identical). Stats carry `n_priced`, `priced_frac`.
2. **`code/rdgrowth/train_rd_adapt_v9d.py`** (`build/make_v9d.py` from v9c): imports from `rd_prune_v2`; `edge_scores` uses the same price (so n_lam, the rank threshold and the shadow price see only priced edges); `select_collapses` gets the masks; bytes-per-collapse prior b_v + 2 b_f; the prune line prints `lam-admitted N of M priced edges`; config event carries `prune_version`. The exact kept-aware accounting of v7 is unchanged and now matches the price.
3. Sbatch (`build/make_sbatch_v9d.py` from v9c): names `adapt9d` / `adapt9db`, tables `adapt-v9d*.txt`, VERSION v9d-r1 / v9db-r1; file list and version greps include `rd_prune_v2.py`; PRUNE (30 rounds) and OVER (1.25 / 1.1) as v9c.

Principle: "the same price in reverse" must be the price of the *file*, not of the soup. An edge that frees no delivered bytes has no price and no reason to collapse under a rate objective.

## Tests
`build/test_v9d_cpu.py` (CPU torch): (1) rd_prune_v2 without masks ≡ rd_prune (same pairs and stats on a 24×24 grid); (2) with masks on a grid whose right half is unkept with zero residual (the trap v9b/v9c fell into): every selected pair has dR > 0 and touches a kept face, the unkept region is never collapsed, n_priced < E; (3) the trainer's `edge_scores` returns +inf exactly on unpriced edges; (4) diff audit vs v9c. Paste block replayed byte-exact; submit scripts dry-run (6 test jobs + gate; 43 / 28 / 15 sweep jobs by mode).

## Batch16
Same six test rows as batch15 (room 5, counter 5, kitchen 10, counter 10, kitchen 20, lego 2.5), same gate pass bar (≥ 4/5 within 0.95–1.05, none outside 0.90–1.08, PSNR ≥ hull − 0.15; lego within 0.95–1.05 and ≥ 32.233) → 43-job v9d sweep. Expected: 5–7 B per pair, the five rows at 0.97–1.05, lego within 0.1 dB. `commands/node-batch16-v1.md` is the paste.
