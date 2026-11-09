# G0 on the current protocol (node-batch19, 2026-11-09): 7 outdoor / T&T scenes × 3 budgets, 21/21 for training to the budget

Rows: `g0-refresh-2026-11-09.txt` (budget-sweep-v2.sbatch; train-at-budget = upstream `--max_points B`, post-hoc = the dense FP32 original collapsed to the same vertex count, same 3000-step finetune, same K4096 R256 codec). λ-grown reference = the scene's four fixed-λ assets (rd-prune3 tables), interpolated log-linearly at the budget run's delivered bytes.

| scene | V (= λ asset ÷0.9 cap) | train-at-budget MB / PSNR | post-hoc collapse MB / PSNR | Δ budget − collapse (equal V) | λ-grown at equal bytes | Δ budget − λ |
|---|---|---|---|---|---|---|
| bicycle | 2.29 M | 22.23 / 22.67 | 22.45 / 22.66 | +0.01 | 22.94 | −0.27 |
| bicycle | 1.07 M | 11.11 / 22.16 | 10.62 / 21.14 | +1.02 | 22.63 | −0.47 |
| bicycle | 0.38 M | 4.56 / 21.51 | 3.87 / 17.56 | +3.95 | 21.98 | −0.48 |
| flowers | 1.95 M | 18.09 / 18.98 | 18.57 / 18.64 | +0.34 | 19.13 | −0.15 |
| flowers | 1.09 M | 10.76 / 18.80 | 10.46 / 17.11 | +1.68 | 19.16 | −0.36 |
| flowers | 0.49 M | 5.46 / 18.46 | 4.79 / 14.14 | +4.31 | 18.82 | −0.37 |
| garden | 1.99 M | 21.51 / 24.30 | 21.69 / 23.88 | +0.42 | 24.37 | −0.08 |
| garden | 0.90 M | 10.61 / 23.48 | 10.10 / 21.73 | +1.75 | 23.69 | −0.21 |
| garden | 0.44 M | 5.61 / 22.64 | 5.07 / 19.37 | +3.26 | 22.93 | −0.29 |
| stump | 1.68 M | 16.37 / 24.08 | 16.33 / 23.77 | +0.31 | 24.53 | −0.45 |
| stump | 0.66 M | 7.22 / 23.35 | 6.50 / 21.20 | +2.15 | 23.89 | −0.54 |
| stump | 0.31 M | 3.95 / 22.61 | 3.19 / 16.59 | +6.02 | 23.22 | −0.62 |
| train | 1.99 M | 21.45 / 18.77 | 21.48 / 18.68 | +0.09 | 18.67 | +0.11 |
| train | 1.10 M | 12.20 / 18.57 | 12.29 / 18.18 | +0.39 | 18.66 | −0.09 |
| train | 0.51 M | 6.18 / 18.14 | 5.93 / 16.78 | +1.36 | 18.55 | −0.40 |
| treehill | 2.40 M | 22.71 / 20.45 | 22.99 / 20.10 | +0.35 | 20.32 | +0.12 |
| treehill | 1.14 M | 11.54 / 20.48 | 10.97 / 18.92 | +1.56 | 20.37 | +0.11 |
| treehill | 0.43 M | 5.07 / 20.14 | 4.21 / 15.60 | +4.53 | 20.30 | −0.17 |
| truck | 1.44 M | 15.59 / 22.39 | 15.72 / 22.09 | +0.30 | 22.25 | +0.14 |
| truck | 0.76 M | 8.64 / 22.10 | 8.52 / 20.65 | +1.45 | 22.28 | −0.19 |
| truck | 0.33 M | 4.41 / 21.30 | 3.98 / 17.07 | +4.22 | 21.64 | −0.34 |

**Train-at-budget vs post-hoc collapse at equal vertex count: +0.01 to +6.02 dB, mean +1.88, 21 of 21.** By budget tier: at the λ 1.85e-3 count (≈32×) +0.26 mean; at the λ 6.2e-3 count (≈73×) +1.43; at the λ 1.85e-2 count (≈150×) +3.95. The gap grows as the budget shrinks, as in the October measurement (+0.3 to +4.3 dB), now on 7 scenes including T&T and the two scenes (flowers, treehill) that were never in G0.

**λ-grown vs budget-capped at equal bytes: −0.62 to +0.14 dB for the cap, mean −0.24; λ wins 17 of 21**, consistent with the earlier "λ ≥ cap on 10/10, +0.0 to +0.57". The three-way ordering is post-hoc collapse < fixed vertex cap < byte-priced growth, at every budget tier.

Why post-hoc collapse loses so much at small counts: the collapsed meshes end with far fewer faces per vertex (stump 310k V: 104k faces post-hoc vs 671k trained; bicycle 380k V: 282k vs 774k) — collapsing a dense soup to a sparse vertex set destroys connectivity that training-to-budget keeps, and the 3000-step finetune cannot regrow it.

Indoor scenes (room, kitchen, counter, bonsai) are not in this refresh: upstream's `--indoor` preset overwrites `--max_points` after parsing; they need the train_ckpt_v4 override path (smoke test first). The October indoor G0 rows (room +1.32/+2.05, bonsai +1.56/+4.30, counter +0.70/+2.71, kitchen +0.32/+1.82, without `--indoor`) remain as the indoor evidence with that caveat.
