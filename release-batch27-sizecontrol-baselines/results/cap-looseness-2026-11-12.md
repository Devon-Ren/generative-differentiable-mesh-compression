# How tightly does a vertex cap fix the final vertex count and file size? (existing runs only, no new GPU; 2026-11-12)

Source: all MeshSplatting `--max_points` runs on the 13 captured scenes (72 runs before batch27 from g0-refresh / batch21 / batch24, plus the 6 batch27 runs). Same code, same options, same seed in every run; only the cap differs.

## Final vertices / cap
- 72 runs before batch27: 0.68 to 1.06; 78 runs with batch27: mean 0.88, standard deviation 0.078.
- The ratio is not a smooth function of the cap inside one scene (runs before batch27): stump 888k 0.75 / 983k 0.92; bonsai 1005k 0.87 / 1328k 0.68 / 2090k 0.84; bicycle 440k 0.89 / 474k 0.80 / 986k 0.95; kitchen 622k 0.92 / 928k 0.81.
- So the batch27 observation (garden 1,105k 0.80 vs 1,073k 0.94; playroom 1,081k 0.78 vs 986k 0.95) is the same behaviour that the older runs already show; both batch27 runs of a scene were submitted with the same job file and options (submit-batch27-v1.sh, cap-pass2-v1.sbatch), so it is not a configuration difference.
- Not shown here: the cause (which training stage removes the vertices). That needs the training logs on the cluster.

## What a "measure, rescale the cap, retrain" step achieves
Every ordered pair of caps of one scene whose caps differ by at most 25% is read as (first training, second training): the second cap is what the update rule would pick for the target T = c + (MB1 - c) * cap2 / cap1; error = MB2 / T - 1.

| runs | update rule | pairs | mean abs error | median | worst | within 10% | within 5% |
|---|---|---|---|---|---|---|---|
| before batch27 | proportional (c = 0) | 40 | 5.3% | 2.8% | 19.2% | 32/40 (80%) | 24/40 |
| before batch27 | fixed bytes removed (c = 2 MB) | 40 | 5.3% | 4.0% | 22.5% | 36/40 (90%) | 25/40 |
| all, with batch27 | proportional (c = 0) | 62 | 7.4% | 6.2% | 25.9% | 44/62 (71%) | 28/62 |
| all, with batch27 | fixed bytes removed (c = 2 MB) | 62 | 6.9% | 4.6% | 22.5% | 48/62 (77%) | 33/62 |

Reading: a second cap training lands within 10% in roughly 8 to 9 of 10 cases, not in all of them; the misses come from the jumps of the vertices/cap ratio above. Limits: pairs are ordered (each pair counted in both directions); the caps of a pair differ by up to 25%, more than a typical retry step; captured scenes only.
