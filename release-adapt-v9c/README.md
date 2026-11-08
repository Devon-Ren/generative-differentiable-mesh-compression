# release-adapt-v9c — adaptive λ v9c: no negative landing slope + 30 prune rounds (2026-11-08)

## Why (from the v9b sweep, release-adapt-v9b/results/adapt-v9b-sweep-analysis.txt)
v9b landed 1.006 / 1.020 / 0.995 × target (5 / 10 / 20 MB; 22/33 within ±5 %, 32/33 within ±10 %; equal-bytes ΔPSNR −0.05). Seven indoor rows ended 5–13 % over target although phase B ran 7–11 active rounds: the landing **estimate** read under target for the first 3–9 rounds (kitchen 10: probe 9.43 MB at 12500, estimates 8.51 / 9.24 / 9.79 MB at 13000–14000, final 11.34). `extrapolate_probe` fits a least-squares slope over the probes so far and projects it 11 000 iterations ahead; with 2–3 probes a small negative sample slope (the probe fluctuates a few percent under the opacity ramp) becomes −0.8 MB at the horizon and holds the gate closed. The kept set only grows under the opacity ramp (kappa-analysis-v1 on 59 runs: final / first probe 1.00–1.39; collapsed bytes are added back into the virtual size), so a negative slope is never information.

## What changed vs v9b
1. **Code, one line** (`code/rdgrowth/train_rd_adapt_v9c.py`, 945 lines; `build/make_v9c.py`): `max(0.0, slope)` in the estimate; the raw slope is still logged. `build/test_v9c_cpu.py`: diff audit (1 fix + 4 version strings), kitchen-10 replay (v9b estimate 8.49 → v9c 9.39 at 13000; gate opens at 13500 instead of 14500; counter 5 opens at round 2 instead of 6).
2. **Sbatch**: prune rounds every 250 from 12500 to 19750 (30; the over-target rows collapsed only 0.8–1 % of V per round, the foldover / independent-set limit of `select_collapses`, and ran out of rounds). +15 probes ≈ +8 min per Mip run. OVER unchanged (1.25 / 1.1). Names `adapt9c` / `adapt9cb`, tables `adapt-v9c.txt` / `adapt-v9c-blender.txt`, VERSION v9c-r1 / v9cb-r1.

## Files
`build/` generators and CPU test; `scripts/gate_batch15_v1.py` (pass bar with the room / counter / kitchen hull points embedded), `gate-batch15.sbatch`, `submit-batch15-sweep-v1.sh all|mip|blender` (43 jobs; mic / ship run with SMOKE=0), `submit-batch15-v1.sh` (checks, 6 v9c test jobs, gate, plus mic / ship 2.5 with **v9b** SMOKE=0 to complete the v9b Blender table); `commands/node-batch15-v1.md` = the paste.

## Test batch and pass bar
room 5, counter 5, kitchen 10, counter 10, kitchen 20 (v9b 1.061 / 1.092 / 1.135 / 1.079 / 1.100) + lego 2.5 (v9b 0.999, 32.383). Gate: Mip ≥ 4/5 rows, ≥ 4 within 0.95–1.05, none outside 0.90–1.08, PSNR ≥ hull(delivered bytes) − 0.15; lego within 0.95–1.05, PSNR ≥ 32.233. Pass → the 43-job v9c sweep (≈38 GPU-h) is submitted by the gate.

## Still open after v9c (not touched)
Outdoor 20 MB under-target (0.93–0.96): the last densify interval admitted 88–92 % of the asked splits (rank-yield error with no interval left) on top of a thin margin for the outdoor class. Blender 4 MB: ficus / mic are at lam_min with the upstream limiter binding (model-size class limit); chair / drums / hotdog / materials limiter-bound in the last 1–2 intervals; `growth_capped` should be defined on the last-interval shortfall.
