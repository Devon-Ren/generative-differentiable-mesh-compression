---
name: adaptive-lambda-history-nov2026
description: 11/05-06 adaptive-lambda controller history v1 -> v2 -> v3 (what each version got wrong, measured), plus ba-sens and 11-scene fan results; current state is in adaptive-lambda-nov2026
metadata:
  type: project
---

v1 (commands/node-batch2-v1.md, rdgrowth/train_rd_adapt_v1.py; 16 jobs incl. ba-sens + fan, all exit 0 11/05): phase A desired splits = (target - R_hat)/(3 B_v remaining), lambda *= (n_pass/desired)^0.5; phase B lambda *= (R_hat/target)^2. RESULT (rd/adapt-v1-analysis.txt): quality on the fixed-lambda hull (-0.12..+0.07 dB) but size 52-77 % of target: projection bug (B_v = B_a + 2 B_f double counted connectivity) x upstream shrink after the last prune round (0.85) x phase A/B shortfalls.
v2 (commands/node-batch3-v1.md, train_rd_adapt_v2.py 590 l + probe_bytes_v1.py 66 l + rd/adapt-v2.sbatch 109 l): real delivery encoder as byte meter at every prune round (probe/delivered 0.998-1.000, 24/24 runs since), b_v/b_f/overhead from measured streams, phase B count-driven + shadow-price lambda. RESULT (rd/adapt-v2-analysis.txt): hit 0.65-1.05; phase B never collapsed (cleaned probe at the CURRENT opacity floor reads 1.1-1.6x low); quality -0.08 dB (no prune); phase A 0.5-0.8 of aim. Densify dump (commands/adapt-v2-densify-dump.md): upstream prune kills 30-70 % of each interval's new vertices (sigma 0.33-0.5).
v3 (commands/node-batch4-v1.md, train_rd_adapt_v3.py 652 l + rd/adapt-v3.sbatch 107 l): SurvivalMeter sigma in the phase-A budget, lambda_next = rank price of the planned count (gain constants removed), probe at the final opacity floor 0.9999, phase B = fixed-lambda prune + count collapses, overshoot 1.15. RESULT 11/06 (rd/adapt-v3-analysis.txt): hit 0.73-1.03 (mean 0.87), quality -0.10..+0.07 dB (mean -0.007) = on the hull; still short because (1) probe at 1x keeps 0.30-0.59 of faces, final cleaning runs at scaling 4 (final/probe 1.03-1.39), (2) fixed-lambda prune removed 11-16 % of vertices while under target, (3) phase A limiter-bound late on 20 MB (upstream add_percentage 1.23/1.27).
Lessons: a smoke shorter than densify_from cannot test phase A; check the size model against one delivered file before a controller run; the probe must reproduce the FINAL cleaning conditions (floor AND supersampling).

ba-sens (rd/ba-sens-v1-analysis.txt, report §2.10): kitchen/garden x B_v {8,13} on the B_v 10 curve (-0.01..+0.07 dB), lam_eff/lam 0.78 / 1.31 vs predicted 0.80 / 1.30 -> price error = lambda rescale.
Fan 11 scenes (rd/fan-analysis-v2.txt, §2.7 + App. B): joint hull <= 0.16 dB above lambda-only; K1024 on hull at the two low lambdas, larger banks at the top; DC 8 bit / K16384 pay indoors only.
See [[adaptive-lambda-nov2026]], [[advisor-rd-questions-nov2026]], [[rd-price-constants-rationale]].
