---
name: rd-price-constants-rationale
description: Why split/prune use fixed B_v=10 B, B_f=2 B for all scenes (11/03): Lagrangian derivation, meaning of D_u+D_v-D_uv, measured bytes/vertex 9.5-13.3 B on 44 assets, reviewer answer and optional sensitivity run
metadata:
  type: project
---

Note: plan/split-prune-rationale-2026-11-03.md (Chinese explanation + English paper paragraph).

- All v3 runs used BPV=10.0, BPF=2.0, MU=1.0, MAXF=0.15 (sbatch exports). Origin: 10/18 Room q14 stream budget (topology ~4 + xyz ~2.3 + DC ~1.5 + index ~2.4 B/vertex; 4 B topology / ~2 faces per vertex = 2 B/face).
- NOTATION (user 11/03, binding for paper/report): only two independent constants, B_a = 6 B attribute bytes per vertex (pos residual + DC + SH indices) and B_f = 2 B connectivity per face; never write B_v; formulas additive only: split dR = 3B_a + 6B_f = 30 B, collapse dR = B_a + n_uv B_f = 10 B, dD_uv = sum over the union of the two vertex rings (no D_u+D_v-D_uv). Applied in report-latex-v2.
- Code facts: split dR = 3*B_v (no B_f term); collapse dR = (B_v - 2B_f) + n_faces(uv)*B_f (interior = B_v). The report-latex-v1 TikZ boxes and §1.2 originally stated these wrongly; corrected lines sent 11/03.
- Measured delivered bytes/vertex excluding the fixed 0.61 MB banks, 44 lambda assets: 9.5-13.3 B, median 10.8; rises with lambda (median 10.4 -> 11.4); kitchen highest (faces/vertex 3.1-3.4). Faces/vertex 1.8-3.4, median 2.3.
- Reviewer answer: price must exist before training; per-scene B_v = rescaling lambda by 0.95-1.33 (< 1/4 of the 3x lambda grid step); one constant keeps lambda a common exchange rate across scenes. Both dD are proxies (split: upper bound of removable error; collapse: weaker, justified by prune results -4..-8 % bytes, +0.02 dB).
- Optional sensitivity run proposed: B_v in {8,10,13} x kitchen/garden x 1 lambda, ~6 GPU-h (not submitted; compute exhausted, see [[hpc-request-nov2026]]).
See [[rdgrowth-v3-prune-design]], [[rd-hull6-results-nov2026]].
