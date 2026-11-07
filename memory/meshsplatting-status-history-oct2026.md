---
name: meshsplatting-status-history-oct2026
description: Earlier measured findings of the MeshSplatting compression project (Room byte breakdown, 10/13 direction reset, batches 1-4, G0 train-at-budget result), split out of the status file on 2026-10-19
metadata:
  type: project
---

Split out of [[meshsplatting-compression-status]] (kept under 4 KB).

Room dense 41.4 MB: topology 30.4 bits/v, XYZ 29.9, DC 15.1; FP32 ≈742 MB; near-soup mesh (45% boundary edges, 8277 components).
Collapse retraining: finetune after collapse plateaus (loss is in what gets removed) → [[c2-collapse-experiments]]; equal faces ≠ equal bytes, compare at equal vertices.
Coding-only ceiling ≈4 MB; vertex_weight drop lossless. Don't edit user's code / push without asking.
DIRECTION RESET 2026-10-13 (user + advisor): stop stacking methods for 0.1-dB gains. Paper thesis (advisor): "differentiable opaque mesh does not need independent SH; the representation is highly redundant". Plan = two RD axes + convex hull: axis A appearance (K × R dictionaries), axis B geometry (vertex count 100→50% via --target-vertices, geo-only default). Key open question: how flat is the geometry curve near 90–80% vertices. DRACO: splits soup 4.5× vertices, total larger than ours → topology-coding CLOSED; levers = fewer vertices or manifold mesh (future). Rate-aware results: [[rate-aware-results]]. Assessment 2026-10-13: hull alone is not CVPR; needs the WHY, a method framing, and mesh-based competitors on MB + FPS (see [[baseline-numbers-literature]]).
BATCH 1-3 DONE 2026-10-16/17: cross-scene codebook falsified ([[crossbank-index-locality-results]]); garden + bonsai backbones = Room's ([[garden-scene-results]], [[bonsai-scene-results]]); greedy allocator = hull. BATCH 4 DONE 2026-10-17: counter/kitchen/bicycle/stump axis B ([[scenes4-results]]): q14 sharing+quant cost 0.4-1.5 dB.
G0 RESULT 2026-10-17 (DECISIVE, job 20133555, rd/g0-budget-2026-10-17.txt): Room trained with --max_points 1.6M stops at 1.35M v (42 %), test 28.165 (dense 28.263), delivered 13.56 MB / 27.984 / SSIM .862; post-hoc collapse to the same 1.35M v: 13.93 MB / 26.66 / .837 -> +1.3 dB at equal bytes, and 13.6 MB beats the old default v0.8 25.5 MB / 27.883. => dense training overgrows; PhD's 'rate controls generation' is RIGHT for geometry; the method = budget/lambda-controlled growth; post-hoc collapse was the wrong lever. 1.0M budget: 967k v (30 %), test 27.843, delivered 10.38 MB / 27.711 / .855 (72x, -0.55 vs original) vs post-hoc 10.14 / 25.665 -> +2.05 dB; dense->1.35M costs 0.006 dB/MB (flat), 1.35M->0.97M 0.086 dB/MB: knee at or below 1.35M. 7/7 scenes confirmed 10/18 ([[scenes7-budget-vs-collapse]]): means budget 1.6M 14.4 MB/-0.68 dB/46x, 1.0M 9.8/-0.97/67x vs collapse default 23.6/-0.44/28x. NOTE run_garden.sh trains originals WITHOUT --indoor (outdoor params for all scenes); keep that for comparability.
