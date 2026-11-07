---
name: advisor-reaction-nov2026
description: 11/03 Prof. Yao Wang saw the garden 73x viewer render and questioned the whole mesh direction (why mesh, 3DGS better at same size, test simple objects, only a bounded/compact/physically-correct mesh is a contribution); diagnosis and venue options in plan/advisor-questions-2026-11-03.md; decision pending
metadata:
  type: project
---

What she saw: garden lambda 6.2e-3 (10.2 MB, 23.65 dB, tree canopy as flat triangle fragments). Breakdown given: ~2.6 dB = MeshSplatting class gap (FP32 24.75 vs 3DGS 27.3), 1.1 dB = the 73x point, codec <= 0.3 dB; HAC++ at 12.8 MB = 27.19 (3.5 dB above us at equal bytes). Vegetation fragments exist at FP32 (soup needs soft blending, P2).

Her asks: (1) why mesh at all / what it brings over Gaussians; (2) people will ask why 3DGS compressed to the same size is much better; (3) test simple object-centric data (a table, a person) first; (4) the real contribution is a bounded, compact, physically correct differentiable mesh = the +++ line ([[meshsplat-plus-vision]]).

Answer file: plan/advisor-questions-2026-11-03.md (per-question honest answers; mesh buys fixed-function raster, collision/physics, shadows/relighting with baked vertex colour (-0.3..-0.7 dB), DCC editing, CPU decode; costs 3 dB on unbounded scenes). Venue options: P1 submit as planned (3.3-3.6), P2 skip CVPR and fold compression into +++, P3 reposition as a "how compressible, what it costs" analysis + object-centric chapter (NeRF-Synthetic, 8 scenes x 3 runs ~ 8 GPU-h, needs a white_background smoke test) + engine demo (Blender shadows/collision, CPU). Recommended: P3 decided by the object-centric result, else P2. No GPU runs possible until the HPC node arrives ([[hpc-request-nov2026]]). Wenhao's decision pending.
See [[meshsplatting-compression-status]], [[quality-gap-diagnostics-oct2026]].
