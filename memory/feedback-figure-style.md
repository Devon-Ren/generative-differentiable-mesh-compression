---
name: feedback-figure-style
description: Wenhao's figure/table rules for the CVPR paper and advisor reports (11/01): one unified style, no exotic markers, absolute MB axes with "better" arrows, smooth hulls, ratios in tables not figures, baselines 2024+ only, MPEG codec required by the advisor
metadata:
  type: feedback
---

Rules given 2026-11-01 after seeing the first RD figures ("有点丑", "奇形怪状的玩意"):
- One visual language for every figure: analysis/report/figstyle_v1.py (ours green PCHIP-smoothed curve, post-hoc orange, cap grey, mesh class black hollow circles + text labels, Gaussian class cool colours, generic codecs purple). Circles only; labels as text next to points, not legend symbols.
- Every RD panel: x = absolute delivered MB (log, plain ticks), a "better" arrow, no per-point compression-ratio labels (ratios go in the tables).
- Compression-ratio comparisons need one unambiguous uncompressed reference; otherwise use absolute sizes.
- Baselines: only 2024 and newer, comparable protocol (BakedSDF/MobileNeRF -> supplement); mesh methods appear as points (they have no RD).
- The advisor wants an MPEG mesh codec in the generic-codec comparison (plan: Open3DGC/SC3DMC first, V-DMC optional) alongside Draco.
- Tables: per-scene PSNR/SSIM/LPIPS/FPS (3DGS class "-" where unreported), ratios in the table.
- He wants an intuitive "we are far ahead among meshes" figure, not necessarily RD (-> fig-mesh-class-v1 (b) lollipop "size for equal PSNR").

**Why:** figures go to the advisor/PhDs and to the HPC resource request; mixed styles and odd markers read as sloppy.
**How to apply:** build every new figure on figstyle_v1 and version it (-vN); see [[feedback-versioned-deliverables]], [[rd-hull6-results-nov2026]].
