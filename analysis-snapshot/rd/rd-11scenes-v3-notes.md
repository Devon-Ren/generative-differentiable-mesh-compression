# rd-11scenes-v3 figure notes (2026-11-01)

Files: plot_11scenes_v3.py (input rd-11scenes-v2.csv + baselines-rd-v1.csv) -> rd-11scenes-v3.png/.pdf.

**Per-scene panels**: our v3 points at λ 6.2e-4 / 1.85e-3 / 6.2e-3 / 1.85e-2; the solid line is the Pareto hull (a point is dropped from the hull when a smaller asset has ≥ test PSNR); dominated points are hollow. Orange = dense training + codec: the post-hoc collapse curve (7 scenes) or the level-1.0 point (4 new scenes). Dashed = FP32 original. Purple = HAC per-scene RD curve (HAC appendix Tables C/D, λe 0.5e-3…4e-3), the only 3DGS-class method with per-scene multi-rate numbers in print. x axis log.

**Average panels**: Mip-360 (9 scenes) and T&T (2): our 4-λ means + level 1.0 + FP32 star; HAC (5 points, appendix AVG rows), HAC++ low/high (Table II; λ 4e-3 / 0.5e-3 by size matching, inferred), RDO-Gaussian 6 points (appendix Tables A.3 / A.1, extracted from a summary of the HTML, verify against the PDF before the paper), ContextGS low/high, LightGaussian, Compact3DGS (Lee et al.), 3DGS, Scaffold-GS (HAC++ Table II).

**Why flowers / treehill / truck / train (and bicycle at 6.2e-4) are not monotone**: the larger assets fit the training distribution but not the test split. Validation PSNR (held-out train-trajectory views) vs test PSNR, λ 6.2e-4 minus λ 1.85e-2:
flowers val +1.17 / test +0.18; treehill val +1.17 / test +0.01; truck val +0.66 / test +0.15; train val +0.52 / test +0.00; bicycle val +1.32 / test +0.91; room (for contrast) val +3.07 / test +1.76.
MeshSplatting's own per-scene numbers are equally low there (flowers 19.34, treehill 20.53, truck 22.32, train 18.72), i.e. the representation saturates on these scenes at a few hundred thousand vertices; more vertices only add train-view fit (R9). The hull is the Pareto subset; the paper plots hull + hollow dominated points.

**Gap to the 3DGS class at equal bytes** (Mip-360 means): ours λ 6.2e-4 28.1 MB / 24.47 vs HAC 25.0 MB / 27.83 (−3.4 dB), HAC++ 18.5 MB / 27.82, RDO-Gaussian 23.5 MB / 27.05 (−2.6 dB); at ~9 MB ours 23.8 vs HAC++ 27.60 / RDO 26.5. T&T: ours 11.4 MB / 20.5 vs HAC++ 8.6 / 24.32, RDO 12.0 / 23.34. The gap is the opaque-mesh class gap (MeshSplatting FP32 itself is 3.0 dB below 3DGS on Mip-360), not the codec: our curve sits 0.0–0.3 dB under the FP32 original down to ~30×.
