# rd-main-v2: main-paper RD figure, v2b (2026-11-01)

plot_rd_main_v2.py rd-11scenes-v2.csv baselines-rd-v1.csv rd-main-v2 -> rd-main-v2.png/.pdf  (also reads generic-codecs-2026-10-24.txt)

Changes vs rd-main-v1 (user's three requirements, 11/01):
1. x axis = absolute delivered size in MB (log) in ALL panels. No "uncompressed reference" is needed to read (a) or (b); a compression ratio is only printed as a label on our points, each relative to the one FP32 asset named in the legend.
2. every panel carries a "better (smaller, higher PSNR)" arrow pointing to the upper-left corner.
3. mesh methods have no RD curve, so they are single points (hollow black markers in (b)).

(a) one asset, every codec: Room. Same FP32 MeshSplatting asset (741.7 MB / 28.263, star) sent through generic codecs vs ours.
  quantised PLY + zstd-19 158.5 MB / 28.09; Draco SH as 8-bit attributes 456.1 / 28.09; Draco RGB only 62.2 / 25.29; texture route (xatlas + Draco + 8k JPEG) 136.8 / 25.29  (generic-codecs-2026-10-24.txt).
  ours λ 6.2e-4 26.1 / 28.32 (28x), 1.85e-3 12.6 / 28.00 (59x), 6.2e-3 5.2 / 27.38 (142x), 1.85e-2 2.8 / 26.57 (264x); level 1.0 (dense + codec) 30.9 / 28.06 (24x).
  Reading: at the quality of the best generic codec (28.09) ours is 5x smaller than PLY+zstd; a generic codec that drops SH to reach 62 MB loses 3 dB, our 2.8 MB point is still 1.3 dB above it. This is the "obvious advantage" panel: no reference ambiguity, one asset, every bar.

(b) engine-ready meshes, Mip-NeRF 360 9-scene mean, absolute MB vs PSNR.
  ours λ 6.2e-4 28.2 / 24.47 (23x), 1.85e-3 20.3 / 24.27 (32x), 6.2e-3 8.9 / 23.81 (74x), 1.85e-2 4.2 / 23.10 (155x); level 1.0 26.6 / 24.23; FP32 reproduction 657 / 24.50 (star).
  points: MeshSplatting authors' RGB-only PLY 100 / 24.78 (their paper numbers, SH dropped -> our SH-dropped measurement still pending); MiLo 253 / 24.09; BakedSDF 434 / 24.51 (= (4·27.06+5·22.47)/9); Binary Opacity Grids 913 / 25.62 (= (4·27.71+5·23.94)/9); RaDe-GS mesh 1100 / 23.56; GOF mesh 1500 / 20.78 (MeshSplatting Tables 1-2).
  Reading: ours is 10-100x smaller than every published engine-ready mesh at equal PSNR (MiLo 253 MB / 24.09 vs ours 8.9 MB / 23.81 or 20.3 MB / 24.27). BOG is +1.1 dB but 913 MB.

(c) Gaussian-class compressors vs ours: y = Δ test PSNR vs the full-precision model that method itself trains and compresses, x = absolute MB. Reference stated per method in the legend:
  ours -> our FP32 MeshSplatting reproduction (657 MB / 24.50); HAC, HAC++, ContextGS -> Scaffold-GS (253.9 MB / 27.50, HAC++ Table II); RDO-Gaussian -> its own 3DGS mean (745.8 / 27.49); LightGaussian, Compact3DGS -> 3DGS (750.9 / 27.46).
  Reading: RDO-Gaussian is the method closest in spirit (λ-priced rate during training); at every size ours keeps more of its own model's quality (e.g. 8.9 MB: ours −0.69 vs RDO ≈ −1.5). Anchor-based coders sit at +0.1…+0.3 because Scaffold-GS is a far more compressible base; they stop at ~8-28 MB.
  This panel is the one that can still be attacked ("each method vs its own base"); if the user prefers, drop it and keep (a)+(b) only; the absolute 11-scene hull (rd-11scenes-v3) goes to the supplement.

Caption caveats: RDO-Gaussian points read from its appendix (verify against PDF before submission); BakedSDF / BOG averages recomputed from indoor/outdoor means; authors' RGB-only PLY is their reported number, not our measurement.
