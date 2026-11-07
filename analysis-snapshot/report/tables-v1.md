# Report tables v1 (2026-11-01)

Conventions: PSNR/SSIM/LPIPS on the test split (every 8th view), MB = 1e6 bytes, ratio = FP32 size / delivered size. FP32 = our reproduction WITHOUT upstream --indoor (indoor rerun in progress). LPIPS: the "FP32 (train.log)" columns use the upstream train.py / 3DGS-protocol convention ([0,1] inputs); our delivered-asset columns use the pipeline evaluator ([-1,1] inputs), which reads ~0.03 higher on the same images (garden FP32: 0.216 vs 0.249) -> the unified pass will put every number on the 3DGS convention. Room FP32 row is pipeline-evaluated (no train.log).

## T1. Per scene: FP32 model, the MeshSplatting paper, and our delivered assets

| scene | FP32 MB | FP32 PSNR / SSIM / LPIPS (train.log) | paper PSNR / SSIM / LPIPS | λ 6.2e-4: MB (ratio) PSNR / SSIM / LPIPS | λ 1.85e-3: MB (ratio) PSNR / SSIM / LPIPS | λ 6.2e-3: MB (ratio) PSNR / SSIM / LPIPS | λ 1.85e-2: MB (ratio) PSNR / SSIM / LPIPS |
|---|---|---|---|---|---|---|---|
| room | 742 | 28.26 / 0.883 / 0.339 | 28.52 / 0.873 / 0.271 | 26.1 (28×) 28.32 / 0.875 / 0.357 | 12.6 (59×) 28.00 / 0.863 / 0.395 | 5.2 (142×) 27.38 / 0.839 / 0.447 | 2.8 (264×) 26.57 / 0.816 / 0.484 |
| counter | 575 | 26.07 / 0.842 / 0.284 | 26.51 / 0.846 / 0.279 | 25.5 (23×) 25.90 / 0.833 / 0.371 | 14.8 (39×) 25.79 / 0.823 / 0.399 | 6.7 (86×) 25.26 / 0.796 / 0.444 | 3.4 (171×) 24.59 / 0.766 / 0.482 |
| kitchen | 456 | 26.39 / 0.827 / 0.267 | 27.42 / 0.858 / 0.227 | 26.7 (17×) 26.65 / 0.842 / 0.302 | 16.7 (27×) 26.24 / 0.825 / 0.334 | 7.5 (61×) 25.41 / 0.780 / 0.398 | 3.7 (124×) 24.27 / 0.722 / 0.461 |
| bonsai | 662 | 27.36 / 0.866 / 0.307 | 28.19 / 0.876 / 0.294 | 26.7 (25×) 27.49 / 0.872 / 0.366 | 14.0 (47×) 26.65 / 0.843 / 0.425 | 6.1 (109×) 26.16 / 0.819 / 0.478 | 3.4 (192×) 25.19 / 0.786 / 0.517 |
| garden | 740 | 24.75 / 0.762 / 0.216 | 24.70 / 0.762 / 0.217 | 36.4 (20×) 24.73 / 0.761 / 0.253 | 24.6 (30×) 24.50 / 0.740 / 0.287 | 10.2 (73×) 23.65 / 0.668 / 0.385 | 5.0 (149×) 22.78 / 0.591 / 0.467 |
| bicycle | 680 | 23.04 / 0.640 / 0.351 | 23.04 / 0.641 / 0.348 | 31.6 (22×) 22.99 / 0.639 / 0.395 | 27.0 (25×) 23.02 / 0.636 / 0.405 | 11.9 (57×) 22.69 / 0.590 / 0.470 | 5.2 (131×) 22.08 / 0.526 / 0.535 |
| stump | 698 | 24.99 / 0.686 / 0.308 | 24.78 / 0.678 / 0.316 | 27.6 (25×) 24.81 / 0.675 / 0.372 | 21.4 (33×) 24.71 / 0.667 / 0.390 | 8.9 (79×) 24.12 / 0.620 / 0.461 | 3.9 (178×) 23.21 / 0.548 / 0.534 |
| flowers | 668 | 19.45 / 0.492 / 0.409 | 19.34 / 0.480 / 0.417 | 25.7 (26×) 19.02 / 0.481 / 0.457 | 24.2 (28×) 19.07 / 0.483 / 0.459 | 12.0 (56×) 19.21 / 0.462 / 0.494 | 5.6 (118×) 18.84 / 0.417 / 0.542 |
| treehill | 692 | 20.39 / 0.539 / 0.427 | 20.53 / 0.540 / 0.428 | 27.0 (26×) 20.31 / 0.517 / 0.513 | 27.4 (25×) 20.45 / 0.524 / 0.505 | 11.9 (58×) 20.38 / 0.509 / 0.558 | 5.1 (136×) 20.30 / 0.475 / 0.611 |
| truck | 432 | 22.39 / 0.807 / 0.226 | 22.32 / 0.799 / 0.237 | 27.3 (16×) 21.78 / 0.788 / 0.276 | 19.3 (22×) 22.20 / 0.796 / 0.278 | 9.4 (46×) 22.37 / 0.784 / 0.312 | 4.4 (98×) 21.64 / 0.743 / 0.376 |
| train | 485 | 18.90 / 0.712 / 0.319 | 18.72 / 0.693 / 0.337 | 30.1 (16×) 18.55 / 0.687 / 0.387 | 25.4 (19×) 18.66 / 0.695 / 0.384 | 13.4 (36×) 18.68 / 0.691 / 0.404 | 6.4 (76×) 18.55 / 0.664 / 0.443 |
| **Mip-360 mean (9)** | 657 | 24.52 / 0.726 / 0.323 | 24.78 / 0.728 / 0.311 | 28.1 (23×) 24.47 / 0.722 / 0.376 | 20.3 (32×) 24.27 / 0.712 / 0.400 | 8.9 (74×) 23.81 / 0.676 / 0.460 | 4.2 (155×) 23.09 / 0.628 / 0.515 |
| **T&T mean (2)** | 459 | 20.64 / 0.760 / 0.272 | 20.52 / 0.746 / 0.287 | 28.7 (16×) 20.17 / 0.737 / 0.332 | 22.3 (21×) 20.43 / 0.745 / 0.331 | 11.4 (40×) 20.52 / 0.737 / 0.358 | 5.4 (85×) 20.10 / 0.704 / 0.409 |

## T2. Cross-method, dataset means (published numbers verbatim; FPS as reported, hardware differs; "–" = not reported)

| class | method (venue) | Mip-360 PSNR / SSIM / LPIPS | Mip-360 MB | T&T PSNR / SSIM / LPIPS | T&T MB | FPS | engine-ready mesh | ratio vs own FP32 |
|---|---|---|---|---|---|---|---|---|
| mesh | MeshSplatting, paper (2025) | 24.78 / 0.728 / 0.310 | 100 (RGB-only PLY, SH dropped) | 20.52 / 0.745 / 0.287 | – | 220 / 190 (M4 laptop) | yes | – |
| mesh | MeshSplatting FP32, our reproduction (no --indoor) | 24.52 / 0.726 / 0.323 | 657 | 20.64 / 0.760 / 0.272 | 459 | – | yes (FP32 PLY) | 1× |
| mesh | MiLo (SIGGRAPH Asia 2025) | 24.09 / 0.688 / 0.323 | 253 | 21.46 / 0.706 / 0.348 | – | 170 / 160 (M4) | yes | – |
| mesh | Binary Opacity Grids (SIGGRAPH 2024) | 25.62 / – / – | 913 | – | – | 927 (desktop GPU) | yes | – |
| mesh | RaDe-GS mesh (2024) | 23.56 / 0.668 / 0.361 | 1100 | 20.51 / 0.659 / 0.344 | – | OOM on M4 | yes | – |
| mesh | GOF mesh (2024) | 20.78 / 0.573 / 0.465 | 1500 | 21.69 / 0.690 / 0.326 | – | OOM on M4 | yes | – |
| mesh | **ours, λ 6.2e-4** | 24.47 / 0.722 / 0.376† | 28.1 | 20.17 / 0.737 / 0.332† | 28.7 | 350 phone / 820 laptop (Room) | yes, + CPU decode 2–6 s | 23× / 16× |
| mesh | **ours, λ 1.85e-3** | 24.27 / 0.712 / 0.400† | 20.3 | 20.43 / 0.745 / 0.331† | 22.3 | 350 phone / 820 laptop (Room) | yes, + CPU decode 2–6 s | 32× / 21× |
| mesh | **ours, λ 6.2e-3** | 23.81 / 0.676 / 0.460† | 8.9 | 20.52 / 0.737 / 0.358† | 11.4 | 350 phone / 820 laptop (Room) | yes, + CPU decode 2–6 s | 74× / 40× |
| 3DGS | 3DGS (SIGGRAPH 2023) | 27.46 / 0.812 / 0.222 | 750.9 | 23.69 / 0.844 / 0.178 | 431.0 | 99 (desktop GPU) | no | 1× |
| 3DGS | Scaffold-GS (CVPR 2024) | 27.50 / 0.806 / 0.252 | 253.9 | 23.96 / 0.853 / 0.177 | 86.5 | 135 | no | 1× (base of HAC family) |
| 3DGS | HAC++ low / high (2025) | 27.60 / 0.803 / 0.253 ; 27.82 / 0.811 / 0.231 | 8.34 ; 18.48 | 24.22 / 0.849 / 0.190 ; 24.32 / 0.854 / 0.178 | 5.18 ; 8.63 | 130–151 | no | 30× / 14× vs Scaffold-GS |
| 3DGS | RDO-Gaussian (ECCV 2024) | 27.05 / 0.802 / 0.239 | 23.46 | 23.34 / 0.835 / 0.195 | 12.03 | – | no | 32× vs 3DGS |
| 3DGS | Compact3DGS (CVPR 2024) | 27.08 / 0.798 / 0.247 | 48.80 | 23.32 / 0.831 / 0.201 | 39.43 | – | no | 15× vs 3DGS |
| 3DGS | LightGaussian (NeurIPS 2024) | 27.00 / 0.799 / 0.249 | 44.54 | 22.83 / 0.822 / 0.242 | 22.43 | – | no | 17× vs 3DGS |

† our LPIPS on the pipeline convention ([-1,1]); ~0.03 lower on the 3DGS convention used by every other row (unified pass pending). Our indoor scenes will move after the --indoor rerun.

## T3. Deployability (from rd/table2-deployability-v1.md, λ 6.2e-3 point)

| scene | asset MB | ratio | train min, original -> ours | CPU decode s (1 core) | laptop ms (fps) | phone ms (fps) |
|---|---|---|---|---|---|---|
| room | 5.2 | 142× | 51 -> 27 (0.52×) | 2.3 | 1.2 (820) @ 3114×2075 | 2.8 (352) @ 3114×2075 |
| garden | 10.2 | 73× | 49 -> 24 (0.48×) | 4.4 | 1.8 (566) @ 5187×3361 | 6.3 (160) @ 5187×3361 |
| bicycle | 11.9 | 57× | 41 -> 25 (0.60×) | 5.4 | 1.9 (523) @ 4946×3286 | 6.9 (145) @ 4946×3286 |
| 11-scene mean | 9.4 | 73× | 0.57× | 4.1 | – | – |

Laptop = MacBook Pro M2 Max (Chrome), phone = iPhone 17 Pro (Safari), WebGL2 viewer, SH degree 3, full capture resolution.
