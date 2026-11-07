# Baselines on the same scenes (Mip-NeRF 360 room / garden) — v1, 2026-10-16

Legend: **[paper]** = number reported by the cited paper (per scene unless marked avg); **[ours]** = measured in this project;
**[est]** = inferred, must be re-measured. Protocol: room = indoor, images_2; garden = outdoor, images_4; test = every 8th view.
Our numbers are decoded-asset test PSNR (39 views on room).

## 1. Our method (MeshSplatting + rate-aware codec, final format const/ctx)

| scene | point | MB | PSNR | SSIM | LPIPS | note |
|---|---|---|---|---|---|---|
| room | MeshSplatting original (our training, FP32 minimal) | 742 | 28.263 | | | [ours] |
| room | default v0.8 K4096 R256 q14/q7 | 25.51 | 27.883 | 0.8686 | 0.3703 | [ours] 29x, -0.38 dB |
| room | low-rate v0.6 K4096 R1 | 17.60 | 27.351 | 0.8526 | 0.4005 | [ours] 42x |
| room | high-rate v1.0 K4096 R1024 | 31.82 | 28.091 | 0.8737 | 0.3581 | [ours] |
| garden | MeshSplatting original | 740 | 24.732 | | | [ours] |
| garden | default v0.8 K4096 R256 q15/q7 | 27.34 | 24.371 | 0.7288 | 0.2981 | [ours] 27x, -0.36 dB |

## 2. Same representation, naive export (the baseline reviewers will ask for first)

| scene | method | MB | PSNR | note |
|---|---|---|---|---|
| room/garden | MeshSplatting authors' PLY (RGB vertex colours only) | ~100 [paper: "+-100MB"] | original minus ~2 dB (avg) [paper] | https://meshsplatting.github.io/ ; downloadable for room & garden -> we can measure exactly |
| room | MeshSplatting + Draco connectivity + raw XYZ/SH | 32.8 MB topo+XYZ alone (qp14) | | [ours, 2026-10-13]; SH would add 1536 bit/v uncompressed |

## 3. Mesh-based real-time methods (the comparable class). Mostly dataset-level numbers; none report room alone.

| method | venue | Mip-360 PSNR | asset size | FPS | mesh colours | source |
|---|---|---|---|---|---|---|
| MeshSplatting | CVPR 2026 | avg 24.78 (3M vertices avg); 3DGS 27.21 in same table | not reported (PLY ~100 MB RGB-only) | not reported | per-vertex SH3 | arXiv 2512.06818 Table 1 |
| MiLo (mesh from GS) | SIGGRAPH Asia 2025 | avg 24.09 (MeshSplatting table) | 60-400 MB incl. vertex colours; garden 301-354 MB | not reported | vertex colours | anttwo.github.io/milo |
| RaDe-GS / GOF / 2DGS meshes | 2024 | avg 23.56 / 20.78 / 15.36 (31M / 33M / 2M vertices) | not reported | | | MeshSplatting Table 1 |
| Triangle Splatting (opaque) | 2025 | avg 21.05 (3M) | | | | MeshSplatting Table 1 |
| BakedSDF (real-time) | SIGGRAPH 2023 | indoor 27.06 / outdoor 22.47 | 434.5 MB avg disk | 72 FPS (MacBook browser) | baked | arXiv 2302.14859 |
| MobileNeRF | CVPR 2023 | outdoor 21.95; **garden 23.54** | **garden 239 MB**; outdoor avg 345 MB | garden 296 FPS desktop | textures + MLP | CVPR23 supp (no indoor/room) |
| Binary Opacity Grids | SIGGRAPH 2024 | indoor 27.71 / outdoor 23.94 | 0.85 GiB avg total (0.20 mesh + 0.65 appearance), 10M faces | 927 FPS desktop | baked features | arXiv 2402.12377 |
| Radiance Meshes (volumetric tets, not opaque) | CVPR 2026 | indoor 30.61 / outdoor 24.38 | not reported (9.3M / 4.2M prims) | 240 / 384 FPS | SH + linear gradient | arXiv 2512.04076 |

## 4. 3DGS and 3DGS compression on the same scenes (the class we cannot beat on PSNR/MB; cite, do not claim)

| scene | method | MB | PSNR | SSIM | LPIPS | source |
|---|---|---|---|---|---|---|
| room | 3DGS | 353.1 | 31.74 (30.63 in the 3DGS paper) | 0.926 | 0.197 | HAC supp |
| room | Scaffold-GS | 133.0 | 31.93 | 0.925 | 0.202 | HAC supp |
| room | HAC low..high | 5.53 .. 9.16 | 31.55 .. 32.19 | 0.921..0.929 | 0.208..0.194 | HAC supp |
| room | HAC++ low..high | 3.41 .. 7.55 | 31.75 .. 32.05 | 0.920..0.928 | 0.217..0.196 | arXiv 2501.12255 |
| room | Mip-NeRF 360 | n/a | 31.64 | 0.912 | | nerfbaselines |
| garden | 3DGS | 1328.0 | 27.30 (27.41 in paper) | 0.856 | 0.122 | HAC supp |
| garden | Scaffold-GS | 271.0 | 27.17 | 0.842 | 0.146 | HAC supp |
| garden | HAC low..high | 22.69 .. 36.27 | 27.28 .. 27.50 | | | HAC supp |
| garden | HAC++ low..high | 12.83 .. 27.22 | 27.19 .. 27.49 | | | arXiv 2501.12255 |
| avg | HAC++ / HAC / ContextGS / CompGS / Compact3DGS / LightGaussian / Reduced3DGS | 8.3-18.5 / 15.3-21.9 / 12.7-18.4 / 8.8-16.5 / 48.8 / 44.5 / 29.5 | 27.60-27.82 / 27.53-27.77 / 27.62-27.75 / 26.37-27.26 / 27.08 / 27.00 / 27.19 | | | HAC++ Table II |

## 5. Reading

- The 3.5 dB gap to 3DGS on room (28.26 vs 31.7) is MeshSplatting's, not the codec's (-0.38). HAC++ room = 3.4 MB at 31.75: 7x smaller and 3.9 dB higher than our default. PSNR-per-MB against 3DGS compressors is lost by construction; the paper must compare within the "decodes to a standard opaque mesh" class and add FPS / triangle count.
- Within that class, our assets (17-32 MB) are 4-30x smaller than MobileNeRF / BakedSDF / BOG / MiLo (239-850 MB) at similar or better PSNR (garden: 24.37 at 27 MB vs MobileNeRF 23.54 at 239 MB; indoor: 27.9 on room vs BOG indoor avg 27.71 / BakedSDF 27.06 -- but those are 4-scene averages, so we need all 9 scenes before claiming).
- The cheapest honest baseline is the authors' own RGB-only PLY (~100 MB, -2 dB): we keep view dependence at 20 bit/vertex for -0.2 dB where RGB-only (24 bit/vertex) loses 2 dB. Measure it on room and garden from their Google Drive files.
- Missing to be CVPR-complete: all 9 Mip-360 scenes (+ T&T), FPS of the decoded mesh in a rasterizer vs triangle count, LPIPS/SSIM curves, and one generic-codec baseline (Draco + 8-bit SH + zstd) built from the same MeshSplatting model.

Sources: MeshSplatting https://arxiv.org/html/2512.06818v1 , https://meshsplatting.github.io/ , https://github.com/meshsplatting/mesh-splatting ; HAC supp https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01178-supp.pdf ;
HAC++ https://arxiv.org/pdf/2501.12255 ; 3DGS.zip survey https://arxiv.org/pdf/2407.09510v1 ; BakedSDF https://arxiv.org/pdf/2302.14859v2 ; BOG https://arxiv.org/pdf/2402.12377v1 ;
MobileNeRF supp https://openaccess.thecvf.com/content/CVPR2023/supplemental/Chen_MobileNeRF_Exploiting_the_CVPR_2023_supplemental.pdf ; MiLo https://anttwo.github.io/milo/ ; Radiance Meshes https://arxiv.org/html/2512.04076v1 ; nerfbaselines https://nerfbaselines.github.io/m-mipnerf360
