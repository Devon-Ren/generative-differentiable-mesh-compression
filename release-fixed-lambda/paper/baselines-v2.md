# Baselines for Table 2 (mesh class + 3DGS class), v2, 2026-10-30

Replaces analysis/baselines/baselines-room-garden-v1.md as the source for the paper's cross-method table. Every published number below was copied today from the arXiv HTML/PDF tables named in the "source" column (fetched 10/30); our numbers come from analysis/rd/rd-11scenes-v1.csv and rd/rd-prune3-*.csv. Protocol: Mip-NeRF 360 = 9 scenes, T&T = truck + train (same two scenes for HAC++, MeshSplatting and us). Columns are the dataset means as the papers print them; "x" marks a number the paper does not report.

## A. Our rows (decoded asset, test views; SSIM/LPIPS from our evaluator)

| point | M360 PSNR | SSIM | LPIPS | MB | T&T PSNR | SSIM | LPIPS | MB | standard mesh | decode | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MeshSplatting, our reproduction (FP32) | 24.50 | x | x | ~650 | 20.64 | x | x | 458 | yes | - | - |
| ours lambda 1.85e-3 | 24.27 | 0.712 | 0.400 | 20.3 | 20.43 | 0.745 | 0.331 | 22.3 | yes | 3-6 s CPU | 145-350 (phone), >500 (laptop) |
| ours lambda 6.2e-3 | 23.81 | 0.676 | 0.460 | 8.9 | 20.52 | 0.737 | 0.358 | 11.4 | yes | 2-5 s CPU | same |

Two things to settle before these rows go in: (1) our reproduced originals are below the MeshSplatting paper on three indoor scenes (bonsai -0.90, kitchen -0.97, counter -0.57 dB; the other eight within +-0.26), so the table should carry the paper's MeshSplatting row AND our reproduction row, with the loss of our codec stated relative to the reproduction; (2) our LPIPS (0.40 at lambda 1.85e-3) vs the paper's 0.310 for the same model class is most likely an evaluator difference (VGG vs AlexNet LPIPS); re-evaluate the originals and our assets with one evaluator before printing LPIPS (readiness item "Table 1 其余场景 + 同一 evaluator 重评 LPIPS").

## B. Mesh class (decodes to a mesh an engine can load)

| method | venue | M360 PSNR / SSIM / LPIPS | T&T PSNR / SSIM / LPIPS | size | FPS | primitives | source |
|---|---|---|---|---|---|---|---|
| MeshSplatting | 2025 (arXiv 2512.06818) | 24.78 / 0.728 / 0.310 | 20.52 / 0.745 / 0.287 | 100 MB ("Memory", Table 2; RGB-only PLY) | 220 HD / 190 Full HD (MacBook M4) | 3 M v (M360), 2 M v (T&T) | Table 1, Table 2; per-scene supp Tables 6-8 |
| MiLo | SIGGRAPH Asia 2025 | 24.09 / 0.688 / 0.323 | 21.46 / 0.706 / 0.348 | 253 MB | 170 / 160 | 7 M v / 4 M v | MeshSplatting Tables 1-2 |
| RaDe-GS mesh | 2024 | 23.56 / 0.668 / 0.361 | 20.51 / 0.659 / 0.344 | 1.1 GB | OOM on M4 | 31 M v / 10 M v | MeshSplatting Tables 1-2 |
| GOF mesh | 2024 | 20.78 / 0.573 / 0.465 | 21.69 / 0.690 / 0.326 | 1.5 GB | OOM | 33 M v / 12 M v | same |
| 2DGS mesh | 2024 | 15.36 / 0.498 / 0.474 | 14.23 / 0.569 / 0.485 | x | x | 2 M v / 16 M v | same |
| Triangle Splatting, opaque-only | 2025 | 21.05 / 0.558 / 0.462 | 17.27 / 0.600 / 0.402 | x | x | 3 M / 6 M | MeshSplatting Table 1 (dagger row) |
| BakedSDF (real-time) | SIGGRAPH 2023 | indoor 27.06 / outdoor 22.47 (PSNR) | x | 434 MB avg | 72 (browser) | x | v1 file (arXiv 2302.14859); per-scene to re-check |
| Binary Opacity Grids | SIGGRAPH 2024 | indoor 27.71 / outdoor 23.94 (PSNR) | x | 0.85 GiB avg | 927 | 10 M faces | v1 file (arXiv 2402.12377) |
| MobileNeRF | CVPR 2023 | outdoor 21.95 (no indoor) | x | 345 MB outdoor avg | 296 (garden) | x | v1 file |

Not mesh assets, cite as context only: Triangle Splatting soft (27.14-27.16 / 0.814 / 0.191 on M360; 23.14 / 0.857 / 0.143 T&T; 97 FPS; "memory" 795 MB, Table 2 of arXiv 2505.19175), Radiance Meshes (volumetric tets; indoor 30.61 / outdoor 24.38, v1 file).

## C. 3DGS class (HAC++ Table II, arXiv 2501.12255, verbatim; size in MB)

| method | M360 PSNR / SSIM / LPIPS / MB | T&T PSNR / SSIM / LPIPS / MB |
|---|---|---|
| 3DGS | 27.46 / 0.812 / 0.222 / 750.9 | 23.69 / 0.844 / 0.178 / 431.0 |
| Scaffold-GS | 27.50 / 0.806 / 0.252 / 253.9 | 23.96 / 0.853 / 0.177 / 86.50 |
| Compact3DGS (Lee et al.) | 27.08 / 0.798 / 0.247 / 48.80 | 23.32 / 0.831 / 0.201 / 39.43 |
| Compressed3D (Niedermayr) | 26.98 / 0.801 / 0.238 / 28.80 | 23.32 / 0.832 / 0.194 / 17.28 |
| EAGLES | 27.14 / 0.809 / 0.231 / 58.91 | 23.28 / 0.835 / 0.203 / 28.99 |
| LightGaussian | 27.00 / 0.799 / 0.249 / 44.54 | 22.83 / 0.822 / 0.242 / 22.43 |
| SOG | 26.56 / 0.791 / 0.241 / 16.70 | 23.15 / 0.828 / 0.198 / 9.30 |
| Navaneet et al. | 27.12 / 0.806 / 0.240 / 19.33 | 23.44 / 0.838 / 0.198 / 12.50 |
| Reduced3DGS | 27.19 / 0.807 / 0.230 / 29.54 | 23.57 / 0.840 / 0.188 / 14.00 |
| RDO-Gaussian | 27.05 / 0.802 / 0.239 / 23.46 | 23.34 / 0.835 / 0.195 / 12.03 |
| MesonGS-FT | 26.99 / 0.796 / 0.247 / 27.16 | 23.32 / 0.837 / 0.193 / 16.99 |
| HAC low / high | 27.53 / 0.807 / 0.238 / 15.26 ; 27.77 / 0.811 / 0.230 / 21.87 | 24.04 / 0.846 / 0.187 / 8.10 ; 24.40 / 0.853 / 0.177 / 11.24 |
| ContextGS low / high | 27.62 / 0.808 / 0.237 / 12.68 ; 27.75 / 0.811 / 0.231 / 18.41 | 24.20 / 0.852 / 0.184 / 7.05 ; 24.29 / 0.855 / 0.176 / 11.80 |
| CompGS low / high | 26.37 / 0.778 / 0.276 / 8.83 ; 27.26 / 0.803 / 0.239 / 16.50 | 23.11 / 0.815 / 0.236 / 5.89 ; 23.70 / 0.837 / 0.208 / 9.60 |
| HAC++ low / high | 27.60 / 0.803 / 0.253 / 8.34 ; 27.82 / 0.811 / 0.231 / 18.48 | 24.22 / 0.849 / 0.190 / 5.18 ; 24.32 / 0.854 / 0.178 / 8.63 |

FPS (HAC++ Table XI, Mip-360): 3DGS 99, Scaffold-GS 135, HAC++ 130-151 (desktop GPU); the other codecs report no FPS. Per-scene 3DGS / Scaffold-GS / HAC / HAC++ numbers for room and garden are in the v1 file (HAC supp, HAC++ appendix).

## D. What the table says and how to write it

- Against the 3DGS class we lose 3.3 dB on Mip-360 (24.27 vs HAC++ 27.60 at a similar 8-20 MB) and 3.8 dB on T&T, with lower SSIM and higher LPIPS. This is the representation's gap (MeshSplatting 24.78 vs 3DGS 27.46 in the authors' own tables), not the codec's (-0.22 dB). Write it as a fact, in the same table, no greying; the columns that separate the classes are "standard mesh" (yes / no), decode on CPU (seconds / needs the method's own decoder and renderer), and FPS on a phone (ours 145-350; no 3DGS codec reports a phone number).
- Within the mesh class we are the only compressed entry: MeshSplatting's own file is 100 MB RGB-only (about -2 dB), MiLo 253 MB, BakedSDF 434 MB, BOG 850 MB, RaDe-GS / GOF > 1 GB; ours 9-22 MB at the original's quality within 0.2-0.6 dB. Indoor-vs-outdoor splits (BakedSDF, BOG) need our per-scene means in the same split to be comparable (ours lambda 1.85e-3: indoor room/counter/kitchen/bonsai mean 26.67, outdoor five-scene mean 22.35; compute from rd-11scenes-v1.csv when the table is typeset).
- Recommendation for the row set: Table 2 = 3DGS, Scaffold-GS, LightGaussian, Compact3DGS, RDO-Gaussian, HAC, HAC++ (the ones reviewers know) + MeshSplatting (paper), MeshSplatting (our repro), MiLo, BakedSDF, BOG + our two lambda points. The remaining 3DGS codecs go to a supplementary table.

## E. Baselines we have to run ourselves (nothing else is needed)

1. MeshSplatting authors' RGB-only PLY on room and garden (their Google Drive files): measure MB and test PSNR with our evaluator, to replace "about -2 dB" by a number. CPU, minutes.
2. One evaluator for LPIPS/SSIM over originals + our assets on all 11 scenes (the readiness item; ~10 min GPU per scene).
3. Optional: 3DGS official checkpoints' sizes per scene are already in HAC's supp; no run needed.

## F. Citation keys to add
HAC++ 2501.12255; HAC 2403.14530; ContextGS 2405.20721; CompGS 2404.09458; MesonGS 2409.09756; SOG 2312.13299; Navaneet (CompGS k-means) 2311.18159; Reduced3DGS 2406.xxxx (Papantonakis et al., i3D 2024); Compressed3D (Niedermayr) 2401.02436; EAGLES 2312.04564; LightGaussian 2311.17245; Compact3DGS (Lee) 2311.13681; Scaffold-GS 2312.00109; RDO-Gaussian 2406.01597; MeshSplatting 2512.06818; Triangle Splatting 2505.19175; MiLo (anttwo.github.io/milo); BakedSDF 2302.14859; BOG 2402.12377; MobileNeRF 2208.00277. arXiv ids marked xxxx or taken from memory must be checked when the bib is written.

## G. Table vs RD policy (decided with the user 10/31; this is the rule for the paper)

**Rule.** An RD *curve* is run by us only where the input is the same asset: Table 1 (generic codecs on our MeshSplatting model) and our own lambda sweep. Cross-representation baselines (3DGS class, other mesh methods) enter as **published points** at the standard protocol (Mip-360 9 scenes / T&T truck+train, test = every 8th view): one point per method, two for methods that publish a low/high-rate pair (HAC, HAC++, ContextGS, CompGS), drawn as markers (or two-point segments) in the same RD figure. Nothing in the 3DGS class is re-run: re-running reproduces their published numbers with noise and the comparison is cross-class with a 3 dB representation gap; nothing in the mesh class can be re-run at CVPR cost (BakedSDF / BOG are JAX pipelines with hours per scene, MiLo has no rate knob). So: table rows = published points, RD curves = ours + generic codecs, both in one figure.

**What this means per table.**
- Table 1 (same input, RD): ours level 1.0 + lambda points + post-hoc collapse curve vs quantised PLY+zstd at 2-3 precisions, Draco at 2-3 qp, texture route. Already measured on room + garden; extend to the other 9 scenes (CPU, the readiness item). Our own curve needs **at least 3 lambda points per scene** to be a curve: today 11 scenes x 2 lambdas + Room 6-point sweep. Add lambda 6.2e-4 (high-rate point near the original) on the 10 remaining scenes: ~0.9 GPU-h each (training 0.9x of original), ~9 GPU-h total, no code change (evaluation of the finalized method, not optimisation).
- Table 2 (cross-representation, points): rows below, dataset means, columns PSNR / SSIM / LPIPS / MB / mesh output (yes-no) / FPS with hardware footnote / decode. Two of our lambda points as rows; the full curve is in the figure.
- Figure RD (Mip-360 and T&T panels, x = MB log, y = PSNR): our curve solid, generic codecs dashed, published baselines as markers (3DGS class one colour, mesh class another, MeshSplatting paper + RGB-only PLY as reference points). Per-scene RD (Room full sweep + the 3-point curves) in the supplement.

**Row selection (keep the table short).**
- Mesh class, newest and most comparable first: MeshSplatting (paper; our reproduction; authors' RGB-only PLY measured by us), MiLo (SIGGRAPH Asia 2025), Binary Opacity Grids (SIGGRAPH 2024), BakedSDF (SIGGRAPH 2023, the real-time mesh classic), Triangle Splatting opaque-only (2025, the other triangle-primitive method). Drop from the main table: MobileNeRF (2023, outdoor only), RaDe-GS / GOF / 2DGS meshes (weak, from MeshSplatting's table) -> supplement. Radiance Meshes: one sentence as context (not an opaque mesh).
- 3DGS class, the ones reviewers know plus the SOTA pair: 3DGS, Scaffold-GS, LightGaussian, Compact3DGS, RDO-Gaussian, HAC, HAC++. ContextGS, CompGS, EAGLES, Reduced3DGS, Compressed3D, SOG, Navaneet, MesonGS -> supplement (all from HAC++ Table II, already in §C).
- Total main-table rows: 8 mesh-side (incl. 3 MeshSplatting variants and our 2 points) + 7 3DGS-side = 15.

**Number formats (fixed now, do not revisit).** PSNR two decimals, SSIM three, LPIPS three with the evaluator stated once in the caption, size in MB one decimal (MiB vs MB: HAC++ prints MB; we print MB = 1e6 bytes and say so), ratio as "Nx" relative to the FP32 original, dB loss relative to our reproduced original. LPIPS evaluator: the 3DGS `metrics.py` convention (LPIPS-VGG) which HAC++ inherits and MeshSplatting most likely inherits (verify in their repo); every one of our LPIPS numbers in the paper comes from that one script, run once over originals + assets, then frozen.

**The indoor deficit (issue 1 of 10/30).** All our originals and lambda runs were trained without upstream's `--indoor` flag (upstream's `update_indoor` changes the optimisation hyper-parameters for indoor scenes; confirmed in commands/batchA-originals-v1.md "no --indoor, as for the 7 scenes" and in train_rd_v3's argparse default False). That is the most likely cause of bonsai -0.90 / kitchen -0.97 / counter -0.57 vs the paper (room -0.26). Options: (a) report the reproduction as is, with the flag difference stated in one sentence, codec losses relative to our reproduction (no GPU); (b) retrain the four indoor originals with --indoor (~4 GPU-h) and, for consistency, their lambda points (~6 GPU-h more). Recommendation: (a) now, (b) only if time remains after Table 1 / qualitative figure; the paper's claims are relative, so (a) is sound.

## H. Final shortlist (user 10/31: at most 4 + 4), arXiv ids verified 10/31, RD-curve status read from the papers' HTML

Mesh class (4):
| # | paper | arXiv | why this one | has RD curve |
|---|---|---|---|---|
| M1 | MeshSplatting: Differentiable Rendering with Opaque Meshes (Held et al., 2025) | https://arxiv.org/abs/2512.06818 | the representation we compress; rows: paper numbers, our reproduction, authors' RGB-only PLY (measured by us) | no (single points; appendix has a triangle-count section, to check) |
| M2 | MILo: Mesh-In-the-Loop Gaussian Splatting (Guedon et al., ACM TOG 2025) | https://arxiv.org/abs/2506.24096 | newest mesh-from-splats method, same Mip-360/T&T protocol, reports mesh size (253 MB) and FPS | no (base/dense variants, one size per scene) |
| M3 | Binary Opacity Grids (Reiser et al., SIGGRAPH 2024) | https://arxiv.org/abs/2402.12377 | best-quality baked mesh pipeline, reports size (0.85 GiB) and FPS (927) | no |
| M4 | BakedSDF (Yariv et al., SIGGRAPH 2023) | https://arxiv.org/abs/2302.14859 | the real-time mesh classic, reports size (434 MB) and browser FPS | no |

3DGS class (4):
| # | paper | arXiv | why this one | has RD curve |
|---|---|---|---|---|
| G1 | 3D Gaussian Splatting (Kerbl et al., TOG 2023) | https://arxiv.org/abs/2308.04079 | the uncompressed reference point (750.9 MB Mip-360 / 431 MB T&T per HAC++ Table II) | no |
| G2 | HAC++: Towards 100X Compression of 3DGS (Chen et al., 2025) | https://arxiv.org/abs/2501.12255 | SOTA compressor, publishes RD curves (Fig. 5, 7) and the low/high pair we quote | yes |
| G3 | End-to-End Rate-Distortion Optimized 3D Gaussian Representation, RDO-Gaussian (Wang et al., ECCV 2024) | https://arxiv.org/abs/2406.01597 | the method closest in spirit (lambda-priced rate inside training), RD curves on all four datasets (Fig. 3) | yes |
| G4 | LightGaussian (Fan et al., 2023/24) | https://arxiv.org/abs/2311.17245 | the representative post-hoc prune + distil + VQ pipeline, the "post-hoc works on Gaussians" counterpart of our finding | mostly single points (Fig. 6 sweeps prune/VQ ratio) |

Swaps if wanted: G4 -> Compact 3D Gaussian Representation (Lee et al., https://arxiv.org/abs/2311.13681, RD curves in appendix Fig. 8) or HAC (https://arxiv.org/abs/2403.14530, RD curves Fig. 4). Dropped from the main table: Scaffold-GS, ContextGS, CompGS, EAGLES, Reduced3DGS, Compressed3D, SOG, MesonGS, Triangle Splatting opaque-only, MobileNeRF, RaDe-GS/GOF/2DGS meshes -> supplement table only, numbers already in §B/§C.

How RD is handled with this set: G2/G3 publish RD curves at the standard protocol -> their published low/high points (HAC++) and curve points (RDO-Gaussian, if the per-point values are tabulated; otherwise the two ends) go into our RD figure as markers; G1, G4 and all four mesh rows are single points. Our curve and the generic-codec curves are the only ones we compute (rule in §G).
