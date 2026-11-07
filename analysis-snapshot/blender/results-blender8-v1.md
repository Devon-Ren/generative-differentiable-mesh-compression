# NeRF-Synthetic 8-scene probe — results v1 (2026-11-04, jobs 21850758-21850777, general, ~8 GPU-h total)

Protocol: upstream MeshSplatting `--indoor`, **black background** (delivery chain hardcodes it), checkpoint = best OFFICIAL val split PSNR (val_select_v2, importance<=0.5 prune reproduced), then level-1.0 delivery (K4096 R256, xyz q14, dc q7, coarse finetune) → test = 200 official test views. Originals: train.py via train_ckpt_v2; λ rows: train_rd_v3 (prices BPV 10, BPF 2, MU 1, MAXF 0.15, prune rounds 12k/14k/16k/18k).
Raw tables: /scratch/zr2245/analysis/originals-blender-val4.txt, rd-prune3b4-<scene>.txt.

## 8-scene means

| config | val-best (raw mesh) | finetune test | **delivered test** | SSIM | LPIPS | **MB** | kV | MF | bytes vs FP32 | ΔdB vs FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| FP32 original (val-selected, pruned) | 31.55 | 31.24 | 30.93 | 0.9502 | 0.0721 | 4.48 | 258 | 1.00 | 1.00 | — |
| λ = 6.2e-4 | 31.87 | 31.55 | 31.21 | 0.9519 | 0.0684 | 5.10 | 276 | 1.20 | 1.14 | +0.29 |
| λ = 6.2e-3 | 31.42 | 31.21 | 30.93 | 0.9502 | 0.0712 | 2.63 | 123 | 0.54 | 0.59 | +0.01 |

3DGS (paper, white bg) mean 33.31. Codec loss FP32 finetune→delivered 0.32 dB; val-best→delivered-test 0.62 dB (split difference included).
Gate (plan/object-dataset-decision-2026-11-04.md): FP32 val-best mean ≥ 31 → **passed (31.55) → P3**.

## Per-scene (delivered MB / test PSNR; black bg)

| scene | FP32 val | FP32 | λ6.2e-4 | λ6.2e-3 | 3DGS (white) | gap FP32→3DGS | best iters (FP32/λa/λb) |
|---|---|---|---|---|---|---|---|
| chair | 32.33 | 4.01 / 31.73 | 5.58 / 32.20 | 2.65 / 32.16 | 35.83 | 4.1 | 22k/26k/21k |
| drums | 26.32 | 4.46 / 25.36 | 5.38 / 25.51 | 3.33 / 25.49 | 26.15 | 0.8 | 26k/26k/26k |
| ficus | 31.84 | 1.69 / 31.92 | 2.07 / 32.10 | 1.54 / 31.98 | 34.87 | 3.0 | 26k/26k/26k |
| hotdog | 35.71 | 4.96 / 35.21 | 4.16 / 35.81 | 2.19 / 35.44 | 37.72 | 2.5 | 21k/21k/21k |
| lego | 34.88 | 7.89 / 33.20 | 9.26 / 33.50 | 3.10 / 32.58 | 35.78 | 2.6 | 22k/26k/21k |
| materials | 28.69 | 3.89 / 27.81 | 4.46 / 27.76 | 2.43 / 27.74 | 30.00 | 2.2 | 22k/22k/22k |
| mic | 32.94 | 2.56 / 33.10 | 2.96 / 33.21 | 1.91 / 32.76 | 35.36 | 2.3 | 21k/21k/21k |
| ship | 29.65 | 6.40 / 29.08 | 6.95 / 29.61 | 3.90 / 29.31 | 30.80 | 1.7 | 26k/26k/26k |

## Reading
1. **Rate control transfers to objects unchanged**: λ6.2e-3 gives the same delivered PSNR as the FP32 original at 59 % of the bytes (8/8 scenes within ±0.6 dB, 5/8 above). λ6.2e-4 grows *more* than the --indoor schedule (276k vs 258k V) and is +0.29 dB at +14 % bytes → the FP32 original is a dominated RD point on this dataset too.
2. **Val-selection is mandatory**: every run peaks at 21-26k; 30k is 0.4-2.9 dB lower on val. Same schedule artefact as lego.
3. **Class gap to 3DGS ≈ 2.4 dB mean**, but background differs (ours black, 3DGS white) → not yet a paper number. Largest gaps: chair 4.1, ficus 3.0 (thin leaves), lego 2.6, hotdog 2.5. Smallest: drums 0.8, ship 1.7.
4. Codec cost on objects ≈ 0.3 dB (same as Mip-360). Normal prior: no gain (lego v4n). Supersample: no gain.
5. Open before any table: white-background chain (-v2, new files), real raw-size ratios (du of pruned state dict), chair/ficus render inspection, literature baselines on Synthetic-NeRF (3DGS/HAC/HAC++/Compact3DGS/LightGaussian/MobileNeRF/NeRF2Mesh).

## Baseline sources on NeRF-Synthetic (checked 2026-11-04 15:40; HAC++ values from a web-page extraction, verify against the PDF before use)
- MeshSplatting (arXiv 2512.06818): no NeRF-Synthetic results at all (Mip-360 / T&T / DTU only); qualitative figs = bicycle/truck/train vs MiLo, Triangle Splatting. Our 8-scene FP32 run is the first MeshSplatting number on Blender.
- HAC++ (arXiv 2501.12255) Table II Synthetic-NeRF (PSNR/SSIM/LPIPS/MB): 3DGS 33.80/.970/.031/68.46; Scaffold-GS 33.41/.966/.035/19.36; HAC low 33.24/.967/.037/1.18, high 33.71/.968/.034/1.86; ContextGS low 32.79/.965/.040/1.01, high 33.51/.968/.035/1.56; EAGLES 32.51/.964/.039/4.26; LightGaussian 32.73/.965/.037/7.84; RDO-Gaussian 33.12/.967/.035/2.31; Lee et al. (Compact3DGS?) 33.33/.968/.034/5.54; Compressed3D 32.94/.967/.033/3.68; HAC++ low 33.03/.966/.039/0.88, high 33.76/.969/.033/1.84. No rendered Synthetic figures (Fig. 9 = bit allocation on lego/materials). NOTE their 3DGS = 33.80 vs 3DGS paper 33.32.
- MobileNeRF: live web demos for all 8 scenes: https://storage.googleapis.com/jax3d-public/projects/mobilenerf/mobilenerf_viewer_mac/zdeferred_syn_mac_faster.html?obj=<scene>
- NeRF2Mesh: all 8 scene meshes released, HF ashawkey/nerf2mesh (MIT): mesh_0.obj + feat0_0.png + feat1_0.png + mlp.json per scene (lego 83 MB).
- Same-view qualitative figure needs our own renders: proposed (1) 3DGS self-trained, black bg, 8 scenes ~1 GPU-h; (2) NeRF2Mesh released meshes via their renderer; (3) HAC++ official code ~3 GPU-h (optional). Awaiting user's choice.
- Mesh-class on NeRF-Synthetic (verbatim tables, white bg): MobileNeRF PSNR 30.90 / SSIM .947 / LPIPS .062, disk 125.75 MB, 494k V / 224k tris, 800x800 (Tables 3/4/5); NeRF2Mesh "Ours (mesh)" 29.76 / .940 / .072, disk 73.53 MB (OBJ+PNG uncompressed), 200k V / 192k F; w/o L_smooth 31.04 but mic alone 133 MB (Tables 2/3/4/5). Our λ6.2e-3 2.63 MB / 30.93 (black bg) = 48x smaller than MobileNeRF at equal PSNR, 28x smaller / +1.2 dB vs NeRF2Mesh; HAC++ low 0.88 MB / 33.03 = 3x smaller and +2.1 dB (class gap, 3.3 dB on Mip-360).

## lego high-rate point (2026-11-04, rd-hirate-blender-v1, λ 6.2e-5, add_percentage 1.5, max_points 1.5 M, --indoor, black bg)
V@delaunay 727,500 (λ6.2e-4: 540,120); final 679,183 V / 2,556,482 F; val-best 22k 34.806 (pruned 34.839); finetune test 33.575; **delivered 10.902 MB / test 33.276 / SSIM .9711 / LPIPS .0385**. λ6.2e-4 = 9.259 MB / 33.496 / val 35.195 → the high-rate point is DOMINATED (+18 % bytes, −0.2 dB test, −0.4 dB val). Did not reach 25-30 MB: upstream per-interval opacity/size pruning sets the equilibrium mesh size, not the growth limiter. Reading: lego RD hull saturates at ~9 MB; extra vertices are pruned or fit training views. msv: viewer-v2/msv/lego-hi-lam0.000062.msv (53 MB, decode 3.9 s). Decision: no further high-rate runs.

## Per-scene baseline rows, NeRF-Synthetic (verbatim from appendices, 2026-11-05; white bg)
- 3DGS paper Table 2 (PSNR only): Mic 35.36, Chair 35.83, Ship 30.80, Materials 30.00, Lego 35.78, Drums 26.15, Ficus 34.87, Hotdog 37.72, Avg 33.32. Also Mip-NeRF lego 35.70, INGP-Base 36.39, Plenoxels 34.10, Point-NeRF 35.04.
- HAC Table G (3DGS re-run) lego 35.98/.982/.017/80.53 MB; Table H (Scaffold-GS) lego 35.48/.981/.018/19.84 MB.
- HAC Table B lego (λe → PSNR/SSIM/LPIPS/MB): 0.004 34.25/.976/.027/0.97; 0.003 34.65/.977/.024/1.07; 0.002 35.04/.979/.022/1.25; 0.001 35.51/.981/.019/1.61; 0.0005 35.67/.981/.019/1.90.
- HAC++ Table A lego: 4e-3 34.33/.976/.027/0.80; 3e-3 34.64/.977/.025/0.97; 2e-3 35.10/.979/.022/1.08; 1e-3 35.52/.981/.019/1.49; 0.5e-3 35.77/.982/.018/1.83.
- MobileNeRF supp Tables 8-11 lego: 34.18/.975/.025, disk 199 MB (SNeRG lego 33.82, 114 MB; JAXNeRF 33.24; NeRF 32.54). Per-scene MobileNeRF disk MB: chair 107, drums 120, ficus 80, hotdog 88, lego 199, materials 191, mic 50, ship 171 (mean 125.75).
- NeRF2Mesh Table 7 lego: mesh 32.07/.957/.047; volume 33.66/.972/.055; mesh w/o L_smooth 34.90/.977/.025. Other scenes (same table, summary-extracted earlier, verify): chair mesh 31.93, hotdog 34.11, ship 28.69. No per-scene size.
- Lego reading: ours λ6.2e-4 9.26 MB / 33.50 (black) vs MobileNeRF 34.18 @ 199 MB, NeRF2Mesh 32.07; HAC++ 34.3-35.8 @ 0.8-1.8 MB → gap at equal bytes 2.5-3 dB on lego (worse than the 2.1 dB mean).
