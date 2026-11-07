# Table 2 deployability columns, draft v1 (2026-10-25)

Sources: rd-11scenes-v1.csv (bytes / delivered test PSNR), walltime-v1-2026-10-25.txt (training wall-clock, L40S, 30k iters incl. 4 prune rounds; Room original = lambda-0 proxy), decode-times-v1-2026-10-25.txt (single CPU core, cs655), fps-bench-v1-2026-10-25.txt (viewer.html, 300 GPU-bound frames at full capture resolution, SH degree 3; laptop = MacBook Pro M2 Max Chrome 154; phone = iPhone 17 Pro, iOS 18.7 Safari 27).

## Per scene, lambda 6.2e-3 point (the ~73x operating point)

| scene | original MB / test | asset MB | ratio | dB vs orig | V / F | train min orig -> ours | decode s | laptop ms (fps) @ res | phone ms (fps) @ res |
|---|---|---|---|---|---|---|---|---|---|
| room | 742 / 28.26 | 5.2 | 142x | -0.88 | 0.44M / 1.02M | 51 -> 27 (0.52x) | 2.3 | 1.2 (820) @ 3114x2075 | 2.8 (352) @ 3114x2075 |
| garden | 740 / 24.73 | 10.2 | 73x | -1.08 | 0.83M / 2.16M | 49 -> 24 (0.48x) | 4.4 | 1.8 (566) @ 5187x3361 | 6.3 (160) @ 5187x3361 |
| bonsai | 662 / 27.29 | 6.1 | 109x | -1.13 | 0.52M / 1.13M | 53 -> 29 (0.55x) | 2.4 | - | - |
| counter | 575 / 25.94 | 6.7 | 86x | -0.68 | 0.52M / 1.36M | 58 -> 33 (0.57x) | 2.8 | - | - |
| kitchen | 456 / 26.45 | 7.5 | 61x | -1.05 | 0.55M / 1.64M | 54 -> 34 (0.63x) | 3.2 | - | - |
| bicycle | 680 / 23.03 | 11.9 | 57x | -0.35 | 1.08M / 2.41M | 41 -> 25 (0.60x) | 5.4 | 1.9 (523) @ 4946x3286 | 6.9 (145) @ 4946x3286 |
| stump | 698 / 24.93 | 8.9 | 79x | -0.81 | 0.80M / 1.74M | 41 -> 20 (0.49x) | 3.7 | - | - |
| flowers | 668 / 19.45 | 12.0 | 56x | -0.24 | 1.10M / 2.36M | 44 -> 26 (0.59x) | 5.1 | - | - |
| treehill | 692 / 20.39 | 11.9 | 58x | -0.01 | 1.12M / 2.26M | 43 -> 26 (0.60x) | 5.1 | - | - |
| truck | 432 / 22.39 | 9.4 | 46x | -0.02 | 0.77M / 1.75M | 31 -> 17 (0.54x) | 3.9 | - | - |
| train | 485 / 18.90 | 13.4 | 36x | -0.22 | 1.14M / 2.75M | 33 -> 21 (0.66x) | 6.4 | - | - |
| **mean** | | | 73x | -0.59 | | 0.57x (11 scenes) | 4.1 | | |

Room original training time is the lambda-0 run of the same code (50.9 min); the other originals are the upstream run (<scene>-vis-collapse). Room ratio uses the delivered 5.23 MB asset.

## lambda 1.85e-3 point (the ~32x operating point): training time only

| scene | asset MB | ratio | dB | train min orig -> ours | decode s (Room only) |
|---|---|---|---|---|---|
| room | 12.6 | 59x | -0.26 | 51 -> 34 (0.66x) | 5.96 |
| garden | 24.6 | 30x | -0.23 | 49 -> 38 (0.77x) | - |
| bonsai | 14.0 | 47x | -0.64 | 53 -> 38 (0.71x) | - |
| counter | 14.8 | 39x | -0.15 | 58 -> 43 (0.74x) | - |
| kitchen | 16.7 | 27x | -0.21 | 54 -> 45 (0.84x) | - |
| bicycle | 27.0 | 25x | -0.02 | 41 -> 40 (0.99x) | - |
| stump | 21.4 | 33x | -0.21 | 41 -> 32 (0.77x) | - |
| flowers | 24.2 | 28x | -0.38 | 44 -> 39 (0.88x) | - |
| treehill | 27.4 | 25x | +0.06 | 43 -> 43 (1.00x) | - |
| truck | 19.3 | 22x | -0.19 | 31 -> 28 (0.92x) | - |
| train | 25.4 | 19x | -0.24 | 33 -> 34 (1.06x) | - |

Means: lambda 1.85e-3 trains in 0.87x of the original (10 scenes, range 0.71-1.06), lambda 6.2e-3 in 0.57x (0.48-0.66). The saving is the 2x/4x supersampling phase after 20k/25k iterations running on fewer vertices; Room lambda sweep: lambda 0 / 3e-4 / 1e-3 = 51-52 min (limiter-bound at 2.9M vertices), 3e-3 34, 1e-2 26, 3e-2 24 min.

## Sentences the table supports

- Decode: 2.3-6.4 s on one CPU core for 0.44-1.14 M vertices, 4.6-5.8 us/vertex, no GPU; output is a standard indexed mesh plus per-vertex codebook indices.
- Rendering: the decoded asset renders through the ordinary rasterisation pipeline (index buffer + vertex-shader SH from two codebook textures). On a laptop GPU every scene renders in < 2 ms at full capture resolution (16 Mpx for bicycle / garden); on an iPhone 17 Pro the 142x Room asset renders in 2.8 ms (350 fps) at 3114x2075, and the 2.2-2.4 M-face garden / bicycle assets in 6.3-6.9 ms (145-160 fps) at 16-17 Mpx, i.e. 4x the training resolution. Fig. 1 inset: analysis/figures/viewer-iphone17pro-bicycle-2026-10-25.jpg.
- Training: rate control shortens training (0.57x at lambda 6.2e-3), it never lengthens it; no extra pass.

## R9: train / test PSNR gap (gap-table-v1-2026-10-25.txt; "train" = the 6-view training sample upstream evaluates at iteration 30000, so +-0.3 dB noise)

| scene | orig train/test/gap | lambda 1.85e-3 | lambda 6.2e-3 | d train (hi-lo) | d test (hi-lo) | test/train share |
|---|---|---|---|---|---|---|
| room | - | 30.23/28.27/+1.96 | 29.25/27.56/+1.69 | +0.98 | +0.71 | +0.72 |
| garden | 25.22/24.75/+0.48 | 25.09/24.61/+0.49 | 23.85/23.66/+0.19 | +1.25 | +0.95 | +0.76 |
| bonsai | 26.66/27.36/-0.70 | 26.40/26.85/-0.45 | 25.22/26.17/-0.95 | +1.18 | +0.68 | +0.57 |
| counter | 26.82/26.07/+0.75 | 26.89/26.16/+0.73 | 26.04/25.55/+0.48 | +0.85 | +0.61 | +0.72 |
| kitchen | 27.70/26.39/+1.30 | 28.03/26.53/+1.50 | 26.89/25.61/+1.27 | +1.14 | +0.92 | +0.80 |
| bicycle | 23.39/23.04/+0.35 | 23.70/23.23/+0.47 | 23.01/22.85/+0.16 | +0.69 | +0.38 | +0.55 |
| stump | 25.53/24.99/+0.54 | 25.57/25.04/+0.54 | 24.54/24.36/+0.18 | +1.03 | +0.67 | +0.65 |
| flowers | 20.46/19.45/+1.01 | 20.72/19.53/+1.19 | 20.25/19.55/+0.70 | +0.47 | -0.03 | -0.05 |
| treehill | 17.96/20.39/-2.43 | 18.28/20.53/-2.25 | 17.69/20.46/-2.77 | +0.59 | +0.07 | +0.11 |
| truck | 23.55/22.39/+1.16 | 23.53/22.26/+1.27 | 23.24/22.38/+0.86 | +0.29 | -0.12 | -0.40 |
| train | 20.20/18.90/+1.30 | 20.44/18.80/+1.64 | 20.34/18.83/+1.51 | +0.10 | -0.03 | -0.26 |

Reading: on the seven standard Mip-360 scenes the bytes bought by the smaller lambda raise test PSNR by 55-80 % of the training-view gain (generalising detail); on flowers / treehill / truck / train the same step raises training PSNR (+0.1..+0.6) but test PSNR not at all (-0.12..+0.07): there the extra vertices fit the training views only, so the 36-58x asset is the better operating point and the RD curve is scene-dependent. The generalisation gap itself does not widen with rate control: original vs lambda 1.85e-3 gap differs by <=0.35 dB on 10/10 scenes (treehill/bonsai negative gaps come from the 6-view training sample).
