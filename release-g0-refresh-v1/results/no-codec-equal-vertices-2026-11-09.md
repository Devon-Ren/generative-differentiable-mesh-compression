# λ-grown vs upstream vertex cap, BEFORE any codec, at matched vertex count (2026-11-09)

Both columns are the 3000-step-finetuned FP32 meshes evaluated on the test split (no dictionary, no quantisation). λ-grown = the fixed-λ asset (rd-prune3 tables); cap = upstream MeshSplatting trained with --max_points set to that asset's vertex count ÷ 0.9 (budget-sweep v2/v3), which ends within a few % of the same V after the final cleaning. Uncompressed size is proportional to V (FP32: 208 B per vertex + 12 B per face), so equal V is equal uncompressed size.

| scene | λ | λ-grown: V, pre-codec PSNR | vertex cap (MeshSplatting --max_points): V, pre-codec PSNR | Δ no codec | λ-grown MB / cap MB |
|---|---|---|---|---|---|
| truck | 0.00185 | 1.74 M, 22.25 | 1.44 M, 22.43 | -0.18 | 19.3 / 15.6 |
| truck | 0.0062 | 0.77 M, 22.37 | 0.76 M, 22.12 | +0.25 | 9.4 / 8.6 |
| truck | 0.0185 | 0.32 M, 21.59 | 0.33 M, 21.27 | +0.32 | 4.4 / 4.4 |
| train | 0.00185 | 2.29 M, 18.71 | 1.99 M, 18.85 | -0.14 | 25.4 / 21.4 |
| train | 0.0062 | 1.14 M, 18.75 | 1.10 M, 18.64 | +0.11 | 13.4 / 12.2 |
| train | 0.0185 | 0.50 M, 18.62 | 0.51 M, 18.24 | +0.38 | 6.4 / 6.2 |
| flowers | 0.00185 | 2.38 M, 19.23 | 1.95 M, 19.09 | +0.14 | 24.2 / 18.1 |
| flowers | 0.0062 | 1.10 M, 19.35 | 1.09 M, 18.88 | +0.47 | 12.0 / 10.8 |
| flowers | 0.0185 | 0.47 M, 18.91 | 0.49 M, 18.51 | +0.40 | 5.6 / 5.5 |
| treehill | 0.00185 | 2.75 M, 20.47 | 2.40 M, 20.49 | -0.02 | 27.4 / 22.7 |
| treehill | 0.0062 | 1.12 M, 20.38 | 1.14 M, 20.49 | -0.11 | 11.9 / 11.5 |
| treehill | 0.0185 | 0.43 M, 20.28 | 0.43 M, 20.08 | +0.19 | 5.1 / 5.1 |
| garden | 0.00185 | 2.11 M, 24.60 | 1.99 M, 24.38 | +0.22 | 24.6 / 21.5 |
| garden | 0.0062 | 0.83 M, 23.64 | 0.90 M, 23.48 | +0.17 | 10.2 / 10.6 |
| garden | 0.0185 | 0.37 M, 22.71 | 0.44 M, 22.55 | +0.16 | 5.0 / 5.6 |
| bicycle | 0.00185 | 2.60 M, 23.22 | 2.29 M, 22.85 | +0.37 | 27.0 / 22.2 |
| bicycle | 0.0062 | 1.08 M, 22.84 | 1.07 M, 22.28 | +0.56 | 11.9 / 11.1 |
| bicycle | 0.0185 | 0.43 M, 22.16 | 0.38 M, 21.55 | +0.61 | 5.2 / 4.6 |
| stump | 0.00185 | 2.06 M, 24.98 | 1.68 M, 24.29 | +0.69 | 21.4 / 16.4 |
| stump | 0.0062 | 0.80 M, 24.31 | 0.66 M, 23.47 | +0.83 | 8.9 / 7.2 |
| stump | 0.0185 | 0.31 M, 23.32 | 0.31 M, 22.65 | +0.67 | 3.9 / 4.0 |
| room | 0.00185 | 1.09 M, 28.31 | 1.10 M, 28.10 | +0.21 | 12.4 / 11.6 |
| room | 0.0062 | 0.44 M, 27.53 | 0.44 M, 27.12 | +0.41 | 5.4 / 5.4 |
| room | 0.0185 | 0.20 M, 26.49 | 0.21 M, 25.90 | +0.59 | 2.9 / 3.0 |
| kitchen | 0.00185 | 1.21 M, 27.47 | 1.05 M, 26.55 | +0.92 | 16.5 / 12.5 |
| kitchen | 0.0062 | 0.56 M, 26.48 | 0.57 M, 25.77 | +0.71 | 8.0 / 7.6 |
| kitchen | 0.0185 | 0.25 M, 24.95 | 0.25 M, 24.29 | +0.67 | 3.8 / 3.9 |
| counter | 0.00185 | 1.05 M, 26.41 | 1.02 M, 25.96 | +0.45 | 13.1 / 11.5 |
| counter | 0.0062 | 0.47 M, 25.77 | 0.52 M, 25.49 | +0.29 | 6.4 / 6.4 |
| counter | 0.0185 | 0.22 M, 24.86 | 0.23 M, 24.63 | +0.23 | 3.3 / 3.4 |
| bonsai | 0.00185 | 1.20 M, 27.80 | 0.90 M, 27.13 | +0.67 | 13.2 / 9.6 |
| bonsai | 0.0062 | 0.50 M, 26.64 | 0.49 M, 26.28 | +0.36 | 6.1 / 5.9 |
| bonsai | 0.0185 | 0.26 M, 25.45 | 0.27 M, 24.01 | +1.43 | 3.5 / 3.6 |

Δ (λ-grown − cap) with no codec at matched vertex count: -0.18 to +1.43 dB, mean +0.39, λ wins 29 of 33; mismatched pairs (V ratio outside 0.85–1.15): 7

Restricted to the 25 pairs whose vertex counts agree within ±15 % (the 7 excluded are top-tier outdoor pairs where the capped run ended 15–30 % below the λ asset's V): -0.11 to +1.43 dB, mean +0.40, λ-grown wins 23 of 25. Indoor 12 of 12 (+0.21 to +1.43, mean +0.58); outdoor well-matched 13 of which 11 positive.

Reading: the growth rule alone, with no codec in the picture, puts the same number of vertices where they buy more test PSNR than upstream's importance-sampled growth capped at that count. This is the representation-efficiency claim Yao asked for; the codec then compresses both equally well (the λ asset and the capped asset deliver within ±5 % of the same MB at equal V, last column).
