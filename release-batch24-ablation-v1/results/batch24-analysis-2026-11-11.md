
## Mip-NeRF 360 / T&T, target 10 MB (11 scenes; capacity-limited and excluded: none)

| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |
|---|---|---|---|---|---|---|
| full | 11 | +2.0 % | -4.7 .. +13.5 % | 10/11 | -0.12 (-0.28 .. -0.06)* | 0 (reference) |
| nocollapse | 11 | +4.5 % | -4.0 .. +20.4 % | 9/11 | -0.16 (-0.30 .. -0.05)* | -0.03 (-0.15 .. +0.04), 4/11 |
| constants | 11 | +5.0 % | -2.2 .. +19.7 % | 10/11 | -0.09 (-0.27 .. +0.00)* | +0.03 (-0.12 .. +0.10), 9/11 |
| matched | 11 | -27.2 % | -35.8 .. -16.0 % | 0/11 | -0.52 (-0.84 .. -0.31)* | -0.40 (-0.60 .. -0.19), 0/11 |

rate-aware ranking vs MeshSplatting importance sampling at the same split counts (both without collapse): nocollapse - matched = +0.55 dB mean (+0.26 .. +0.92), rate-aware better on 11/11; file size of the rate-aware mesh / importance mesh - 1 = +43.9 % (+34.9 .. +53.0 %)

## Mip-NeRF 360 / T&T, target 20 MB (11 scenes; capacity-limited and excluded: none)

| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |
|---|---|---|---|---|---|---|
| full | 11 | -0.5 % | -6.6 .. +10.0 % | 10/11 | -0.13 (-0.60 .. +0.28)* | 0 (reference) |
| nocollapse | 0 | not run yet | | | | |
| constants | 11 | +3.4 % | -2.4 .. +13.0 % | 10/11 | -0.12 (-0.59 .. +0.23)* | +0.01 (-0.05 .. +0.05), 9/11 |
| matched | 0 | not run yet | | | | |

### constant price, Mip-NeRF 360 / T&T: file size spread across scenes at one price (abl24 constlam rows)

- constlam-lam0.00185: 11 scenes, file size 10.8 .. 24.0 MB (max/min 2.2x)
- constlam-lam0.0062: 11 scenes, file size 4.7 .. 11.7 MB (max/min 2.5x)

## NeRF-Synthetic, target 2.5 MB (7 scenes; capacity-limited and excluded: ficus)

| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |
|---|---|---|---|---|---|---|
| full | 7 | +2.6 % | -6.6 .. +9.2 % | 7/7 | -0.27 (-0.87 .. -0.02)* | 0 (reference) |
| nocollapse | 7 | +9.0 % | -6.2 .. +14.9 % | 2/7 | -0.44 (-0.97 .. -0.01)* | -0.16 (-0.84 .. +0.20), 3/7 |
| constants | 7 | +12.8 % | -4.9 .. +24.8 % | 1/7 | -0.34 (-1.11 .. +0.01)* | -0.07 (-0.24 .. +0.05), 3/7 |
| matched | 7 | -11.3 % | -27.6 .. -6.2 % | 4/7 | -0.39 (-0.61 .. -0.11)* | -0.11 (-0.49 .. +0.34), 2/7 |

rate-aware ranking vs MeshSplatting importance sampling at the same split counts (both without collapse): nocollapse - matched = +0.39 dB mean (+0.06 .. +0.76), rate-aware better on 7/7; file size of the rate-aware mesh / importance mesh - 1 = +23.1 % (+14.0 .. +29.7 %)

## NeRF-Synthetic, target 1.5 MB (8 scenes; capacity-limited and excluded: none)

| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |
|---|---|---|---|---|---|---|
| full | 8 | +4.0 % | +1.0 .. +9.0 % | 8/8 | -0.03 (-0.22 .. +0.11)* | 0 (reference) |
| nocollapse | 0 | not run yet | | | | |
| constants | 8 | +17.9 % | +6.0 .. +28.6 % | 1/8 | -0.01 (-0.24 .. +0.15)* | +0.01 (-0.11 .. +0.09), 6/8 |
| matched | 0 | not run yet | | | | |

### constant price, NeRF-Synthetic: file size spread across scenes at one price (abl24 constlam rows)

- constlam-lam0.0062: 8 scenes, file size 1.3 .. 3.0 MB (max/min 2.3x)
- constlam-lam0.0185: 8 scenes, file size 1.0 .. 1.9 MB (max/min 1.9x)

### per-scene rows (file size MB / PSNR / size error; dPSNR vs the scene's constant-price curve)

- room 10 MB: full: 10.29 / 27.83 / +2.9 % / -0.18 | nocollapse: 10.80 / 27.82 / +8.0 % / -0.24* | constants: 10.49 / 27.80 / +4.9 % / -0.23 | matched: 7.48 / 27.27 / -25.2 % / -0.38 | constlam 0.00185: 10.78 / 28.06, 0.0062: 4.69 / 27.13
- room 20 MB: full: 20.64 / 28.19 / +3.2 % / -0.60* | constants: 20.35 / 28.19 / +1.7 % / -0.59*
- kitchen 10 MB: full: 11.35 / 26.42 / +13.5 % / -0.07 | nocollapse: 12.04 / 26.35 / +20.4 % / -0.22 | constants: 11.97 / 26.53 / +19.7 % / -0.03 | matched: 8.40 / 25.43 / -16.0 % / -0.66 | constlam 0.00185: 14.64 / 26.83, 0.0062: 7.01 / 25.86
- kitchen 20 MB: full: 22.00 / 27.03 / +10.0 % / -0.34* | constants: 22.60 / 27.10 / +13.0 % / -0.30*
- counter 10 MB: full: 10.79 / 25.66 / +7.9 % / -0.07 | nocollapse: 11.29 / 25.64 / +12.9 % / -0.12 | constants: 10.91 / 25.73 / +9.1 % / -0.01 | matched: 8.37 / 25.20 / -16.3 % / -0.35 | constlam 0.00185: 11.64 / 25.78, 0.0062: 5.67 / 25.28
- counter 20 MB: full: 21.09 / 25.97 / +5.4 % / -0.23* | constants: 21.27 / 25.94 / +6.3 % / -0.26*
- bonsai 10 MB: full: 10.09 / 26.97 / +0.9 % / -0.06 | nocollapse: 10.15 / 26.86 / +1.5 % / -0.18 | constants: 10.36 / 26.88 / +3.6 % / -0.18 | matched: 7.12 / 26.27 / -28.8 % / -0.36 | constlam 0.00185: 11.66 / 27.20, 0.0062: 5.39 / 26.32
- bonsai 20 MB: full: 19.32 / 27.56 / -3.4 % / -0.22* | constants: 19.95 / 27.64 / -0.2 % / -0.18*
- garden 10 MB: full: 10.21 / 23.61 / +2.0 % / -0.12 | nocollapse: 10.64 / 23.61 / +6.4 % / -0.16 | constants: 10.71 / 23.73 / +7.1 % / -0.04 | matched: 7.75 / 23.15 / -22.5 % / -0.31* | constlam 0.00185: 20.96 / 24.43, 0.0062: 8.95 / 23.60
- garden 20 MB: full: 20.78 / 24.27 / +3.9 % / -0.16 | constants: 21.48 / 24.32 / +7.4 % / -0.13*
- bicycle 10 MB: full: 9.84 / 22.47 / -1.6 % / -0.14* | nocollapse: 9.87 / 22.48 / -1.3 % / -0.12* | constants: 10.16 / 22.52 / +1.6 % / -0.10* | matched: 6.67 / 21.87 / -33.2 % / -0.56* | constlam 0.00185: 23.19 / 23.01, 0.0062: 10.36 / 22.63
- bicycle 20 MB: full: 19.14 / 22.84 / -4.3 % / -0.08 | constants: 20.23 / 22.87 / +1.2 % / -0.07
- stump 10 MB: full: 9.64 / 23.94 / -3.6 % / -0.28 | nocollapse: 9.75 / 23.93 / -2.5 % / -0.30 | constants: 9.78 / 23.97 / -2.2 % / -0.27 | matched: 6.83 / 23.13 / -31.7 % / -0.84* | constlam 0.00185: 18.54 / 24.72, 0.0062: 7.79 / 24.07
- stump 20 MB: full: 18.72 / 24.42 / -6.4 % / -0.31* | constants: 19.91 / 24.47 / -0.5 % / -0.30*
- flowers 10 MB: full: 9.87 / 19.02 / -1.3 % / -0.16* | nocollapse: 9.96 / 19.02 / -0.4 % / -0.16* | constants: 10.06 / 19.09 / +0.6 % / -0.08* | matched: 6.95 / 18.55 / -30.5 % / -0.69* | constlam 0.00185: 21.07 / 19.04, 0.0062: 10.61 / 19.16
- flowers 20 MB: full: 19.21 / 19.12 / -3.9 % / +0.07 | constants: 20.03 / 19.14 / +0.2 % / +0.09
- treehill 10 MB: full: 9.53 / 20.46 / -4.7 % / -0.13* | nocollapse: 9.60 / 20.47 / -4.0 % / -0.13* | constants: 9.80 / 20.50 / -2.0 % / -0.10* | matched: 6.42 / 20.21 / -35.8 % / -0.41* | constlam 0.00185: 24.01 / 20.52, 0.0062: 10.36 / 20.59
- treehill 20 MB: full: 18.67 / 20.55 / -6.6 % / +0.01 | constants: 19.53 / 20.57 / -2.4 % / +0.04
- truck 10 MB: full: 10.12 / 22.17 / +1.2 % / -0.09 | nocollapse: 10.15 / 22.21 / +1.5 % / -0.05 | constants: 10.62 / 22.27 / +6.2 % / +0.00 | matched: 6.63 / 21.73 / -33.7 % / -0.56* | constlam 0.00185: 16.90 / 22.24, 0.0062: 8.09 / 22.28
- truck 20 MB: full: 18.69 / 22.38 / -6.6 % / +0.15* | constants: 20.68 / 22.43 / +3.4 % / +0.20*
- train 10 MB: full: 10.51 / 18.73 / +5.1 % / -0.07* | nocollapse: 10.73 / 18.72 / +7.3 % / -0.07* | constants: 10.65 / 18.80 / +6.5 % / +0.00* | matched: 7.44 / 18.30 / -25.6 % / -0.59* | constlam 0.00185: 22.38 / 18.59, 0.0062: 11.73 / 18.77
- train 20 MB: full: 20.54 / 18.90 / +2.7 % / +0.28 | constants: 21.38 / 18.84 / +6.9 % / +0.23
- lego 1.5 MB: full: 1.52 / 30.65 / +1.1 % / +0.01* | constants: 1.78 / 31.33 / +18.7 % / +0.09
- lego 2.5 MB: full: 2.50 / 32.38 / -0.1 % / -0.12* | nocollapse: 2.80 / 32.37 / +12.1 % / -0.58* | constants: 2.90 / 32.83 / +16.2 % / -0.24* | matched: 2.31 / 31.61 / -7.5 % / -0.61 | constlam 0.0062: 2.47 / 32.47, 0.0185: 1.55 / 30.71
- chair 1.5 MB: full: 1.54 / 31.28 / +2.7 % / +0.11 | constants: 1.77 / 31.67 / +17.8 % / +0.15
- chair 2.5 MB: full: 2.46 / 32.26 / -1.7 % / -0.13* | nocollapse: 2.83 / 31.79 / +13.4 % / -0.97* | constants: 2.83 / 32.45 / +13.0 % / -0.30* | matched: 2.22 / 31.74 / -11.3 % / -0.38* | constlam 0.0062: 2.16 / 32.05, 0.0185: 1.42 / 30.96
- drums 1.5 MB: full: 1.53 / 25.27 / +2.3 % / -0.03* | constants: 1.77 / 25.33 / +17.9 % / -0.02
- drums 2.5 MB: full: 2.67 / 25.48 / +7.0 % / -0.02* | nocollapse: 2.78 / 25.51 / +11.2 % / -0.01* | constants: 2.84 / 25.53 / +13.7 % / +0.01* | matched: 2.29 / 25.34 / -8.3 % / -0.11 | constlam 0.0062: 2.62 / 25.49, 0.0185: 1.73 / 25.34
- ficus 1.5 MB: full: 1.51 / 31.98 / +1.0 % / -0.22* | constants: 1.59 / 32.06 / +6.0 % / -0.24*
- ficus 2.5 MB: full: 1.96 / 32.11 / -21.6 % / -0.60* | nocollapse: 1.95 / 32.07 / -22.2 % / -0.63* | constants: 1.95 / 32.11 / -22.0 % / -0.59* | matched: 1.45 / 31.89 / -42.1 % / -0.23* | constlam 0.0062: 1.32 / 31.93, 0.0185: 1.00 / 31.39
- hotdog 1.5 MB: full: 1.64 / 35.14 / +9.0 % / +0.07 | constants: 1.93 / 35.43 / +28.6 % / -0.03*
- hotdog 2.5 MB: full: 2.73 / 35.43 / +9.2 % / -0.87* | nocollapse: 2.87 / 35.75 / +14.9 % / -0.67* | constants: 3.12 / 35.52 / +24.8 % / -1.11* | matched: 2.23 / 35.28 / -10.7 % / -0.53* | constlam 0.0062: 1.87 / 35.39, 0.0185: 1.25 / 34.42
- materials 1.5 MB: full: 1.57 / 27.56 / +4.9 % / +0.01 | constants: 1.75 / 27.67 / +16.5 % / +0.03
- materials 2.5 MB: full: 2.60 / 27.79 / +4.0 % / -0.18* | nocollapse: 2.67 / 27.79 / +6.9 % / -0.20* | constants: 2.79 / 27.82 / +11.8 % / -0.21* | matched: 2.34 / 27.69 / -6.2 % / -0.20* | constlam 0.0062: 2.00 / 27.75, 0.0185: 1.40 / 27.46
- mic 1.5 MB: full: 1.54 / 32.64 / +2.6 % / -0.12 | constants: 1.76 / 33.03 / +17.1 % / -0.06*
- mic 2.5 MB: full: 2.34 / 33.30 / -6.6 % / -0.50* | nocollapse: 2.35 / 33.29 / -6.2 % / -0.53* | constants: 2.38 / 33.37 / -4.9 % / -0.48* | matched: 1.81 / 32.76 / -27.6 % / -0.41* | constlam 0.0062: 1.59 / 32.85, 0.0185: 1.19 / 32.12
- ship 1.5 MB: full: 1.62 / 28.39 / +8.3 % / -0.05* | constants: 1.81 / 28.56 / +20.7 % / -0.03*
- ship 2.5 MB: full: 2.66 / 29.01 / +6.4 % / -0.11 | nocollapse: 2.76 / 29.07 / +10.6 % / -0.10 | constants: 2.87 / 29.16 / +14.8 % / -0.06 | matched: 2.32 / 28.45 / -7.4 % / -0.47 | constlam 0.0062: 3.05 / 29.31, 0.0185: 1.91 / 28.66

## E1 at the controller counts: training with a vertex cap vs simplification after training (same vertex count, after codec)

| dataset | pairs | mean gain (dB) | range | trained better | gain at largest / mid / smallest budget |
|---|---|---|---|---|---|
| Mip-NeRF 360 / T&T | 33 | +2.07 | +0.10 .. +5.48 | 33/33 | +0.48 / +1.89 / +3.83 |
| NeRF-Synthetic | 24 | +1.33 | -0.03 .. +4.35 | 23/24 | +0.76 / +1.23 / +2.01 |

## E2 at the controller counts: BudgetMesh (controller run) vs MeshSplatting with a vertex cap

Right half (same file size, after codec): the cap mesh encoded with the same codec; the cap curve of each scene (its 3 budgets, log-linear in file size) read at the controller file size. Same vertex count: the cap run whose budget was set from this controller mesh (V within the stated band). Left half (no codec) needs the pre-codec test PSNR of the controller meshes (CPU collect, see below).

| dataset | target | pairs (file size / vertex count within 15 %) | V_cap / V_ctrl range of the vertex-count pairs | same file size: mean gain (range), BudgetMesh better | same vertex count, after codec: mean gain (range), better |
|---|---|---|---|---|---|
| Mip-NeRF 360 / T&T | 20 MB | 11 / 8 | 0.86 .. 1.05 | +0.12 (-0.21 .. +0.32), 10/11 | +0.22 (+0.04 .. +0.35), 8/8 |
| Mip-NeRF 360 / T&T | 10 MB | 11 / 11 | 0.90 .. 1.11 | +0.23 (-0.02 .. +0.55), 10/11 | +0.27 (+0.03 .. +0.77), 11/11 |
| Mip-NeRF 360 / T&T | 5 MB | 11 / 11 | 0.94 .. 1.14 | +0.28 (-0.15 .. +0.51), 10/11 | +0.27 (-0.16 .. +0.53), 10/11 |
| Mip-NeRF 360 / T&T | all | 33 / 30 | | +0.21, 30/33 | +0.26, 29/30 |
| NeRF-Synthetic | 4 MB | 8 / 3 | 0.88 .. 1.00 | -0.01 (-0.32 .. +0.25), 3/8 | +0.15 (+0.08 .. +0.20), 3/3 |
| NeRF-Synthetic | 2.5 MB | 8 / 6 | 0.92 .. 1.01 | +0.14 (-0.10 .. +0.48), 5/8 | +0.23 (+0.02 .. +0.59), 6/6 |
| NeRF-Synthetic | 1.5 MB | 8 / 6 | 0.97 .. 1.14 | +0.24 (+0.08 .. +0.55), 8/8 | +0.26 (+0.11 .. +0.36), 6/6 |
| NeRF-Synthetic | all | 24 / 15 | | +0.12, 16/24 | +0.22, 15/15 |

### per-scene cap rows (controller: MB / PSNR / V; cap: budget -> V / MB / PSNR / pre-codec test; simplified: PSNR)

- room 20 MB: controller 20.64 / 28.19 / 1978484 | cap 2198000 -> 1710844 / 17.68 / 27.89 / pre 28.25 | simplified 27.23
- room 10 MB: controller 10.29 / 27.83 / 918283 | cap 1020000 -> 906257 / 9.91 / 27.66 / pre 27.95 | simplified 25.58
- room 5 MB: controller 5.30 / 27.19 / 429049 | cap 477000 -> 416014 / 5.12 / 26.80 / pre 26.99 | simplified 23.30
- kitchen 20 MB: controller 22.00 / 27.03 / 1705414 | cap 1895000 -> 1410317 / 16.41 / 26.30 / pre 26.88 | simplified 25.86
- kitchen 10 MB: controller 11.35 / 26.42 / 834818 | cap 928000 -> 751933 / 9.36 / 25.65 / pre 26.18 | simplified 23.45
- kitchen 5 MB: controller 5.31 / 25.26 / 368115 | cap 409000 -> 373358 / 5.24 / 24.72 / pre 25.13 | simplified 21.32
- counter 20 MB: controller 21.09 / 25.97 / 1836028 | cap 2040000 -> 1599683 / 17.25 / 25.62 / pre 26.25 | simplified 25.13
- counter 10 MB: controller 10.79 / 25.66 / 879776 | cap 978000 -> 828485 / 9.44 / 25.31 / pre 25.87 | simplified 23.31
- counter 5 MB: controller 5.46 / 25.11 / 404249 | cap 449000 -> 400693 / 5.29 / 24.82 / pre 25.26 | simplified 21.18
- bonsai 20 MB: controller 19.32 / 27.56 / 1880681 | cap 2090000 -> 1753075 / 17.16 / 27.26 / pre 27.79 | simplified 25.98
- bonsai 10 MB: controller 10.09 / 26.97 / 904824 | cap 1005000 -> 871887 / 9.57 / 26.66 / pre 26.98 | simplified 23.11
- bonsai 5 MB: controller 5.01 / 25.81 / 425518 | cap 473000 -> 436461 / 5.43 / 25.64 / pre 25.79 | simplified 20.16
- garden 20 MB: controller 20.78 / 24.27 / 1813364 | cap 2015000 -> 1875740 / 20.62 / 24.23 / pre 24.31 | simplified 23.74
- garden 10 MB: controller 10.21 / 23.61 / 842517 | cap 936000 -> 935782 / 10.99 / 23.55 / pre 23.55 | simplified 21.85
- garden 5 MB: controller 4.83 / 22.71 / 364051 | cap 405000 -> 416012 / 5.35 / 22.57 / pre 22.47 | simplified 19.18
- bicycle 20 MB: controller 19.14 / 22.84 / 1826127 | cap 2029000 -> 1888752 / 18.48 / 22.54 / pre 22.70 | simplified 22.45
- bicycle 10 MB: controller 9.84 / 22.47 / 887494 | cap 986000 -> 940945 / 9.88 / 22.14 / pre 22.26 | simplified 20.78
- bicycle 5 MB: controller 4.83 / 21.90 / 396370 | cap 440000 -> 392041 / 4.72 / 21.47 / pre 21.51 | simplified 17.70
- stump 20 MB: controller 18.72 / 24.42 / 1813896 | cap 2015000 -> 1709471 / 16.73 / 24.18 / pre 24.41 | simplified 23.81
- stump 10 MB: controller 9.64 / 23.94 / 885081 | cap 983000 -> 901987 / 9.52 / 23.59 / pre 23.73 | simplified 22.23
- stump 5 MB: controller 4.78 / 23.27 / 395994 | cap 440000 -> 373184 / 4.55 / 22.79 / pre 22.86 | simplified 18.52
- flowers 20 MB: controller 19.21 / 19.12 / 1861456 | cap 2068000 -> 1856651 / 17.39 / 18.90 / pre 19.02 | simplified 18.54
- flowers 10 MB: controller 9.87 / 19.02 / 892737 | cap 992000 -> 873880 / 8.88 / 18.68 / pre 18.74 | simplified 16.39
- flowers 5 MB: controller 4.91 / 18.81 / 401860 | cap 447000 -> 430462 / 4.97 / 18.49 / pre 18.53 | simplified 13.61
- treehill 20 MB: controller 18.67 / 20.55 / 1848442 | cap 2054000 -> 1947397 / 18.69 / 20.50 / pre 20.53 | simplified 19.93
- treehill 10 MB: controller 9.53 / 20.46 / 901369 | cap 1002000 -> 880522 / 9.22 / 20.41 / pre 20.41 | simplified 18.22
- treehill 5 MB: controller 4.76 / 20.10 / 404078 | cap 449000 -> 424560 / 5.05 / 20.26 / pre 20.25 | simplified 15.57
- truck 20 MB: controller 18.69 / 22.38 / 1667337 | cap 1853000 -> 1379017 / 14.98 / 22.41 / pre 22.45 | simplified 22.04
- truck 10 MB: controller 10.12 / 22.17 / 836519 | cap 929000 -> 808599 / 9.14 / 22.14 / pre 22.16 | simplified 20.87
- truck 5 MB: controller 5.01 / 21.60 / 379060 | cap 421000 -> 402029 / 5.19 / 21.51 / pre 21.47 | simplified 17.98
- train 20 MB: controller 20.54 / 18.90 / 1836360 | cap 2040000 -> 1557663 / 16.95 / 18.68 / pre 18.75 | simplified 18.54
- train 10 MB: controller 10.51 / 18.73 / 877189 | cap 975000 -> 848788 / 9.70 / 18.54 / pre 18.60 | simplified 17.80
- train 5 MB: controller 5.12 / 18.22 / 391996 | cap 436000 -> 419682 / 5.31 / 17.95 / pre 18.03 | simplified 16.32
- lego 4 MB: controller 4.07 / 33.12 / 233232 | cap 259000 -> 232640 / 3.82 / 32.96 / pre 33.30 | simplified 29.66
- lego 2.5 MB: controller 2.50 / 32.38 / 128086 | cap 142000 -> 125800 / 2.40 / 31.80 / pre 32.09 | simplified 27.62
- lego 1.5 MB: controller 1.52 / 30.65 / 58972 | cap 66000 -> 67086 / 1.61 / 30.28 / pre 30.53 | simplified 25.94
- chair 4 MB: controller 4.01 / 32.12 / 235985 | cap 262000 -> 192705 / 3.38 / 31.73 / pre 32.14 | simplified 31.14
- chair 2.5 MB: controller 2.46 / 32.26 / 127731 | cap 142000 -> 128420 / 2.40 / 31.89 / pre 32.21 | simplified 29.19
- chair 1.5 MB: controller 1.54 / 31.28 / 60333 | cap 67000 -> 69508 / 1.65 / 31.13 / pre 31.41 | simplified 26.88
- drums 4 MB: controller 3.79 / 25.54 / 231804 | cap 258000 -> 186525 / 3.22 / 25.47 / pre 25.58 | simplified 25.08
- drums 2.5 MB: controller 2.67 / 25.48 / 143064 | cap 159000 -> 138120 / 2.56 / 25.42 / pre 25.53 | simplified 24.78
- drums 1.5 MB: controller 1.53 / 25.27 / 61541 | cap 68000 -> 69594 / 1.62 / 25.15 / pre 25.27 | simplified 23.59
- ficus 6 MB: controller 1.96 / 32.12 / 103072 | cap 115000 -> 68153 / 1.50 / 31.88 / pre 32.08 | simplified 31.73
- ficus 4 MB: controller 1.98 / 32.09 / 103897 | cap 115000 -> 68153 / 1.50 / 31.88 / pre 32.08 | simplified 31.73
- ficus 3 MB: controller 1.95 / 32.11 / 101481 | cap 113000 -> 68591 / 1.51 / 31.90 / pre 32.10 | simplified 31.75
- ficus 2.5 MB: controller 1.96 / 32.11 / 101260 | cap 113000 -> 68591 / 1.51 / 31.90 / pre 32.10 | simplified 31.75
- ficus 1.5 MB: controller 1.51 / 31.98 / 67454 | cap 75000 -> 56885 / 1.35 / 31.80 / pre 32.02 | simplified 31.02
- hotdog 4 MB: controller 3.71 / 35.59 / 199969 | cap 222000 -> 152711 / 2.84 / 35.57 / pre 35.98 | simplified 34.84
- hotdog 2.5 MB: controller 2.73 / 35.43 / 131693 | cap 146000 -> 125337 / 2.45 / 35.42 / pre 35.82 | simplified 34.78
- hotdog 1.5 MB: controller 1.64 / 35.14 / 62102 | cap 69000 -> 60305 / 1.56 / 34.82 / pre 35.15 | simplified 33.30
- materials 4 MB: controller 3.59 / 27.83 / 233459 | cap 259000 -> 206265 / 3.24 / 27.75 / pre 28.21 | simplified 27.78
- materials 2.5 MB: controller 2.60 / 27.79 / 148553 | cap 165000 -> 137030 / 2.42 / 27.74 / pre 28.15 | simplified 27.54
- materials 1.5 MB: controller 1.57 / 27.56 / 67704 | cap 75000 -> 67994 / 1.55 / 27.39 / pre 27.74 | simplified 26.66
- mic 4 MB: controller 2.58 / 33.40 / 153365 | cap 170000 -> 107795 / 2.05 / 33.06 / pre 33.34 | simplified 32.84
- mic 2.5 MB: controller 2.34 / 33.30 / 130131 | cap 145000 -> 106942 / 2.04 / 33.01 / pre 33.29 | simplified 32.84
- mic 1.5 MB: controller 1.54 / 32.64 / 62377 | cap 69000 -> 61108 / 1.48 / 32.41 / pre 32.64 | simplified 31.13
- ship 4 MB: controller 4.13 / 29.39 / 264647 | cap 294000 -> 253376 / 3.95 / 29.19 / pre 29.55 | simplified 28.47
- ship 2.5 MB: controller 2.66 / 29.01 / 147153 | cap 164000 -> 144319 / 2.56 / 28.73 / pre 29.05 | simplified 27.58
- ship 1.5 MB: controller 1.62 / 28.39 / 66258 | cap 74000 -> 70253 / 1.64 / 28.05 / pre 28.33 | simplified 26.45
