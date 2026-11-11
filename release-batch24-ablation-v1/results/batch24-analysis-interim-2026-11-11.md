
## Mip-NeRF 360 / T&T, target 10 MB (11 scenes with a controller row)

| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |
|---|---|---|---|---|---|---|
| full | 11 | +2.0 % | -4.7 .. +13.5 % | 10/11 | -0.12 (-0.28 .. -0.06)* | 0 (reference) |
| nocollapse | 11 | +4.5 % | -4.0 .. +20.4 % | 9/11 | -0.16 (-0.30 .. -0.05)* | -0.03 (-0.15 .. +0.04), 4/11 |
| constants | 11 | +5.0 % | -2.2 .. +19.7 % | 10/11 | -0.09 (-0.27 .. +0.00)* | +0.03 (-0.12 .. +0.10), 9/11 |
| matched | 0 | not run yet | | | | |

## Mip-NeRF 360 / T&T, target 20 MB (11 scenes with a controller row)

| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |
|---|---|---|---|---|---|---|
| full | 11 | -0.5 % | -6.6 .. +10.0 % | 10/11 | -0.13 (-0.60 .. +0.28)* | 0 (reference) |
| nocollapse | 0 | not run yet | | | | |
| constants | 11 | +3.4 % | -2.4 .. +13.0 % | 10/11 | -0.12 (-0.59 .. +0.23)* | +0.01 (-0.05 .. +0.05), 9/11 |
| matched | 0 | not run yet | | | | |

### constant price, Mip-NeRF 360 / T&T: file size spread across scenes at one price (abl24 constlam rows)

- constlam-lam0.00185: 11 scenes, file size 10.8 .. 24.0 MB (max/min 2.2x)
- constlam-lam0.0062: 11 scenes, file size 4.7 .. 11.7 MB (max/min 2.5x)

## NeRF-Synthetic, target 2.5 MB (8 scenes with a controller row)

| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |
|---|---|---|---|---|---|---|
| full | 8 | -0.4 % | -21.6 .. +9.2 % | 7/8 | - | 0 (reference) |
| nocollapse | 0 | not run yet | | | | |
| constants | 0 | not run yet | | | | |
| matched | 0 | not run yet | | | | |

## NeRF-Synthetic, target 1.5 MB (8 scenes with a controller row)

| variant | scenes | size error mean | range | within 10 % | dPSNR vs constant-price curve: mean (range) | vs full: mean (range), better |
|---|---|---|---|---|---|---|
| full | 8 | +4.0 % | +1.0 .. +9.0 % | 8/8 | - | 0 (reference) |
| nocollapse | 0 | not run yet | | | | |
| constants | 0 | not run yet | | | | |
| matched | 0 | not run yet | | | | |

### constant price, NeRF-Synthetic: file size spread across scenes at one price (abl24 constlam rows)


### per-scene rows (file size MB / PSNR / size error; dPSNR vs the scene's constant-price curve)

- room 10 MB: full: 10.29 / 27.83 / +2.9 % / -0.18 | nocollapse: 10.80 / 27.82 / +8.0 % / -0.24* | constants: 10.49 / 27.80 / +4.9 % / -0.23 | constlam 0.00185: 10.78 / 28.06, 0.0062: 4.69 / 27.13
- room 20 MB: full: 20.64 / 28.19 / +3.2 % / -0.60* | constants: 20.35 / 28.19 / +1.7 % / -0.59*
- kitchen 10 MB: full: 11.35 / 26.42 / +13.5 % / -0.07 | nocollapse: 12.04 / 26.35 / +20.4 % / -0.22 | constants: 11.97 / 26.53 / +19.7 % / -0.03 | constlam 0.00185: 14.64 / 26.83, 0.0062: 7.01 / 25.86
- kitchen 20 MB: full: 22.00 / 27.03 / +10.0 % / -0.34* | constants: 22.60 / 27.10 / +13.0 % / -0.30*
- counter 10 MB: full: 10.79 / 25.66 / +7.9 % / -0.07 | nocollapse: 11.29 / 25.64 / +12.9 % / -0.12 | constants: 10.91 / 25.73 / +9.1 % / -0.01 | constlam 0.00185: 11.64 / 25.78, 0.0062: 5.67 / 25.28
- counter 20 MB: full: 21.09 / 25.97 / +5.4 % / -0.23* | constants: 21.27 / 25.94 / +6.3 % / -0.26*
- bonsai 10 MB: full: 10.09 / 26.97 / +0.9 % / -0.06 | nocollapse: 10.15 / 26.86 / +1.5 % / -0.18 | constants: 10.36 / 26.88 / +3.6 % / -0.18 | constlam 0.00185: 11.66 / 27.20, 0.0062: 5.39 / 26.32
- bonsai 20 MB: full: 19.32 / 27.56 / -3.4 % / -0.22* | constants: 19.95 / 27.64 / -0.2 % / -0.18*
- garden 10 MB: full: 10.21 / 23.61 / +2.0 % / -0.12 | nocollapse: 10.64 / 23.61 / +6.4 % / -0.16 | constants: 10.71 / 23.73 / +7.1 % / -0.04 | constlam 0.00185: 20.96 / 24.43, 0.0062: 8.95 / 23.60
- garden 20 MB: full: 20.78 / 24.27 / +3.9 % / -0.16 | constants: 21.48 / 24.32 / +7.4 % / -0.13*
- bicycle 10 MB: full: 9.84 / 22.47 / -1.6 % / -0.14* | nocollapse: 9.87 / 22.48 / -1.3 % / -0.12* | constants: 10.16 / 22.52 / +1.6 % / -0.10* | constlam 0.00185: 23.19 / 23.01, 0.0062: 10.36 / 22.63
- bicycle 20 MB: full: 19.14 / 22.84 / -4.3 % / -0.08 | constants: 20.23 / 22.87 / +1.2 % / -0.07
- stump 10 MB: full: 9.64 / 23.94 / -3.6 % / -0.28 | nocollapse: 9.75 / 23.93 / -2.5 % / -0.30 | constants: 9.78 / 23.97 / -2.2 % / -0.27 | constlam 0.00185: 18.54 / 24.72, 0.0062: 7.79 / 24.07
- stump 20 MB: full: 18.72 / 24.42 / -6.4 % / -0.31* | constants: 19.91 / 24.47 / -0.5 % / -0.30*
- flowers 10 MB: full: 9.87 / 19.02 / -1.3 % / -0.16* | nocollapse: 9.96 / 19.02 / -0.4 % / -0.16* | constants: 10.06 / 19.09 / +0.6 % / -0.08* | constlam 0.00185: 21.07 / 19.04, 0.0062: 10.61 / 19.16
- flowers 20 MB: full: 19.21 / 19.12 / -3.9 % / +0.07 | constants: 20.03 / 19.14 / +0.2 % / +0.09
- treehill 10 MB: full: 9.53 / 20.46 / -4.7 % / -0.13* | nocollapse: 9.60 / 20.47 / -4.0 % / -0.13* | constants: 9.80 / 20.50 / -2.0 % / -0.10* | constlam 0.00185: 24.01 / 20.52, 0.0062: 10.36 / 20.59
- treehill 20 MB: full: 18.67 / 20.55 / -6.6 % / +0.01 | constants: 19.53 / 20.57 / -2.4 % / +0.04
- truck 10 MB: full: 10.12 / 22.17 / +1.2 % / -0.09 | nocollapse: 10.15 / 22.21 / +1.5 % / -0.05 | constants: 10.62 / 22.27 / +6.2 % / +0.00 | constlam 0.00185: 16.90 / 22.24, 0.0062: 8.09 / 22.28
- truck 20 MB: full: 18.69 / 22.38 / -6.6 % / +0.15* | constants: 20.68 / 22.43 / +3.4 % / +0.20*
- train 10 MB: full: 10.51 / 18.73 / +5.1 % / -0.07* | nocollapse: 10.73 / 18.72 / +7.3 % / -0.07* | constants: 10.65 / 18.80 / +6.5 % / +0.00* | constlam 0.00185: 22.38 / 18.59, 0.0062: 11.73 / 18.77
- train 20 MB: full: 20.54 / 18.90 / +2.7 % / +0.28 | constants: 21.38 / 18.84 / +6.9 % / +0.23
