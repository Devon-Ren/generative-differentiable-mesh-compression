# Draco carrying our appearance payload (batch25, 63 assets, 2026-11-12)

Per dataset: file size of each route divided by our file (min..max, mean).

| dataset | n | Draco geometry only | Draco geometry + indices/DC via zstd | Draco with indices/DC as generic attribute | our appearance streams (indices + DC via zstd) / our file |
|---|---|---|---|---|---|
| Mip-NeRF 360 / T&T | 33 | 0.85..1.24 (1.05) | 1.26..1.55 (1.41) | 2.73..3.50 (3.02) | 0.28..0.42 |
| Deep Blending | 6 | 1.43..1.56 (1.47) | 1.70..1.83 (1.75) | 3.28..3.50 (3.41) | 0.26..0.30 |
| NeRF-Synthetic | 24 | 0.90..1.49 (1.21) | 1.11..1.74 (1.43) | 2.13..3.48 (2.80) | 0.17..0.27 |

Quality of the Draco routes with our indices equals ours up to Draco position re-quantisation at the same bit depth (bytes-only comparison).

## per asset

- room 5 MB: ours 5.31 MB | Draco geo 1.02x | geo+zstd idx 1.32x | Draco+VQ attr 2.91x
- room 10 MB: ours 10.29 MB | Draco geo 0.99x | geo+zstd idx 1.31x | Draco+VQ attr 2.96x
- room 20 MB: ours 20.64 MB | Draco geo 0.91x | geo+zstd idx 1.26x | Draco+VQ attr 2.92x
- kitchen 5 MB: ours 5.31 MB | Draco geo 1.21x | geo+zstd idx 1.50x | Draco+VQ attr 3.27x
- kitchen 10 MB: ours 11.35 MB | Draco geo 1.24x | geo+zstd idx 1.54x | Draco+VQ attr 3.50x
- kitchen 20 MB: ours 22.00 MB | Draco geo 1.23x | geo+zstd idx 1.54x | Draco+VQ attr 3.47x
- counter 5 MB: ours 5.46 MB | Draco geo 1.24x | geo+zstd idx 1.53x | Draco+VQ attr 3.19x
- counter 10 MB: ours 10.79 MB | Draco geo 1.24x | geo+zstd idx 1.55x | Draco+VQ attr 3.27x
- counter 20 MB: ours 21.09 MB | Draco geo 1.13x | geo+zstd idx 1.47x | Draco+VQ attr 3.18x
- bonsai 5 MB: ours 5.01 MB | Draco geo 1.00x | geo+zstd idx 1.32x | Draco+VQ attr 2.90x
- bonsai 10 MB: ours 10.09 MB | Draco geo 1.13x | geo+zstd idx 1.47x | Draco+VQ attr 3.10x
- bonsai 20 MB: ours 19.32 MB | Draco geo 1.00x | geo+zstd idx 1.36x | Draco+VQ attr 2.98x
- garden 5 MB: ours 4.83 MB | Draco geo 1.03x | geo+zstd idx 1.36x | Draco+VQ attr 2.98x
- garden 10 MB: ours 10.20 MB | Draco geo 1.03x | geo+zstd idx 1.39x | Draco+VQ attr 3.14x
- garden 20 MB: ours 20.78 MB | Draco geo 0.99x | geo+zstd idx 1.37x | Draco+VQ attr 3.13x
- bicycle 5 MB: ours 4.83 MB | Draco geo 1.03x | geo+zstd idx 1.38x | Draco+VQ attr 2.84x
- bicycle 10 MB: ours 9.84 MB | Draco geo 1.03x | geo+zstd idx 1.41x | Draco+VQ attr 3.00x
- bicycle 20 MB: ours 19.14 MB | Draco geo 1.00x | geo+zstd idx 1.40x | Draco+VQ attr 3.01x
- stump 5 MB: ours 4.78 MB | Draco geo 0.98x | geo+zstd idx 1.32x | Draco+VQ attr 2.77x
- stump 10 MB: ours 9.64 MB | Draco geo 1.01x | geo+zstd idx 1.38x | Draco+VQ attr 2.95x
- stump 20 MB: ours 18.72 MB | Draco geo 0.98x | geo+zstd idx 1.37x | Draco+VQ attr 2.99x
- flowers 5 MB: ours 4.91 MB | Draco geo 0.90x | geo+zstd idx 1.26x | Draco+VQ attr 2.76x
- flowers 10 MB: ours 9.87 MB | Draco geo 0.87x | geo+zstd idx 1.26x | Draco+VQ attr 2.88x
- flowers 20 MB: ours 19.21 MB | Draco geo 0.85x | geo+zstd idx 1.26x | Draco+VQ attr 2.88x
- treehill 5 MB: ours 4.76 MB | Draco geo 0.97x | geo+zstd idx 1.34x | Draco+VQ attr 2.73x
- treehill 10 MB: ours 9.53 MB | Draco geo 0.99x | geo+zstd idx 1.39x | Draco+VQ attr 2.87x
- treehill 20 MB: ours 18.67 MB | Draco geo 0.96x | geo+zstd idx 1.38x | Draco+VQ attr 2.91x
- truck 5 MB: ours 5.01 MB | Draco geo 1.19x | geo+zstd idx 1.52x | Draco+VQ attr 2.92x
- truck 10 MB: ours 10.12 MB | Draco geo 1.19x | geo+zstd idx 1.54x | Draco+VQ attr 3.00x
- truck 20 MB: ours 18.69 MB | Draco geo 1.08x | geo+zstd idx 1.46x | Draco+VQ attr 2.88x
- train 5 MB: ours 5.12 MB | Draco geo 1.15x | geo+zstd idx 1.48x | Draco+VQ attr 3.03x
- train 10 MB: ours 10.51 MB | Draco geo 1.16x | geo+zstd idx 1.51x | Draco+VQ attr 3.15x
- train 20 MB: ours 20.54 MB | Draco geo 1.09x | geo+zstd idx 1.47x | Draco+VQ attr 3.07x
- drjohnson 5 MB: ours 5.23 MB | Draco geo 1.43x | geo+zstd idx 1.71x | Draco+VQ attr 3.31x
- drjohnson 10 MB: ours 11.20 MB | Draco geo 1.47x | geo+zstd idx 1.76x | Draco+VQ attr 3.46x
- drjohnson 20 MB: ours 20.73 MB | Draco geo 1.46x | geo+zstd idx 1.76x | Draco+VQ attr 3.50x
- playroom 5 MB: ours 5.56 MB | Draco geo 1.44x | geo+zstd idx 1.70x | Draco+VQ attr 3.28x
- playroom 10 MB: ours 10.83 MB | Draco geo 1.56x | geo+zstd idx 1.83x | Draco+VQ attr 3.48x
- playroom 20 MB: ours 21.93 MB | Draco geo 1.46x | geo+zstd idx 1.76x | Draco+VQ attr 3.43x
- lego 2.5 MB: ours 2.50 MB | Draco geo 1.25x | geo+zstd idx 1.48x | Draco+VQ attr 2.95x
- lego 4 MB: ours 4.07 MB | Draco geo 1.49x | geo+zstd idx 1.74x | Draco+VQ attr 3.48x
- chair 2.5 MB: ours 2.46 MB | Draco geo 1.24x | geo+zstd idx 1.48x | Draco+VQ attr 2.90x
- chair 4 MB: ours 4.01 MB | Draco geo 1.43x | geo+zstd idx 1.70x | Draco+VQ attr 3.32x
- drums 2.5 MB: ours 2.67 MB | Draco geo 1.35x | geo+zstd idx 1.59x | Draco+VQ attr 3.18x
- drums 4 MB: ours 3.79 MB | Draco geo 1.42x | geo+zstd idx 1.69x | Draco+VQ attr 3.32x
- ficus 1.5 MB: ours 1.51 MB | Draco geo 0.90x | geo+zstd idx 1.11x | Draco+VQ attr 2.13x
- ficus 2.5 MB: ours 1.96 MB | Draco geo 1.07x | geo+zstd idx 1.30x | Draco+VQ attr 2.53x
- ficus 4 MB: ours 1.98 MB | Draco geo 1.08x | geo+zstd idx 1.32x | Draco+VQ attr 2.55x
- hotdog 2.5 MB: ours 2.73 MB | Draco geo 1.42x | geo+zstd idx 1.62x | Draco+VQ attr 3.26x
- hotdog 4 MB: ours 3.71 MB | Draco geo 1.49x | geo+zstd idx 1.72x | Draco+VQ attr 3.43x
- materials 1.5 MB: ours 1.57 MB | Draco geo 1.01x | geo+zstd idx 1.19x | Draco+VQ attr 2.27x
- materials 2.5 MB: ours 2.60 MB | Draco geo 1.23x | geo+zstd idx 1.48x | Draco+VQ attr 2.80x
- materials 4 MB: ours 3.59 MB | Draco geo 1.28x | geo+zstd idx 1.55x | Draco+VQ attr 2.91x
- mic 2.5 MB: ours 2.34 MB | Draco geo 1.15x | geo+zstd idx 1.40x | Draco+VQ attr 2.72x
- mic 4 MB: ours 2.58 MB | Draco geo 1.17x | geo+zstd idx 1.44x | Draco+VQ attr 2.74x
- ship 1.5 MB: ours 1.62 MB | Draco geo 1.07x | geo+zstd idx 1.25x | Draco+VQ attr 2.45x
- ship 2.5 MB: ours 2.66 MB | Draco geo 1.30x | geo+zstd idx 1.52x | Draco+VQ attr 2.96x
- ship 4 MB: ours 4.13 MB | Draco geo 1.37x | geo+zstd idx 1.63x | Draco+VQ attr 3.17x
- lego 1.5 MB: ours 1.52 MB | Draco geo 0.98x | geo+zstd idx 1.17x | Draco+VQ attr 2.29x
- chair 1.5 MB: ours 1.54 MB | Draco geo 1.03x | geo+zstd idx 1.21x | Draco+VQ attr 2.37x
- drums 1.5 MB: ours 1.53 MB | Draco geo 1.09x | geo+zstd idx 1.27x | Draco+VQ attr 2.50x
- hotdog 1.5 MB: ours 1.64 MB | Draco geo 1.10x | geo+zstd idx 1.27x | Draco+VQ attr 2.54x
- mic 1.5 MB: ours 1.54 MB | Draco geo 1.03x | geo+zstd idx 1.22x | Draco+VQ attr 2.42x