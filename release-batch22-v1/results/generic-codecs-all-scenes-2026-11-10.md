# Generic codecs on the same original mesh vs our codec (batch21 + batch22, 2026-11-10)

Same FP32 MeshSplatting mesh for every row of a scene. "ours" = our level-1.0 delivery of that mesh (rd-hull6-v2 table). Generic rows: quantised PLY + zstd-19; Draco (positions + faces + 8-bit SH as a generic attribute); MPEG-4 SC3DMC/TFAN via Open3DGC (positions + 8-bit SH); Draco with 8-bit RGB only (no view dependence); baked 4096² texture route. Δ = PSNR vs the FP32 mesh.

| scene | FP32 | ours MB (Δ) | zstd PLY (Δ) | Draco +SH8 (Δ) | MPEG +SH8 (Δ) | Draco RGB only (Δ) | texture 4096 (Δ) | MPEG+SH8 ÷ ours | Draco+SH8 ÷ ours | RGB-only ÷ ours |
|---|---|---|---|---|---|---|---|---|---|---|
| room | 742 MB | 25.7 (-0.49) | 159 (-0.17) | 456 (-0.17) | 108 | 62 (-2.98) | 82 (-2.98) | 4.2× | 17.7× | 2.4× |
| kitchen | 456 MB | 24.3 (-0.76) | 108 (-0.06) | 352 (-0.06) | 74 (-0.06) | 49 (-4.38) | 69 (-4.38) | 3.1× | 14.5× | 2.0× |
| counter | 575 MB | 23.6 (-0.74) | 132 (-0.04) | 386 (-0.04) | 91 (-0.04) | 55 (-3.62) | 75 (-3.62) | 3.8× | 16.3× | 2.4× |
| bonsai | 662 MB | 28.2 (-0.62) | 145 (-0.09) | 347 (-0.09) | 98 (-0.09) | 52 (-3.77) | 68 (-3.77) | 3.5× | 12.3× | 1.8× |
| garden | 740 MB | 32.5 (-0.27) | 172 (-0.14) | 544 (-0.14) | – | 73 (-3.67) | 84 (-3.67) |  | 16.8× | 2.2× |
| bicycle | 680 MB | 28.4 (-0.20) | 142 (-0.18) | 416 (-0.18) | 108 (-0.18) | 58 (-1.82) | 70 (-1.82) | 3.8× | 14.7× | 2.0× |
| stump | 698 MB | 29.2 (-0.28) | 153 (-0.19) | 431 (-0.19) | 114 (-0.19) | 59 (-1.68) | 70 (-1.68) | 3.9× | 14.8× | 2.0× |
| flowers | 668 MB | 26.9 (-0.50) | 143 (-0.21) | 383 (-0.21) | 106 (-0.21) | 52 (-0.85) | 69 (-0.85) | 3.9× | 14.2× | 1.9× |
| treehill | 692 MB | 28.5 (-0.12) | 149 (-0.05) | 416 (-0.05) | 113 (-0.05) | 56 (-0.74) | 69 (-0.74) | 4.0× | 14.6× | 2.0× |
| truck | 432 MB | 20.0 (-0.08) | 95 (-0.02) | 269 (-0.02) | 69 (-0.02) | 42 (-1.73) | 60 (-1.73) | 3.4× | 13.5× | 2.1× |
| train | 485 MB | 22.3 (-0.16) | 112 (-0.02) | 341 (-0.02) | 81 (-0.02) | 49 (-1.74) | 67 (-1.74) | 3.6× | 15.3× | 2.2× |
| drjohnson | 778 MB | n/a | 183 (-0.02) | 655 (-0.02) | 131 (-0.02) | 103 (-2.13) | 116 (-2.13) | | | |
| playroom | 765 MB | n/a | 180 (-0.02) | 595 (-0.02) | 124 (-0.02) | 92 (-2.85) | 112 (-2.85) | | | |

**Over the Mip-360 / T&T scenes: MPEG with 8-bit SH is 3.1–4.2× our file (mean 3.7×, 10 scenes); Draco with 8-bit SH 12.3–17.7× (mean 15.0×, 11 scenes); the only generic route near our size, Draco with RGB only, is still 1.8–2.4× our file and drops view dependence (−0.7 to −4.4 dB on the same mesh, vs our −0.1 to −0.8 dB).**

Caveat for the paper: the generic rows are near-lossless (−0.02 to −0.2 dB) while ours loses 0.1–0.8 dB at this operating point, so this is 'bytes each codec needs for essentially the same mesh', not an RD curve. Garden's MPEG row and the Deep Blending level-1.0 deliveries are not in this collect (garden: in the o3dgc table on HPC; DB: not run).
