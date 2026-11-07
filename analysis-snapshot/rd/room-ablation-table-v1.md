# Room ablation table v1 (2026-10-19) — all rows K4096 R256 q14/q7 stage2-v2, test PSNR of the delivered file; original FP32 742 MB / 28.263

## Low rate (~4–6 MB, 130–170×)
| variant | what decides the vertex count | which triangles split | vertices | MB | PSNR | SSIM | LPIPS | vs λ at equal MB |
|---|---|---|---|---|---|---|---|---|
| dense → post-hoc collapse to 0.97M | nothing / collapse afterwards | importance | 966571 | 10.14 | 25.665 | — | — | far below (−2 dB at 2× the bytes) |
| vertex cap 350k | hand-set cap, hit early | importance | 361813 | 4.48 | 26.628 | — | — | ≈ −0.3 |
| importance sampling on the λ schedule (control) | λ schedule replayed (18 counts) | importance | 348573 | 4.31 | 26.810 | .8314 | .4607 | −0.12 |
| λ = 6.2e-3 (growth only) | λ | residual / bytes > λ | 493772 | 5.70 | 27.370 | .8388 | .4450 | 0 |
| λ = 6.2e-3 + phase-B λ-prune (μ=1, 4 rounds) | λ, both directions | same + collapse if ring residual / bytes saved < λ | 439781 | 5.23 | 27.382 | .8388 | .4467 | +0.16 (−8 % MB at equal PSNR) |

## Mid rate (~13 MB, 55×)
| variant | vertices | MB | PSNR | SSIM | LPIPS | note |
|---|---|---|---|---|---|---|
| dense → post-hoc collapse to 1.35M | 1348124 | 13.93 | 26.660 | — | — | −1.3 dB vs cap at equal vertices |
| vertex cap 1.6M | 1348124 | 13.56 | 27.984 | .862 | — | |
| λ = 1.85e-3 (growth only) | 1323185 | 13.70 | 27.985 | .8618 | .3944 | tie with cap: selection does not matter here |
| λ = 1.85e-3 + phase-B λ-prune | 1187152 | 12.57 | 28.004 | .8627 | .3946 | −8 % MB, +0.02 dB |

## Reading
- Post-hoc removal is the wrong lever on meshes (1.3–2 dB behind at equal vertices).
- Among in-training methods at ~4.3 MB: cap ≈26.5 < importance-on-λ-schedule 26.81 < λ 26.93 (interp): the λ-shaped schedule is worth ~0.3 dB, the error ranking another 0.12. At 13 MB all in-training variants tie.
- Importance-sampled splits are wasted more often: from identical split counts, 370k vertices survive to the Delaunay rebuild vs 525k with error ranking (−30 %).
- Phase-B λ-prune (same λ, same units, reverse direction) removes 10–11 % of the vertices and 8 % of the bytes at no PSNR cost on both rate points; it converges in 4 rounds without the rate limiter binding. The variable-price (bytes-per-face) component is not what drives the selection (selected edges are not biased toward high-savings edges), so the paper presents pruning as the same price applied in reverse.
- Sources: rd/rd-sweep-room-2026-10-18.txt, rd/rd-prune3-room-2026-10-19.txt, rd/budget-sweep-2026-10-17.txt, rd/room-lambda-vs-cap-v2.csv; figure room-lambda-vs-cap-v2.png.
