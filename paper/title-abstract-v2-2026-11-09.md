# Title + abstract, v2 (2026-11-09 evening), for the CVPR 2027 registration (Nov 10 AoE)

Reframed after Yao's guidance: the paper is about compression; the growth rule and the controller are the means. Every number below is in a repo table (cited in brackets).

## Title

**BudgetMesh: Rate-Aware Training and Compression for Opaque Differentiable Meshes**

Name check (web, 11/09): "BudgetMesh" has no paper, code or product collision; "RateMesh" is also free; "BudgetSplat" is taken (feed-forward budget-controllable Gaussian splatting) and "Budgeted-GS" exists, so no -Splat name. Runner-up: "Grow to Budget" as a phrase in the intro, not as the system name.

## Abstract (211 words)

In this work, we study how differentiable opaque meshes, as introduced by MeshSplatting, can be compressed. They render in any engine, but a trained scene occupies hundreds of megabytes. Unlike Gaussian splats, they cannot be simplified after training without a large loss in quality: because every pixel is covered by a single opaque triangle, removing vertices from a trained mesh degrades quality far more than training a smaller mesh from the start. We argue that the right place to decide the size of a mesh is during its growth, while the optimizer can still adapt. Building on this idea, we introduce BudgetMesh, a complete compression scheme spanning training, encoding and decoding: a rate-aware growth rule that charges each refinement step its true cost in encoded bytes, a closed-loop controller that steers growth to a user-specified file size, and a dedicated codec with a standalone decoder that writes one compact, engine-ready file. Built on the MeshSplatting pipeline, BudgetMesh produces a more efficient mesh than its native densification at the same vertex count. Given a target file size, the output file lands within about 10 % of it on four datasets. At 20 to 5 MB the delivered files are 32–125× smaller than the trained model at an average cost of 0.3–1.3 dB.

## Contributions, in Yao's order

1. **Rate-aware growth makes the representation itself more efficient.** Same vertex count, no codec involved: λ-grown meshes beat MeshSplatting trained with a vertex cap by +0.40 dB mean, 23 of 25 matched pairs (indoor 12/12, mean +0.58) [no-codec-equal-vertices-2026-11-09.md]. Same delivered bytes, with the codec: +0.33 dB, 29 of 33 [g0-refresh-analysis]. And the reason it must happen during training: collapsing a trained mesh to the same count instead loses 2.45 dB mean, up to 6.02, 33 of 33. On room, a matched-count control (same number of splits per interval, upstream's importance sampling instead of the price) isolates +0.12 dB from *which* triangles are split and +0.3 dB from the price-shaped schedule [rd-ctrl-v1].
2. **Size as an input.** The encoder in the loop as the byte meter; a controller that plans vertex growth toward the count the budget buys and sets λ as the rank price of that plan; 33 pairs, mean hit 1.007, 32/33 within 10 %, 22/33 within 5 %, −0.05 dB vs the fixed-λ curve at equal bytes; Deep Blending unseen 1.06–1.12. One calibrated margin per capture class, reported.
3. **Where the bytes go, and what is redundant.** The complete encode/decode chain (dictionary SH, grid positions, cached-edge connectivity, context-coded indices; one file, isolated CPU decoder, 2–6 s) is a system contribution and is kept short. The finding: 45 of the 52 floats per vertex are SH, and replacing them by two indices into per-scene banks (4096 + 256) carries the delivery loss almost alone: at the top rate DC 8-bit recovers only 0.04–0.08 dB of the 0.4–0.6 dB loss indoors (SH sharing ≈ 85 % of it) [indoor-diag-v1]. Dictionary size is not one-size-fits-all: the joint λ × dictionary hull uses K1024 at the two smallest budgets on 11/11 scenes and larger banks (16384 + 1024, or residual 1024) at the top rate on 11/11, i.e. K should scale with V; but the gain over fixed 4096 + 256 is ≤ 0.16 dB at every base point (room 0.12, counter 0.15, kitchen 0.09) [fan-analysis-v2], so the paper uses the fixed banks and reports the hull. Generic mesh codecs (Draco, MPEG SC3DMC) on the same asset are 3.5–15× larger.

Said once in the introduction: opaque meshes are ~3 dB below Gaussian codecs at equal bytes; that is the representation's gap, and this paper is about what the mesh class can do per byte.

## Answers for Yao's specific questions

- *"Without compression, is adaptive-λ better than plain MeshSplatting at equal size / equal quality?"* Yes, measured today on 33 pairs: +0.40 dB at equal vertex count with no codec (uncompressed size is proportional to V). The equal-quality reading (how many more vertices the capped mesh needs to match the λ-grown PSNR) will be read off the pre-codec RD curves once plotted; not computed yet. The paper will show the pre-codec RD curves (PSNR vs V) for both, then the post-codec ones.
- *"Is 4096 + 256 always best after the codebook sweep?"* No. Smaller banks win at small budgets and larger banks at the top rate, on every scene; the dictionary should grow with the vertex count. The fixed setting is within 0.16 dB of the joint hull everywhere, which is why it is used for the headline tables; the sweep goes in the appendix with the K ∝ V observation.
- *"Rate-aware" wording:* keep "rate-aware" in the title and text; "byte-priced" for the mechanism (split iff ΔD > λ·ΔR with ΔR in delivered bytes).

## Headline numbers (BudgetMesh runs only; no constant-λ numbers anywhere)


| scene | FP32 | 5 MB | 7.5 MB | 10 MB | 15 MB | 20 MB |
|---|---|---|---|---|---|---|
| room | 597 MB / 28.48 | 5.3 (113×) / 27.19 (-1.29) | 7.4 (80×) / 27.69 (-0.79) | 10.3 (58×) / 27.83 (-0.65) | 15.4 (39×) / 28.08 (-0.40) | 20.6 (29×) / 28.19 (-0.29) |
| kitchen | 507 MB / 27.46 | 5.3 (96×) / 25.26 (-2.20) | 8.2 (62×) / 25.98 (-1.48) | 11.4 (45×) / 26.42 (-1.04) | 16.7 (30×) / 26.83 (-0.63) | 22.0 (23×) / 27.03 (-0.43) |
| counter | 520 MB / 26.48 | 5.5 (95×) / 25.11 (-1.37) | 8.1 (64×) / 25.49 (-0.99) | 10.8 (48×) / 25.66 (-0.82) | 16.2 (32×) / 25.96 (-0.52) | 21.1 (25×) / 25.97 (-0.51) |
| bonsai | 678 MB / 28.28 | 5.0 (135×) / 25.81 (-2.47) | 7.6 (89×) / 26.21 (-2.07) | 10.1 (67×) / 26.97 (-1.31) | 14.9 (46×) / 27.35 (-0.93) | 19.3 (35×) / 27.56 (-0.72) |
| garden | 740 MB / 24.73 | 4.8 (153×) / 22.71 (-2.02) | 7.6 (98×) / 23.24 (-1.49) | 10.2 (73×) / 23.61 (-1.12) | 15.5 (48×) / 24.03 (-0.70) | 20.8 (36×) / 24.27 (-0.46) |
| bicycle | 680 MB / 23.03 | 4.8 (141×) / 21.90 (-1.13) | 7.3 (93×) / 22.23 (-0.80) | 9.8 (69×) / 22.47 (-0.56) | 14.6 (47×) / 22.70 (-0.33) | 19.1 (36×) / 22.84 (-0.19) |
| stump | 698 MB / 24.93 | 4.8 (146×) / 23.27 (-1.66) | 7.2 (98×) / 23.71 (-1.22) | 9.6 (72×) / 23.94 (-0.99) | 14.3 (49×) / 24.18 (-0.75) | 18.7 (37×) / 24.42 (-0.51) |
| flowers | 668 MB / 19.45 | 4.9 (136×) / 18.81 (-0.64) | 7.4 (90×) / 18.92 (-0.53) | 9.9 (68×) / 19.02 (-0.43) | 14.7 (46×) / 19.11 (-0.34) | 19.2 (35×) / 19.12 (-0.33) |
| treehill | 692 MB / 20.39 | 4.8 (145×) / 20.10 (-0.29) | 7.2 (96×) / 20.24 (-0.15) | 9.5 (73×) / 20.46 (+0.07) | 14.2 (49×) / 20.44 (+0.05) | 18.7 (37×) / 20.55 (+0.16) |
| truck | 432 MB / 22.39 | 5.0 (86×) / 21.60 (-0.79) | 7.6 (57×) / 21.98 (-0.41) | 10.1 (43×) / 22.17 (-0.22) | 14.5 (30×) / 22.35 (-0.04) | 18.7 (23×) / 22.38 (-0.01) |
| train | 485 MB / 18.90 | 5.1 (95×) / 18.22 (-0.68) | 7.7 (63×) / 18.65 (-0.25) | 10.5 (46×) / 18.73 (-0.17) | 15.7 (31×) / 18.80 (-0.10) | 20.5 (24×) / 18.90 (+0.00) |
| drjohnson | 778 MB / 26.47 | 5.2 (149×) / 25.42 (-1.05) | – | 11.2 (70×) / 26.09 (-0.38) | – | 20.7 (38×) / 26.40 (-0.07) |
| playroom | 765 MB / 27.38 | 5.6 (138×) / 26.36 (-1.02) | – | 10.8 (71×) / 26.98 (-0.40) | – | 21.9 (35×) / 27.49 (+0.11) |

| target | scenes | mean ratio | mean ΔPSNR | ΔPSNR range | mean hit |
|---|---|---|---|---|---|
| 5 MB | 13 | 125× | -1.28 | -2.47 … -0.29 | 1.017 |
| 7.5 MB | 11 | 81× | -0.92 | -2.07 … -0.15 | 1.010 |
| 10 MB | 13 | 62× | -0.62 | -1.31 … +0.07 | 1.033 |
| 15 MB | 11 | 40× | -0.43 | -0.93 … +0.05 | 1.009 |
| 20 MB | 13 | 32× | -0.25 | -0.72 … +0.16 | 1.006 |

Across all 61 rows: ratio 23–153×, ΔPSNR -2.47 … +0.16 dB; Mip/T&T 20 MB → 5 MB: mean 32× to 125× at -0.25 to -1.28 dB.

## Still to do before Nov 16 (paper)

- Pre-codec RD figure (PSNR vs V, λ-grown vs cap, 11 scenes) from the existing CSVs: no GPU.
- Qualitative renders at 5 / 10 / 20 MB, one indoor one outdoor: GPU, 1 h.
- Optional: DC q8 rows for all delivered assets (CPU re-delivery), answers Yao's comment #28 directly in a table.
