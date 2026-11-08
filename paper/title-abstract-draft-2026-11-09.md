# Title, abstract and contribution list — draft for Yao (2026-11-09, for the Nov 10 AoE registration)

Author: Claude, for Devon Ren. Every number is from the repo tables as of 11/08 night (v9b sweep, Deep Blending one-shot, fixed-λ hull latex-v5). Numbers marked [TBD] wait for tonight's G0 refresh. Three title options; one abstract; the contribution list in the order the paper argues.

## Title options

1. **Grow to Budget: Size-Controlled Differentiable Meshes** (recommended: says the mechanism and the user-facing property in five words)
2. Size-Bounded Differentiable Meshes: Rate-Priced Growth and Collapse for Mesh-Based View Synthesis
3. Byte-Priced Mesh Splatting: Compressing Differentiable Opaque Meshes While They Grow

Avoid: "generative compression" (reviewers read generative models), "adaptive λ" (internal name), "compression" for face reduction (T2-Splat vocabulary; we reserve compression for bytes).

## Abstract (≈190 words)

Differentiable opaque meshes such as MeshSplatting render in any game engine, but a trained scene is 400–800 MB of per-vertex spherical harmonics, and unlike Gaussian splats the mesh cannot be compressed afterwards: collapsing a trained mesh to a given vertex count loses 0.3–4.3 dB against a mesh trained to that count, because every pixel is covered by a single opaque triangle and the optimizer never re-places the survivors. We therefore decide the byte budget while the mesh grows. A split is accepted only if the distortion it removes is worth the bytes it adds to the delivered file, the same price is applied in reverse as an in-training edge collapse that keeps the optimizer state, and the real encoder serves as the byte meter: a shared SH dictionary, grid-quantized positions and a cache-based connectivity coder, written to one file and decoded in seconds on a CPU. On top of this, a closed-loop controller turns a target size into the price, so the asset size becomes an input. On 19 scenes of Mip-NeRF 360, Tanks and Temples and NeRF-Synthetic one setting lands within 10 % of the requested size on 32 of 33 unbounded-scene targets, with quality on the fixed-price curve, and it transfers unchanged to Deep Blending. The resulting meshes are 24–150× smaller than the trained model at −0.2 to −1.7 dB and render at 145–350 fps on a phone.

## Contributions (the order the paper argues)

1. **A measurement that motivates the method.** Opaque differentiable meshes cannot be compressed after training: at equal vertex count, training to the budget beats post-hoc collapse by +0.3 to +4.3 dB on 14 of 14 points (7 Mip-360 scenes); refreshed on the current protocol for 7 outdoor / T&T scenes × 3 budgets [TBD tonight]. The reason is structural: one opaque triangle per pixel, vertices coupled through connectivity.
2. **Byte-priced growth and collapse with the encoder in the loop.** Split iff ΔD > λ·ΔR with ΔR in delivered bytes; the same price in reverse as an edge collapse that preserves Adam state and is priced only on bytes the delivered file actually loses; the byte model is measured on the soup every 1000 iterations and on the mesh every prune round by the real encoder, never assumed. λ is a rank price of the scene's own residual-per-byte distribution; a ±30 % error in the price constants rescales λ and leaves the RD hull unchanged (8/8 points).
3. **Size as an input.** A closed-loop controller: a geometric growth plan toward the vertex count the budget buys, λ as the price of that plan, gated trimming against the extrapolated landing. 19 scenes × 3 targets with one setting: mean hit 1.007, 32/33 within 10 %, 22/33 within 5 %, equal-bytes ΔPSNR −0.05 dB; Deep Blending, unseen and unchanged, 1.06–1.12 with quality on its own curve. The remaining bias has one measured cause (the visible-face set keeps growing after the growth window closes, 1.1–1.5× indoors, 1.05× outdoors) and we show why no probe before the topology rebuild can observe it; the margin for it is calibrated on earlier runs and reported.
4. **The first complete encode/decode chain for this representation, and its compressibility anatomy.** One file, isolated decoder with an audited no-leakage proof, 2–6 s CPU decode, standard indexed mesh. Where the bytes go: SH sharing carries the loss (DC bits do not), a cross-scene codebook does not transfer, connectivity is the floor; generic codecs (Draco, MPEG SC3DMC) are 3.5–15× larger on the same asset. Fixed K=4096/R=256 is within 0.16 dB of the joint λ×dictionary hull on all 11 scenes.

Stated once in the introduction, not hidden: opaque meshes are about 3 dB below Gaussian splatting at equal bytes; that gap is the representation's (MeshSplatting FP32 is already 2.6 dB below 3DGS) and this paper is about what the mesh class can do per byte. T2-Splat (arXiv 2611.06974) improves the base representation with per-face textures and reports face budgets, not bytes; our pricing law and codec are representation-agnostic and would apply to its meshes (no code released; future work).

## Headline numbers available for the abstract / teaser (latex-v5 set, indoor rerun)

| | Mip-360 (9) | T&T (2) | NeRF-Synthetic (8) | Deep Blending (2, one-shot) |
|---|---|---|---|---|
| FP32 | 642 MB / 24.81 dB | 459 MB / 20.64 | 4.48 MB / 30.93 | ≈770 MB / 27.23 |
| λ 6.2e-4 | 26.6 MB / 24× / −0.24 | 28.7 / 16× / −0.47 | 5.10 / +0.29 | [TBD tonight] |
| λ 1.85e-3 | 20.0 / 32× / −0.37 | 22.3 / 21× / −0.21 | — | 15.4 / 50× / −0.36 |
| λ 6.2e-3 | 9.0 / 72× / −0.92 | 11.4 / 40× / −0.12 | 2.63 / 59 % bytes / ±0.00 | 6.3 / 122× / −1.19 |
| λ 1.85e-2 | 4.3 / 151× / −1.68 | 5.4 / 85× / −0.54 | — | [TBD tonight] |
| controller, 5/10/20 MB hit | 1.006 / 1.020 / 0.995 | (in the 33) | 2.5 MB 1.03, 4 MB 0.97 | 1.06 / 1.09 / 1.10 |

## Limitations paragraph (draft)

The byte price is a calibrated constant proxy plus a fixed, decodable entropy coder, not a trainable rate model; a wrong price rescales λ, it does not move the hull. The controller's aim carries one margin per capture class for a quantity that is unobservable before upstream's topology rebuild; making it adaptive requires two-sided control (a growth window after the rebuild), which we leave to future work. Outdoor scenes at 20 MB land 4–7 % under because the last growth interval admits fewer splits than planned. Small objects cannot reach large targets (the model-capacity limit; the controller reports it). Absolute quality is bounded by the opaque-mesh class: about 3 dB below Gaussian codecs at equal bytes, and triangle-soup fragments on vegetation.

## What Yao should react to by Nov 9 evening

1. Title choice (1 recommended).
2. Whether the controller is the headline (claim 3 first in the abstract) or the measurement (claim 1 first). The draft puts the measurement first because it is the "why"; the title carries the controller.
3. The one-sentence treatment of the 3 dB gap in the introduction.
