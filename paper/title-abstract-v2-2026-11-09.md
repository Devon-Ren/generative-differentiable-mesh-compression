# Title + abstract, v2 (2026-11-09 evening), for the CVPR 2027 registration (Nov 10 AoE)

Reframed after Yao's guidance: the paper is about compression; the growth rule and the controller are the means. Every number below is in a repo table (cited in brackets).

## Title

**Grow to Budget: Rate-Aware Training and Compression for Opaque Differentiable Meshes**

## Abstract (202 words)

MeshSplatting optimizes an opaque triangle mesh directly from images, but a trained scene occupies hundreds of megabytes. We first show that, unlike Gaussian splats, such meshes cannot be reduced after training: collapsing a trained mesh to a smaller vertex count loses 2.5 dB on average, and up to 6 dB, against a mesh trained at that count on 33 scene–size pairs. We therefore build on MeshSplatting, keeping its representation, renderer and optimization unchanged, and replace its densification rule by a rate-aware one: a split is accepted only if the distortion it removes exceeds λ times the bytes it adds to the encoded file, and the same criterion in reverse drives an in-training edge collapse. Byte costs are measured by the actual encoder during training, and a closed-loop controller maps a requested file size to λ, so size becomes an input: one configuration lands within 10 % of the target on 32 of 33 pairs over Mip-NeRF 360 and Tanks and Temples, and transfers unchanged to Deep Blending. At equal vertex count, before entropy coding, rate-aware growth improves MeshSplatting by 0.4 dB; with our dictionary-SH, grid-position and connectivity coder, files are 24–150× smaller at 0.2–1.7 dB and decode in seconds on a CPU.

## Contributions, in Yao's order

1. **Rate-aware growth makes the representation itself more efficient.** Same vertex count, no codec involved: λ-grown meshes beat MeshSplatting trained with a vertex cap by +0.40 dB mean, 23 of 25 matched pairs (indoor 12/12, mean +0.58) [no-codec-equal-vertices-2026-11-09.md]. Same delivered bytes, with the codec: +0.33 dB, 29 of 33 [g0-refresh-analysis]. And the reason it must happen during training: collapsing a trained mesh to the same count instead loses 2.45 dB mean, up to 6.02, 33 of 33. On room, a matched-count control (same number of splits per interval, upstream's importance sampling instead of the price) isolates +0.12 dB from *which* triangles are split and +0.3 dB from the price-shaped schedule [rd-ctrl-v1].
2. **Size as an input.** The encoder in the loop as the byte meter; a controller that plans vertex growth toward the count the budget buys and sets λ as the rank price of that plan; 33 pairs, mean hit 1.007, 32/33 within 10 %, 22/33 within 5 %, −0.05 dB vs the fixed-λ curve at equal bytes; Deep Blending unseen 1.06–1.12. One calibrated margin per capture class, reported.
3. **Where the bytes go, and what is redundant.** The complete encode/decode chain (dictionary SH, grid positions, cached-edge connectivity, context-coded indices; one file, isolated CPU decoder, 2–6 s) is a system contribution and is kept short. The finding: 45 of the 52 floats per vertex are SH, and replacing them by two indices into per-scene banks (4096 + 256) carries the delivery loss almost alone: at the top rate DC 8-bit recovers only 0.04–0.08 dB of the 0.4–0.6 dB loss indoors (SH sharing ≈ 85 % of it) [indoor-diag-v1]. Dictionary size is not one-size-fits-all: the joint λ × dictionary hull uses K1024 at the two smallest budgets on 11/11 scenes and larger banks (16384 + 1024, or residual 1024) at the top rate on 11/11, i.e. K should scale with V; but the gain over fixed 4096 + 256 is ≤ 0.16 dB at every base point (room 0.12, counter 0.15, kitchen 0.09) [fan-analysis-v2], so the paper uses the fixed banks and reports the hull. Generic mesh codecs (Draco, MPEG SC3DMC) on the same asset are 3.5–15× larger.

Said once in the introduction: opaque meshes are ~3 dB below Gaussian codecs at equal bytes; that is the representation's gap, and this paper is about what the mesh class can do per byte.

## Answers for Yao's specific questions

- *"Without compression, is adaptive-λ better than plain MeshSplatting at equal size / equal quality?"* Yes, measured today on 33 pairs: +0.40 dB at equal vertex count with no codec (uncompressed size is proportional to V). The equal-quality reading (how many more vertices the capped mesh needs to match the λ-grown PSNR) will be read off the pre-codec RD curves once plotted; not computed yet. The paper will show the pre-codec RD curves (PSNR vs V) for both, then the post-codec ones.
- *"Is 4096 + 256 always best after the codebook sweep?"* No. Smaller banks win at small budgets and larger banks at the top rate, on every scene; the dictionary should grow with the vertex count. The fixed setting is within 0.16 dB of the joint hull everywhere, which is why it is used for the headline tables; the sweep goes in the appendix with the K ∝ V observation.
- *"Rate-aware" wording:* keep "rate-aware" in the title and text; "byte-priced" for the mechanism (split iff ΔD > λ·ΔR with ΔR in delivered bytes).

## Still to do before Nov 16 (paper)

- Pre-codec RD figure (PSNR vs V, λ-grown vs cap, 11 scenes) from the existing CSVs: no GPU.
- Qualitative renders at 5 / 10 / 20 MB, one indoor one outdoor: GPU, 1 h.
- Optional: DC q8 rows for all delivered assets (CPU re-delivery), answers Yao's comment #28 directly in a table.
