# Controlled vocabulary for the paper (2026-11-10)

One term per concept. Defined once, in plain words, the first time it appears; used identically everywhere after, including tables, captions, figure labels and the appendix. The "do not write" column lists the synonyms that already crept into our notes and would make a reader think there are two things when there is one.

## Objects

| concept | write | do not write | first-use definition (one sentence) |
|---|---|---|---|
| the base method | MeshSplatting | upstream, the original method, the base system, vanilla | MeshSplatting [ref] optimizes an opaque triangle mesh directly from posed images. |
| our method | BudgetMesh | the controller, adaptive-λ, the system, our pipeline, grow-to-budget (as a name) | BudgetMesh trains a MeshSplatting mesh so that its encoded file meets a size the user specifies, and encodes it. |
| the thing being compressed | opaque triangle mesh / the mesh | triangle soup, differentiable mesh (keep only in the title and intro), asset, model | A mesh of opaque triangles with per-vertex position, color and view-dependent appearance. |
| the stored result | the file | asset, delivery, delivered asset, bitstream, payload | The single file our encoder writes and our decoder reads. |
| its size | file size (in MB) | bytes, rate, bitrate, budget (except in "size budget" once) | File size is the size of that file in megabytes. |
| what the user asks for | target size | budget, target budget, requested size, target MB | The target size is the file size the user asks for. |
| the uncompressed model | the FP32 model | original, trained model, dense model, full model | The FP32 model is the trained MeshSplatting mesh stored in 32-bit floats. |
| vertex count | vertex count | V (in prose), number of vertices, point count, primitive count | — |
| a growth step | split | subdivision, densification (use once to connect to MeshSplatting's term), refinement step | A split replaces one triangle by four by inserting its edge midpoints. |
| a shrink step | collapse | edge collapse (first use only), merge, prune, decimation | A collapse merges the two endpoints of an edge into one vertex and removes the faces on that edge. |
| reducing a trained mesh afterwards | simplification after training | post-hoc collapse, post-hoc simplification, decimation, downsampling, G0 | Simplification after training reduces a finished mesh to fewer vertices by collapses, followed by fine-tuning. |
| training with a vertex limit | training with a vertex cap | train-at-budget, max_points run, capped run | Training with a vertex cap runs MeshSplatting with an upper limit on the vertex count. |
| our growth rule | rate-aware growth | byte-priced growth, priced split, λ-growth, phase A | Rate-aware growth accepts a split only if the error it removes is worth the bytes it adds to the file. |
| our in-training shrink rule | rate-aware collapse | λ-prune, phase B, trimming, the prune | Rate-aware collapse applies the same test in reverse during training. |
| the trade-off weight | the price λ | lambda, the Lagrangian multiplier, exchange rate, threshold | The price λ converts bytes into error units; a split is accepted when error removed > λ × bytes added. |
| how λ is set in BudgetMesh | the size controller | adaptive λ, closed loop, rate control, the controller (after first use OK) | The size controller adjusts the price during training so that the final file meets the target size. |
| measuring bytes during training | byte probe | probe, size probe, encoder-in-the-loop | A byte probe encodes a copy of the current mesh with the real encoder to measure its file size. |
| estimated bytes of a change | byte cost | ΔR, rate term, price tag | — |
| error of a triangle | error (of a triangle / of a region) | distortion, residual, ΔD, loss mass | The error of a triangle is the sum of pixel errors over the pixels it covers in the training views. |
| our encoder/decoder | the codec (encoder, decoder) | stage-2, delivery chain, compression pipeline | The codec is our encoder, which writes the file from a trained mesh, and our decoder, which reads it. |
| SH compression | shared appearance dictionary | SH codebook, bank, dictionary SH, K4096 R256 (numbers only when needed) | The shared appearance dictionary replaces each vertex's 45 view-dependent coefficients by two indices into per-scene tables. |
| constant λ runs | constant price (controller off) | fixed-λ, fixed lambda method, the hull, rd-prune3 | In the ablation "constant price", λ is held fixed for the whole training instead of being set by the controller. |
| hit accuracy | size error | hit ratio, landing, hit | Size error is (file size − target size) / target size. |
| the held-out dataset | Deep Blending, used only for transfer (not touched during development) | unseen, held-out (define if used), one-shot | — |

## Phrases we keep, with their fixed wording

- "at the same vertex count" (never "equal V", "matched count", "iso-vertex")
- "at the same file size" (never "equal bytes", "iso-rate")
- "quality" = PSNR unless stated; write "PSNR" in tables, "quality" in prose only when the sentence does not quote a number
- "the four datasets" after they have been listed once

## Writing rules for this paper

1. Mechanism first, name second: say what happens, then give it its name in parentheses the first time. After that, the name alone.
2. No Greek letters in the abstract or figure titles except λ, and λ only after "price".
3. Every table caption says what is being compared, on what, and what "better" means (higher or lower), in that order.
4. Numbers in prose: at most one per sentence; the rest in the table the sentence points to.
5. One symbol per quantity: V vertex count, F face count, R file size in bytes, D error, λ price. Do not reuse letters.
6. Do not use: post-hoc, ad hoc, a priori, de facto, in the loop, closed-loop (say "the controller measures the file size during training and adjusts the price"), upstream, hull (say "curve"), soup, landing, trim, prune.
7. When a result is partial, the limitation is in the same paragraph, not in a later section only.
8. Captions and section titles are complete sentences or noun phrases a non-specialist can read without the text.

## Sentences to reuse (already checked)

- Why: "Because each pixel is covered by exactly one opaque triangle, removing vertices from a trained mesh degrades quality far more than training a smaller mesh from the start."
- What: "BudgetMesh accepts a split only when the error it removes is worth the bytes it adds to the file, applies the same test in reverse to collapse edges during training, and adjusts the price so that the final file meets the target size."
- Codec: "The encoder stores positions on a grid, connectivity with a context-based arithmetic coder, and view-dependent appearance as indices into a shared per-scene dictionary; a standalone decoder reconstructs a standard mesh in a few seconds on a CPU."
