# Experiments section plan (2026-11-10)

Principle: every table in the main paper answers one of the three contributions, uses BudgetMesh outputs or MeshSplatting itself, and covers all four datasets. Anything that explains *why* rather than *whether* goes to the appendix. Status column: ✔ data in hand, ◐ data in hand but a figure/table still to be made, ✎ needs a run (all small, CPU or minutes of GPU).

## Setup paragraph (one, at the top)

Datasets: Mip-NeRF 360 (9), T&T (2), NeRF-Synthetic (8), Deep Blending (2, used only as a transfer test). Protocol: MeshSplatting training unchanged; images_2 indoor / images_4 outdoor, standard test splits; metrics PSNR/SSIM/LPIPS on the decoded file at the renderer's supersampling; sizes are bytes of the single delivered file. Baseline MeshSplatting FP32 model sizes and PSNR per scene. One sentence on what "the same vertex count" means (after the final cleaning).

## Main paper

### E1. Motivation: a trained mesh cannot be simplified (Contribution 1) ✔
Table: training with a vertex cap vs simplification after training at the same vertex count, three budgets per scene, all 21 scenes (71 pairs). Report per dataset: mean gain, range, wins. Numbers: Mip/T&T +2.45 (33/33), DB +2.71 (6/6), NeRF-Syn +1.27 (32/32). One sentence on why (faces per vertex collapse; the `room-g0` montage in Fig. 1 or the qualitative figure shows it).
Reviewer concern it closes: "why not just simplify it after training like everyone does with meshes".

### E2. Rate-aware growth is a better representation at equal size (Contribution 1) ✔
Table: BudgetMesh (rate-aware growth, before any entropy coding) vs MeshSplatting with a vertex cap, same vertex count: PSNR gain. Mip/T&T 25 matched pairs +0.40 (23/25), indoor 12/12 +0.58; add DB and NeRF-Syn columns from batch21 (derived, no run). Second half of the same table: the same comparison after the codec at the same file size (λ wins 62/71).
Reviewer concern: "is the gain from the growth rule or from the codec?" The two halves answer it separately.

### E3. Size as an input (Contribution 2) ✔
Table: size error per dataset × target: Mip/T&T 5/7.5/10/15/20 MB (55 rows, 53 within 10 %), NeRF-Syn 1.5/2.5/3/4/6 MB (28 non-capped rows, 26), Deep Blending 5/10/20 (6 rows, 4; all within 12 %). Capacity-limited rows (λ at floor) reported in a separate line, not hidden. Column "quality cost of control": equal-bytes ΔPSNR against the constant-price ablation (E7) mean −0.05 dB.
Figure: the RD curve of BudgetMesh per dataset (PSNR vs delivered MB, five targets), FP32 point marked; this is the paper's RD figure.
Reviewer concern: "does the controller trade quality for size?" and "does it generalize?" (Deep Blending column, untouched settings).

### E4. Compression results (Contribution 3, headline) ✔
Table: per scene, FP32 MB/PSNR and BudgetMesh at 20/10/5 MB: MB (ratio) / PSNR (Δ). Means per target: 32× −0.25, 62× −0.62, 125× −1.28. Deep Blending rows included. Decode time (2–6 s CPU) and render fps in the caption or a small column.
Reviewer concern: "how much compression, at what cost, is the file real".

### E5. Against generic mesh codecs (Contribution 3) ◐
Table: same original mesh, MPEG SC3DMC/TFAN (geometry only; geometry + 8-bit SH), Draco (geometry; + SH), zstd of the quantised PLY, vs our file. All 13 COLMAP scenes (MPEG ✔ 13; Draco ✔ 4 + 9 from batch22). Statement: MPEG with attributes is 3–4× our file; MPEG's geometry alone is 0.5–0.9× our whole file.
Reviewer concern: "why a new codec instead of Draco/MPEG".

### E6. Where the bytes go and what is redundant (Contribution 3) ✔
Small table or stacked bar: bytes per stream (positions, connectivity, SH indices, DC, dictionaries) for a typical asset at 20 and 5 MB. Two findings as text with numbers: (i) SH sharing carries the loss, DC 8-bit recovers only 0.04–0.08 dB of the 0.4–0.6 dB top-rate loss; (ii) dictionary size: fixed 4096+256 within 0.16 dB of the joint hull on Mip/T&T and DB, but on the smallest objects (mic, ficus) a 1024-entry dictionary at a lower price is +0.5–0.6 dB at the same file size, so K should scale with V.
Reviewer concern: "is 4096+256 cherry-picked" (Yao's own question).

### E7. Ablations of the training rule (one table, compact) ✔ / ✎
Rows, all on a fixed subset (e.g. room, garden, truck, lego) at one target:
- constant price (controller off): same quality at the same file size, size unpredictable (the only use of constant price runs) ✔
- no in-training collapse (phase B off): size error and quality ✔ (v7b-style rows exist; check)
- price from constants instead of the encoder (no probe): size error ✔ (v7b vs v8/v9 data)
- importance-sampled splits at matched count (MeshSplatting's rule, our schedule): +0.12 dB from *which* triangles ✔ (room only; ✎ one more scene would help, ~1 h GPU)
Reviewer concern: "which part of the method does the work".

### E8. Qualitative figure ✎ (batch23 running)
Two rows of views: GT, FP32, 20/10/5 MB with error maps, one indoor one outdoor; plus the G0 triptych (same vertex count: grown vs capped vs collapsed).

### Limitations paragraph ✔
Per-class aim margin (1.25 / 1.1, calibrated); indoor rows land 5–12 % over because the visible-face set grows after the growth window; small objects cannot fill large budgets (capacity); opaque meshes are ~3 dB below Gaussian codecs at the same file size (stated, not benchmarked in a table).

## Appendix

- A1. Controller internals: byte-probe accuracy (probe/delivered 0.999–1.001), per-interval λ schedule plot for one scene, phase-B rounds, timing overhead (+10 min per run). ✔
- A2. Price sensitivity: ±30 % on the byte constants rescales λ, hull unchanged (8/8 points). ✔
- A3. Full dictionary fan tables (21 scenes × 4 prices × 6 variants). ✔
- A4. Full per-scene tables for E1–E4 and NeRF-Synthetic per-scene controller rows incl. capacity-limited ones. ✔
- A5. Decoder audit (no-leakage proof, standalone decode, format description, stream list). ✔
- A6. Mesh statistics of delivered assets (manifoldness, slivers) and the un-stacking measurement that motivates the aim margin. ✔
- A7. Constant-price RD points (4 prices × 19 scenes) as the ablation's full data. ✔

## Not in the paper
Trainable entropy model (future work, one sentence); the v9c/v9d controller variants (mention as "variants we tried did not change the landing", one sentence, appendix table optional); 3DGS codec comparison table (one sentence in the intro per Yao).

## Remaining runs
- batch22 (Draco rows, 9 scenes): submitted.
- batch23 (renders): block ready.
- E7 matched-count control on 1–2 more scenes: optional, ~1 h GPU each; decide after seeing the queue.
Everything else is table-making from existing files.
