# Virtual reviewer 1 (2026-11-11): independent agent, read only title/abstract, experiments-draft-v1.tex (batch24 numbers), figures, related-work notes

Overall 4 (Borderline); confidence 4; novelty 3; soundness 3; rigor 3; clarity 3. "If W1-W3 are addressed convincingly I would move to 5."

## Weaknesses (ranked by the reviewer)
- W1 (rating): ablation does not isolate the byte term. The ranking variant swaps in importance sampling (changes criterion and byte term, lands 27 % small). Needed control: error-only ranking (dD without dR), same split count, all 11 scenes.
- W2 (rating): vertex-cap margin +0.27 dB with pairs matched within +-15 % V; which side has more vertices not stated; Fig g0 shows cap-486k (28.58) >= grown-5MB (28.55) contradicting its caption; contribution list (+0.40, 23/25) disagrees with Tab 2 (+0.27, 29/30); NeRF-Synthetic same-file-size +0.12 on 16/24 "coin flip".
- W3 (rating): PSNR only; LPIPS 0.309 -> 0.411 (20 MB) -> 0.508 (5 MB), SSIM 0.731 -> 0.704 -> 0.632 on Mip-360; all main tables need SSIM/LPIPS.
- W4: cost-of-control numbers inconsistent (0.15 / -0.12 / -0.03 / -0.27 / -0.05); reference curve = 2 points per scene, interpolation error unbounded.
- W5: ratios vs unquantized FP32 inflated; quantised PLY + zstd is only 5.2x ours; no decomposition fewer vertices vs fewer bytes per vertex.
- W6: generic-codec table not a fair RD comparison; natural baseline = Draco + VQ-SH (the same two codebook indices as a generic attribute); "level 1.0" undefined.
- W7: transfer claim on 2 indoor Deep Blending scenes with the worst size error (10 MB row +10.2 % mean); same capture class as the tuned aim; "one setting" is one constant per class.
- W8: simplification baseline under-specified (decimator, fine-tune length, what is optimised); face collapse suggests degenerate topology.

## Questions
1 error-only ranking on all scenes; 2 signed V difference in Tab 2 + exact-V interpolation; 3 reconcile +0.40 vs +0.27 and -0.05 vs -0.12/-0.27; 4 MeshSplatting paper 24.78 vs our FP32 24.81 with "-2 dB RGB-only" footnote; 5 decimation protocol; 6 seed variance; 7 how/when aims 1.25/1.1 chosen; 8 SSIM/LPIPS per target and vs cap at equal bytes; 9 missing 3/6 MB NeRF-Synthetic rows, capacity limit = controller failure?

## Missing experiments (reviewer's essential list)
error-only ranking; cap at exact V with SSIM/LPIPS; 3-4 prices per scene for the reference curve with stated interpolation error; Draco + VQ-SH; codec component ablation (dictionary vs per-vertex quantised SH, context coder vs Draco connectivity, grid bits); one 3DGS-class row in the main text; transfer evidence beyond 2 indoor scenes (or drop the claim). Nice-to-have: ratio decomposition, seed variance, compressed competitors.

## Clarity
Private vocabulary (price, rank price, growth aim, byte probe, landing estimate, capacity limit, level 1.0, reference curve) defined once; Tab 1 vs Fig g0 mismatch (two vs three methods); Tab 3 vs Tab 4 different scene sets for the same target; NeRF-Synthetic FP32 reference differs (val-selected) -> positive dPSNR; Fig rd panels compressed by 100-1500 MB diamonds; qualitative crops indistinguishable except error maps; Fig g0 caption vs numbers; Sec e9 "by construction" contradicted by -6.6 % undershoots; draft marks left.

## Three changes
1 error-only ranking ablation (+ price-schedule control) on all 11 scenes; 2 cap comparison at exact V with PSNR/SSIM/LPIPS, reconcile numbers, SSIM/LPIPS everywhere incl. abstract; 3 fair generic baseline (Draco + VQ-SH) + codec component ablation + one 3DGS row in main.
