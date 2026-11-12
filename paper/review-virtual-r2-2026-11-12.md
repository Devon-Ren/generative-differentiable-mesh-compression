# Virtual reviewer 2 (2026-11-12): independent agent; read the CVPR draft (tex, PDF, Fig. 1) and three result files not yet in the draft

Scores (5-point scale): draft as it is 2 (Weak Reject, upper end); with the new measurements integrated honestly and the contributions re-framed 3 (Borderline).
Confidence 4, novelty 3, soundness 2, rigor 2, clarity 3. "A 4 is reachable if the cap is shown to be intrinsically noisy and the paper is re-centred on Table 1a plus the system."

## Weaknesses (reviewer's ranking)
- W1 size-control claim has no baseline in the draft; the intro sentence "Today it cannot be ... several times" is refuted by the cap-from-prior data (27/33 interpolated, 3/3 measured captured scenes); at 10 MB on real scenes full / no-probe / cap-prior all sit at 10/11.
- W2 "one byte price" is narrative: dR_split is constant, so growth is top-k by error with k from the plan; lambda is an output; probe + plan with MeshSplatting's selection lands 3/4.
- W3 selection-rule gain overstated: Table 5's -0.42 dB comes from runs that landed -27% and an interpolated curve; controlled measurement +0.17 / +0.14 / +0.07; vs cap on captured scenes +0.13 / 0.00 / -0.09.
- W4 calibration in-sample; hidden constants: the controller plans with fixed 2.5 faces per vertex during growth although the draft says the probe measures f; aim needed spans 1.04-1.30 / 0.96-1.41.
- W5 ratios vs raw floats (7-27x vs quantized PLY+zstd); Fig. 1 still shows a "MeshSplatting (paper)" point at ~100 MB while Table 3a lists 642 MB; the 1.4x row (Draco geometry + zstd payload) is never discussed; the index context coder gains little over zstd.
- W6 perceptual loss under-reported (LPIPS 0.309 -> 0.411 at 20 MB); no qualitative figure in the main paper.
- W7 Table 1a depends on a 3000-step fine-tune budget that is not ablated.
- W8 inconsistencies: capacity-limited counts (three objects in Sec. 4.1 vs 6 (+2) in Table 2; 3 MB row), Fig. 1 uses 5 objects / 84 MB vs Table 3b 8 objects / 65.7 MB, TBD cells, VERIFY bib entries.
- W9 the new data are not internally consistent: interpolation predicted the two-training cap at 11/11 and the Mip prior on playroom at +22..32%; measured +16% and +9.6%. In garden and playroom the FIRST batch27 cap runs kept 0.78-0.80 of the cap, the second 0.94-0.95 (which matches the older runs): bimodal seed behaviour or a configuration difference; "loose handle" is an untested hypothesis.

## Advice on size control
- Acceptable claim: "With one constant per capture class and no scene-specific size prior, BudgetMesh lands within 10% on 83/89 in-capacity runs (4/6 held-out) at about 1.3x the wall-clock of one MeshSplatting run. A vertex cap from a size prior fitted on similar scenes reaches the same tolerance in one run on roughly four of five in-distribution scenes, but does not transfer across datasets or to small budgets, and needs a second training otherwise."
- Table rows: cap from LOO prior; same + conditional retry; probe + plan with MeshSplatting selection; BudgetMesh without collapse; full. Columns: calibration data required, trainings (mean/max), GPU-minutes, hit rate at 5% / 10% after first and final training, mean abs / worst error, dPSNR at same size; mark measured vs interpolated.
- Retry protocol: retry only on a miss; affine update (subtract the fixed dictionary bytes); max 3 trainings; same rule for BudgetMesh's misses; give the baseline an in-domain prior (the lego -41.8% used the Mip prior while BudgetMesh's aim was calibrated on NeRF-Synthetic).
- Priority runs: (1) diagnose the cap non-monotonicity: diff configs/logs, repeat garden cap 1,105,000 and playroom 1,081,000; (2) single aim 1.25 on lego + one more object; (3) fair lego baseline: leave-one-object-out prior + affine retry; (4) kitchen at 10 MB with the cap prior; (5) CPU: LOO aim hit rates, error CDF, cumulative hit rate vs GPU-minutes.
- Growth aim: say it stands for final faces per vertex (r = -0.97), show the scatter and the negative prediction result, explain why it is set above the mean, list every constant in one table.

## Text edits
1 rewrite abstract / intro par. 2-3 / contribution 1; 2 demote the Lagrangian framing, fix "f is probed"; 3 reconcile the 100 MB point, lead with 7-27x next to 31-122x, add dSSIM / dLPIPS; 4 discuss the 1.4x row, replace "3-18x", replace Table 5's -0.42; 5 Fig. 1 zoom or dPSNR-vs-cap plot, label overlaps, same object sets, qualitative figure in main, TBD cells, capacity counts, "dB" macro spacing, line-number overlaps, VERIFY bib.
