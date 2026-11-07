---
name: training-speed-review-oct2026
description: 10/31 profile of MeshSplatting RD-growth training on the L40S (where the time goes, what is result-preserving) and the train_rd_v4 speed variant awaiting a Room check
metadata:
  type: project
---

SHELVED by user 10/31 (accepts the 16-17 GPU-h hull batch; v4 never run). Profile (Room lam 1.85e-3 final mesh, 1.19 M vertices, analysis/diag/profile_train_v1b.py, commands/profile-train-v1b.md): per-iteration 34.6 / 38.5 / 65.1 / 171.3 ms at it5000 / it15000 / 2x / 4x -> ~32 min of iterations per run (upper bound). Rasterizer backward kernel = 43 % (102 ms/step at 4x), forward ~25 %, Adam 6.6 ms constant (10 %), identity area pools 4 ms at scale 1, GT h2d copy 1.8 ms, zero-fills 6-16 ms at 2x/4x. fused_ssim is NOT installed in the env (python SSIM). Compiled .so has sm_89 SASS. upstream training_report evaluates all test views + LPIPS every 1000 its (pure logging, cost unmeasured, est 3-5 min). Images on GPU fit even at 4x (peak 10.8 GB); the v1 profiler OOM was a missing no_grad around training_report.

v4 (commands/speed-v4-v1.md; rdgrowth/render_v4.py, rdgrowth/train_rd_v4.py, commands/rd-prune-v4.sbatch, outputs /scratch/zr2245/<scene>-rd4-lam<L>-mu<mu>, table rd-prune4-<scene>.txt): eval only at --test_iterations, skip identity pools, fused Adam, --data_device cuda. Expected 7-9 min saved of ~40 (20 %); Room check job proposed, pass = test PSNR within 0.05 dB and MB within 2 % of the v3 row. Rule from user: tolerate small floating-point changes, never change what is optimised. Open: warp-reduced atomics in backward.cu (CUDA rewrite, 5-7 min more) — user decides.

**How to apply:** when launching new training batches after the Room check passes, use rd-prune-v4.sbatch; plotting must read both rd-prune3 and rd-prune4 tables. See [[meshsplatting-compression-status]], [[rdgrowth-v3-prune-design]].
