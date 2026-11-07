# rd-metrics-v1 (PROVISIONAL, 2026-11-01): PSNR / SSIM / LPIPS vs delivered MB, dataset means

plot_rd_metrics_v1.py rd-metrics-v1 -> rd-metrics-v1.png/.pdf. Our SSIM/LPIPS = delivery-log values (training_report evaluator = upstream train.py: pip lpips VGG on [0,1]).
Published points: HAC++ Table II (3DGS class), MeshSplatting Tables 1-2 (mesh class). Mip-360 FP32 / level-1.0 rows omitted (5 originals have no logged SSIM/LPIPS); T&T has them.

Our means (MB / PSNR / SSIM / LPIPS):
  Mip-360: 6.2e-4 28.1 / 24.47 / 0.722 / 0.376; 1.85e-3 20.3 / 24.27 / 0.712 / 0.400; 6.2e-3 8.9 / 23.81 / 0.676 / 0.460; 1.85e-2 4.2 / 23.09 / 0.628 / 0.515
  T&T:     6.2e-4 28.7 / 20.17 / 0.737 / 0.332; 1.85e-3 22.3 / 20.43 / 0.745 / 0.331; 6.2e-3 11.4 / 20.52 / 0.737 / 0.358; 1.85e-2 5.4 / 20.10 / 0.704 / 0.409
  T&T FP32 reproduction 458.6 / 20.64 / 0.759 / 0.272; level 1.0 21.2 / 20.52 / 0.739 / 0.318

Readings:
1. SSIM agrees with the MeshSplatting paper where PSNR agrees: T&T 0.745 (ours, 1.85e-3) vs 0.745 (paper); Mip-360 0.722 vs 0.728 (PSNR -0.3, the indoor-flag deficit). LPIPS does NOT: T&T 0.331 vs 0.287, Mip-360 0.376 vs 0.310, i.e. +0.04..+0.07 with equal SSIM. CORRECTED the same day with the paper's per-scene Tables 6-8: outdoor and T&T LPIPS agree with our FP32 within +-0.02 (garden 0.217 vs 0.249, flowers 0.417 vs 0.409, treehill 0.428 vs 0.427, truck 0.237 vs 0.226, train 0.337 vs 0.319) -> same evaluator (VGG). The gap is indoor only (room 0.271 vs 0.339; counter/kitchen level-1.0 0.389/0.363 vs 0.279/0.227) = the missing --indoor flag, plus codec loss on the delivered lambda assets. The unified pass is still needed for consistency, not to change the network.
2. vs MiLo (253 MB): at 20 MB ours is higher in PSNR (+0.18) and SSIM (+0.024) but lower in LPIPS as printed (0.400 vs 0.323) -> cross-paper LPIPS cannot be compared until the network is known; the paper must state the network and recompute ours with the same one.
3. SSIM and PSNR tell the same story as PSNR alone (same ordering, same knee at 6.2e-3); LPIPS degrades faster at the low end (0.376 -> 0.515 over 28 -> 4 MB on Mip-360), so the 150x point should be sold on size, not perceptual quality.
4. T&T is non-monotone in all three metrics (6.2e-3 is the best point): same val-vs-test effect as the PSNR hull (R9), consistent across metrics.
Final version (supplement S2) after eval_all_v1 (unified pass); then add Mip-360 FP32 / level-1.0 rows and the generic-codec points.
