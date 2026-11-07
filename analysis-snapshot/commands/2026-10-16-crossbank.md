# 2026-10-16 — Main line batch 1: cross-scene codebook + index spatial locality

User approved candidate-1 story ("我觉得这个主线可以，我们快速推进进度做吧"). First cheap-novelty batch, one sbatch job on the GPU node.

## Files
- `analysis/commands/crossbank-v1.sbatch` — garden v0.8 mesh (`$G/axisB/v0.8/finetune/best.pt`), q15/q7, stage2-v2 const/ctx. Three deliveries:
  - `own`        : `--historical-config none --k 4096 --r 1024 --dictionary-rounds 4` (garden-trained banks, reference)
  - `xfer-frozen`: `--historical-config $A/frozen-banks.json --k 0 --seed-asset $SEED --dictionary-rounds 0` (Room K4096/R1024 banks, bank_lr = residual_lr = 0, pure assignment)
  - `xfer-adapt` : `--historical-config none --k 0 --seed-asset $SEED --dictionary-rounds 0` (Room banks as init, appearance finetune allowed)
  - SEED = `seed_asset` from `/scratch/zr2245/room-shared-collapse-QGQgyr/run/config.json`, fallback `/scratch/zr2245/room-fixed-sharing-0N4OSU/run/shared/residual_k4096_r1024_f1/asset.mslb`.
  - Output `/scratch/zr2245/garden-vis-collapse/crossbank/{own,xfer-frozen,xfer-adapt}` + summary `/scratch/zr2245/garden-vis-collapse/crossbank.txt` (MB, test psnr/ssim/lpips, val coarse/shared_initial/fitted/decoded, banks MB, mapping MB). Log must end `crossbank exit 0`.
- `analysis/index_locality.py` v1 (CPU) — per deliver dir: loads shared-final.pt, undirected mesh edges, per code stream (mapping z with K, residual u with R): neighbour hit rate, independent Σp², lift, entropy H, conditional entropy H(z|nbr) (sparse pair counts; biased low when K² ≈ #edges), ctx coder real bits/vertex from summary.json. Run over every Room K×R dir (`$W/axisA-v0.8/k*_r*`, `$W/axisA-q14q7-v*/k*_r*`) + the three garden crossbank dirs -> `/scratch/zr2245/garden-vis-collapse/index-locality.csv`.

## Decision rule (told to user)
xfer-frozen within 0.3 dB of own => "codebook is universal" (texture-sample claim strengthens: appearance codes are scene-agnostic).
xfer-frozen more than 1 dB below own => scene-specific codebook (still a result: dictionary must be delivered per scene; xfer-adapt measures how much a Room init recovers).

## Expected outputs from user
`crossbank.txt` (3 rows) and `index-locality.csv` -> analysis + index-locality figure (lift / ctx bits vs K across levels).

## Results (2026-10-16, job on L40S, 32G would have sufficed)
Seed actually used: `/scratch/zr2245/room-delivery-quant-0i2F5s/run/xyz-q18_dc-q8-_luz1ezw/asset.mslb` (config.json seed_asset; Room K4096 R1024).
garden v0.8 q15/q7 (MB / test PSNR / SSIM / LPIPS): own 28.056 / 24.393 / .7302 / .2970 | xfer-frozen 27.721 / 23.745 / .7172 / .3080 (-0.65 dB) | xfer-adapt 27.714 / 24.162 / .7232 / .3032 (-0.23 dB). Val decoded 24.814 / 24.160 / 24.556. banks 0.77 MB all; mapping 6.60 vs 6.26 MB (Room codebook used less uniformly on garden: H 10.90 vs 11.77 bits).
Verdict: middle zone -> "transferable appearance prior / warm start", not "universal codebook". Confound: own used --dictionary-rounds 4, xfer runs rounds 0 -> crossbank-v2 controls (rand, own-r0, reverse direction garden->Room).
Raw: analysis/rd/crossbank-2026-10-16.txt, index-locality-2026-10-16.log, index-locality-v1.csv (45 rows; 0.8 group = axisA-v0.8 v1 zstd dirs, coder column is zstd there).
Index locality (Room, ctx levels 0.6/0.7/1.0): hit_z 0.12-0.22 (indep 1e-3..1e-4, lift 100-780x); H(z) = log2K - 0.3 (uniform usage); H(z|nbr) 6.0-7.5 bits ~ independent of K; coder 7.1-7.7 (K1024), 9.2-9.8 (K4096), 11.5-12.1 (K16384). Residual: hit_u 0.02-0.05, H(u|nbr) = log2R - 1, coder ~ raw. hit_z by level: 1.0 0.219 -> 0.8 0.174 -> 0.7 0.172 -> 0.6 0.171 (first 20 % removed = same-code neighbours). Garden own hit_z 0.038 (Room 0.127). Figure: analysis/rd/room-index-locality-v1.png/.pdf (plot_index_locality.py).
Next: commands/crossbank-v2.sbatch (general account, 32G): garden rand / own-r0, Room room-own-r0 / rev-frozen / rev-adapt -> $G/crossbank2.txt.

## Controls crossbank-v2 (done 2026-10-16, ~16 min on L40S, 32G)
garden: rand 27.771 / 24.346 | own-r0 27.993 / 24.384 (own rounds 4 = 24.393). Room v0.8 q14/q7: room-own-r0 26.222 / 27.890 | rev-frozen 25.499 / 27.658 | rev-adapt 25.464 / 27.743.
Reading: refinement rounds worth 0.01 dB; data-sampled codebook within 0.05 dB of k-means; foreign-scene init (24.162) worse than random own sample -> no cross-scene transfer; outdoor vocabulary covers indoor (-0.23) but not vice versa (-0.65). Codebook content is nearly irrelevant; information = index assignment + joint fit.
Pending from user: initialize() source head from the crossbank2 log (confirm --kmeans-rounds 0 = data sample). Batch 2 = commands/garden-axisA-v1.sbatch.

## Greedy allocator check (2026-10-16, local, no GPU)
analysis/rd/allocator_greedy.py: from the richest config, repeatedly take the single-lever cut (R, K, or vertex level one notch) with the lowest dB/MB price.
Room (42 measured points): the greedy path = the global hull exactly, 9/9 points, 0.000 dB gap (allocator-greedy-room-v1.txt). Implication: the hull needs only local slopes (3 neighbour evaluations per step), not the full grid; "lever switching" is a greedy rule.

## bonsai allocator check (2026-10-17)
rd/bonsai-codebook-levels-v1.csv (18 points: levels 1.0/0.8/0.7 x backbone). Global hull: v0.7K1024R1 -> v0.7K4096R1 -> v1.0K4096R1 -> v1.0K4096R256 -> v1.0K4096R1024 -> v1.0K4096R4096 -> v1.0K16384R4096 (level 0.8 never on the hull).
Greedy single-lever path: 8 points, 7 on the hull, the one off-hull point (v0.8 K4096 R1) sits 0.005 dB under the chord. Room: 9/9 exact. So the greedy rule reproduces the hull on both a "vertices first" scene and a "dictionary first" scene. Figure rd/bonsai-rd-codebook-levels-v1.png, output rd/allocator-greedy-bonsai-v1.txt.
