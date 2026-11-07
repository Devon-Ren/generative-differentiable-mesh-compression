# 2026-10-16 — Room hull-v2 re-export (final-format numbers for the codebook RD figure)

Figure v1 (stage2-v1 format, q14/q8 axis-A sweep): `analysis/rd/room-rd-codebook-v1.png/.pdf`, data `room-codebook-v1.csv`,
script `plot_codebook_rd.py`. Figure v2 = same plot from `hull-v2.txt` (const/ctx q14/q7) once the job below has run.

## D. 提交 hull-v2 重导出（GPU，~40–60 min，无训练，可重跑）
```bash
mkdir -p /scratch/zr2245/analysis
# (1) 把 analysis/commands/hull-v2-reexport-v1.sh 和 hull-v2-v1.sbatch 上传到 /scratch/zr2245/analysis/
ls -l /scratch/zr2245/analysis/hull-v2-reexport-v1.sh /scratch/zr2245/analysis/hull-v2-v1.sbatch
ls -d /scratch/zr2245/room-vis-collapse/axisA-v0.8/k*_r* | wc -l     # expect 12
sbatch /scratch/zr2245/analysis/hull-v2-v1.sbatch
```
Done when `hull-v2.slurm.<job>.log` ends with `hull-v2 exit 0`; send back `/scratch/zr2245/room-vis-collapse/hull-v2.txt`
(one line per point: `axisA-v0.8/k4096_r256 | vertices … | K 4096 R 256 | v2 q14/q7 <MB> MB test <psnr> <ssim> <lpips> | v1-format …`).
Then (CPU, here): build `room-codebook-v2.csv` from the 12 axisA lines and run
`python3 plot_codebook_rd.py room-codebook-v2.csv room-rd-codebook-v2 --title ... --default 4096,256`.

## E. 多 level 的 K×R sweep（用户 10/16 要的图：每个顶点保留率 12 点 + 各自 hull，一张图多条 hull）
Script `commands/axisA-levels-v1.sh` (delivered inline as heredoc): levels 1.0 0.9 0.7 0.6 0.5 (0.8 last, optional), K {1024,4096,16384} x R {1,256,1024,4096},
final format const/ctx q14/q7, waits for `$W/hull-v2.txt` before using the GPU, restartable, table `/scratch/zr2245/room-vis-collapse/axisA-levels.txt`
(one line per point: `level L | K k R r | vertices n | q14/q7 v2 const/ctx | MB test psnr ssim lpips | val ... | mapping banks | seconds`).
Plot: `rd/plot_codebook_levels.py <csv level,K,R,MB,test_psnr> <out-stem> --default 0.8,4096,256` -> colour = level (blue ramp, dark = more vertices),
marker = K, hull per level, label column at the right. Output will be `rd/room-rd-codebook-levels-v1.png/.pdf` from `rd/room-codebook-levels-v1.csv`.
Timing (10/16): first point 886 s was a cold start; steady state 212-228 s per point (K1024), ~5 min for K16384 -> 60 points serial ~4-5 h. Optional: 5 parallel sbatch jobs, one level each
(`axisA-levels-v1.sbatch`, `sbatch --export=ALL,LEVELS=<L>,WAIT_FOR=`), ~3-4 h each; all append to the same axisA-levels.txt. v0.8 not re-run (hull-v2.txt).
First point: v1.0 K1024 R1 = 26.706 MB / 27.797 (vs v0.8 K4096 R256 25.513 / 27.883: fewer vertices + larger dictionary wins).
Level 1.0 done (12 points, 78 min; rows in rd/axisA-levels-2026-10-16-partial.txt); interactive node then expired. Level-1.0 hull:
K1024R1 -> K4096R1 -> K1024R1024 -> K4096R256 -> K4096R1024 -> K4096R4096 -> K16384R4096; K16384 R1/R256/R1024 below hull (0.025/0.013/0.004).
Reduced plan (user asked to cut): levels 0.7 + 0.6 only (0.9 dominated by 0.8, 0.5 deferred), grid K1024/K4096 x 4 R + K16384 R4096 = 9 points/level,
`axisA-levels-v2.sbatch` (~2 h). Final figure = 4 hulls (100/80/70/60), v0.8 from hull-v2.txt.

## Result (10/16 06:10): axisA-levels.txt 30 rows (exit 0 0) -> rd/axisA-levels-2026-10-16.txt; + 12 v0.8 rows from hull-v2 -> rd/room-codebook-levels-v1.csv (42 rows)
Figure: rd/room-rd-codebook-levels-v1.png/.pdf (plot_codebook_levels.py, 4 hulls 100/80/70/60).
Per-level hulls: 1.0 K1024R1->K4096R1->K1024R1024->K4096R256->K4096R1024->K4096R4096->K16384R4096 | 0.8 K1024R1->K4096R1->K4096R256->K4096R1024->K4096R4096->K16384R4096
| 0.7 K1024R1->K4096R1->K1024R1024->K4096R256->K4096R1024->K4096R4096->K16384R4096 | 0.6 K1024R1->K4096R1->K1024R1024->K4096R1024->K4096R4096->K16384R4096.
GLOBAL hull: v0.6K1024R1 (16.6/27.222) -> v0.6K4096R1 (17.6/27.351) -> v0.7K4096R1 (20.4/27.594) -> v0.8K4096R1 (23.1/27.759) -> v0.8K4096R256 (25.5/27.883) -> v0.8K4096R1024 (26.2/27.907)
-> v1.0K4096R1024 (31.8/28.091) -> v1.0K4096R4096 (33.0/28.115) -> v1.0K16384R4096 (35.8/28.140). 0.7 hull lies entirely under 0.8's; 0.6 only wins below ~19 MB; 1.0 crosses above 0.8 at ~28 MB.
