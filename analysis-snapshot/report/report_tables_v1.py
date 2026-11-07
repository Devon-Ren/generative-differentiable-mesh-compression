"""Report tables v1 (2026-11-01) -> tables-v1.md.  T1 per-scene ours vs FP32 vs the MeshSplatting paper; T2 cross-method dataset means; T3 deployability (copied)."""
import sys; sys.path.insert(0, '/mnt/project-files/analysis/report'); from figstyle_v1 import *
rows, SL = load_rows()
TL = {r['scene']: (float(r['psnr']), float(r['ssim']), float(r['lpips'])) for r in csv.DictReader(open(D + 'originals-trainlog-metrics-2026-11-01.csv'))}
PAPER = {'bicycle': (23.04, .641, .348), 'flowers': (19.34, .480, .417), 'garden': (24.70, .762, .217), 'stump': (24.78, .678, .316), 'treehill': (20.53, .540, .428),
         'room': (28.52, .873, .271), 'counter': (26.51, .846, .279), 'kitchen': (27.42, .858, .227), 'bonsai': (28.19, .876, .294), 'truck': (22.32, .799, .237), 'train': (18.72, .693, .337)}
L = ['# Report tables v1 (2026-11-01)', '', 'Conventions: PSNR/SSIM/LPIPS on the test split (every 8th view), MB = 1e6 bytes, ratio = FP32 size / delivered size. '
     'FP32 = our reproduction WITHOUT upstream --indoor (indoor rerun in progress). LPIPS: the "FP32 (train.log)" columns use the upstream train.py / 3DGS-protocol convention ([0,1] inputs); '
     'our delivered-asset columns use the pipeline evaluator ([-1,1] inputs), which reads ~0.03 higher on the same images (garden FP32: 0.216 vs 0.249) -> the unified pass will put every number on the 3DGS convention. Room FP32 row is pipeline-evaluated (no train.log).', '']
# ---------- T1
L += ['## T1. Per scene: FP32 model, the MeshSplatting paper, and our delivered assets', '',
      '| scene | FP32 MB | FP32 PSNR / SSIM / LPIPS (train.log) | paper PSNR / SSIM / LPIPS | λ 6.2e-4: MB (ratio) PSNR / SSIM / LPIPS | λ 1.85e-3: MB (ratio) PSNR / SSIM / LPIPS | λ 6.2e-3: MB (ratio) PSNR / SSIM / LPIPS | λ 1.85e-2: MB (ratio) PSNR / SSIM / LPIPS |', '|---|---|---|---|---|---|---|---|']
def cell(s, lam):
    mb, p = ours_lambda(rows, s, lam); ss, lp = SL[(s, round(lam, 6))]; return f'{mb:.1f} ({ORIG[s][1] / mb:.0f}×) {p:.2f} / {ss:.3f} / {lp:.3f}'
for s in M360 + TNT:
    t = TL[s]; pp = PAPER[s]
    L.append(f'| {s} | {ORIG[s][1]:.0f} | {t[0]:.2f} / {t[1]:.3f} / {t[2]:.3f} | {pp[0]:.2f} / {pp[1]:.3f} / {pp[2]:.3f} | ' + ' | '.join(cell(s, l) for l in LAMS) + ' |')
def means(scenes):
    out = {}
    for l in LAMS:
        mb = sum(ours_lambda(rows, s, l)[0] for s in scenes) / len(scenes); p = sum(ours_lambda(rows, s, l)[1] for s in scenes) / len(scenes)
        ss = sum(SL[(s, round(l, 6))][0] for s in scenes) / len(scenes); lp = sum(SL[(s, round(l, 6))][1] for s in scenes) / len(scenes); fp = sum(ORIG[s][1] for s in scenes) / len(scenes)
        out[l] = (mb, fp / mb, p, ss, lp)
    return out
for name, sc in (('Mip-360 mean (9)', M360), ('T&T mean (2)', TNT)):
    m = means(sc); t = [sum(TL[s][i] for s in sc) / len(sc) for i in range(3)]; pp = [sum(PAPER[s][i] for s in sc) / len(sc) for i in range(3)]
    L.append(f'| **{name}** | {sum(ORIG[s][1] for s in sc) / len(sc):.0f} | {t[0]:.2f} / {t[1]:.3f} / {t[2]:.3f} | {pp[0]:.2f} / {pp[1]:.3f} / {pp[2]:.3f} | ' + ' | '.join(f'{m[l][0]:.1f} ({m[l][1]:.0f}×) {m[l][2]:.2f} / {m[l][3]:.3f} / {m[l][4]:.3f}' for l in LAMS) + ' |')
# ---------- T2
m9 = means(M360); m2 = means(TNT); fp9 = sum(ORIG[s][1] for s in M360) / 9; fp2 = sum(ORIG[s][1] for s in TNT) / 2
t9 = [sum(TL[s][i] for s in M360) / 9 for i in range(3)]; t2 = [sum(TL[s][i] for s in TNT) / 2 for i in range(3)]
L += ['', '## T2. Cross-method, dataset means (published numbers verbatim; FPS as reported, hardware differs; "–" = not reported)', '',
      '| class | method (venue) | Mip-360 PSNR / SSIM / LPIPS | Mip-360 MB | T&T PSNR / SSIM / LPIPS | T&T MB | FPS | engine-ready mesh | ratio vs own FP32 |', '|---|---|---|---|---|---|---|---|---|',
      '| mesh | MeshSplatting, paper (2025) | 24.78 / 0.728 / 0.310 | 100 (RGB-only PLY, SH dropped) | 20.52 / 0.745 / 0.287 | – | 220 / 190 (M4 laptop) | yes | – |',
      f'| mesh | MeshSplatting FP32, our reproduction (no --indoor) | {t9[0]:.2f} / {t9[1]:.3f} / {t9[2]:.3f} | {fp9:.0f} | {t2[0]:.2f} / {t2[1]:.3f} / {t2[2]:.3f} | {fp2:.0f} | – | yes (FP32 PLY) | 1× |',
      '| mesh | MiLo (SIGGRAPH Asia 2025) | 24.09 / 0.688 / 0.323 | 253 | 21.46 / 0.706 / 0.348 | – | 170 / 160 (M4) | yes | – |',
      '| mesh | Binary Opacity Grids (SIGGRAPH 2024) | 25.62 / – / – | 913 | – | – | 927 (desktop GPU) | yes | – |',
      '| mesh | RaDe-GS mesh (2024) | 23.56 / 0.668 / 0.361 | 1100 | 20.51 / 0.659 / 0.344 | – | OOM on M4 | yes | – |',
      '| mesh | GOF mesh (2024) | 20.78 / 0.573 / 0.465 | 1500 | 21.69 / 0.690 / 0.326 | – | OOM on M4 | yes | – |']
for l, lab in ((0.00062, 'λ 6.2e-4'), (0.00185, 'λ 1.85e-3'), (0.0062, 'λ 6.2e-3')):
    a, b = m9[l], m2[l]
    L.append(f'| mesh | **ours, {lab}** | {a[2]:.2f} / {a[3]:.3f} / {a[4]:.3f}† | {a[0]:.1f} | {b[2]:.2f} / {b[3]:.3f} / {b[4]:.3f}† | {b[0]:.1f} | 350 phone / 820 laptop (Room) | yes, + CPU decode 2–6 s | {a[1]:.0f}× / {b[1]:.0f}× |')
L += ['| 3DGS | 3DGS (SIGGRAPH 2023) | 27.46 / 0.812 / 0.222 | 750.9 | 23.69 / 0.844 / 0.178 | 431.0 | 99 (desktop GPU) | no | 1× |',
      '| 3DGS | Scaffold-GS (CVPR 2024) | 27.50 / 0.806 / 0.252 | 253.9 | 23.96 / 0.853 / 0.177 | 86.5 | 135 | no | 1× (base of HAC family) |',
      '| 3DGS | HAC++ low / high (2025) | 27.60 / 0.803 / 0.253 ; 27.82 / 0.811 / 0.231 | 8.34 ; 18.48 | 24.22 / 0.849 / 0.190 ; 24.32 / 0.854 / 0.178 | 5.18 ; 8.63 | 130–151 | no | 30× / 14× vs Scaffold-GS |',
      '| 3DGS | RDO-Gaussian (ECCV 2024) | 27.05 / 0.802 / 0.239 | 23.46 | 23.34 / 0.835 / 0.195 | 12.03 | – | no | 32× vs 3DGS |',
      '| 3DGS | Compact3DGS (CVPR 2024) | 27.08 / 0.798 / 0.247 | 48.80 | 23.32 / 0.831 / 0.201 | 39.43 | – | no | 15× vs 3DGS |',
      '| 3DGS | LightGaussian (NeurIPS 2024) | 27.00 / 0.799 / 0.249 | 44.54 | 22.83 / 0.822 / 0.242 | 22.43 | – | no | 17× vs 3DGS |',
      '', '† our LPIPS on the pipeline convention ([-1,1]); ~0.03 lower on the 3DGS convention used by every other row (unified pass pending). Our indoor scenes will move after the --indoor rerun.']
L += ['', '## T3. Deployability (from rd/table2-deployability-v1.md, λ 6.2e-3 point)', '',
      '| scene | asset MB | ratio | train min, original -> ours | CPU decode s (1 core) | laptop ms (fps) | phone ms (fps) |', '|---|---|---|---|---|---|---|',
      '| room | 5.2 | 142× | 51 -> 27 (0.52×) | 2.3 | 1.2 (820) @ 3114×2075 | 2.8 (352) @ 3114×2075 |', '| garden | 10.2 | 73× | 49 -> 24 (0.48×) | 4.4 | 1.8 (566) @ 5187×3361 | 6.3 (160) @ 5187×3361 |',
      '| bicycle | 11.9 | 57× | 41 -> 25 (0.60×) | 5.4 | 1.9 (523) @ 4946×3286 | 6.9 (145) @ 4946×3286 |', '| 11-scene mean | 9.4 | 73× | 0.57× | 4.1 | – | – |',
      '', 'Laptop = MacBook Pro M2 Max (Chrome), phone = iPhone 17 Pro (Safari), WebGL2 viewer, SH degree 3, full capture resolution.']
open('tables-v1.md', 'w').write('\n'.join(L) + '\n'); print('wrote tables-v1.md'); print('\n'.join(L[6:9]))
