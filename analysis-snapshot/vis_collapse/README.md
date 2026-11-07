# vis_collapse — C2 第一个实验：pixel-space（visibility-aware）collapse criterion

历史 collapse（26.9 MB 结果）的 score 是 `max(位置半径/中位边长, appearance 半径)`，和相机无关。
这里只改 score，其余（QEM 端点合并、合法性规则、appearance 特征、80° 法线限制）与
`stage2/legacy_hierarchy/native_qem.cpp` 完全相同（`native/native_qem_vis.cpp` 是其派生副本，
uniform 模式下与原版逐 bit 一致，已在合成网格上验证 merge 序列相同）：

    new score = max( 位置半径 × pixel_scale ,  appearance 半径 × appearance_unit )   [单位：像素]

- `pixel_scale[v]` = 该顶点在所有训练视角（去掉 validation 子集）里被渲染到时的 max(focal_px / depth)，
  即"顶点动 1 个世界单位在屏幕上动多少像素"；cluster 取成员最大值。没被任何视角渲染到的顶点取 seen 顶点的 5% 分位数。
- `appearance_unit` 默认 = 中位边长的像素长度，所以 1 个 appearance 单位的代价和以前一样等于 1 条中位边。
- 默认按 cost 顺序 collapse 到 `--target-faces 4041039`（和历史 error=2 层面数一致），公平比较。

| 文件 | 作用 | 节点 |
|---|---|---|
| `build_vis.py` | 编译 `native/_native_qem_vis` | 任意 |
| `vertex_footprint.py` | 原版 mesh 每顶点 pixel_scale / views_seen / 残差（用 upstream 渲染器） | GPU |
| `coarse_vis.py` | 建 coarse 层，输出 `coarse-initial.pt`（和 Stage-2 同格式）+ 报告 | CPU，需 ~48 GB RAM |
| `../coarse_finetune.py` | 评估（`--steps 0`）和 appearance-only 重训（`--steps 3000`） | GPU |
| `deliver_coarse.sh` → `deliver_coarse.py` | 重训后的 best.pt → stage2 sharing（seed 字典）→ 400 步 shared fit → q18/q8 + codec 导出 → 真实 MB 和 test 指标；只 import stage2 包，默认读历史 run 的 config.json | GPU |

比较基准（历史 coarse-initial.pt，同为 4,041,039 faces）：val 28.972 → appearance-only 29.319 → +geo 29.371；test 27.479 / 27.510。

## 10/12 实验 1 结果 → 实验 2：render sensitivity 权重

实验 1（纯 pixel_scale）**明显更差**：同 4,041,039 faces 下 val initial 27.66（历史 28.97），最差 val 视角 21.9（历史 25.7），
appearance-only 3000 步后 28.02（历史 29.32），test 26.50（历史 27.48）。虽然删除集重合 93%，但那 7% 的差别是：
pixel_scale 低（离相机远）的顶点被允许合并到 28/162 ≈ 0.17 世界单位 ≈ 4.9 条中位边，而近处只有 1.4 条。
MeshSplatting 的颜色在顶点上，顶点密度就是纹理分辨率，远处的顶点每像素承担的纹理反而更多；
"屏幕空间位移 ≤ N 像素"是经典 LOD 的指标，对 vertex-coloured mesh 是错的单位。→ 论文里一个有用的负结果。
因此几何项的单位回到"中位边长"（`--base uniform`，默认），权重只在此基础上逐顶点放缩。

实验 2 用渲染敏感度做每顶点权重（`vertex_footprint.py` 默认多算三组数组，每视角反传一次）：
`sens_pos[v] = Σ_views ‖∂(ΣI)/∂v‖`，`sens_dc[v] = Σ_views ‖∂(ΣI)/∂DC_v‖`，`l1_pos[v] = Σ_views ‖∂L1/∂v‖`。
`coarse_vis.py --base uniform --geo-weight sens_pos --app-weight sens_dc --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale`
把几何/appearance 项按 clip((s/median)^power) 逐顶点放缩（C++ 支持每顶点 appearance_scale，uniform 模式仍逐 bit 等价）。
`--weight-divide-pixel-scale` 先把敏感度除以 footprint，去掉"离相机近就大"的因子，只留对比度/结构信息。
`--base uniform` 无权重时与历史选择完全一致（Jaccard 1.0，已验证）；`--base pixel` 无权重时 = 实验 1。

## v4：交付
`bash deliver_coarse.sh --coarse <finetune>/best.pt --output <dir> --png`，对应 coarse_once.py 的后半段，
历史基准 26.884 MB / test 27.3467。

## v5 (2026-10-13): `--target-vertices`

Equal face count is not equal rate: at 4,041,038 faces the geo-only mesh kept 1,984,634 vertices and the geo+app
mesh 1,863,742, and every payload item (XYZ residual, DC, mappings, vertex_weight) scaled with the vertex count
(26.70 vs 25.47 MB). Each accepted merge removes exactly one vertex but a variable number of faces (boundary /
soup edges remove fewer), so criteria that prefer different regions reach the same face target with different
vertex counts. `--target-vertices N` collapses in cost order until N vertices remain (overrides --target-faces);
use it to compare criteria at equal vertex count, i.e. roughly equal bytes, or sweep it for an RD curve.
```
python coarse_vis.py --source $C --footprint $W/footprint-sens.npz --base uniform --target-vertices 1863742 \
    --geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale \
    --old-map $OLD --output $W/geo-only-v1864k
```

## v6 (2026-10-13): RD axes and rate-aware fit

* `deliver_coarse.py --k K --r R` builds the SH dictionaries by k-means on the delivered mesh (axis A: K x R sweep,
  `scripts/run_axisA.sh`), `--xyz/--dc` pick the delivery precision, `--qat` quantises the geometry before the shared
  fit and quantises DC with a straight-through estimator inside the fit (rate-aware step 2).
* `precision_sweep.py` re-exports an existing `shared-final.pt` at several precisions without training
  (rate-aware step 1, `scripts/run_rate_aware.sh`).
* `scripts/run_axisB.sh` = vertex-count sweep (geo-only criterion, 100..50 %), `scripts/run_draco.sh` = Draco
  connectivity reference, `stage2_run.sh <script.py>` runs any script in the stage2 env.
* Modes outside the stage2 lists (q12 XYZ; q7..q4 DC) are enabled by extending `delivery.XYZ_MODES/DC_MODES` at
  runtime in this package only; the stage2 decoder accepts any 1..24 bits, nothing in stage2 is edited.

## v7 (2026-10-13): index_rd.py (rate-aware step 2)

Neighbour-context RD reassignment of `app_mapping` / `app_residual_mapping` on an existing delivery: each vertex may take a
neighbour's codeword when `D + lambda * unit * bits` is lower (Jacobi sweeps on the mesh graph), with an estimated
context-coder rate, the REAL zstd bytes after re-export, validation PSNR before/after a short appearance re-fit, and
decoded test metrics.  `scripts/run_index_rd.sh`.  All sweep scripts in v7 point at `/scratch/zr2245/analysis/vis_collapse-v7`.

## v9

Bug fix only: `index_rd.py` crashed at import on Python 3.11 (`f-string: unmatched '['`, nested same-quote f-string that only Python >= 3.12 accepts). The sweep scripts in `scripts/` point at `/scratch/zr2245/analysis/vis_collapse-v9`. Install: `cd /scratch/zr2245/analysis && unzip -q vis_collapse-v9.zip` (new directory `vis_collapse-v9`), then rebuild the native module there as for v8.

## v10

Real bytes for a neighbour-context index coder.  index_rd (v7-v9) estimated the context-coder rate; the delivered
asset still codes the two index arrays with plain zstd, which cannot see mesh neighbours (measured gap on Room:
3.4 bits/vertex = 0.85 MB at lambda 3).  `native/index_ctx.cpp` is a decodable coder (hit bit + neighbour slot,
else raw code through an adaptive prefix tree) built on the arithmetic coder of `stage2/codec/native.cpp`
(`#include` with the `edge.cpp` trick, so the build needs `-I <stage2>/codec`).  `index_context_codec.py` re-runs
the lambda reassignment, codes z and u, verifies the round trip and tabulates zstd vs ctx MB next to the index_rd
PSNR columns.  Install: `cd /scratch/zr2245/analysis && unzip -q vis_collapse-v10.zip`, copy the
`_native_qem_vis*.so` from v8/v9, then `bash scripts/run_index_ctx.sh` (builds `_index_ctx` on first use).

## v11

`scripts/run_axisA.sh` now defaults to the axis-B 80%-vertex mesh (`$W/axisB/v0.8/finetune/best.pt`, the candidate default operating point) and writes to `$W/axisA-<level>/` + `$W/axisA-<level>.txt`, where `<level>` is derived from the mesh path (`TAG=`/`MESH=` override), so sweeps on different meshes never collide.

## v12 (2026-10-15): stage2-v2 format, default point, garden

* All stage2 entry points (`deliver_coarse.sh`, `stage2_run.sh`, `*.py --stage2`) now default to
  `/scratch/zr2245/shared-collapse-stage2-v2` (= a patched COPY of stage2-v1, see `analysis/stage2-v2-patch-v1/`;
  `STAGE2_PKG=...` overrides).  The two format changes are in the stage2 package, not here: `vertex_weight` is replaced by
  one constant opacity logit (`weight='const'`) and `app_mapping` / `app_residual_mapping` are coded by the
  neighbour-context arithmetic coder (`index_codec='ctx'`, the decodable version of v10's `native/index_ctx.cpp`).
* `deliver_coarse.py --weight {const,keep} --index-codec {ctx,zstd}` (defaults const / ctx) are forwarded to
  `export_asset` when the package supports them (inspect.signature); with stage2-v1 the defaults silently become
  keep / zstd and are recorded as `weight` / `index_codec` / `stage2_version` in config.json, summary.json and the
  DELIVERY log line (plus `mapping_MB`, `banks_MB`, `vertex_weight_MB`).  `--seed-asset` is only required with `--k 0`;
  `--historical-config none` turns the Room defaults off (needed for other scenes).
* `index_rd.py` / `run_axisA.sh` sum whichever of app_mapping / app_residual_mapping / ctx_mapping / ctx_residual exist.
  `precision_sweep.py` and `index_rd.py` call `export_asset` with the package defaults, i.e. const / ctx under stage2-v2.
* `codec_variants.py` + `scripts/run_codec_variants.sh`: 2 x 2 (weight x index_codec) re-export of an existing
  shared-final.pt at q14/q7, no training; proves ctx is lossless (identical PSNR) and shows the bytes of each change.
* `scripts/run_default_room.sh`: the candidate default (v0.8 mesh, K4096 R256, q14/q7, stage2-v2) end to end.
* `scripts/run_garden.sh`: second scene (MipNeRF360 garden, `images_4`) in one GPU session: train original ->
  footprint -> levels 1.0 / 0.8 / 0.7 (collapse, 3000-step appearance finetune, K4096 R256 q14/q7 delivery);
  `$G/garden.txt`.  Restartable; `FRACS=`, `K=`, `R=`, `XYZ=`, `DC=` override.
* Install: `cd /scratch/zr2245/analysis && unzip -q vis_collapse-v12.zip`, copy `native/_native_qem_vis*.so` (and
  `native/_index_ctx*.so`) from v11; stage2-v2 must exist (`bash stage2-v2-patch-v1/apply_stage2_v2.sh`).
