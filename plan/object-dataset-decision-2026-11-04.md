# 单物体数据集 vs Mip-360：决策备忘（2026-11-04）

背景：Yao 已把 GPU 节点申请转发给 NYU HPC（等待回复）；general 账户 FairShare 0.24，可承受 ~10-20 GPU-h 的小批量。

## 1. 结论（先答问题）

| 问题 | 答案 |
|---|---|
| 不用 Mip-360 会被审稿人说吗？ | 会，而且是硬伤。**不要替换，要追加。** Mip-360 + T&T 继续做主表（和 MeshSplatting、MiLo、HAC/HAC++ 全部可比），单物体数据集作为新增一节。 |
| 现在先准备数据集还是先测？ | 两件事并行：CPU 下载/整理数据集（不花 FairShare），GPU 上只跑一个 lego smoke test（0.5 GPU-h）+ NeRF-Synthetic 8 场景探针（≈ 10 GPU-h）。general 0.24 够用。 |
| baseline 在单物体数据集上有没有数？ | 3DGS 类压缩方法**有**（见 §2），无需复现。mesh 类**没有**，但 mesh 类的参照就是 MeshSplatting FP32，我们自己跑；MiLo 可选。 |
| 要换 baseline 或全部复现吗？ | 不需要。NeRF-Synthetic 上 3DGS-class 全部引用论文数；mesh-class 只跑 MeshSplatting FP32（本来就要跑）。人物数据集（THuman2.0 / ZJU-MoCap）任何 baseline 都没有压缩数据，**划入 +++，不进 CVPR**。 |

## 2. 各 baseline 在哪些数据集上有已发表数字（请在 arXiv 表格里核对后再写进论文）

| 方法 | 类别 | Mip-360 | T&T | NeRF-Synthetic (Blender, 8 场景) | DTU (Chamfer) |
|---|---|---|---|---|---|
| 3DGS | gauss | ✓ | ✓ | ✓ (≈33.3 dB) | – |
| HAC / HAC++ | gauss-codec | ✓ | ✓ | ✓ (HAC 原文 Table 1 含 Synthetic-NeRF，~1-2 MB) | – |
| Compact3DGS (Lee) / LightGaussian / Compressed3DGS (Niedermayr) | gauss-codec | ✓ | ✓ | ✓ | – |
| MeshSplatting | mesh | ✓ | ✓ | **✗** | – |
| MiLo | mesh | ✓ | ✓ | ✗ | ✓ |
| 2DGS / GOF / RaDe-GS | surface-gauss | ✓ | ✓ | ✗ (GOF 无) | ✓ |
| MobileNeRF / NeRF2Mesh | mesh (2022-23) | ✓ | – | ✓ (≈30-31 dB) | – |

含义：
- NeRF-Synthetic 上 **3DGS-class 压缩 baseline 列可以完全引用**，这是选它而不是 DTU / 人物数据集的决定性原因。
- mesh-class 在 NeRF-Synthetic 的"参照"= MeshSplatting FP32（我们跑），再加 MobileNeRF / NeRF2Mesh 的论文数作为"老 mesh 方法"行。MiLo 复现可选（公开代码，8 × ~0.5 GPU-h）。
- DTU 是几何（Chamfer）基准，不是 RD 基准，CVPR 不加。

## 3. 审稿人"为什么换数据集"怎么答

不换，所以这个问题不成立。论文写法：
> Mip-360 / T&T：与 MeshSplatting 及所有 mesh/3DGS baseline 保持同一基准；
> NeRF-Synthetic：目标应用是 engine-ready 单资产（object-centric），且全部 3DGS 压缩工作都在该数据集报数，可直接对比。

## 4. 对预期要诚实

- 3DGS 在 NeRF-Synthetic ≈ 33.3 dB；不透明 mesh 方法历史上 ≈ 30-31 dB。MeshSplatting FP32 大概率落在 30-32。
- 所以差距**缩小**（3 dB → 约 1.5-2.5 dB），且视觉上没有天空/背景碎片，图会好看很多；但不会"可比"。不要在报告里承诺 comparable。
- 这一节的价值：展示 mesh 资产在它的主场（单物体、白底、可进引擎）仍然 24-70× 压缩、损失 < 0.3 dB，并配 Blender/Unity 导入截图。

## 5. 决策门（探针跑完再决定 P2 / P3）

| 探针结果 (NeRF-Synthetic 8 场景均值) | 决定 |
|---|---|
| MeshSplatting FP32 ≥ 31 dB 且 λ 6.2e-4 点 ≤ −0.3 dB | P3：加 object-centric 一节 + 引擎 demo |
| FP32 30-31 | P3 但图为主、数字为辅，正文明说 class gap |
| FP32 < 30 或 blender loader 跑不通（normal prior 依赖 COLMAP） | P2：不加；CVPR 维持 Mip-360/T&T，object 工作留给 +++ |

## 6. 现在可做（不等 HPC 回复）

CPU（不花 FairShare）：
1. 下载 NeRF-Synthetic（nerf_synthetic.zip，8 场景，transforms_{train,val,test}.json，800×800）到 /scratch。
2. 检查 upstream：`scene/dataset_readers.py` 是否保留 3DGS 的 `readNerfSyntheticInfo`；`extract_normals.py` 是否只接受 COLMAP 目录（它读 `-i IMAGES`）。这是 blender 路径最可能卡住的地方。
3. 整理 §2 的 baseline 数字到 paper/baselines-v3.md（新文件，不覆盖 v2）。

GPU（general，≈ 10 GPU-h）：
4. lego 一次 smoke test：FP32 + `--white_background`，30k iter，看 PSNR 与 mesh 是否正常（0.5 GPU-h）。
5. 通过后 8 场景 × {FP32, λ 6.2e-4, λ 6.2e-3} ≈ 8 × 3 × 0.4 h ≈ 10 GPU-h。
6. 人物数据集：先不下载；等 +++ 启动再定（无 baseline，无 loader）。

备注：rd-indoor-v1 的 sbatch 以 `-i IMAGES` + extract_normals 为前提，blender 数据需要新建 `originals-blender-v1.sbatch` / `rd-prune-blender-v1.sbatch`（不改旧文件）。待 smoke test 确认 loader 后再写。
