# f3v1：从照片构造"自由空间边界"（F3）

设计：plan/f3-design-2026-11-22.md。所有脚本用 `bash 脚本名` 运行（不要 source），失败时打印 `FAILED: ...` 和日志末尾，不会关掉终端。

## 目录（HPC 上）
/scratch/zr2245/memory-medium/
  f3v1/            代码（本 zip）
  data/sponza-synth-v1/   合成数据（另一个 zip）
  out/             所有输出和日志（自动创建）
  Depth-Anything-3/       install_login.sh 克隆
/scratch/zr2245/conda-envs/da3   DA3 专用环境（install_login.sh 创建，不碰 MeshSplatting 环境）
/scratch/zr2245/hf-cache         DA3 权重

## 步骤
1. login 节点（有网）：`bash run/install_login.sh`（一次性，约 10–20 分钟）
2. 任意节点（CPU 即可）：`python scripts/selftest_cpu.py ../data/sponza-synth-v1` → 末行 `SELFTEST (quick) OK`
3. GPU 节点：`bash run/f3_1.sh`（先 2 分钟 smoke，过了才跑完整版）
4. GPU 节点：`bash run/f3_2.sh`
5. GPU 节点：`bash run/f3_3.sh`（可选 `MS_CKPT=<room 的 point_cloud_state_dict.pt> bash run/f3_3.sh`）

## 每个运行的检查点
[0] 环境（CUDA、nvdiffrast）→ [1] 数据与深度文件 → [2] 闭合曲面（boundary_edges 必须为 0）→
[3] GPU 坐标约定检查（渲染深度与深度图一致、且明显好于上下翻转版本，否则训练前退出）→ [4] 优化 → [5] 评估
输出：out/<run>/run.log、result.json、surface_final.npz
