---
name: hpc-layout-nyu
description: Wenhao's NYU HPC paths for MeshSplatting compression: env activation, code dirs per route, Room result dirs (recorded 2026-10-12, not live-verified)
metadata:
  type: reference
  modified: 2026-10-12T01:47:37.834Z
---

User-supplied 2026-10-12 from past run records; Claude cannot reach HPC directly. CLUSTER IS NYU TORCH, NOT GREENE (2026-10-16: user corrected 'nyu hpc早就不是greene了'; login host `login.torch.hpc.nyu.edu` (ssh config also has cs601.hpc.nyu.edu and a Shanghai host); VS Code goes through torch_connect.sh -> `cpu-torch`; never say `ssh greene`). Slurm account: torch_pr_38_tandon_advanced (L40S nodes).

Env: `source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate` (Py3.11, torch 2.7.1+cu126, CUDA 12.6).

Code: original `/scratch/zr2245/meshsplatting-original-lossless-v1` (upstream/ has train.py); dense primary dict `directional-sharing-v1`; residual `directional-sharing-residual-v1`; quant `delivery-quantization-v1`; XYZ codec `mesh-local-codec-v1`; final 41 MB encoder `topology-order-v1`; collapse `shared-collapse-stage2-v1` (entry coarse_once.py; dir also holds older entries). All under /scratch/zr2245/. Consolidated packages meshsplatting-{dense,collapse}-complete-v1 may or may not be extracted there.

Room results: original ckpt `/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original/point_cloud/iteration_30000/point_cloud_state_dict.pt`; shared `.../run/shared/residual_k4096_r1024_f1/asset.mslb`; dense 41.399 MB asset `/scratch/zr2245/room-topology-xmawwL/codec/strip-fetch_edge-arithmetic_xyz-ac-yd9hmv0b/asset.mslb`; collapse run `/scratch/zr2245/room-shared-collapse-QGQgyr/run` (delivery/asset.mslb). Dataset `/scratch/zr2245/datasets/m360/room`.

See [[meshsplatting-compression-status]].

How Wenhao uses HPC (2026-10-12): `torch_connect.sh` grabs a small CPU job (cpu_short, 2 CPU, 8 GB, 6 h, account torch_pr_38_general) and VS Code connects to it as host `cpu-torch`. CPU-only work: just `cd` + `python ...` there, no sbatch. GPU: interactive (sbatch a sleep job, ssh into the node) for tuning/smoke runs, which is enough for MeshSplatting; non-interactive .sh jobs only for long batch runs (e.g. many scenes). Give instructions in that form. Anything needing >8 GB RAM won't fit the cpu-torch node — but when he already holds a GPU node he runs CPU-heavy steps there too; do not send him to a separate CPU node (see [[user-wenhao-working-style]]).
