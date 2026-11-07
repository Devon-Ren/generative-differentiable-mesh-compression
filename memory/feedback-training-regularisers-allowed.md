---
name: feedback-training-regularisers-allowed
description: 11/05 Wenhao reversed the "no training changes" stance after seeing close-up soup renders: he allows ANY training-process change (loss penalties) to get smooth surfaces and no jump connections; mesh-quality regularisers move from +++ into the CVPR line (confirmed 11/05 with a one-shot rule; the one shot failed -> back to +++, see mesh-reg-v4-line)
metadata:
  type: feedback
---

On 2026-11-05, after viewing the lego/ship high-rate .msv close-ups in viewer-v3 (spikes, faceted surfaces, long sliver triangles bridging gaps), Wenhao said: "我允许改任何训练过程 ... 表面smooth总得做到吧，然后不要有跳跃连接，这种东西别告诉我不能避免，loss function里加入惩罚".

**Why:** the compression story is fine (48x smaller than MobileNeRF at equal PSNR) but the delivered mesh is not usable up close; he wants a mesh that looks like a mesh, and accepts PSNR cost (precedent NeRF2Mesh lego: smooth 32.07 vs unsmoothed 34.90).

**How to apply:** regularisers (jump-edge penalty + post-Delaunay long-edge filter, Laplacian on shared vertices after 11k, adjacent-face normal consistency) are now in scope as analysis/rdgrowth/train_rd_v4.py (new file, no upstream edit); weights normalised by scene scale, set on lego, verified on ship, never per-scene. Judge by val PSNR + mesh_stats_v2 + viewer. Paper metric becomes quality-per-byte of a usable mesh. Supersedes the 11/04 geometry-control conclusion "constraints = +++"; see [[nerf-synthetic-probe-nov2026]], [[meshsplat-plus-vision]].
