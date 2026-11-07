---
name: stage2-v2-bitstream-spec
description: Verified facts about the stage2-v2 delivery bitstream (10/24, from the user's stage2-v2-src-1024.zip); spec file analysis/baselines/stage2-v2-bitstream-spec-v1.md
metadata:
  type: project
  modified: 2026-10-24T00:00:00Z
---

Source: user's stage2-v2-src-1024.zip. Spec: analysis/baselines/stage2-v2-bitstream-spec-v1.md. Verified: XYZ grid per axis; predictor = smallest-index earlier neighbour (mode 2, NOT parallelogram); edge cache 64 directed edges, new vertex = varint 0; index ctx = earlier-index 1-ring distinct codes <=32; index streams near raw (11.2/12, 7.7/8 bits); q7/q15 modes come from vis_collapse extending delivery lists at runtime.

See [[meshsplatting-compression-status]].
