---
name: hpc-request-nov2026
description: 11/02 HPC allocation request plan (plan/hpc-request-2026-11-02.md): CVPR rating assessment 3.3-3.6 as-is, CVPR line needs 60-100 GPU-h max, MeshSplat+++ ~200 GPU-h; node-reservation vs FairShare-reset advice
metadata:
  type: project
  modified: 2026-11-02T15:00:34.164Z
---

User (11/02) plans to ask NYU HPC for a whole node (~4 GPUs) for ~5 days before the CVPR deadline; wanted a judgment on whether the CVPR paper can reach a mean 4/5 and how to split GPU hours between the CVPR line and MeshSplat+++ ([[feedback-keep-plusplus-separate]]).

Assessment given: as-is 3.3-3.6; GPU-fixable objections = incomplete ablation (11 scenes x 2 lambda x {fixed cap, split-only} = 44 GPU-h, 6-scene version 24) and "MeshSplatting-only" (same codec on MiLo meshes, 9 scenes, ~12 GPU-h), plus a near-lossless top point (lambda 2e-4 + finer codec, 11 GPU-h) -> 3.6-3.9; the 3 dB class gap and "known parts" novelty critique are the ceiling. CVPR line total 50 (min) to 100 (full) GPU-h; +++ line ~200 (C2 manifold-growth prototype 60, baselines 25, debug slack 50).

Advice: a 5-day reservation is 480 nominal GPU-h but idles count; realistic 250-300. Fill days 1-2 with scripted CVPR sweeps on all GPUs, develop +++ on one GPU, days 3-5 +++ sweeps. Ask HPC for either the node reservation or a FairShare reset / temporary high-priority QOS (~350 GPU-h); the latter is usually easier to grant. File: plan/hpc-request-2026-11-02.md. No email drafted (user asked for discussion only).
