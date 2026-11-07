---
name: meshsplatting-compression-status
description: State of Wenhao's MeshSplatting compression project (CVPR 2027 target): key measured findings, chosen paper story, files in project folder
metadata:
  type: project
  modified: 2026-11-01T08:00:00Z
---

Goal: CVPR 2027 paper compressing MeshSplatting (arXiv 2512.06818); deadline ~mid-Nov 2026. Reply Chinese + English terms.

Earlier measured findings (Room byte breakdown, direction reset 10/13, batches 1-4, G0 result): [[meshsplatting-status-history-oct2026]].
Plan: /mnt/project-files/plan/cvpr2027-research-plan.md. C2 collapse done; C4 Delaunay connectivity DEAD.

Paths: [[hpc-layout-nyu]]. C2: [[c2-collapse-experiments]].

STORY v2 10/18: rate-controlled growth, vertex count is an OUTPUT -> [[rdgrowth-v1-design]].
FRAMING 10/19 (user): engineering need -> full codec + compressibility analysis + rate-in-the-loop at 50-100x; lossy by design. Story v3: plan/story-v3-engineering-need-2026-10-19.md.

QUALITY GAP CLOSED 10/20 ([[quality-gap-diagnostics-oct2026]]). BATCH A + B DONE 10/24 -> [[rdgrowth-v3-scenes7-results]]. XYZ=auto in the sbatch files is BROKEN (.numpy() on requires_grad params): use xyz_rule_v1.py + explicit XYZ=qN (commands/batchA-resume-v1.md, batchB §2-v2). READINESS 10/25: plan/cvpr-readiness-2026-10-25.md + v2 (after deployability: weak-accept material once qualitative fig + related-work para exist; R9 = operating-point observation, not a discovery), speed-direction-2026-10-25.md (measure deployability, don't re-engineer training).

ADVISOR CONCERNS 10/23 + GENERIC CODECS 10/24: Table 1 = generic codecs on the same asset; ours 5x (quant+zstd) / 15-17x (Draco/texture) smaller at equal quality.

BITSTREAM SPEC 10/24: [[stage2-v2-bitstream-spec]].
VIEWER+DEPLOYABILITY 10/25: analysis/viewer-v1/, rd/table2-deployability-v1.md: train 0.57x/0.87x of original; decode 2.3-6.4 s one core; iPhone 17 Pro 145-350 fps; R9: small-lambda bytes generalise on 7 Mip-360 scenes only. Attach cap hit: send links.
PACKAGE 10/26: see [[rd-complete-package-v1]] (not yet run on HPC).
10/30: +++ probes closed ([[meshsplat-plus-vision]]); user chose paper first; paper/related-work-v1.md, technical-report-v1.md, baselines-v2.md (§G RD policy, §H 4+4 shortlist) drafted. Open: qualitative figure, Table 1 other 9 scenes, 3rd lambda point, unified LPIPS pass, indoor flag decision.
11/01 FIGURE REVIEW (plan/figure-review-2026-11-01.md): INDOOR FLAG NEVER USED in any run (originals + all lambda runs) -> indoor PSNR -0.6..-1.0 / LPIPS +0.07..+0.14 vs the paper; LPIPS evaluator verified identical to the paper (VGG, outdoor per-scene match). Proposed R1 = rerun 4 indoor scenes with --indoor (4 originals + 16 lambda runs, ~22 GPU-h, new rd-prune-v3c.sbatch) -> user decision pending; then R3 unified eval_all_v1 (~3 GPU-h), Table 1 to 9 scenes (CPU), qualitative. Main paper = 6 figs + 3 tables, baselines 2024+ only, evolution story = Fig 4 ablation not a history figure. Provisional 3-metric figure rd/rd-metrics-v1.png.
11/01 evening: user submitted the 12 indoor jobs (full rerun); FairShare will be exhausted -> advisor/HPC request. Report package analysis/report/ (report-v1.md, tables-v1.md, 4 figures on figstyle_v1; see [[feedback-figure-style]]). Pipeline LPIPS = [-1,1] convention, ~0.03 above the 3DGS/train.log convention -> unified pass must use [0,1] (3DGS) convention.
11/01 MPEG baseline (Open3DGC SC3DMC/TFAN): Room 108 MB, strongest generic row, ours 3.5x smaller -> [[o3dgc-mpeg-baseline]]; v3 package (geo-only + MPEG-geometry texture rows) pending the user's run.
11/01 INDOOR RERUN: 4 --indoor originals match the paper (+-0.09 dB, LPIPS exact) -> reproduction closed; lambda hull at equal bytes unchanged (room) / +0.7 dB (kitchen); but codec loss vs own FP32 doubled (level 1.0 -0.5..-0.76 dB) -> rd/indoor-first-look-2026-11-01.md; proposed 0.5 GPU-h re-export diagnostic (DC q8 / K16384 R1024 / true weight), user decision pending; counter/bonsai lambda jobs moved to general.
