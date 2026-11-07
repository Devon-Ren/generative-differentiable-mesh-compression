---
name: handover-package-nov2026
description: Stage handover zips - CURRENT = /mnt/project-files/handover/meshsplatting-handover-v2.zip (11/07, 88 MB, v7b sweep + v8 code); read its 00-README then 03-HPC in any new thread; v3 must add batch11/batch12 outcomes
metadata:
  type: project
  modified: 2026-11-07T08:00:00Z
---

**v2 (current)**: /mnt/project-files/handover/meshsplatting-handover-v2.zip built 2026-11-07 ~08:00Z (88.45 MB, 901 entries, md5 3f5bda523a773612745e8bacfecee869). Docs also loose in /mnt/project-files/handover/docs-v2/ (00-04, VERSION, README-release-adapt-v7/v8.md). v1 (11/06, 88 MB, md5 bef6a56f...) kept untouched; docs-v1/ still there.

Contents: 00-README (idea, evolution, Yao feedback table, §4 state at 11/07 08:00Z incl. batch8-v2 / v7b sweep numbers + causes / v8 design / batch11 plan + expectations, §5 next steps, working rules, key numbers, tree); 01-EVOLUTION (+ stage 6 v7b sweep -> v8); 02-EXPERIMENTS-INDEX (+ §H v7b sweep rows, §I v8); 03-HPC (+ §7 pending batch11, gotchas 13 probe needs >=K kept vertices, 14 smoke harness); 04-CODE-MAP (+ v8 row); release-fixed-lambda/ (unchanged); release-adapt-v7/ (+ results/ 12 files: batch8-v2 rows+analysis, sweep paste, adapt-v7*.txt tables, adapt_analysis_v7b_sweep.py, analysis txt/csv/tex; commands/ + batch8-v2, 9-v2, 10-v1); release-adapt-v8/ (README, code/rdgrowth/train_rd_adapt_v8.py 887 l md5 a150276d..., sbatch adapt-v8*.sbatch, commands/node-batch11-v1.md, build/ generators + CPU test); analysis-snapshot (583 files), plan (36), paper, memory (49), uploads, slides, legacy-packages.

**How to apply:** new thread reads 00-README then 03-HPC, then memory/MEMORY.md. Next revision = meshsplatting-handover-v3.zip (never overwrite) and must add: batch11-v1 outcome vs pass bar (a)-(e), batch12 (33 Mip/T&T v8 reruns) tables + regenerated paper tables, any lambda-prune ablation. Build recipe: scratch generators make_docs_v2.py-style anchored edits on docs-v2/ (anchors must assert exactly one hit). See [[adaptive-lambda-nov2026]], [[user-wenhao-working-style]], [[feedback-versioned-deliverables]].
