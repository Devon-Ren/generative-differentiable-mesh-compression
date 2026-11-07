---
name: rd-complete-package-v1
description: Stage-summary package meshsplatting-rd-complete-v1.zip (2026-10-26) - layout, one-shot chain, isolation proof, what is untested on HPC
metadata:
  type: project
  modified: 2026-10-26T00:00:00Z
---

/mnt/project-files/meshsplatting-rd-complete-v1.zip (18.6 MB, 2507 files, VERSION "meshsplatting-rd-complete-v1 2026-10-26"); source tree in scratchpad pkg/meshsplatting-rd-complete-v1 (session-only). Built from meshsplatting-collapse-complete-v1 + stage2-v2 patch + rdgrowth (train_rd_v3) + vis_collapse v12 + coarse_finetune + viewer + diag + results + docs.

Chain (rd_pipeline.py, `bash run_rd.sh --dataset S --images images_2 --lam 0.0062 --output OUT`, or `sbatch submit_rd.sh ...` on torch_pr_38_general, 48G, 4h): reconstruction_rd -> precision (xyz_rule_v1) -> footprint -> level (coarse_vis --target-vertices V0) -> finetune (3000) -> share_fit (deliver_coarse K4096 R256, 400 steps) -> encode (encode_v2.py: ONLY bitstream/asset.mslb; diagnostics kept encoder-side) -> standalone_decode (verify_isolation.py: child decode.py in scrubbed env + work dir holding only the file, sys.addaudithook classifies every open/Popen; VIOLATION => exit 1; exactness vs restore(prepare(shared-final.pt)) up to vertex order) -> decoded_render (render_asset.py from the file) -> viewer export; result.json has isolation_pass, encoder_deterministic, file_MB, ratios, dB. Stages restartable (completed/<stage>.json, --resume).

Results in package: results/all-results-v1.csv (287 points; series original / rd-v3 / rd-v2 / train-at-budget / collapse / codebook / precision / index-rd / format-ablation / control), summary-by-scene-v1.md, RESULTS.md index, build_master_table_v1.py. Headline (11 scenes, originals' eval PSNR): lam 1.85e-3 20.7 MB 32x -0.22 dB train 0.85x; lam 6.2e-3 9.4 MB 73x -0.59 dB train 0.57x.

**Why:** user asked 10/26 for a stage summary zip with the complete rd-controlled architecture, encoder, decoder, all data, and a guarantee that encoder emits one file / decoder reads only that file (no leakage).
**How to apply:** verified locally only (verify.py, tests/test_isolation.py, test_rd_pipeline.py, dry-run from a fresh extract); setup.sh + a real run_rd.sh on HPC still pending. Next revision must be -v2 (new zip name) per [[feedback-versioned-deliverables]]. See [[meshsplatting-compression-status]].
