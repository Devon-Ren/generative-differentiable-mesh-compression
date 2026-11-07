---
name: o3dgc-mpeg-baseline
description: MPEG-4 Part 16 SC3DMC/TFAN (Open3DGC) baseline for Table 1 (11/01): package versions, Room 108 / garden 126 MB, geometry-only 20 MB, texture route 146 MB, geometry-vs-SH ablation, pending GPU eval
metadata:
  type: project
---

Package: analysis/baselines/o3dgc-v1/ (o3dgc_driver.cpp, SH as 6x8-dim attributes because Open3DGC caps attribute dim at 8) + o3dgc-v2/ (run_o3dgc_v2.sh mirrors, o3dgc_codec_v2.py + run_o3dgc_v3.sh); paste blocks commands/o3dgc-v1/v2/v3.md. amd/rest3d is deleted from GitHub; mirror rbsheth/Open3DGC works and the driver compiled first try on cpu-torch (11/01). GitHub is unreachable from the Claude container, so the API was never verified here.

Room result (11/01): int-SH row 107.98 MB, 3.19 M vertices kept (no split), PSNR ~28.09 by construction (= quant-PLY row); float-SH row 102.52 MB, PSNR pending (EVAL=1 on a GPU node). MPEG = strongest generic codec (1.5x below zstd, 4.2x below Draco); ours 3.5x smaller at equal quality, 7.9x at -0.28 dB. Files: rd/generic-codecs-2026-11-01-o3dgc.txt, baselines/generic-codecs-2026-11-01-v2.md, report/fig-codecs-room-v2, report-v2.md.

Paper usage (decided 11/01): all-in MPEG row = Table 1 main row; MPEG geometry + baked JPEG = texture-tier row; MPEG geometry + our SH = ablation. v3 RESULTS (11/01): MPEG geometry only 19.9 (Room) / 20.7 (garden) MB -> MPEG SH attributes 88.1 MB on Room (4.6 bit/channel); garden int-SH 126.0 MB (ours 32.5, 3.9x). Texture route with MPEG geometry 145.7 / 148.6 MB (atlas 5.4x duplicated vertices, larger than Draco's 136.8) -> texture tier 90-150 MB / -3 dB regardless of codec. Ablation: geometry coder swap -> ~35 MB (+13 %), SH coder swap -> ~104 MB: SH codebook = 73 MB of the 108->31 gain, geometry coder = 4 MB (our ~16/15 MB stream split is approximate until the asset's stream table is dumped). Files: baselines/generic-codecs-2026-11-01-v3.md, report/report-v3.md.

See [[meshsplatting-compression-status]], [[feedback-figure-style]].
