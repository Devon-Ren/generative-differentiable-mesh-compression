---
name: stage2-v2-codec-patch
description: stage2-v2 delivery-format patch (constant vertex_weight + neighbour-context SH index coder) built 2026-10-15, where it lives, how it is installed, what was verified
metadata:
  type: project
---

stage2-v2 (2026-10-15, user-approved): /mnt/project-files/analysis/stage2-v2-patch-v1/ (patch + apply_stage2_v2.sh + roundtrip_check.py) creates /scratch/zr2245/shared-collapse-stage2-v2 as a patched COPY of stage2-v1 (v1 untouched). Changes: delivery.prepare(weight='const') replaces vertex_weight by one logit scalar (vertex_weight_const/_cols); topology.encode/decode call codec/index_context.py -> codec/index_ctx.cpp (CLI, includes native.cpp via the edge.cpp main-rename trick) coding app_mapping/app_residual_mapping as ctx_mapping/ctx_residual; runtime.export_asset(wire, folder, xyz, dc, weight='const', index_codec='ctx'). Verified locally on CPU: keep/zstd output byte-identical to v1; v2 decodes v1 assets; pytest encoder/delivery/seed/fit pass. HPC numbers pending (run_codec_variants.sh / run_default_room.sh).

**Why:** advisor framing (per-vertex SH redundant) + Room measurements: ctx coder saves ~0.55 MB lossless at 1.98M vertices, vertex_weight ~1.3-2 MB is all-equal logits. User approved writing it as a patch on a COPY (v1 stays untouched).
**How to apply:** vis_collapse v12+ defaults to STAGE2_PKG=/scratch/zr2245/shared-collapse-stage2-v2; deliver_coarse --weight/--index-codec fall back to keep/zstd on v1 and record the effective format in summary.json. See [[c2-collapse-experiments]], [[rate-aware-results]].
