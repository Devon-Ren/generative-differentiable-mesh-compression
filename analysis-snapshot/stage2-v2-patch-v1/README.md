# stage2-v2 patch (v1 of the patch, 2026-10-15)

Two delivery-format changes to `shared-collapse-stage2-v1`, applied to a **copy** (`shared-collapse-stage2-v2`):

1. **Constant vertex weight** (`delivery.prepare(..., weight='const')`): the per-vertex opacity logits (`vertex_weight`,
   1.25–2.1 MB on Room) are replaced by one scalar (`vertex_weight_const`, the mean logit); `restore()` rebuilds the
   per-vertex tensor so `SharedMesh` and the renderer are unchanged.  `export_asset` reports
   `vertex_weight_const.max_abs_opacity_error` (opacity = 0.999 + 0.001·sigmoid, so the error is ≤ 1e-3 by construction).
2. **Neighbour-context index coding** (`codec/index_ctx.cpp` + `codec/index_context.py`, hooked into
   `topology.encode/decode` after the topology stream): `app_mapping` and `app_residual_mapping` are arithmetic-coded
   with the codes of already-decoded mesh neighbours as context (hit bit + slot, else adaptive order-0 raw code).  The
   decoder rebuilds the context from the decoded faces, nothing extra is transmitted.  Payload fields become
   `ctx_mapping` / `ctx_residual` (+ `_k`, `_u32` scalars) instead of `app_mapping` / `app_residual_mapping`.

`export_asset(wire, folder, xyz, dc, weight='const', index_codec='ctx')` — pass `weight='keep'` / `index_codec='zstd'`
for the v1 behaviour (byte-identical to v1, verified).  v2 decodes v1 assets unchanged (`read_asset`).  Everything
else (XYZ/topology codecs, sharing, fit, renderer) is untouched; `test_cpu.py` encoder/delivery/fit tests pass.

Local verification (synthetic 200×200 grid, 40k vertices, 50 % neighbour-correlated codes, CPU): v1 536,977 B →
v2 395,820 B; vertex_weight 125 KB → 0; ctx_mapping 28,975 B vs zstd 39,321; ctx_residual 23,242 vs 29,216;
semantic audit and SharedMesh render both pass; max opacity error 2.7e-6.

## Install (any node, no GPU needed)

```bash
cd /scratch/zr2245/analysis && unzip -q stage2-v2-patch-v1.zip
bash /scratch/zr2245/analysis/stage2-v2-patch-v1/apply_stage2_v2.sh
```

## Check on HPC (CPU, ~1 min)

```bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
cd /scratch/zr2245/shared-collapse-stage2-v2 && python /scratch/zr2245/analysis/stage2-v2-patch-v1/roundtrip_check.py . /scratch/zr2245/stage2-v2-check
```

vis_collapse ≥ v12 points `stage2_run.sh` / `deliver_coarse.sh` at stage2-v2 by default (`STAGE2_PKG=` overrides).
