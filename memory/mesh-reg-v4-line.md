---
name: mesh-reg-v4-line
description: 11/05 train_rd_v4 one-shot FAILED on lego lam 6.2e-4 (a sliver-only -0.9 dB dominated; b/c lap+normal collapsed to 19 dB; grad ratio reg/main ~1000 = weights 3 orders too strong) -> filed as +++ exploration, NOT CVPR; results analysis/blender/results-reg-v1.md
metadata:
  type: project
---

Files (all new, no upstream/v3 edit): analysis/rdgrowth/mesh_reg_v1.py (VERSION 2026-11-05b), test_mesh_reg_v1.py (CPU, prints MESH_REG_TEST OK), train_rd_v4.py (patched copy of v3; identical behaviour with all --w_* 0 and --filter_aspect 0), analysis/blender/rd-reg-blender-v1.sbatch (one scene x one CFG: v4 train -> val_select_v2 -> mesh_stats_v2 incl. v3-ref row -> level-1.0 -> .msv), paste block §15 in analysis/commands/nerf-synthetic-v1.md (check-reg-v1.sh = line counts / py_compile / bash -n / CPU self-test, then 3 lego jobs).

Design: aspect = longest^2/(2 area) ~ length/base; sliver loss mean relu(log(aspect/20)) from iter 500 (soup phase too), capped at 8; Laplacian + adjacent-face normal only after the Delaunay rebuild (11000, vertices shared before that are per-face), open-boundary vertices excluded from the Laplacian; hard jump filter (aspect>20 AND longest>6x median edge) at delaunay / each prune round / final, prune_triangles keep-mask asserted. Logs: [REG it] with |g_reg|/|g_main| vertex-gradient ratio every 500, [REG jumpfilter], [REG mesh] (sliver %, lap, normal, dihedral>60 %). Configs: a = sliver 0.05 only; b = 0.05/0.05/0.05; c = 0.2 x3. Reference: lego lam 6.2e-4 v3 = 9.26 MB / 33.50 test.

**Why:** user (11/05) demands smooth surfaces and no jump connections, accepts PSNR cost; weights are dimensionless so one set should transfer (set on lego, verify on ship, never per-scene).
RESULT 11/05: a = 6.35 MB / 32.60 (v3 9.26 / 33.50; lam 6.2e-3 3.10 / 32.58 dominates it), slivers <10deg 28 -> 9 % but dihedral/non-manifold unchanged (sliver term active only in iters 500-1000, ratio 1083 there); b = 3.60 / 18.89, c = 3.11 / 19.04 (ratio 260-3100 at 30k, mesh flattened to 0.4 M faces). Jump filter by aspect found ~10 faces: the visual jumps are moderate-aspect bridging faces, need a normal/depth criterion. Verdict per the user's one-shot rule: +++ exploration, no CVPR use, no follow-up runs unless the user asks. +++ starting point recorded: w_lap = w_normal = 1e-4, w_sliver 1e-3, reg_from 2000, a0 8, normal-angle jump criterion.
**How to apply:** do not propose regulariser runs for the CVPR paper; keep train_rd_v4/mesh_reg_v1 untouched as the +++ baseline. See [[feedback-training-regularisers-allowed]], [[nerf-synthetic-probe-nov2026]].
