---
name: sweep-scene-settings-nov2026
description: per-scene sbatch settings used by every Mip-360 / T&T / NeRF-Synthetic batch (XYZ precision, IMAGES, INDOOR, DATA path, run-dir naming of the fixed-lambda references and originals), as of 11/06
metadata:
  type: reference
  modified: 2026-11-06T15:01:10.259Z
---

Mip-360 (DATA default /scratch/zr2245/datasets/m360/$SCENE): room q14, kitchen q16, counter q15, bonsai q15 -> INDOOR=1, IMAGES=images_2; garden q15, bicycle q16, stump q16, flowers q15, treehill q16 -> INDOOR=0, IMAGES=images_4. T&T: truck q18, train q18 -> INDOOR=0, IMAGES=images, DATA=/scratch/zr2245/datasets/tandt/$SCENE. NeRF-Synthetic (datasets/nerf_synthetic/$SCENE, transforms_{train,val,test}.json): all 8 scenes (chair drums ficus hotdog lego materials mic ship) XYZ q14, --indoor, black bg, IMAGES=train.

Reference run dirs under /scratch/zr2245: Blender fixed-lambda hull $S-rd3b4-lam{0.00062,0.0062}-mu1.0 (v2 delivery in $RUN-best, val_select_v3 re-delivery in $RUN-best3); Blender FP32 originals $S-blender2; adapt runs $S-adapt{N}-t${T}mb (Mip-360) / $S-adapt{N}b-t${T}mb (+ -best for the val-selected copy, Blender); probe self-check meshes $S-adapt1-t10mb (room/kitchen/garden/bicycle only) and $S-rd3b4-lam0.0062-mu1.0-best (Blender).

Slurm on the reserved node (11/05-11/12): --reservation=yw523-l40s --account=torch_pr_38_general --cpus-per-task=6 for parallel GPU jobs (the node's cores are shared; see [[compute-status-nov2026]]).
See [[hpc-layout-nyu]], [[adaptive-lambda-nov2026]].
