#!/usr/bin/env bash
# verify_hpc_md5.sh: is the code on HPC the same as the code in this backup (hold-pipeline/)?  Read-only.  Run on the login node.
LAB=${LAB:-/scratch/zr2245/meshsplatting-original-lossless-v1}; AN=${AN:-/scratch/zr2245/analysis}; PKG=${PKG:-$AN/vis_collapse-v12}; STAGE2=${STAGE2:-/scratch/zr2245/shared-collapse-stage2-v2}
same=0; diff=0; miss=0
chk () { if [ ! -f "$1" ]; then echo "MISSING  $1"; miss=$((miss+1)); return; fi; m=$(md5sum < "$1" | cut -c1-32); if [ "$m" = "$2" ]; then same=$((same+1)); else echo "DIFFERENT  $1  (HPC $m, backup $2, backup file hold-pipeline/$3)"; diff=$((diff+1)); fi; }
chk "$AN/rdgrowth/train_rd_adapt_v9b_abl7.py" 1eeb64ab5032e903f69ad1a21466d99f rdgrowth/train_rd_adapt_v9b_abl7.py
chk "$AN/rdgrowth/probe_bytes_v1.py" b03b945dce50345ea404c4f6862b928f rdgrowth/probe_bytes_v1.py
chk "$AN/rdgrowth/train_rd_v3.py" 8975fc7c7c0382e8d71f261781551ef2 rdgrowth/train_rd_v3.py
chk "$AN/rdgrowth/rd_growth_v2.py" f272280c9692a58961dc160e1777f6a7 rdgrowth/rd_growth_v2.py
chk "$AN/rdgrowth/rd_prune.py" 5d8b88c40018ef9a4760c60b32bb667e rdgrowth/rd_prune.py
chk "$AN/coarse_finetune.py" 9ecb13a01a146e7ebb659c06081bd71b coarse_finetune.py
chk "$AN/blender/val_select_v3.py" 129b0aecdac3c7fef7f76d7d996899e4 blender/val_select_v3.py
chk "$AN/blender/train_ckpt_v2.py" eba8a8b8bceb3e67d27a71bdf7e9f5f6 blender/train_ckpt_v2.py
chk "$AN/ablation-b24-r8.sbatch" 82259b73557bfcdb760648f42fad08cc jobs/ablation-b24-r8.sbatch
chk "$AN/ablation-b24-r8-blender.sbatch" d204fb12d9bc3b5c634ab9870b1f38b4 jobs/ablation-b24-r8-blender.sbatch
chk "$AN/capjob-v1.sh" 5a42c14029c6bbb3db8abcf1e921f5ae jobs/capjob-v1.sh
chk "$AN/budget-sweep-v2.sbatch" 247b78f2bad723ecd7cae26e236076d9 jobs/budget-sweep-v2.sbatch
chk "$AN/budget-sweep-v3.sbatch" 3c4ba68fe54bc67ddd7852f39d5365f5 jobs/budget-sweep-v3.sbatch
chk "$AN/budget-sweep-blender-v1.sbatch" d720b8ccba85622f90f25ee0b52b77ee jobs/budget-sweep-blender-v1.sbatch
chk "$PKG/stage2_run.sh" ce7c5ed0c8f63bbb62854895351293fa vis_collapse/stage2_run.sh
chk "$PKG/deliver_coarse.sh" c29c1cce4ee6a5514d1b2ac6743dfb89 vis_collapse/deliver_coarse.sh
chk "$PKG/deliver_coarse.py" 1877cc0ce8562d0f620b90b9ce85b788 vis_collapse/deliver_coarse.py
chk "$PKG/coarse_vis.py" 201cf1ceaef9cfa58ee4b11ba10f6664 vis_collapse/coarse_vis.py
chk "$PKG/vertex_footprint.py" 1f5e9fa41e51a56bcbbbb45e4d094fdc vis_collapse/vertex_footprint.py
chk "$PKG/scripts/run_garden.sh" 9faa7f6547aa02b42f3be535ed634d62 vis_collapse/scripts/run_garden.sh
chk "$PKG/index_context_codec.py" 968ec0132fc509f2cf181045fa3f3baa vis_collapse/index_context_codec.py
chk "$STAGE2/STAGE2_VERSION" 3e4b1a39f99be012a80b1b7eb861ac59 stage2/STAGE2_VERSION
chk "$STAGE2/runtime.py" f5d04b147d2d9654b3dc7a1ded73b520 stage2/runtime.py
chk "$STAGE2/sharing_after.py" 492b64244cb49af07a6349d7547924bf stage2/sharing_after.py
chk "$STAGE2/model.py" 7a85cf5b10c418b877a7798b7373a9d2 stage2/model.py
chk "$LAB/upstream/train.py" a0b4724a322213f2722cc42d005435ef upstream/train.py
chk "$LAB/upstream/arguments/__init__.py" b40ac5e4694990abb0785a3c0942b473 upstream/arguments/__init__.py
echo "same $same | different $diff | missing $miss"
