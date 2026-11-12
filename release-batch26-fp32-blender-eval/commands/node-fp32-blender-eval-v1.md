# FP32 SSIM / LPIPS on NeRF-Synthetic (one small GPU job, about 10 min, fairshare)

Fills the two TBD cells of Table 3 (MeshSplatting FP32 on NeRF-Synthetic). Evaluates the 8 val-selected FP32 originals
($S-blender4-best, the reference of every NeRF-Synthetic number in the paper) on the test split with the same evaluator as all
other rows (eval_ckpt_v1.py: upstream renderer, 4x supersampling, black background, LPIPS-vgg).

```bash
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets/nerf_synthetic; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
for S in lego chair drums ficus hotdog materials mic ship; do f=/scratch/zr2245/$S-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; [ -f $f ] || echo "MISSING $f"; done
cat > $AN/fp32-blender-eval.sbatch <<'SB'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=fp32bl
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/zr2245/analysis/fp32bl-%j.out
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets/nerf_synthetic; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; mkdir -p $AN/fp32bl
for S in lego chair drums ficus hotdog materials mic ship; do
  M=/scratch/zr2245/$S-blender4-best/original
  python $AN/generic-v1/eval_ckpt_v1.py --lab $LAB --model $M --dataset $D/$S --images train --ckpt $M/point_cloud/iteration_30000/point_cloud_state_dict.pt --out $AN/fp32bl/$S.json --label $S-fp32 2>&1 | grep "^EVAL\|FAILED\|Error" || echo "FAILED $S"
done
echo "fp32bl exit 0"
SB
sbatch $AN/fp32-blender-eval.sbatch
```

When it finishes: `grep "^EVAL\|FAILED" $(ls -t /scratch/zr2245/analysis/fp32bl-*.out | head -1)` and send me the 8 lines.
