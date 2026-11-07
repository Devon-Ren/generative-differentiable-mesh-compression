# batchA-resume-v1 (2026-10-24): resume batch A after "PRECISION RULE FAILED"

**Cause (inferred, confirm with the grep the script prints first):** the XYZ=auto rule inside originals-v1.sbatch calls
`w['triangles_points'].numpy()` on the ORIGINAL checkpoint. That checkpoint stores trainable tensors
(`requires_grad=True`), and PyTorch refuses `.numpy()` on them. Earlier rule runs (geom_precision.py) read
`shared-final.pt`, which has no grad, so the bug never showed. All our other loaders call `.detach()` first.

**Fix:** a standalone rule script `xyz_rule_v1.py` (detach, same formula b = ceil(log2(extent / (0.1 × median edge))),
clamped 13..18), run once at submit time; the result is passed as `XYZ=qN`, so originals-v1.sbatch skips its broken
auto block. No sbatch file changes. Normals and the 30k training are already done and are skipped automatically
(restartable stages), so each job only runs the level-1.0 protocol (~40 min) → `--time=02:00:00`.
q17 is not in the codec's XYZ list (q16 / q18 are), so a q17 verdict is promoted to q18 (finer, still satisfies the rule).

## One paste (cs658 or any node; the rule needs ~2 GB RAM, ~30 s per scene)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/xyz_rule_v1.py <<'PYEOF'
"""xyz_rule_v1.py (2026-10-24): scene XYZ precision rule b = ceil(log2(extent / (0.1 * median edge))), clamped to 13..18.
Replaces the inline XYZ=auto block of originals-v1.sbatch / rd-prune-v3b.sbatch, which called .numpy() on requires_grad
tensors of the ORIGINAL checkpoint and failed.  usage: python xyz_rule_v1.py <point_cloud_state_dict.pt>
stdout: qN   stderr: the numbers behind it"""
import sys, math, torch, numpy as np
def arr(t): return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)
w = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
x = arr(w['triangles_points']).astype(np.float64).reshape(-1, 3); f = arr(w['_triangle_indices']).astype(np.int64).reshape(-1, 3)
extent = float((x.max(0) - x.min(0)).max()); rng = np.random.default_rng(0); sel = rng.choice(len(f), min(len(f), 1000000), replace=False)
e = np.concatenate([f[sel][:, [0, 1]], f[sel][:, [1, 2]], f[sel][:, [2, 0]]]); med = float(np.median(np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)))
raw = math.log2(extent / (0.1 * med)); b = int(min(18, max(13, math.ceil(raw))))
print(f'q{b}')
print(f'xyz rule: V {len(x)} F {len(f)} extent {extent:.3f} median edge {med:.5f} log2 {raw:.2f} -> q{b}', file=sys.stderr)
PYEOF
cat > $AN/resubmit-orig-v1.sh <<'SHEOF'
#!/usr/bin/env bash
# resubmit-orig-v1.sh (2026-10-24): resume batch A with an explicit XYZ per scene (normals + training are skipped).
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
echo "--- error line in the old job .err files (expect: Can't call numpy() on Tensor that requires grad):"
for e in $AN/orig-*-orig-*.err; do [ -f "$e" ] && echo "$(basename $e): $(grep -h -m1 -i 'error' $e)"; done
[ -f $AN/originals-v1.sbatch ] || { echo "FAILED: $AN/originals-v1.sbatch missing"; exit 1; }
for S in flowers treehill truck train; do
  C=/scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  [ -f $C ] || { echo "SKIP $S: no original checkpoint at $C"; continue; }
  Q=$(python $AN/xyz_rule_v1.py $C) || { echo "FAILED: xyz rule on $S"; continue; }
  [ "$Q" = "q17" ] && { echo "$S: rule says q17, not in codec list -> q18"; Q=q18; }
  case $S in
    truck|train) EX="SCENE=$S,IMAGES=images,DATA=$D/tandt/$S,XYZ=$Q" ;;
    *)           EX="SCENE=$S,IMAGES=images_4,XYZ=$Q" ;;
  esac
  j=$(sbatch --parsable --time=02:00:00 --export=ALL,$EX --job-name=$S-orig $AN/originals-v1.sbatch) && echo "submitted $S-orig $j  XYZ $Q" || echo "FAILED: sbatch $S"
done
squeue -u $USER -o "%.10i %.14j %.24a %.2t %.10M %.6D %R"
SHEOF
bash $AN/resubmit-orig-v1.sh
```

Paste back the whole output (the error lines, four `xyz rule:` lines and the job ids). When the jobs finish:
`cat /scratch/zr2245/analysis/originals.txt`.

**Cost:** 4 × ~0.7 h ≈ 3 GPU-h (training already paid).

## Batch B §2 (the 4 new scenes) also changes

rd-prune-v3b.sbatch has the same XYZ=auto block, so do NOT pass `XYZ=auto` there. Use
[batchB-rd3-v1.md §2-v2](batchB-rd3-v1.md) (added 2026-10-24), which computes XYZ with `xyz_rule_v1.py` at submit time.
Batch B §1 (6 existing scenes) passes explicit q15/q16 and is not affected.
