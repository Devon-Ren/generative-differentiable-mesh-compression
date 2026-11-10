# node-batch22-v1 (2026-11-10): Draco / zstd generic-codec rows for the 9 remaining Mip-360 / T&T scenes, on the reserved node

batch21 ran all 59 jobs; the only failures were `run_generic_v1` on 9 scenes, because that runner's built-in scene table only knows room and garden (the o3dgc runner knows all 11 and succeeded). This block re-runs it with the scene settings exported, on gl030 through the reservation (no fairshare), as two jobs of about 45 minutes so neither sits below the GPU-utilisation threshold for 2 hours (the Draco / zstd part is CPU work; the test-view evaluation uses the GPU).

## §1 write, check, submit (one paste)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/generic-codecs-b22.sbatch <<'SBEOF_GENC22'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=genc22
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zr2245/analysis/genc22-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/genc22-%x-%j.err
# generic-codecs-b22.sbatch (2026-11-10, VERSION genc-b22 = genc-b21 + RUN/DATA/IMAGES/QP exported for the 9 Mip-360 / T&T scenes the Draco runner does not know (its own table has room and garden only; the o3dgc runner knows all 11), same originals as the o3dgc rows ($SCENE-vis-collapse); b21 text follows): the generic-codec baselines (Draco / zstd / texture route: generic-v1/run_generic_v1.sh,
# and MPEG-4 SC3DMC-TFAN: o3dgc-v1/run_o3dgc_v3.sh) on the ORIGINAL mesh of every scene in SCENES, one after another, in one job.  So far only
# room and garden have rows (generic-codecs.txt, o3dgc rows).  Both runners are restartable and non-fatal per row; a scene whose runner fails
# prints FAILED and the job moves to the next scene.  Deep Blending scenes use the --indoor originals ($SCENE-indoor) through the runners'
# generic branch (RUN / DATA / IMAGES / QP exported).  TEX="4096" (one texture resolution: the xatlas bake is the slow part).
#   sbatch --export=ALL,SCENES="kitchen counter bonsai drjohnson playroom" --job-name=b21-genc-a .../generic-codecs-b21.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENES=${SCENES:?set SCENES="kitchen counter ..." via --export}; export TEX=${TEX:-"4096"}
AN=/scratch/zr2245/analysis; G=$AN/generic-v1; O=$AN/o3dgc-v1; D=/scratch/zr2245/datasets
for f in $G/run_generic_v1.sh $G/generic_codec_v1.py $G/eval_ckpt_v1.py $O/run_o3dgc_v3.sh $O/o3dgc_codec_v2.py; do [ -f $f ] || { echo "FAILED: MISSING $f (generic-codecs-v1 / o3dgc-v3 blocks were never installed?)"; exit 1; }; done
echo "generic-codecs-b21 | scenes $SCENES | TEX '$TEX' | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nf=0
for S in $SCENES; do
  unset RUN DATA IMAGES QP
  case $S in
    drjohnson|playroom) export RUN=/scratch/zr2245/$S-indoor/original DATA=$D/db/$S IMAGES=images QP=15;;
    room|garden) ;;   # known to both runners' case tables
    kitchen) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/m360/$S IMAGES=images_2 QP=16;;
    counter|bonsai) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/m360/$S IMAGES=images_2 QP=15;;
    flowers) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/m360/$S IMAGES=images_4 QP=15;;
    bicycle|stump|treehill) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/m360/$S IMAGES=images_4 QP=16;;
    truck|train) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/tandt/$S IMAGES=images QP=18;;
    *) echo "FAILED: no settings for scene $S"; nf=$((nf+1)); continue;;
  esac
  if [ -n "$RUN" ] && [ ! -f $RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then echo "FAILED: $S original missing at $RUN"; nf=$((nf+1)); continue; fi
  echo "=== $S: generic-v1 (zstd / Draco / texture) $(date) ==="
  bash $G/run_generic_v1.sh $S > $G/$S-b22.log 2>&1 || { echo "FAILED: run_generic_v1 $S"; tail -12 $G/$S-b22.log; nf=$((nf+1)); }
  grep -h "^$S \|FAILED\|WARNING" $G/$S-b22.log | tail -14
  echo "=== $S: o3dgc-v3 (MPEG SC3DMC / TFAN) $(date) ==="
  EVAL=1 bash $O/run_o3dgc_v3.sh $S > $O/$S-b21.log 2>&1 || { echo "FAILED: run_o3dgc_v3 $S"; tail -12 $O/$S-b21.log; nf=$((nf+1)); }
  grep -ih "ROW\|FAILED\|done" $O/$S-b21.log | tail -8
done
echo "===== generic-codecs.txt (these scenes) ====="; for S in $SCENES; do grep -h "^$S " $AN/generic-codecs.txt 2>/dev/null; done
echo "generic-codecs-b22 exit 0 ($nf failures)  $(date)"
SBEOF_GENC22
AN=/scratch/zr2245/analysis; RES=yw523-l40s
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/generic-codecs-b22.sbatch 50 c1b75cb09fdc9eff4d9e533323b619e6; bash -n $AN/generic-codecs-b22.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for S in kitchen counter bonsai bicycle stump flowers treehill truck train; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || echo "WARNING: $S original missing at $S-vis-collapse (its rows will FAIL inside the job)"; done
scontrol show node gl030 | grep -oE "State=[^ ]*|AllocTRES=[^ ]*"; busy=$(squeue -u $USER -h -o %j | grep -c b22); [ "$busy" = 0 ] || { echo "CHECK FAILED: b22 jobs already queued"; exit 1; }
echo "CHECK OK"
sb () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 "$@") && echo "submitted $j" || echo "FAILED: sbatch $*"; }
sb --time=02:00:00 --export=ALL,SCENES="kitchen counter bonsai truck train" --job-name=b22-genc-a $AN/generic-codecs-b22.sbatch
sb --time=02:00:00 --export=ALL,SCENES="bicycle stump flowers treehill" --job-name=b22-genc-b $AN/generic-codecs-b22.sbatch
sleep 30; squeue -u $USER -o "%.10i %.16j %.2t %.10M %R" | grep -E "b22|JOBID"
```

## §2 collect (about an hour later)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch22-collect-$(date +%m%d-%H%M).txt
{ echo "===== sacct"; sacct -X -u $USER -S now-6hours -o JobID,JobName%20,State,Elapsed,ExitCode | grep -E "b22|JobID"
echo "===== generic-codecs.txt (all scenes)"; cat $AN/generic-codecs.txt
echo "===== FAILED"; grep -h "FAILED\|WARNING" $AN/genc22-*.out 2>/dev/null; } > $O 2>&1; wc -l $O; echo "send me $O"
```
