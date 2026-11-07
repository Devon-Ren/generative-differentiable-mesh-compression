# lambda-k-fan-v1: garden resubmit (2026-11-05 13:55)

garden-lkfan (21965003) failed its pre-flight check: the §1 block submitted every scene with `IMAGES=images_2`, but garden (outdoor, rd3 assets) was trained and delivered with `images_4` (see commands/2026-10-15-stage2v2-v12-garden.md). room / kitchen (rd3i, images_2) are unaffected and running.

The block below reads the images folder recorded in the garden lam-0.0062 delivery config, checks the dataset folder exists, and only then resubmits garden with that folder. Nothing else changes (same sbatch v1, same table file).

```bash
AN=/scratch/zr2245/analysis
IM=$(python -c "import json; print(json.load(open('/scratch/zr2245/garden-rd3-lam0.0062-mu1.0/axisB/v1.0/deliver/config.json')).get('images'))")
echo "garden delivery config says images = $IM"
ls -d /scratch/zr2245/datasets/m360/garden/images* 
if [ -n "$IM" ] && [ "$IM" != None ] && [ -d /scratch/zr2245/datasets/m360/garden/$IM ]; then
  sbatch --parsable --account=torch_pr_38_general --export=ALL,SCENE=garden,TAG=rd3,IMAGES=$IM --job-name=garden-lkfan $AN/lambda-k-fan-v1.sbatch
else
  echo "NOT SUBMITTED: config images='$IM' or folder missing -- paste this output"
fi
squeue -u $USER -o "%.10i %.14j %.3t %.10M %.6D %R" | grep -E "lkfan|idiag|JOBID"
```

Then the §2 view block of commands/lambda-k-fan-v1.md as before. The table `$AN/lambda-k-fan-v1.txt` is created by the first lambda that finishes (room lam 0.00062, roughly 40-60 min after its 08:35 start), so `cat: No such file` before that is expected.
