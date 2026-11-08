#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=gate15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/zr2245/analysis/gate15-%j.out
# gate-batch15.sbatch (2026-11-08): CPU job after the six v9c test jobs (--dependency=afterany); applies gate_batch15_v1.py and submits the v9c sweep part(s) that passed.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
AN=/scratch/zr2245/analysis
echo "gate-batch15 $(date) on $(hostname) job $SLURM_JOB_ID"
sacct -X -u $USER -S now-8hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt9c|adapt9xb|JobID"
for f in $AN/adapt9c-*-adapt9c-*.out $AN/adapt9cb-*-adapt9cb-*.out; do [ -f $f ] || continue; grep -q "FAILED" $f && { echo "--- $f FAILED:"; grep -h "FAILED" $f | head -3; tail -5 $f | cut -c1-300; }; done
python3 $AN/gate_batch15_v1.py $AN/adapt-v9c.txt $AN/adapt-v9c-blender.txt 2>&1 | tee $AN/gate-batch15.log || { echo "GATE FAILED: gate script crashed"; exit 1; }
mode=$(grep "^GATE SUBMIT" $AN/gate-batch15.log | awk '{print $3}')
case "$mode" in
  all|mip|blender) echo "GATE: submitting the v9c sweep ($mode) $(date)" | tee -a $AN/gate-batch15.log; bash $AN/submit-batch15-sweep-v1.sh $mode 2>&1 | tee -a $AN/gate-batch15.log;;
  *) echo "GATE: nothing submitted (mode '$mode'); read $AN/gate-batch15.log" | tee -a $AN/gate-batch15.log;;
esac
echo "gate-batch15 done $(date)"
