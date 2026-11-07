#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=gate14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/zr2245/analysis/gate14-%j.out
# gate-batch14.sbatch (2026-11-07): CPU job that runs after the six v9b test jobs (--dependency=afterany), applies the pass bar
# (gate_batch14_v1.py) to the rows they wrote, and submits the v9b sweep part(s) that passed (submit-batch14-sweep-v1.sh all|mip|blender).
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
AN=/scratch/zr2245/analysis
echo "gate-batch14 $(date) on $(hostname) job $SLURM_JOB_ID"
sacct -X -u $USER -S now-8hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt9x|JobID"
for f in $AN/adapt9x-*-adapt9x-*.out $AN/adapt9xb-*-adapt9xb-*.out; do [ -f $f ] || continue; grep -q "FAILED" $f && { echo "--- $f FAILED:"; grep -h "FAILED" $f | head -3; tail -5 $f | cut -c1-300; }; done
python3 $AN/gate_batch14_v1.py $AN/adapt-v9b.txt $AN/adapt-v9b-blender.txt 2>&1 | tee $AN/gate-batch14.log || { echo "GATE FAILED: gate script crashed"; exit 1; }
mode=$(grep "^GATE SUBMIT" $AN/gate-batch14.log | awk '{print $3}')
case "$mode" in
  all|mip|blender) echo "GATE: submitting the v9b sweep ($mode) $(date)" | tee -a $AN/gate-batch14.log; bash $AN/submit-batch14-sweep-v1.sh $mode 2>&1 | tee -a $AN/gate-batch14.log;;
  *) echo "GATE: nothing submitted (mode '$mode'); read $AN/gate-batch14.log" | tee -a $AN/gate-batch14.log;;
esac
echo "gate-batch14 done $(date)"
