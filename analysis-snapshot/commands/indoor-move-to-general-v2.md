# indoor-move-to-general-v2 (2026-11-01): move ONLY the 2 pending bonsai indoor λ jobs (counter jobs already started on tandon) from tandon_advanced to general

sshare 11/01: general FairShare 0.173 > tandon_advanced 0.116, and the four tandon jobs have sat pending while the general ones ran.
Two routes, in one script: first try `scontrol update JobId=… Account=torch_pr_38_general` (keeps the job and its queue age;
works on PD jobs when the user has the target account); if Slurm refuses, `scancel` + resubmit with the identical `--export`
(the sbatch scripts skip training when a checkpoint exists, and a PD job has produced nothing, so nothing is lost either way).

```bash
AN=/scratch/zr2245/analysis
cat > $AN/move-indoor-to-general-v2.sh <<'SHEOF'
#!/usr/bin/env bash
# move-indoor-to-general-v2.sh (2026-11-01): bonsai indoor lambda jobs (PD only) -> torch_pr_38_general (scontrol first, scancel+resubmit as fallback)
AN=/scratch/zr2245/analysis; A=torch_pr_38_general
moved=0; resub=0
for S in bonsai; do
  for LAMS in "0.00062 0.0185" "0.00185 0.0062"; do
    T=$(echo $LAMS | tr ' ' '_' | sed 's/0\.0*//g'); NAME=$S-i$T
    J=$(squeue -u $USER -h -n $NAME -o "%i %t %a" | awk '$2=="PD"{print $1}' | head -1)
    if [ -z "$J" ]; then echo "skip $NAME: no pending job with that name (running or gone)"; squeue -u $USER -h -n $NAME -o "   %i %t %a %M"; continue; fi
    if scontrol update JobId=$J Account=$A 2> $AN/move-$NAME.err; then
      sleep 2; NOW=$(squeue -u $USER -h -j $J -o "%a"); echo "moved $NAME job $J -> account $NOW"; moved=$((moved+1)); continue
    fi
    echo "scontrol refused for $NAME ($(cat $AN/move-$NAME.err | head -1)); cancelling $J and resubmitting on $A"
    scancel $J; sleep 2
    Q=q15
    j=$(sbatch --parsable --account=$A --time=03:30:00 --export=ALL,SCENE=$S,IMAGES=images_2,XYZ=$Q,INDOOR=1,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$NAME $AN/rd-prune-v3c.sbatch) \
      && { echo "resubmitted $NAME as $j (lams $LAMS, XYZ $Q, account $A)"; resub=$((resub+1)); } || echo "FAILED: sbatch $NAME"
  done
done
echo "moved $moved, resubmitted $resub (expected 2 in total)"
squeue -u $USER -o "%.10i %.16j %.28a %.2t %.10M %R"
SHEOF
bash $AN/move-indoor-to-general-v2.sh
```

bonsai uses XYZ q15 (same as submit-indoor-v1.sh). The two running counter jobs are left alone (the script only touches PD jobs anyway). Job names stay `bonsai-i62_185`, `bonsai-i185_62`.
