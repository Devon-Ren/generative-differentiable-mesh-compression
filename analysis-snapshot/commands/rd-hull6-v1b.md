# rd-hull6-v1b: 同 rd-hull6-v1，只把 11 个 job 分到两个账户，2026-10-31

sshare（10/31 16:09）：general FairShare 0.289（RawUsage 214043），tandon 0.201（RawUsage 0）。11 个 job 全压 general 会把它的 FairShare 再拉低；tandon 这周没用过，可以分担。分法：6 个室内 + garden 走 general（优先级高、先启动），bicycle / stump / flowers / treehill / truck / train 走 tandon_advanced（改为 **5 室内 + garden = 6 on general，5 on tandon**）。其余全同 v1（同一个 rd-prune-v3b.sbatch，LAMS="0.00062 0.0185"，可重启）。

## §1 写脚本并提交（任意节点，整块粘贴）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-rd3-hull6-v1b.sh <<'SHEOF_HULL6'
#!/usr/bin/env bash
# submit-rd3-hull6-v1b.sh (2026-10-31): = v1 + account split (general: room kitchen counter bonsai garden stump; tandon_advanced: bicycle flowers treehill truck train)
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; LAMS="0.00062 0.0185"
[ -f $AN/rd-prune-v3b.sbatch ] || { echo "FAILED: missing $AN/rd-prune-v3b.sbatch"; exit 1; }
xyz_of () { case $1 in room) echo q14;; garden|bonsai|counter|flowers) echo q15;; kitchen|bicycle|stump|treehill) echo q16;; truck|train) echo q18;; *) echo "";; esac; }
acc_of () { case $1 in room|kitchen|counter|bonsai|garden|stump) echo torch_pr_38_general;; *) echo torch_pr_38_tandon_advanced;; esac; }
n=0
for S in room kitchen counter bonsai garden stump bicycle flowers treehill truck train; do
  Q=$(xyz_of $S); [ -n "$Q" ] || { echo "FAILED: no XYZ for $S"; continue; }
  A=$(acc_of $S)
  case $S in room|kitchen|counter|bonsai) IM=images_2; DX="" ;; truck|train) IM=images; DX=",DATA=$D/tandt/$S" ;; *) IM=images_4; DX="" ;; esac
  for L in $LAMS; do [ -f /scratch/zr2245/$S-rd3-lam$L-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "note: $S lam $L checkpoint exists, sbatch will skip its training"; done
  j=$(sbatch --parsable --account=$A --time=02:30:00 --export=ALL,SCENE=$S,IMAGES=$IM,XYZ=$Q,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15,SMOKE=0$DX --job-name=$S-h6 $AN/rd-prune-v3b.sbatch) \
    && { echo "submitted $S-h6 $j  XYZ $Q  IMAGES $IM  account $A"; n=$((n+1)); } || echo "FAILED: sbatch $S"
done
echo "submitted $n jobs"
squeue -u $USER -o "%.10i %.14j %.28a %.2t %.10M %.6D %R"
SHEOF_HULL6
bash $AN/submit-rd3-hull6-v1b.sh
```

## §2 看进度 / 结果（同 v1）

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.14j %.28a %.2t %.10M %R"; echo; grep -h "lam 0.00062\|lam 0.0185" $AN/rd-prune3-*.txt | cut -c1-200; echo; grep -l "FAILED" $AN/rd3-*-h6-*.out 2>/dev/null
```

若某个账户排队明显慢，可把还在 PD 的 job 挪账户：`scontrol update JobId=<id> Account=torch_pr_38_general`（或 tandon_advanced）。
