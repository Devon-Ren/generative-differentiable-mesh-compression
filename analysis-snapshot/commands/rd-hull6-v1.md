# rd-hull6-v1：补齐 11 场景各 6 点的 RD 曲线（2026-10-31）

**现状**：每个场景已有 3 个点——FP32 原始（≈740 MB）、codec level 1.0（不删顶点）、λ 6.2e-3、λ 1.85e-3（v3，含 phase-B prune）。**补两档 λ**：6.2e-4（高码率点，Room 的 v2 结果是 28.4 MB / 28.27 dB，≈ 原始质量 / 26x）和 1.85e-2（低码率点，Room v2 为 3.0 MB / 26.5 dB）。四档 λ 成等比序列（×3），加原始和 level 1.0 共 **6 点**。Room 的这两档虽有 v2（无 prune）结果，但 hull 要同一序列（v3），所以 Room 也重跑。
**代码不变**：复用已装好的 `$AN/rd-prune-v3b.sbatch`（batch B 同一个），只是 LAMS 不同；输出到 `/scratch/zr2245/<scene>-rd3-lam<L>-mu1.0/`，行追加到 `$AN/rd-prune3-<scene>.txt`（和已有两档同一张表，按 lam 列区分）。可重启（已完成的 λ 跳过）。
**成本**：每场景 λ 6.2e-4 训练 ≈ 0.9× 原始（40–55 min）+ λ 1.85e-2 ≈ 25 min + 两次 level-1.0 交付各 ≈ 10 min ≈ **1.5 GPU-h**；11 场景 ≈ **16–17 GPU-h**，11 个 job 各 2.5 h walltime（general 并发 3–4 个时约 6–8 h 墙钟，看队列）。XYZ 精度沿用 batch B 的每场景值（room q14；garden/bonsai/counter/flowers q15；kitchen/bicycle/stump/treehill q16；truck/train q18）。

## §1 提交 11 个 job（login 或任意节点，整块粘贴；写脚本后用 bash 跑，不会退出终端）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-rd3-hull6-v1.sh <<'SHEOF_HULL6'
#!/usr/bin/env bash
# submit-rd3-hull6-v1.sh (2026-10-31): two extra lambda points (6.2e-4, 1.85e-2) on all 11 scenes with rd-prune-v3b.sbatch,
# same settings as batch B (BPV 10, BPF 2, MU 1, MAXF 0.15, SMOKE 0), explicit XYZ per scene.  Rows -> $AN/rd-prune3-<scene>.txt
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; LAMS="0.00062 0.0185"
[ -f $AN/rd-prune-v3b.sbatch ] || { echo "FAILED: missing $AN/rd-prune-v3b.sbatch"; exit 1; }
xyz_of () { case $1 in room) echo q14;; garden|bonsai|counter|flowers) echo q15;; kitchen|bicycle|stump|treehill) echo q16;; truck|train) echo q18;; *) echo "";; esac; }
n=0
for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train; do
  Q=$(xyz_of $S); [ -n "$Q" ] || { echo "FAILED: no XYZ for $S"; continue; }
  case $S in room|kitchen|counter|bonsai) IM=images_2; DX="" ;; truck|train) IM=images; DX=",DATA=$D/tandt/$S" ;; *) IM=images_4; DX="" ;; esac
  for L in $LAMS; do [ -f /scratch/zr2245/$S-rd3-lam$L-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && echo "note: $S lam $L checkpoint exists, sbatch will skip its training"; done
  j=$(sbatch --parsable --time=02:30:00 --export=ALL,SCENE=$S,IMAGES=$IM,XYZ=$Q,LAMS="$LAMS",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15,SMOKE=0$DX --job-name=$S-h6 $AN/rd-prune-v3b.sbatch) \
    && { echo "submitted $S-h6 $j  XYZ $Q  IMAGES $IM"; n=$((n+1)); } || echo "FAILED: sbatch $S"
done
echo "submitted $n jobs"
squeue -u $USER -o "%.10i %.14j %.24a %.2t %.10M %.6D %R"
SHEOF_HULL6
bash $AN/submit-rd3-hull6-v1.sh
```

## §2 看进度 / 结果

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.14j %.2t %.10M %R"; echo
grep -h "lam 0.00062\|lam 0.0185" $AN/rd-prune3-*.txt | sort; echo
for f in $AN/rd3-*-h6-*.err; do [ -s $f ] && { echo "== $f"; tail -n 3 $f; }; done 2>/dev/null | tail -n 40
```

跑完把 `cat $AN/rd-prune3-*.txt` 贴给我，我出 11 场景的 6 点 hull 图（CPU，几分钟）。
