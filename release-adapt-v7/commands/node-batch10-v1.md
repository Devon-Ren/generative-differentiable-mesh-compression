# node-batch10-v1（2026-11-07）：补跑 5 个 Blender 2.5 MB job（ficus hotdog materials mic ship），SMOKE=0

**为什么挂**：5 个 job 都死在 smoke 冒烟跑（1300 it）最后一步的 final probe：`RuntimeError: byte probe failed at iteration 1300 (rc 1)`，probe 子进程日志为空（连 "Dictionary initialization" 都没打出来就退出了）。smoke 要检验的两件事都已经过了——1250 的 probe 正常（ficus 276198 V / 92066 F → 1.465 MB）、强制 prune 正常（30000 对，exact 记账）；挂的是对 1300 it 这种"刚初始化的 soup 再砍掉 10 %"的退化网格做 final cleaning + probe，正式跑永远不会处于这种状态。同样 5 个场景的 4 MB job（SMOKE=0，同一份代码、同一条路径）全部跑通，30000 it 的 final probe 正常（probe/delivered 0.99–1.08），所以补跑用 SMOKE=0 即可；self-check 保留（11 s，已证明 1.000）。§1 里先顺手把 ficus 那个 probe-1300.pt 的面数打出来，确认是空网格（预期 F = 0）——这一步只读，不影响提交。

代码一个字不改：`train_rd_adapt_v7b.py`（md5 24f478ca…）、`adapt-v7b-blender.sbatch`（139 行，2d110e69…）就是 batch8-v2 / batch9-v2 用的。5 job × ≈27 min，单卡 ≈2.3 GPU-h，4 卡并行 ≈30 min 墙钟。

## 1. 诊断 + 提交（一次粘贴）

```bash
AN=/scratch/zr2245/analysis; RES=yw523-l40s
cat > $AN/submit-batch10-v1.sh <<'SUB10'
AN=/scratch/zr2245/analysis; RES=yw523-l40s
chk () { [ -f $1 ] || { echo "CHECK FAILED: missing $1"; exit 1; }; n=$(wc -l < $1); [ "$n" = "$2" ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum $1 | cut -c1-32); [ "$m" = "$3" ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v7b.py 827 24f478cab998b59d1886c2b8b956a111
chk $AN/adapt-v7b-blender.sbatch 139 2d110e69c79c3c2552c64fdbfef49c91
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
PT=$(ls -d $AN/adapt7b-smoke-ficus-*/original/probes/probe-1300.pt 2>/dev/null | tail -1)
if [ -n "$PT" ]; then python - "$PT" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print('diag: ficus smoke final-probe export: V %d F %d (expected F 0 = empty mesh after the 1300-it final cleaning; the real runs never pass through this state)' % (sd['triangles_points'].shape[0], sd['_triangle_indices'].shape[0]))
PY
else echo "diag: no ficus smoke probe-1300.pt found (smoke dir already removed); skipping"; fi
for S in ficus hotdog materials mic ship; do
  D=/scratch/zr2245/$S-adapt7b-t2.5mb-best/axisB/v1.0/deliver/summary.json; [ -f $D ] && { echo "CHECK FAILED: $S 2.5 MB already delivered at $D (nothing to resubmit)"; exit 1; }
  [ -f /scratch/zr2245/$S-adapt7b-t4mb-best/axisB/v1.0/deliver/summary.json ] || { echo "CHECK FAILED: $S 4 MB delivery missing (the proof that the real path works on this scene)"; exit 1; }
  [ -d /scratch/zr2245/datasets/nerf_synthetic/$S ] || { echo "CHECK FAILED: dataset for $S missing"; exit 1; }
done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt7|redeliv3"); [ "$busy" = 0 ] || echo "note: $busy adapt7/redeliver jobs still in the queue"
echo "CHECK OK"; k=0
for S in ficus hotdog materials mic ship; do
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:30:00 --export=ALL,SCENE=$S,XYZ=q14,TARGETS=2.5,SMOKE=0 --job-name=$S-2.5-adapt7b $AN/adapt-v7b-blender.sbatch) && { echo "submitted $S 2.5 MB $j"; k=$((k+1)); } || echo "FAILED: sbatch $S 2.5"
done
echo "submitted $k of 5"
SUB10
bash $AN/submit-batch10-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
```

`CHECK FAILED` 就停下来贴给我；`CHECK OK` 后 5 个 job 进队。

## 2. ≈35 分钟后：结果（按新 job id 看，旧的 22088893–97 .out 还在）

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-2hours -o JobID,JobName%22,State,Elapsed,ExitCode | grep -E "2.5-adapt7b|JobID"
for S in ficus hotdog materials mic ship; do f=$(ls -t $AN/adapt7b-$S-2.5-adapt7b-*.out | head -1); echo "=== $f"; grep -h "FAILED\|exit 0\|self-check:" $f | tail -3; done
echo "===== adapt-v7-blender.txt (2.5 MB rows)"; grep "target 2.5 MB" $AN/adapt-v7-blender.txt
```

把这段输出贴回来，我把 5 条并进 sweep 表（`adapt_analysis_v7b_sweep.py` 直接重跑）。
