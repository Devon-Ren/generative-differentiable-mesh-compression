"""regenerates commands/node-batch7-v1.md from the four files it embeds (heredocs byte-identical, line counts in the submit script computed)."""
A = '/mnt/project-files/analysis/'
f_tr = open(A + 'rdgrowth/train_rd_adapt_v6.py').read(); f_vs = open(A + 'blender/val_select_v3.py').read()
f_sb = open(A + 'rd/adapt-v6-blender.sbatch').read(); f_rd = open(A + 'rd/redeliver-v3-blender.sbatch').read()
for name, txt, delim in (('tr', f_tr, 'PYEOF_TRAIN6'), ('vs', f_vs, 'PYEOF_VALSEL3'), ('sb', f_sb, 'SBEOF_ADAPT6B'), ('rd', f_rd, 'SBEOF_REDELIV3')):
    assert delim not in txt and txt.endswith('\n') and '```' not in txt, name
n_tr, n_vs, n_sb, n_rd = (t.count('\n') for t in (f_tr, f_vs, f_sb, f_rd))
md = f'''# node-batch7-v1（2026-11-06）：v5 整条链审查后的一次修正（adapt-v6 + val_select_v3），仍只在 lego / chair 上验证；3 个 job ≈ 3 GPU-h

**审查结论**（完整表：`analysis/rd/adapt-v5-blender-analysis.txt`，脚本 `rd/adapt_analysis_v5b.py`）。v5 在 lego / chair × {{2.5, 4, 6}} 上交付 hit 0.91–1.32，但按控制器自己的 probe 表是 1.01–1.17（chair 6 不可达除外），差的那 6–20 % 不在控制器里：

1. **交付链多交了 6–20 % 字节（和控制器无关）**：`val_select_v2` 用 checkpoint 里存的 `importance_score.pt` 剪枝，而 upstream 把这个量当成 Delaunay 之后的 **running max**（只在 29900 清零），软 floor 时期露过一次脸的面就一直"重要"；upstream 自己的最终清理是在 30000 重新渲染一遍算的。lego 2.5：交付 F 666976 vs 最终模型 F 561481（顶点 154537 vs 154281，几乎相同——多出来的全是面）。11/04 那批 8 场景固定 λ hull 也是这么交付的，字节同样偏大（相对比较公平，绝对值要重交）。**修法**：`blender/val_select_v3.py`（{n_vs} 行）在选中的 checkpoint 上按交付链的渲染设置（upstream 加载时固定的 floor 0.999、coarse_finetune 默认的 4× supersampling）重新渲染 100 张训练视图取每面最大 blending，用它剪枝；同时记录 final-floor/4× 的保留数（和 probe 对表）和 running-max 的保留数（和 v2 对表），剪完在 val 上自检。
2. **超标的主因是 phase A 那个 1.15 的余量**：它假设 phase B 能砍掉的比后面长出来的多；实测两个数据集都相反——12k 之后 kept 面继续"摊开"（Blender +9–40 %，Mip-360 室内 +6–23 %），而固定 λ prune 只去掉 7–10 % 的 V。phase B 只能减不能加，所以 aim 就是 target 本身（`--phase_a_overshoot` 默认 1.0）。
3. **phase B 每轮只有 ~3 % V 的权限**：~22k 个可砍边里 12–16k 被 foldover 过滤拒掉，只剩 2–6k；4 轮合计 7–10 %。决策规则不动，轮数从 4 改成 **16（12000 起每 500 一轮到 19500）**，Blender 上每轮 probe ~20 s，一个 run 多 5 min。
4. 外推改成对所有 probe 的最小二乘（500 一轮时两点斜率太抖：0.01 MB 的误差到 24000 放大成 0.2 MB）；在 v5 的 6 条 trend 上 CPU 回放，est@18k / final 1.01–1.04，和 v5 的 1.00–1.04 一样，两点时与 v5 公式完全相同。
5. 新增诊断：每轮砍掉的顶点里有多少在 probe 的 kept 集合里（`kept_removed`），用来核对 removed_mb 的记账。
6. chair 6 MB 是不可达目标（13/17 个 interval 被 upstream add_percentage 卡住，V 303652 / V* 572727，外推 4.63 < 6），照报不算错。

**代码变动范围**：`rdgrowth/train_rd_adapt_v6.py`（{n_tr} 行）= v5 + 上面 2/4/5 三处，其余逐字相同（diff 可查）；`blender/val_select_v3.py`（{n_vs} 行）= v2 + 重算 importance；`rd/adapt-v6-blender.sbatch`（{n_sb} 行，VERSION v6b）= v5b2 + OVER=1.0、PRUNE 每 500、val_select_v3、新增自检、重交时的 val_select 守卫改成测 state dict；`rd/redeliver-v3-blender.sbatch`（{n_rd} 行，VERSION rd3）= 把已有的 run（参考点）不重训、只用 val_select_v3 重交付到 `$RUN-best3`，一行对比 v2 / v3 的字节和 PSNR。四个文件另外请了一个独立审查过一遍（6 条意见全部处理：重交守卫、importance_score.pt 改为可选、floor/supersampling 按交付链、少渲染一遍 val）。v5 / v2 / upstream / probe_bytes_v1 / train_ckpt_v2 都不动，正在 HPC 上的文件不覆盖。

**这批 3 个 job（单卡顺序跑）**：(a) 4 个参考点重交付（lego/chair × λ 6.2e-4 / 6.2e-3，≈20 min，不重训）→ 修正后的 hull；(b) lego-adapt6b、(c) chair-adapt6b 各 3 个目标（每个 run ≈25 min + 自检 smoke，≈1.3 h/job）。合计 ≈3 GPU-h。通过标准：交付 / probe ≈ 1.00（第 1 条修好）、可达目标 hit 在 0.9–1.1、等字节 ΔPSNR 不比 v5 的 +0.05 差。

## 1. 写四个文件 + 提交 3 个 job（一次粘贴；检查和提交都在脚本里，不会碰你的终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth $AN/blender
cat > $AN/rdgrowth/train_rd_adapt_v6.py <<'PYEOF_TRAIN6'
{f_tr}PYEOF_TRAIN6
cat > $AN/blender/val_select_v3.py <<'PYEOF_VALSEL3'
{f_vs}PYEOF_VALSEL3
cat > $AN/adapt-v6-blender.sbatch <<'SBEOF_ADAPT6B'
{f_sb}SBEOF_ADAPT6B
cat > $AN/redeliver-v3-blender.sbatch <<'SBEOF_REDELIV3'
{f_rd}SBEOF_REDELIV3
cat > $AN/submit-batch7.sh <<'SUB7'
AN=/scratch/zr2245/analysis; RES=yw523-l40s
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }}; }}
chk $AN/rdgrowth/train_rd_adapt_v6.py {n_tr}; chk $AN/blender/val_select_v3.py {n_vs}; chk $AN/adapt-v6-blender.sbatch {n_sb}; chk $AN/redeliver-v3-blender.sbatch {n_rd}; chk $AN/rdgrowth/probe_bytes_v1.py 66
grep -q "version='adapt-v6'" $AN/rdgrowth/train_rd_adapt_v6.py || {{ echo "CHECK FAILED: train_rd_adapt_v6.py is not adapt-v6"; exit 1; }}
grep -q "n_kept_removed" $AN/rdgrowth/train_rd_adapt_v6.py || {{ echo "CHECK FAILED: train_rd_adapt_v6.py lacks the kept-removed diagnostic"; exit 1; }}
grep -q "VERSION = 'val_select_v3" $AN/blender/val_select_v3.py || {{ echo "CHECK FAILED: val_select_v3.py is not val_select_v3"; exit 1; }}
grep -q "VERSION v6b" $AN/adapt-v6-blender.sbatch || {{ echo "CHECK FAILED: adapt-v6-blender.sbatch is not v6b"; exit 1; }}
grep -q "VERSION rd3" $AN/redeliver-v3-blender.sbatch || {{ echo "CHECK FAILED: redeliver-v3-blender.sbatch is not rd3"; exit 1; }}
grep -q "PROBE_VERSION = 'probe-v1'" $AN/rdgrowth/probe_bytes_v1.py || {{ echo "CHECK FAILED: probe_bytes_v1.py is not probe-v1"; exit 1; }}
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v6.py $AN/blender/val_select_v3.py $AN/rdgrowth/probe_bytes_v1.py || {{ echo "CHECK FAILED: python files do not compile"; exit 1; }}
bash -n $AN/adapt-v6-blender.sbatch && bash -n $AN/redeliver-v3-blender.sbatch || {{ echo "CHECK FAILED: sbatch syntax"; exit 1; }}
for f in $AN/vis_collapse-v12/stage2_run.sh $AN/vis_collapse-v12/scripts/run_garden.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py $AN/blender/train_ckpt_v2.py $AN/coarse_finetune.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || {{ echo "CHECK FAILED: MISSING $f"; exit 1; }}; done
REFS=""; for S in lego chair; do [ -f /scratch/zr2245/datasets/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}
  for L in 0.00062 0.0062; do R=$S-rd3b4-lam$L-mu1.0; [ -d /scratch/zr2245/$R/original/point_cloud ] || {{ echo "CHECK FAILED: reference run $R has no checkpoints"; exit 1; }}; REFS="$REFS $R"; done; done
echo "reference runs to re-deliver:$REFS"
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt5b|adapt6b|redeliv3"); [ "$busy" = 0 ] || echo "note: $busy adapt/redeliver jobs still in the queue"
echo "CHECK OK"; k=0
j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,RUNS="$REFS",XYZ=q14 --job-name=refs-redeliv3 $AN/redeliver-v3-blender.sbatch) && {{ echo "submitted refs-redeliv3 $j"; k=$((k+1)); }} || echo "FAILED: sbatch redeliver"
sub () {{ j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=$1,XYZ=$2,TARGETS="2.5 4 6" --job-name=$1-adapt6b $AN/adapt-v6-blender.sbatch) && {{ echo "submitted $1-adapt6b $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1"; }}
sub lego q14; sub chair q14
echo "submitted $k of 3 (one GPU: they run one after another, redeliver first)"
SUB7
bash $AN/submit-batch7.sh
squeue -u $USER -o "%.10i %.16j %.3t %.8M %R"
```

## 2. 25 分钟后：重交付的表 + lego 的自检 / smoke

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.16j %.3t %.8M %R"
for f in $AN/redeliv3-*.out; do echo "=== $f"; grep -h "importance over\\|pruned best\\|VAL_SELECT_RESULT\\|FAILED\\|exit 0" $f | tail -14; echo "----- table"; grep -A9 "^=====" $f | tail -6 | cut -c1-330; done
for f in $AN/adapt6b-*-adapt6b-*.out; do echo "=== $f"; grep -h "self-check\\|smoke probe scaling\\|smoke prune\\|smoke final\\|^\\[probe\\]\\|FAILED\\|note:\\|=== .* target" $f | tail -12; done
```

期望：重交付 4 行里 `bytes x0.8x–0.9x`（v3 比 v2 少 6–20 %），`dPSNR` 在 ±0.05 内；lego 到 `=== lego target 2.5 MB`。再过 30 min 可以看第一个 run 的 phase A / 16 轮 probe：

```bash
for d in /scratch/zr2245/*-adapt6b-t*mb; do echo "--- $d"; grep -h "^\\[RD adapt\\|^\\[RD prune\\|^\\[RD [0-9]*\\] V" $d/train.log 2>/dev/null | tail -6 | cut -c1-330; done
```

## 3. 全部跑完（≈3 h）后

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-8hours -o JobID,JobName%18,State,Elapsed,ExitCode | grep -E "adapt6b|redeliv3|JobID"
for f in $AN/adapt6b-*-adapt6b-*.out $AN/redeliv3-*.out; do echo "=== $f"; tail -3 $f; done
echo "===== redeliver-v3-blender.txt"; cat $AN/redeliver-v3-blender.txt
echo "===== adapt-v6-blender.txt"; cat $AN/adapt-v6-blender.txt
echo "===== val_select_v3 on the adapt runs"; for d in /scratch/zr2245/*-adapt6b-t*mb; do echo "--- $d"; grep -h "cleaning render\\|importance over\\|pruned best\\|pruned val" $d/val_select.log 2>/dev/null | cut -c1-300; done
echo "===== phase A / probes"; for d in /scratch/zr2245/*-adapt6b-t*mb; do echo "--- $d"; grep -h "^\\[RD [0-9]*\\] V\\|^\\[RD adapt\\|^\\[RD prune\\|^Training is done" $d/train.log 2>/dev/null | tail -52 | cut -c1-340; done
```

把三段（redeliver 4 行、adapt-v6-blender 6 行、phase A / probes）贴回来。

## 4. 通过后的全量 sweep（粘贴块等 §3 出来再给）

- NeRF-Synthetic 8 场景 × {{2.5, 4, 6}} MB：24 run × ~25 min ≈ 10 GPU-h（adapt-v6-blender.sbatch 直接用）；另加 8 场景固定 λ hull 的重交付（24 × 5 min ≈ 2 GPU-h，不重训，论文 Blender 表的字节要用修正后的）。
- Mip-360 / T&T 11 场景 × {{5, 10, 20}} MB：33 run × ~50 min ≈ 27 GPU-h（adapt-v6.sbatch = adapt-v5.sbatch + 同样三处改动，等 §3 通过再写；20 MB 在大场景上仍受 upstream add_percentage 上限，照跑、照报）。
'''
open(A + 'commands/node-batch7-v1.md', 'w').write(md)
print('md lines', md.count('\n'), '| files', n_tr, n_vs, n_sb, n_rd)
