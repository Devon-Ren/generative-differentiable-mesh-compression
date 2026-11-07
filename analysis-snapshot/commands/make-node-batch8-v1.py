"""regenerates commands/node-batch8-v1.md from the two files it embeds (heredocs byte-identical, line counts in the submit script computed)."""
A = '/mnt/project-files/analysis/'
f_tr = open(A + 'rdgrowth/train_rd_adapt_v7.py').read(); f_sb = open(A + 'rd/adapt-v7-blender.sbatch').read()
for name, txt, delim in (('tr', f_tr, 'PYEOF_TRAIN7'), ('sb', f_sb, 'SBEOF_ADAPT7B')):
    assert delim not in txt and txt.endswith('\n') and '```' not in txt, name
n_tr, n_sb = (t.count('\n') for t in (f_tr, f_sb))
md = f'''# node-batch8-v1（2026-11-06）：adapt-v7 = v6 的 phase B 三处修正，lego / chair × {{2.5, 4}} MB 四个 job 并行验证（一波 ≈40 min，≈2.5 GPU-h）

**v6 的三个 phase-B 故障都量化了**（`rd/adapt-v6-blender-analysis.txt`），v7 只改这三处，phase A 一行不动：

1. **记账**：v6 每轮按 `pairs × (b_v + b_f × 每对删掉的全部面)` 记 removed_mb，把不在 probe kept 集合里的顶点和面也算进去，高估 1.5–1.9×，虚拟尺寸的斜率被抬高 → 第 2 轮外推 est/final = 1.20–1.44，前 10 轮都按"会超标"砍；最后一轮仍 1.04–1.06。v7 改成精确计数：`(两端都在 kept 集合的 pair 数) × b_v + (删掉的面里在 kept 面集合的数目) × b_f`，probe 顺手返回面 mask，`faces_removed_by()` 在塌缩前按 `collapse_edges` 同一条规则算出会删哪些面（塌缩后用实际删面数核对，不符就 fail）。在 v6 六条日志上回放（`rd/adapt-v7-replay-on-v6-logs.txt`）：最后一轮 est/final 1.006–1.030。
2. **门控 + 上限**：prune 是字节杠杆，只在外推落点 > 目标时开；砍的数量 `wanted = min(max(每轮计划, λ 准入数), 整个超出量 / 每对字节)`，永远不砍过目标。欠目标 → mode `none`，只记录 λ 准入数。上限或 rank 生效时 λ := 所取 pair 的 shadow price（mode `capped` / `count`）。v6 不管 est 说什么 16 轮全砍，删掉 V 的 14–19 %（其中 4–14 % 发生在控制器本身想要更多字节的轮次），6 个 run 全部落在 0.67–0.95。
3. **growth-capped 标记**：最后一轮外推 < 目标 且 limiter 卡住过半 interval → 行里标 `growth-capped True`（phase B 没有加字节的杠杆；回放里 chair 4 / 6 即使一刀不砍也只到 0.88 / 0.76）。
4. 诊断（不参与决策）：最后一轮之后在 22000、26000 各 probe 一次，看 19.5k→30k 之间 +0.1–0.5 MB 的"摊开"增长到底在 24000（floor 终点）前还是后结束——决定 sweep 时外推 horizon 要不要改。每次 ~20 s。

**回放给出的预期**（在 v6 轨迹上，只到门第一次打开为止是准的）：lego 2.5 不砍会到 1.09，门在 12.5k–13k 打开往回修 → 预期 ≈1.00–1.05；chair 2.5 不砍 1.03 → 门不开或只开一两轮，≈1.0；lego 4 不砍 0.98 → 门不开，≈0.98；chair 4 不砍 0.88 → `growth-capped True`，≈0.88（这是 upstream add_percentage 的上限，不是控制器的错）。6 MB 两个目标不跑（chair 6 必不可达 0.76，lego 6 ≈1.00 和 lego 4 同一类）。

**通过标准**：(a) lego 2.5 / chair 2.5 / lego 4 的 delivered hit 在 0.90–1.10，probe/delivered 仍 ≈1.0x；(b) chair 4 标 growth-capped True；(c) 每轮 `est` 对最终 probe 的误差 ≤ ±5 %（最后一轮）；(d) 等字节 PSNR 不低于 v6（v6 已在 hull 上方）。四条都过 → 定版 v7，给场景 sweep 块（§4）。

**改动范围**：`rdgrowth/train_rd_adapt_v7.py`（{n_tr} 行，v6 + 上面 1–4，其余逐字相同，diff 112 行）；`rd/adapt-v7-blender.sbatch`（{n_sb} 行，VERSION v7b = v6b + 新文件名 / 新 grep / smoke 多 5 条断言 / 行里加 growth-capped、limiter 计数、extra probes、每轮 kept 面数和 removed MB）。val_select_v3、probe_bytes_v1、train_ckpt_v2、v6 及更早版本都不动。

## 1. 写两个文件 + 提交 4 个 job（一次粘贴；检查和提交都在脚本里，不碰你的终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/train_rd_adapt_v7.py <<'PYEOF_TRAIN7'
{f_tr}PYEOF_TRAIN7
cat > $AN/adapt-v7-blender.sbatch <<'SBEOF_ADAPT7B'
{f_sb}SBEOF_ADAPT7B
cat > $AN/submit-batch8.sh <<'SUB8'
AN=/scratch/zr2245/analysis; RES=yw523-l40s
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }}; }}
chk $AN/rdgrowth/train_rd_adapt_v7.py {n_tr}; chk $AN/adapt-v7-blender.sbatch {n_sb}; chk $AN/blender/val_select_v3.py 254; chk $AN/rdgrowth/probe_bytes_v1.py 66
grep -q "version='adapt-v7'" $AN/rdgrowth/train_rd_adapt_v7.py || {{ echo "CHECK FAILED: train_rd_adapt_v7.py is not adapt-v7"; exit 1; }}
grep -q "def faces_removed_by" $AN/rdgrowth/train_rd_adapt_v7.py || {{ echo "CHECK FAILED: train_rd_adapt_v7.py lacks faces_removed_by"; exit 1; }}
grep -q "VERSION v7b" $AN/adapt-v7-blender.sbatch || {{ echo "CHECK FAILED: adapt-v7-blender.sbatch is not v7b"; exit 1; }}
grep -q "VERSION = 'val_select_v3" $AN/blender/val_select_v3.py || {{ echo "CHECK FAILED: val_select_v3.py is not val_select_v3"; exit 1; }}
grep -q "PROBE_VERSION = 'probe-v1'" $AN/rdgrowth/probe_bytes_v1.py || {{ echo "CHECK FAILED: probe_bytes_v1.py is not probe-v1"; exit 1; }}
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v7.py $AN/blender/val_select_v3.py $AN/rdgrowth/probe_bytes_v1.py || {{ echo "CHECK FAILED: python files do not compile"; exit 1; }}
bash -n $AN/adapt-v7-blender.sbatch || {{ echo "CHECK FAILED: sbatch syntax"; exit 1; }}
for f in $AN/vis_collapse-v12/stage2_run.sh $AN/vis_collapse-v12/scripts/run_garden.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py $AN/blender/train_ckpt_v2.py $AN/coarse_finetune.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || {{ echo "CHECK FAILED: MISSING $f"; exit 1; }}; done
for S in lego chair; do [ -f /scratch/zr2245/datasets/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for d in /scratch/zr2245/lego-adapt7b-t2.5mb /scratch/zr2245/chair-adapt7b-t2.5mb /scratch/zr2245/lego-adapt7b-t4mb /scratch/zr2245/chair-adapt7b-t4mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt6b|adapt7b|redeliv3"); [ "$busy" = 0 ] || echo "note: $busy adapt/redeliver jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show node gl030 | grep -o "CfgTRES=[^ ]*"
echo "CHECK OK"; k=0
sub () {{ j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --export=ALL,SCENE=$1,XYZ=q14,TARGETS="$2" --job-name=$1-$2-adapt7b $AN/adapt-v7-blender.sbatch) && {{ echo "submitted $1 target $2 $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $2"; }}
sub lego 2.5; sub chair 2.5; sub lego 4; sub chair 4
echo "submitted $k of 4 (one GPU each, in parallel; 6 cores each so the four fit beside cpu-torch)"
SUB8
bash $AN/submit-batch8.sh
squeue -u $USER -o "%.10i %.20j %.3t %.8M %R"
```

## 2. 5–8 分钟后：四个 job 都应过了自检 + smoke，进到 `=== <scene> target <T> MB`

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.20j %.3t %.8M %R"
for f in $AN/adapt7b-*-adapt7b-*.out; do echo "=== $f"; grep -h "self-check\\|smoke probe scaling\\|smoke prune\\|smoke final\\|note:\\|FAILED\\|=== .* target" $f | tail -8 | cut -c1-700; done
```

四个 job 都要有 `smoke prune: {{... 'mode': 'count'/'capped'/'lambda', ... 'kept_faces_removed': N, 'removed_mb': X, 'bytes_per_pair_measured': Y ...}}` 和 `smoke final: {{... 'growth_capped': False ...}}`，没有 FAILED。有 FAILED 就把那个 `.out` 的最后 40 行贴我，不要重交。

## 3. 提交后 ≈45 分钟：四行结果 + 每轮 probe / prune / 诊断 probe

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-3hours -o JobID,JobName%22,State,Elapsed,ExitCode | grep -E "adapt7b|JobID"
for f in $AN/adapt7b-*-adapt7b-*.out; do echo "=== $f"; tail -3 $f | cut -c1-200; done
echo "===== adapt-v7-blender.txt"; cat $AN/adapt-v7-blender.txt
echo "===== val_select_v3 on the adapt runs"; for d in /scratch/zr2245/*-adapt7b-t*mb; do echo "--- $d"; grep -h "importance over\\|pruned val" $d/val_select.log 2>/dev/null | cut -c1-300; done
echo "===== phase A / probes"; for d in /scratch/zr2245/*-adapt7b-t*mb; do echo "--- $d"; grep -h "^\\[RD adapt\\|^\\[RD prune\\|^Training is done" $d/train.log 2>/dev/null | tail -40 | cut -c1-420; done
```

把 `adapt-v7-blender.txt` 的 4 行和 `phase A / probes` 段贴回来（val_select 段可省）。我按 §0 的四条判：全过 → §4 的 sweep 块马上给。

## 4. 定版后的场景 sweep（§3 过了再给粘贴块）

- NeRF-Synthetic 8 场景 × {{2.5, 4}} MB：16 run × ~25 min；按场景 8 个 job（每 job 两个目标顺序跑，≈1 h/job），4 卡两波 ≈2 h 墙钟，≈8 GPU-h。sbatch 就是本批的 `adapt-v7-blender.sbatch`，不再改。
- Mip-360 / T&T 11 场景 × {{5, 10, 20}} MB：33 run × ~50–60 min（大场景每次 probe 更贵，轮数改 8 轮：12000 起每 1000 到 19000）；11 个 job 各 3 个目标顺序跑（≈3 h/job），4 卡三波 ≈9 h 墙钟，≈33 GPU-h。sbatch `adapt-v7.sbatch` = adapt-v5.sbatch + 和 v7b 相同的改动，§3 出结果前写好、审完。
- 8 场景固定 λ hull 的 val_select_v3 重交付（论文 Blender 表的字节用修正后的）：24 × 5 min ≈ 2 GPU-h，不重训，和 sweep 同批。
'''
open(A + 'commands/node-batch8-v1.md', 'w').write(md)
print('md lines', md.count('\n'), '| files', n_tr, n_sb)
