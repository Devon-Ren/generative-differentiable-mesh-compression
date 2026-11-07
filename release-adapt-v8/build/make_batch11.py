"""make_batch11.py: write analysis/commands/node-batch11-v1.md (one paste: three files by heredoc + submit script with checks + follow-ups)."""
import hashlib, sys
AN = sys.argv[1]; out = sys.argv[2]


def read(p):
    s = open(p).read(); assert s.endswith('\n'), p
    return s, len(s.splitlines()), hashlib.md5(s.encode()).hexdigest()


tr, tr_n, tr_md5 = read(AN + '/rdgrowth/train_rd_adapt_v8.py')
sb, sb_n, sb_md5 = read(AN + '/rd/adapt-v8-blender.sbatch')
sm, sm_n, sm_md5 = read(AN + '/rd/adapt-v8.sbatch')
for body, tag in ((tr, 'PYEOF_TRAIN8'), (sb, 'SBEOF_ADAPT8B'), (sm, 'SBEOF_ADAPT8')):
    assert tag not in body and 'SUB11' not in body

head = f'''# node-batch11-v1（2026-11-07）：adapt-v8 = phase A 用 byte probe 现场标定 —— lego / chair × {{2.5, 4}} MB + room / bicycle × 10 MB，6 个 job 一波（≈5.5 GPU-h，4 卡 ≈2 h 墙钟）

**v8 改了什么**（`rdgrowth/train_rd_adapt_v8.py`，{tr_n} 行 = v7b 827 行 + 60 行，diff 84 行；phase B 一字不动）：sweep 证明 phase A 的 V* 来自常数先验（11 B/顶点 + 0.6 MB），而 12k 实测 B/V 在 5/10/20 MB 上是 11–12.8 / 9–11 / 7.7–10，大目标上 V* 偏低 10–30 %，10000 之后没有加字节的杠杆。v8 在 phase A 里也跑 byte probe（v2–v7 同一个 probe：最终 floor + 4x 的 cleaning 模拟 + 真实 encoder）：

1. **时机**：upstream 开始 prune 之后（`iteration > start_pruning`，之前是未训练的初始 soup，第一次 prune 砍掉 70–95 %）的 densify interval，每隔 `--probe_a_every`（1000）一次 → Blender 3500…9500 共 7 次（每次 ≈15 s），Mip/T&T 6–7 次（每次 ≈20–45 s）。
2. **取什么**（`SizeModel.update_attributes`）：只取 **属性价 b_v**（文件字节 − 拓扑流 − 字典 − 头 / kept 顶点数）和 **固定 overhead**（字典 + 头）。拓扑价（b_f × faces/vertex）**不取**：phase A 的模型是 soup（每个面自带 3 个顶点），它的连接流和交付的 restricted-Delaunay 网格无关，仍用初值直到 12k 的网格 probe 替换（和 v2–v7b 一样）。拓扑占 35–45 % 字节（Mip 2.2–2.9 × 1.75 = 3.9–5.1 B，Blender 6 B，先验 5 B），所以 V* 剩余先验误差 ≈ ±10 %；随 V 和数据集变的属性价（3–9 B）和 overhead（0.67 MB，占目标 1–13 %）是量出来的。kept share（probe 顶点 / 模型顶点）只记录不用：12k 时 0.8–1.0，训练结束 ≈0.98（v6/v7 的"摊开"），交付顶点数 = 模型顶点数，v7b 的假设不变。
3. V* = `vertices_for(target)` 每个 interval 从更新后的模型重算，v4 的几何计划 / limiter 模拟 / front-load 按实测 V* 走。事件 `probe_a`（字节、kept share、b_v、overhead、更新前后的 V*），densify 事件带 `size_source`，rd_summary 带 `probes_a` / `probe_a_count`。kept 顶点少于字典大小 max(K,R)=4096 时 probe 跳过（事件 `probe_a_skipped`，下个 interval 再试）。
4. `--probe_a_every 0` = v7b。smoke 延长到 1600 it：1500 的第一个 densify interval 强制 probe（`--probe_a_smoke`），1550 强制 prune，断言 probe_a 事件 / `size_source` / rd_summary 字段。

**sbatch**：`adapt-v8-blender.sbatch`（{sb_n} 行，VERSION v8b-r1 = v7b-r2 + 新文件名 / grep / smoke 1600 / 行里加 7 次 phase-A probe）；`adapt-v8.sbatch`（{sm_n} 行，VERSION v8-r1 = v7-r2 + 同样改动 + **phase B 改成 16 轮**（12000 起每 500 到 19500，和 Blender 一样：v8 瞄得更高，indoor 20 MB 预期要砍 10–25 %，每轮吞吐只有上限的 4–12 %，8 轮不够；多 8 次 probe ≈+4–8 min/run））。输出目录全新：`$S-adapt8b-t{{T}}mb`（Blender，+ `-best`）、`$S-adapt8-t{{T}}mb`（Mip），表 `adapt-v8-blender.txt` / `adapt-v8.txt`；v7b 的目录和表不动。

**为什么带两个 Mip 场景**：lego/chair 在 2.5/4 MB 上 limiter 全程卡住（bound 11–13/17），只能验证机制不出错、b_v 标定和 12k 一致、不比 v7b 差；**先验误差本身**要在 limiter 不卡的 Mip 10 MB 上看：room 10（v7b hit 0.814，bound 1/17，indoor）和 bicycle 10（0.775，bound 1/17，outdoor）。这两个过了再给 33 个 Mip/T&T 的重跑块（≈25 GPU-h，一夜）；Blender 16 个结果不用重跑（v8 在 Blender 上预期和 v7b 持平：先验偏高的方向被 limiter 盖住）。

**通过标准**（我按这个判）：
- (a) 6 个 job exit 0；smoke 里有 `smoke probe-a: {{... 'b_v': >0, 'source': 'probe_a@1500' ...}}`；
- (b) **标定成立**：每行 `phase-A probes N (last b_v X vs 12k b_v Y ...)`，9500 的 b_v 和 12k 网格 probe 的 b_v 相差 ≤ 10 %，overhead 相差 ≤ 1 %（soup 上量的属性价能转到网格上，这是 v8 的前提）；
- (c) **room 10 / bicycle 10 的 hit 进 0.90–1.10**（v7b 0.814 / 0.775），等字节 PSNR 比 v7b 行不低超过 0.05 dB；
- (d) lego/chair 四行 hit 0.90–1.10（growth-capped True 的行除外），test PSNR 和 v7b 同目标行相差 ≤ 0.05 dB；
- (e) phase-A probe 总开销：Blender ≤ 3 min/run，Mip ≤ 8 min/run（行里每个 probe 的秒数）。
(b)(c) 过 → batch12（33 个 Mip/T&T）；(b) 过 (c) 不过 → 看 probes-a 的 V* 轨迹和 limiter 计数再定，不盲跑。

## 1. 写三个文件 + 提交 6 个 job（一次粘贴；检查和提交都在脚本里，不碰你的终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/train_rd_adapt_v8.py <<'PYEOF_TRAIN8'
'''
mid1 = '''PYEOF_TRAIN8
cat > $AN/adapt-v8-blender.sbatch <<'SBEOF_ADAPT8B'
'''
mid2 = '''SBEOF_ADAPT8B
cat > $AN/adapt-v8.sbatch <<'SBEOF_ADAPT8'
'''
submit = f'''SBEOF_ADAPT8
cat > $AN/submit-batch11-v1.sh <<'SUB11'
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }}; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || {{ echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }}; }}
chk $AN/rdgrowth/train_rd_adapt_v8.py {tr_n} {tr_md5}
chk $AN/adapt-v8-blender.sbatch {sb_n} {sb_md5}
chk $AN/adapt-v8.sbatch {sm_n} {sm_md5}
chk $AN/rdgrowth/train_rd_adapt_v7b.py 827 24f478cab998b59d1886c2b8b956a111
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
grep -q "version='adapt-v8'" $AN/rdgrowth/train_rd_adapt_v8.py && grep -q "def update_attributes" $AN/rdgrowth/train_rd_adapt_v8.py || {{ echo "CHECK FAILED: train_rd_adapt_v8.py is not adapt-v8"; exit 1; }}
grep -q "VERSION v8b-r1" $AN/adapt-v8-blender.sbatch && grep -q "VERSION v8-r1" $AN/adapt-v8.sbatch || {{ echo "CHECK FAILED: sbatch files are not v8-r1"; exit 1; }}
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v8.py $AN/blender/val_select_v3.py $AN/rdgrowth/probe_bytes_v1.py || {{ echo "CHECK FAILED: python files do not compile"; exit 1; }}
bash -n $AN/adapt-v8-blender.sbatch && bash -n $AN/adapt-v8.sbatch || {{ echo "CHECK FAILED: sbatch syntax"; exit 1; }}
for f in $AN/vis_collapse-v12/stage2_run.sh $AN/vis_collapse-v12/scripts/run_garden.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py $AN/blender/train_ckpt_v2.py $AN/coarse_finetune.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || {{ echo "CHECK FAILED: MISSING $f"; exit 1; }}; done
for S in lego chair; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in room bicycle; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in lego chair; do [ -f /scratch/zr2245/$S-rd3b4-lam0.0062-mu1.0-best/axisB/v1.0/deliver/summary.json ] || echo "note: no lam 6.2e-3 delivery for $S, its probe self-check will be skipped"; done
for S in room bicycle; do [ -f /scratch/zr2245/$S-adapt1-t10mb/axisB/v1.0/deliver/summary.json ] || echo "note: no adapt-v1 10 MB delivery for $S, its probe self-check will be skipped"; done
for d in /scratch/zr2245/lego-adapt8b-t2.5mb /scratch/zr2245/chair-adapt8b-t2.5mb /scratch/zr2245/lego-adapt8b-t4mb /scratch/zr2245/chair-adapt8b-t4mb /scratch/zr2245/room-adapt8-t10mb /scratch/zr2245/bicycle-adapt8-t10mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt7|adapt8|redeliv3"); [ "$busy" = 0 ] || echo "note: $busy adapt/redeliver jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show node gl030 | grep -o "CfgTRES=[^ ]*"
echo "CHECK OK"; k=0
b () {{ EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt8b $AN/adapt-v8-blender.sbatch) && {{ echo "submitted $1 $2 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $2"; }}
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt8 $AN/adapt-v8.sbatch) && {{ echo "submitted $1 $5 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $5"; }}
b lego 2.5; b chair 2.5; m room 1 images_2 q14 10; m bicycle 0 images_4 q16 10; b lego 4; b chair 4
echo "submitted $k of 6 (4 GPUs: lego/chair 2.5 + room/bicycle 10 first, lego/chair 4 start as GPUs free up)"
SUB11
bash $AN/submit-batch11-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
```

## 2. 8–12 分钟后：自检 + smoke（四个带 smoke 的 job：lego 2.5、chair 2.5、room 10、bicycle 10）

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
for f in $AN/adapt8b-*-adapt8b-*.out $AN/adapt8-*-adapt8-*.out; do echo "=== $f"; grep -h "self-check\\|smoke probe-a\\|smoke probe scaling\\|smoke prune\\|smoke final\\|note:\\|FAILED\\|=== .* target" $f | tail -9 | cut -c1-700; done
```

要看到 `smoke probe-a: {{'iteration': 1500, ... 'b_v': X, 'overhead_mb': Y, 'source': 'probe_a@1500' ...}}`、`smoke prune: {{... 'iteration': 1550 ...}}`、`smoke final`，没有 FAILED。有 FAILED 把那个 `.out` 最后 40 行贴我，不要重交；同场景另一个目标的 job 可以 `scancel`。

## 3. ≈40 min 后 lego/chair 四行，≈2 h 后 room/bicycle 两行（一次贴全）

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-4hours -o JobID,JobName%22,State,Elapsed,ExitCode | grep -E "adapt8|JobID"
for f in $AN/adapt8b-*-adapt8b-*.out $AN/adapt8-*-adapt8-*.out; do echo "=== $f"; tail -2 $f | cut -c1-200; done
echo "===== adapt-v8-blender.txt"; cat $AN/adapt-v8-blender.txt
echo "===== adapt-v8.txt"; cat $AN/adapt-v8.txt
echo "===== phase A probes / probes / prune"; for d in /scratch/zr2245/*-adapt8b-t*mb /scratch/zr2245/*-adapt8-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\\[RD probe-a\\|^\\[RD adapt\\|^\\[RD prune\\|^Training is done" $d/train.log | cut -c1-420; done
echo "===== densify (V*, limiter) at 3500/6000/9500"; for d in /scratch/zr2245/*-adapt8b-t*mb /scratch/zr2245/*-adapt8-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\\[RD 3500\\]\\|^\\[RD 6000\\]\\|^\\[RD 9500\\]" $d/train.log | cut -c1-330; done
```

两张表的 6 行 + `phase A probes` 段 + `densify` 段贴回来。我按 §0 的 (a)–(e) 判：过 → batch12 的 33 个 Mip/T&T 重跑块（和 batch9 同一套场景参数，sbatch 就是本批的 `adapt-v8.sbatch`，不再改）。
'''
md = head + tr + mid1 + sb + mid2 + sm + submit
open(out, 'w').write(md)
print(f'wrote {out}: {len(md.splitlines())} lines | train {tr_n} {tr_md5} | blender sbatch {sb_n} {sb_md5} | mip sbatch {sm_n} {sm_md5}')
