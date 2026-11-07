"""make_batch12.py: write analysis/commands/node-batch12-v1.md (one paste: three files by heredoc + submit script with checks + follow-ups)."""
import hashlib, sys
AN = sys.argv[1]; out = sys.argv[2]


def read(p):
    s = open(p).read(); assert s.endswith('\n'), p
    return s, len(s.splitlines()), hashlib.md5(s.encode()).hexdigest()


tr, tr_n, tr_md5 = read(AN + '/rdgrowth/train_rd_adapt_v9.py')
sb, sb_n, sb_md5 = read(AN + '/rd/adapt-v9-blender.sbatch')
sm, sm_n, sm_md5 = read(AN + '/rd/adapt-v9.sbatch')
for body, tag in ((tr, 'PYEOF_TRAIN9'), (sb, 'SBEOF_ADAPT9B'), (sm, 'SBEOF_ADAPT9')):
    assert tag not in body and 'SUB12' not in body

head = f'''# node-batch12-v1（2026-11-07）：adapt-v9 = v8 + phase-A 计划的两处计数修正 —— lego / chair 2.5 + chair 4 + room / bicycle 10 + room 20，6 个 job 一波（≈6.5 GPU-h，4 卡 ≈2.3 h 墙钟）

**batch11 判定**：(a)(b)(e) 过——soup 上量的 b_v 和 12k 网格的 b_v 相差 1–8 %，overhead 相差 < 1 %，probe 总开销 2–3 min；(c)(d) 不过——6 行全比 v7b 低（hit 0.71–0.92 vs 0.78–1.01）。**原因在 densify 日志里，不在 b_v**：

1. **计划多数了一个 interval**。`remaining = (until − it) // interval + 1` 在 9500 给 2，但 upstream 只在 `iteration < densify_until`（10000）时 densify，9500 就是最后一个（1500…9500 共 17 个，计划以为 18 个）。几何计划把还差的增长摊到"两个"interval，最后一个只长到 sqrt(V·V*)：6 个 run 最后一次 upstream prune 后的模型 = 0.81–0.92 V*，和 sqrt(V_9500/V*) = 0.83–0.92 一致（CPU 重放 6 条轨迹，旧计划预测误差 0.3–5 %）。v4–v7b 全部带着这个 bug（sweep 分析里"phase A 只到 V* 的 0.83–0.97"就是它）；v7b 的常数先验在 Blender 上把 V* 抬高 30 % 恰好盖住了，v8 把先验量准了，缺口就露出来。
2. **limiter 单位错了 3 倍**。upstream `add_new_gs` 每个 interval 最多分裂 add_pct·V − V 个**三角形**（rd_select 的 max_split 也是这个数），plan_splits 的前向模拟却把它除以 3 当成顶点数——以为 limiter 紧 3 倍，过早 front-load，而 front-load 时又只要了允许量的 1/3。4 MB Blender 行的"limiter-bound 12–13/17"大半是这个假象。

**v9**（`rdgrowth/train_rd_adapt_v9.py`，{tr_n} 行 = v8 887 行 + 头注释，代码 diff 4 行）：`remaining = ceil((until − it)/interval)`；plan_splits 的 limiter 用三角形数；densify 行打印 `cap N limiter M`（以前把 min(cap, limiter) 打成 limit，看起来像 limiter 卡住，其实是计划自己的上限），事件带 `limiter_max`。phase-A probe、phase B 一字不动。CPU 检查：生成脚本精确替换（11 处）、py_compile/pyflakes、intervals 公式对 5 种 schedule 穷举比对、6 条 batch11 轨迹重放（旧计划复现实测缺口，v9 在 9500 全部到 V*；从 3500 整段重放 6/6 到 1.000 V*、limiter 不卡）、smoke 断言块在假日志上正反各跑一次。

**sbatch**：`adapt-v9-blender.sbatch`（{sb_n} 行，VERSION v9b-r1）、`adapt-v9.sbatch`（{sm_n} 行，VERSION v9-r1）= v8 的两个 + 文件名 / 目录名 adapt9 / 表 adapt-v9*.txt / smoke 断言 densify 事件的 `remaining` 等于按 config 算的值且带 `limiter_max`。

**这 6 个 job**：lego/chair 2.5（机制 + 和 v7b/v8 直接比；预期 phase A 到 V*，先验在 Blender 偏高 3–10 % → 落点 1.00–1.10，phase B 这次要真砍）、chair 4（v8 0.797 capped，看 limiter 修正后是否还 capped）、room 10 / bicycle 10（v8 0.795 / 0.710；预期 ≈0.91 / ≈0.87，剩下的差是拓扑先验 5 B/V vs Mip 实测 3.7–4.5）、room 20（v7b 0.75，看 20 MB 档）。

**通过标准**：
- (a) 6 个 job exit 0，smoke 里有 `smoke v9 plan: {{... 'remaining': 17 ...}} | expected remaining 17`；
- (b) **计划到位**：每行 `prune rounds 16 ... (V M -> ...)` 的 M（最后一次 upstream prune 后的模型顶点数）≥ 0.95 × 9500 那行的 V*（`[RD 9500] ... = V* vertices`），limiter-bound 的行除外；
- (c) room 10 ≥ 0.88、bicycle 10 ≥ 0.85（v8 0.795 / 0.710），等字节 PSNR 比 v7b 不低超过 0.05 dB；
- (d) lego/chair 2.5 落点 0.95–1.05（phase B 砍过之后），PSNR 比 v7b 同目标行不低超过 0.05 dB；chair 4 ≥ 0.85；
- (e) room 20 ≥ 0.80（v7b 0.75）。
(b)(c) 过 → batch13 = 33 个 Mip/T&T 重跑（`adapt-v9.sbatch` 不再改，场景参数同 batch9-v2）+ Blender 16 行重跑（v9 的 phase B 在 Blender 上会动，v7b 那批不等价）。

## 1. 写三个文件 + 提交 6 个 job（一次粘贴；检查和提交都在脚本里，不碰你的终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/train_rd_adapt_v9.py <<'PYEOF_TRAIN9'
'''
mid1 = '''PYEOF_TRAIN9
cat > $AN/adapt-v9-blender.sbatch <<'SBEOF_ADAPT9B'
'''
mid2 = '''SBEOF_ADAPT9B
cat > $AN/adapt-v9.sbatch <<'SBEOF_ADAPT9'
'''
submit = f'''SBEOF_ADAPT9
cat > $AN/submit-batch12-v1.sh <<'SUB12'
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }}; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || {{ echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }}; }}
chk $AN/rdgrowth/train_rd_adapt_v9.py {tr_n} {tr_md5}
chk $AN/adapt-v9-blender.sbatch {sb_n} {sb_md5}
chk $AN/adapt-v9.sbatch {sm_n} {sm_md5}
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
grep -q "version='adapt-v9'" $AN/rdgrowth/train_rd_adapt_v9.py && grep -q "def update_attributes" $AN/rdgrowth/train_rd_adapt_v9.py || {{ echo "CHECK FAILED: train_rd_adapt_v9.py is not adapt-v9"; exit 1; }}
grep -q "max_split_up = max_split" $AN/rdgrowth/train_rd_adapt_v9.py && grep -q "remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))" $AN/rdgrowth/train_rd_adapt_v9.py || {{ echo "CHECK FAILED: train_rd_adapt_v9.py lacks the v9 plan fixes"; exit 1; }}
grep -q "VERSION v9b-r1" $AN/adapt-v9-blender.sbatch && grep -q "VERSION v9-r1" $AN/adapt-v9.sbatch || {{ echo "CHECK FAILED: sbatch files are not v9-r1"; exit 1; }}
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9.py $AN/blender/val_select_v3.py $AN/rdgrowth/probe_bytes_v1.py || {{ echo "CHECK FAILED: python files do not compile"; exit 1; }}
bash -n $AN/adapt-v9-blender.sbatch && bash -n $AN/adapt-v9.sbatch || {{ echo "CHECK FAILED: sbatch syntax"; exit 1; }}
for f in $AN/vis_collapse-v12/stage2_run.sh $AN/vis_collapse-v12/scripts/run_garden.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py $AN/blender/train_ckpt_v2.py $AN/coarse_finetune.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || {{ echo "CHECK FAILED: MISSING $f"; exit 1; }}; done
for S in lego chair; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in room bicycle; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in lego chair; do [ -f /scratch/zr2245/$S-rd3b4-lam0.0062-mu1.0-best/axisB/v1.0/deliver/summary.json ] || echo "note: no lam 6.2e-3 delivery for $S, its probe self-check will be skipped"; done
for S in room bicycle; do [ -f /scratch/zr2245/$S-adapt1-t10mb/axisB/v1.0/deliver/summary.json ] || echo "note: no adapt-v1 10 MB delivery for $S, its probe self-check will be skipped"; done
for d in /scratch/zr2245/lego-adapt9b-t2.5mb /scratch/zr2245/chair-adapt9b-t2.5mb /scratch/zr2245/chair-adapt9b-t4mb /scratch/zr2245/room-adapt9-t10mb /scratch/zr2245/bicycle-adapt9-t10mb /scratch/zr2245/room-adapt9-t20mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt7|adapt8|adapt9|redeliv3"); [ "$busy" = 0 ] || echo "note: $busy adapt/redeliver jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show node gl030 | grep -o "CfgTRES=[^ ]*"
echo "CHECK OK"; k=0
b () {{ EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9b $AN/adapt-v9-blender.sbatch) && {{ echo "submitted $1 $2 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $2"; }}
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9 $AN/adapt-v9.sbatch) && {{ echo "submitted $1 $5 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $5"; }}
b lego 2.5; b chair 2.5; m room 1 images_2 q14 10; m bicycle 0 images_4 q16 10; b chair 4; m room 1 images_2 q14 20
echo "submitted $k of 6 (4 GPUs: lego/chair 2.5 + room/bicycle 10 first, chair 4 + room 20 start as GPUs free up)"
SUB12
bash $AN/submit-batch12-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
```

## 2. 8–12 分钟后：自检 + smoke（五个带 smoke 的 job：lego 2.5、chair 2.5、room 10、bicycle 10、room 20）

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
for f in $AN/adapt9b-*-adapt9b-*.out $AN/adapt9-*-adapt9-*.out; do echo "=== $f"; grep -h "self-check\\|smoke v9 plan\\|smoke probe-a\\|smoke probe scaling\\|smoke prune\\|smoke final\\|note:\\|FAILED\\|=== .* target" $f | tail -9 | cut -c1-700; done
```

要看到 `smoke v9 plan: {{'iteration': 1500, 'remaining': 17, ...}} | expected remaining 17`、`smoke probe-a: {{... 'source': 'probe_a@1500' ...}}`、`smoke prune: {{... 'iteration': 1550 ...}}`、`smoke final`，没有 FAILED。有 FAILED 把那个 `.out` 最后 40 行贴我，不要重交；同场景另一个目标的 job 可以 `scancel`。

## 3. ≈40 min 后 lego/chair 三行，≈2.3 h 后 room/bicycle 三行（一次贴全）

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-4hours -o JobID,JobName%22,State,Elapsed,ExitCode | grep -E "adapt9|JobID"
for f in $AN/adapt9b-*-adapt9b-*.out $AN/adapt9-*-adapt9-*.out; do echo "=== $f"; tail -2 $f | cut -c1-200; done
echo "===== adapt-v9-blender.txt"; cat $AN/adapt-v9-blender.txt
echo "===== adapt-v9.txt"; cat $AN/adapt-v9.txt
echo "===== phase A probes / probes / prune"; for d in /scratch/zr2245/*-adapt9b-t*mb /scratch/zr2245/*-adapt9-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\\[RD probe-a\\|^\\[RD adapt\\|^\\[RD prune\\|^Training is done" $d/train.log | cut -c1-420; done
echo "===== densify (V*, limiter) at 3500/6000/9500"; for d in /scratch/zr2245/*-adapt9b-t*mb /scratch/zr2245/*-adapt9-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\\[RD 3500\\]\\|^\\[RD 6000\\]\\|^\\[RD 9500\\]" $d/train.log | cut -c1-330; done
```

两张表的 6 行 + `phase A probes` 段 + `densify` 段贴回来。我按上面的 (a)–(e) 判：过 → batch13（33 个 Mip/T&T + 16 个 Blender 重跑，sbatch 就是本批的两个，不再改）。
'''
md = head + tr + mid1 + sb + mid2 + sm + submit
open(out, 'w').write(md)
print(f'wrote {out}: {len(md.splitlines())} lines | train {tr_n} {tr_md5} | blender sbatch {sb_n} {sb_md5} | mip sbatch {sm_n} {sm_md5}')
