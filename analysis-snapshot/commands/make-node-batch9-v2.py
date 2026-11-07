"""regenerates commands/node-batch9-v2.md (the scene sweep with the finalised adapt-v7) from the one file it embeds; md5 / line counts of the
files already on HPC (batch7 / batch8) are computed from the local copies and verified by the submit script."""
import hashlib
A = '/mnt/project-files/analysis/'
f_sb = open(A + 'rd/adapt-v7b.sbatch').read()
assert 'SBEOF_ADAPT7' not in f_sb and f_sb.endswith('\n') and '```' not in f_sb
n_sb = f_sb.count('\n')
def md5(p): return hashlib.md5(open(A + p, 'rb').read()).hexdigest()
def nl(p): return open(A + p).read().count('\n')
F = {'rdgrowth/train_rd_adapt_v7b.py': 'rdgrowth/train_rd_adapt_v7b.py', 'adapt-v7b-blender.sbatch': 'rd/adapt-v7b-blender.sbatch', 'adapt-v7b.sbatch': 'rd/adapt-v7b.sbatch',
     'redeliver-v3-blender.sbatch': 'rd/redeliver-v3-blender.sbatch', 'blender/val_select_v3.py': 'blender/val_select_v3.py', 'rdgrowth/probe_bytes_v1.py': 'rdgrowth/probe_bytes_v1.py'}
checks = '\n'.join(f'chk $AN/{h} {nl(l)} {md5(l)}' for h, l in F.items())
md = f'''# node-batch9-v2（2026-11-06）：定版 adapt-v7b 的场景 sweep —— 19 个场景、47 个 job、≈45 GPU-h，4 卡过夜

**v2**：文件名全部换成 batch8-v2 写上去的 v7b 版本（`train_rd_adapt_v7b.py`、`adapt-v7b-blender.sbatch`），Mip-360 / T&T 的 sbatch 改名 `adapt-v7b.sbatch`（VERSION v7-r2 = adapt-v7.sbatch + TR/grep/断言指向 v7b）；逻辑、场景参数、job 顺序与 v1 完全相同。

**只在 batch8（lego/chair × 2.5/4）四条都过了之后才提交。** 代码一个字不改：`train_rd_adapt_v7b.py` 和 `adapt-v7b-blender.sbatch` 就是 batch8 写上去的（提交脚本按 md5 核对），新增的只有 Mip-360 / T&T 的 `adapt-v7b.sbatch`（{n_sb} 行，VERSION v7 = adapt-v5.sbatch + v7 的同一套改动：train_rd_adapt_v7、OVER 1.0、prune 12000 起每 1000 到 19000 共 8 轮（这些场景每次 probe 30–60 s）、诊断 probe 22000/26000、v7b 的 smoke 断言和结果行）。

**跑什么**（每个 job 一个场景一个目标，按目标分波，先出 5 MB / 2.5 MB 的全场景结果）：
- Mip-360 / T&T 11 场景 × {{5, 10, 20}} MB = 33 job，每个 ≈70–100 min（20 MB 的大场景最长）；≈38 GPU-h。20 MB 在大场景上预期 `growth-capped True`（v4 已确认是 upstream add_percentage 的上限），照跑照报，它们排在最后一波。
- NeRF-Synthetic 另外 6 场景（drums ficus hotdog materials mic ship）× {{2.5, 4}} MB = 12 job，每个 ≈30 min；≈5 GPU-h。lego / chair 已在 batch8 里，不重跑。
- 8 场景固定 λ hull（λ 6.2e-4 / 6.2e-3）+ 8 个 FP32 原始模型用 val_select_v3 重交付（不重训）：2 job，≈2 GPU-h。论文 Blender 表的参考字节用这个。
- 每个场景的第一个 job（5 MB / 2.5 MB）带 self-check + smoke，同场景后面的目标 SMOKE=0,SELFCHECK=0（代码路径一样，省 ≈1.5 GPU-h）。
- 顺序：Mip/T&T 5 MB → Blender 2.5 → Mip/T&T 10 MB → Blender 4 → 重交付 → Mip/T&T 20 MB。4 卡 ≈11–12 h 墙钟：早上 8 h 时 5/10 MB 和 Blender 应全部结束，20 MB 大约过半，其余中午前完。

## 1. 写 adapt-v7b.sbatch + 提交 47 个 job（一次粘贴；先核对 batch8 的文件 md5，再提交）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/adapt-v7b.sbatch <<'SBEOF_ADAPT7'
{f_sb}SBEOF_ADAPT7
cat > $AN/submit-batch9-v2.sh <<'SUB9'
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }}; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || {{ echo "CHECK FAILED: $1 md5 $m != $3 (not the file batch8/batch7 wrote?)"; exit 1; }}; }}
{checks}
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v7b.py $AN/blender/val_select_v3.py $AN/rdgrowth/probe_bytes_v1.py || {{ echo "CHECK FAILED: python files do not compile"; exit 1; }}
bash -n $AN/adapt-v7b.sbatch && bash -n $AN/adapt-v7b-blender.sbatch && bash -n $AN/redeliver-v3-blender.sbatch || {{ echo "CHECK FAILED: sbatch syntax"; exit 1; }}
for f in $AN/vis_collapse-v12/stage2_run.sh $AN/vis_collapse-v12/scripts/run_garden.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py $AN/blender/train_ckpt_v2.py $AN/coarse_finetune.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || {{ echo "CHECK FAILED: MISSING $f"; exit 1; }}; done
n_ok=0; for r in lego-2.5 chair-2.5 lego-4 chair-4; do S=${{r%-*}}; T=${{r#*-}}; [ -f /scratch/zr2245/$S-adapt7b-t${{T}}mb-best/axisB/v1.0/deliver/summary.json ] && n_ok=$((n_ok+1)); done
[ "$n_ok" = 4 ] || {{ echo "CHECK FAILED: only $n_ok of the 4 batch8 testbed deliveries exist; do not sweep before batch8 is done and judged"; exit 1; }}
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in drums ficus hotdog materials mic ship; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
LAMS=""; ORIG=""; for S in chair drums ficus hotdog lego materials mic ship; do
  [ -d /scratch/zr2245/$S-blender2/original/point_cloud ] && ORIG="$ORIG $S-blender2" || echo "note: $S-blender2 has no checkpoints, original not re-delivered"
  [ "$S" = lego ] || [ "$S" = chair ] || for L in 0.00062 0.0062; do R=$S-rd3b4-lam$L-mu1.0; [ -d /scratch/zr2245/$R/original/point_cloud ] && LAMS="$LAMS $R" || echo "note: $R has no checkpoints, not re-delivered"; done; done
echo "re-deliver lambda runs:$LAMS"; echo "re-deliver originals:$ORIG"
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt7|redeliv3"); [ "$busy" = 0 ] || echo "note: $busy adapt7/redeliver jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"
echo "CHECK OK"; k=0
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5$6"; [ "$5" = 5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:00:00 --export=ALL,$EX --job-name=$1-$5-adapt7 $AN/adapt-v7b.sbatch) && {{ echo "submitted $1 $5 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $5"; }}
b () {{ EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:30:00 --export=ALL,$EX --job-name=$1-$2-adapt7b $AN/adapt-v7b-blender.sbatch) && {{ echo "submitted $1 $2 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $2"; }}
mip () {{ m room 1 images_2 q14 $1; m kitchen 1 images_2 q16 $1; m counter 1 images_2 q15 $1; m bonsai 1 images_2 q15 $1; m garden 0 images_4 q15 $1; m bicycle 0 images_4 q16 $1; m stump 0 images_4 q16 $1; m flowers 0 images_4 q15 $1; m treehill 0 images_4 q16 $1; m truck 0 images q18 $1 ",DATA=$D/tandt/truck"; m train 0 images q18 $1 ",DATA=$D/tandt/train"; }}
bl () {{ for S in drums ficus hotdog materials mic ship; do b $S $1; done; }}
mip 5; bl 2.5; mip 10; bl 4
[ -n "$LAMS" ] && {{ j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --export=ALL,RUNS="$LAMS",XYZ=q14 --job-name=refs-lams-redeliv3 $AN/redeliver-v3-blender.sbatch) && {{ echo "submitted refs-lams-redeliv3 $j"; k=$((k+1)); }} || echo "FAILED: sbatch redeliver lams"; }}
[ -n "$ORIG" ] && {{ j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --export=ALL,RUNS="$ORIG",XYZ=q14 --job-name=refs-orig-redeliv3 $AN/redeliver-v3-blender.sbatch) && {{ echo "submitted refs-orig-redeliv3 $j"; k=$((k+1)); }} || echo "FAILED: sbatch redeliver originals"; }}
mip 20
echo "submitted $k of 47 (4 GPUs: Slurm starts them in this order as GPUs free up)"
SUB9
bash $AN/submit-batch9-v2.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R" | head -60
```

## 2. 10–15 分钟后：第一波 4 个 job 的自检 / smoke

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R" | head -12
for f in $AN/adapt7-*-adapt7-*.out; do echo "=== $f"; grep -h "self-check\\|smoke prune\\|smoke final\\|note:\\|FAILED\\|=== .* target" $f | tail -6 | cut -c1-600; done
```

有 FAILED 的 job 把 `.out` 最后 40 行贴我；同场景的其它目标 job 可以先 `scancel`（它们会在第一轮 prune 处以同样的错误停下来）。

## 3. 早上：全部结果

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-14hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt7|redeliv3|JobID"
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
echo "===== FAILED jobs"; grep -l "FAILED" $AN/adapt7-*.out $AN/adapt7b-*.out $AN/redeliv3-*.out 2>/dev/null
echo "===== adapt-v7.txt (Mip-360 / T&T)"; cat $AN/adapt-v7.txt
echo "===== adapt-v7-blender.txt (NeRF-Synthetic, incl. batch8 lego/chair)"; cat $AN/adapt-v7-blender.txt
echo "===== redeliver-v3-blender.txt"; cat $AN/redeliver-v3-blender.txt
```

三张表贴回来（行很长，直接整段贴）。每行自带 `growth-capped`、limiter 计数、8/16 轮 probe/est、两次诊断 probe；我做全 19 场景的命中率表、等字节 ΔPSNR vs 固定 λ hull、growth-capped 清单，并给论文 Table 的 LaTeX。
'''
open(A + 'commands/node-batch9-v2.md', 'w').write(md)
print('md lines', md.count('\n'), '| sbatch', n_sb)
