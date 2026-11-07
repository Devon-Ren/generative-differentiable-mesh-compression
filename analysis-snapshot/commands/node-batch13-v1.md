# node-batch13-v1（2026-11-07）：adapt-v9 全场景 sweep —— 43 个 job（Mip/T&T 30 + Blender 13；batch12 已跑的 6 行不重跑，直接留在 adapt-v9*.txt 里）≈34 GPU-h，4 卡一夜

代码一个字不改：`train_rd_adapt_v9.py`（910 行，md5 a120351f64c9472e0353cef29d8ddb04）、`adapt-v9-blender.sbatch`（162 行，68a42385114634c478a721a58edf50b6）、`adapt-v9.sbatch`（154 行，df13ce37898f7dcc34a6c39ccd5d7dba）就是 batch12 用的三个文件，脚本先按 md5 核对。场景参数（INDOOR / IMAGES / XYZ / DATA）和 batch9-v2 完全一样；Mip/T&T 只有 5 MB 的 job 跑 smoke + 自检（其它目标 SMOKE=0,SELFCHECK=0，同 batch9），Blender 只有 2.5 MB 跑。已存在的 6 个 run 目录（lego/chair-adapt9b-t2.5mb、chair-adapt9b-t4mb、room-adapt9-t10mb/t20mb、bicycle-adapt9-t10mb）不在列表里。

## 1. 核对 + 提交 43 个 job（一次粘贴）

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-batch13-v1.sh <<'SUB13'
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (not the file batch12 wrote?)"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9.py 910 a120351f64c9472e0353cef29d8ddb04
chk $AN/adapt-v9-blender.sbatch 162 68a42385114634c478a721a58edf50b6
chk $AN/adapt-v9.sbatch 154 df13ce37898f7dcc34a6c39ccd5d7dba
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9.py || { echo "CHECK FAILED: train_rd_adapt_v9.py does not compile"; exit 1; }
bash -n $AN/adapt-v9.sbatch && bash -n $AN/adapt-v9-blender.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
n_ok=0; for r in lego-adapt9b-t2.5mb chair-adapt9b-t2.5mb chair-adapt9b-t4mb room-adapt9-t10mb bicycle-adapt9-t10mb room-adapt9-t20mb; do [ -f /scratch/zr2245/$r/axisB/v1.0/deliver/summary.json ] || [ -f /scratch/zr2245/$r-best/axisB/v1.0/deliver/summary.json ] && n_ok=$((n_ok+1)); done
[ "$n_ok" = 6 ] || { echo "CHECK FAILED: only $n_ok of the 6 batch12 deliveries exist; batch12 must be done and judged first"; exit 1; }
[ "$(grep -c '' $AN/adapt-v9.txt)" -ge 3 ] && [ "$(grep -c '' $AN/adapt-v9-blender.txt)" -ge 3 ] || { echo "CHECK FAILED: adapt-v9*.txt lack the batch12 rows"; exit 1; }
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in lego chair drums ficus hotdog materials mic ship; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in kitchen counter bonsai garden stump flowers treehill truck train; do [ -f /scratch/zr2245/$S-adapt1-t10mb/axisB/v1.0/deliver/summary.json ] || [ -f /scratch/zr2245/$S-adapt7-t5mb/axisB/v1.0/deliver/summary.json ] || echo "note: no earlier delivery for $S, its probe self-check will be skipped"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt7|adapt8|adapt9|redeliv3"); [ "$busy" = 0 ] || echo "note: $busy adapt/redeliver jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"
echo "CHECK OK"; k=0
m () { EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5$6"; [ "$5" = 5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9 $AN/adapt-v9.sbatch) && { echo "submitted $1 $5 MB $j"; k=$((k+1)); } || echo "FAILED: sbatch $1 $5"; }
b () { EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9b $AN/adapt-v9-blender.sbatch) && { echo "submitted $1 $2 MB $j"; k=$((k+1)); } || echo "FAILED: sbatch $1 $2"; }
mip () { for S in room kitchen counter bonsai; do case $S in room) X=q14;; kitchen) X=q16;; *) X=q15;; esac; m $S 1 images_2 $X $1; done; m garden 0 images_4 q15 $1; m bicycle 0 images_4 q16 $1; m stump 0 images_4 q16 $1; m flowers 0 images_4 q15 $1; m treehill 0 images_4 q16 $1; m truck 0 images q18 $1 ",DATA=$D/tandt/truck"; m train 0 images q18 $1 ",DATA=$D/tandt/train"; }
mip10 () { m kitchen 1 images_2 q16 10; m counter 1 images_2 q15 10; m bonsai 1 images_2 q15 10; m garden 0 images_4 q15 10; m stump 0 images_4 q16 10; m flowers 0 images_4 q15 10; m treehill 0 images_4 q16 10; m truck 0 images q18 10 ",DATA=$D/tandt/truck"; m train 0 images q18 10 ",DATA=$D/tandt/train"; }
mip20 () { m kitchen 1 images_2 q16 20; m counter 1 images_2 q15 20; m bonsai 1 images_2 q15 20; m garden 0 images_4 q15 20; m bicycle 0 images_4 q16 20; m stump 0 images_4 q16 20; m flowers 0 images_4 q15 20; m treehill 0 images_4 q16 20; m truck 0 images q18 20 ",DATA=$D/tandt/truck"; m train 0 images q18 20 ",DATA=$D/tandt/train"; }
bl25 () { for S in drums ficus hotdog materials mic ship; do b $S 2.5; done; }
bl4 () { for S in lego drums ficus hotdog materials mic ship; do b $S 4; done; }
mip 5; bl25; mip10; bl4; mip20
echo "submitted $k of 43 (4 GPUs: Slurm starts them in this order as GPUs free up; room/bicycle 10 and room 20 come from batch12)"
SUB13
bash $AN/submit-batch13-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R" | head -50
```

## 2. 10–15 分钟后：第一波 4 个 5 MB job 的自检 / smoke

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R" | head -12
for f in $AN/adapt9-*-5-adapt9-*.out; do echo "=== $f"; grep -h "self-check\|smoke v9 plan\|smoke probe-a\|smoke prune\|smoke final\|note:\|FAILED\|=== .* target" $f | tail -8 | cut -c1-600; done
```

有 FAILED 的 job 把 `.out` 最后 40 行贴我；同场景的其它目标 job 可以先 `scancel`。

## 3. 早上：全部结果（一次贴全）

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-14hours -o JobID,JobName%22,State,Elapsed,ExitCode | grep -E "adapt9|JobID"
echo "===== adapt-v9.txt"; cat $AN/adapt-v9.txt
echo "===== adapt-v9-blender.txt"; cat $AN/adapt-v9-blender.txt
echo "===== 9500 plan lines + 12k probe + done"; for d in /scratch/zr2245/*-adapt9-t*mb /scratch/zr2245/*-adapt9b-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\[RD 9500\]\|RD adapt 12000\] probe\|^Training is done" $d/train.log | cut -c1-330; done
```

两张表 + 这一段贴回来；我用 `adapt_analysis_v7b_sweep.py` 的 v9 版出 19 场景命中率 / 等字节 ΔPSNR / LaTeX 表，并和 v7b 的表并排。
