AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9d.py 970 e950d66563e5eca3bb8aa18ee77eabe2
chk $AN/rdgrowth/rd_prune_v2.py 241 b500c019c4393d4ba8cfd87da100e1d3
chk $AN/adapt-v9e.sbatch 187 ef9a29c88d6f24b59dd9f2d2910556ff
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
grep -q "VERSION v9e-r1" $AN/adapt-v9e.sbatch && grep -q 'TAB=$AN/adapt-v9e.txt' $AN/adapt-v9e.sbatch && grep -q 'adapt9e-t${T}mb' $AN/adapt-v9e.sbatch || { echo "CHECK FAILED: adapt-v9e.sbatch names"; exit 1; }
grep -q "version='adapt-v9d'" $AN/rdgrowth/train_rd_adapt_v9d.py && grep -q "from rd_prune_v2 import" $AN/rdgrowth/train_rd_adapt_v9d.py || { echo "CHECK FAILED: train_rd_adapt_v9d.py is not adapt-v9d"; exit 1; }
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9d.py $AN/rdgrowth/rd_prune_v2.py || { echo "CHECK FAILED: python files do not compile"; exit 1; }
bash -n $AN/adapt-v9e.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
ls -d /scratch/zr2245/*-adapt9e-t*mb 2>/dev/null | head -3 | sed 's/^/note: exists (restart skips finished stages): /'
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt9|gate1"); [ "$busy" = 0 ] || echo "note: $busy adapt/gate jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0
m () { EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5,OVER=$6$7"; [ "$5" = 5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9e $AN/adapt-v9e.sbatch) && { echo "submitted $1 $5 MB OVER $6 $j"; k=$((k+1)); } || echo "FAILED: sbatch $1 $5"; }
# aim per capture class: indoor 1.16, outdoor 1.27, T&T 1.23 (calibrated on the v7b / v9 / v9b runs, not on these)
tier () { T=$1
  m room 1 images_2 q14 $T 1.16; m kitchen 1 images_2 q16 $T 1.16; m counter 1 images_2 q15 $T 1.16; m bonsai 1 images_2 q15 $T 1.16
  m garden 0 images_4 q15 $T 1.27; m bicycle 0 images_4 q16 $T 1.27; m stump 0 images_4 q16 $T 1.27; m flowers 0 images_4 q15 $T 1.27; m treehill 0 images_4 q16 $T 1.27
  m truck 0 images q18 $T 1.23 ",DATA=$D/tandt/truck"; m train 0 images q18 $T 1.23 ",DATA=$D/tandt/train"; }
tier 5; tier 10; tier 20
echo "submitted $k of 33 (4 GPUs, ~30 GPU-h, ~7.5 h wall; 5 MB jobs run smoke + self-check)"
