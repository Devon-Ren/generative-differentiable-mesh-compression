AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets; MODE=${1:-all}
case $MODE in all|mip|blender) ;; *) echo "FAILED: mode must be all|mip|blender, got '$MODE'"; exit 1;; esac
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (not the file the test batch used?)"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b.py 932 e379f9906f639d5d69acf127543ce690
chk $AN/adapt-v9b-blender.sbatch 173 c6d80dcb8d3af50754079363ffb96dfa
chk $AN/adapt-v9b.sbatch 165 c0f90652afe97b76d0d3148fe1243f15
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b.py || { echo "CHECK FAILED: train_rd_adapt_v9b.py does not compile"; exit 1; }
bash -n $AN/adapt-v9b.sbatch && bash -n $AN/adapt-v9b-blender.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in lego chair drums ficus hotdog materials mic ship; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"
echo "SWEEP CHECK OK (mode $MODE)"; k=0
m () { EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5$6"; [ "$5" = 5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9x $AN/adapt-v9b.sbatch) && { echo "submitted $1 $5 MB $j"; k=$((k+1)); } || echo "FAILED: sbatch $1 $5"; }
b () { EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9xb $AN/adapt-v9b-blender.sbatch) && { echo "submitted $1 $2 MB $j"; k=$((k+1)); } || echo "FAILED: sbatch $1 $2"; }
# the six test rows (room 10, bicycle 10, room 20, garden 5, truck 10, lego 2.5) are not resubmitted
mip5 () { for S in room kitchen counter bonsai; do case $S in room) X=q14;; kitchen) X=q16;; *) X=q15;; esac; m $S 1 images_2 $X 5; done; m bicycle 0 images_4 q16 5; m stump 0 images_4 q16 5; m flowers 0 images_4 q15 5; m treehill 0 images_4 q16 5; m truck 0 images q18 5 ",DATA=$D/tandt/truck"; m train 0 images q18 5 ",DATA=$D/tandt/train"; }
mip10 () { m kitchen 1 images_2 q16 10; m counter 1 images_2 q15 10; m bonsai 1 images_2 q15 10; m garden 0 images_4 q15 10; m stump 0 images_4 q16 10; m flowers 0 images_4 q15 10; m treehill 0 images_4 q16 10; m train 0 images q18 10 ",DATA=$D/tandt/train"; }
mip20 () { m kitchen 1 images_2 q16 20; m counter 1 images_2 q15 20; m bonsai 1 images_2 q15 20; m garden 0 images_4 q15 20; m bicycle 0 images_4 q16 20; m stump 0 images_4 q16 20; m flowers 0 images_4 q15 20; m treehill 0 images_4 q16 20; m truck 0 images q18 20 ",DATA=$D/tandt/truck"; m train 0 images q18 20 ",DATA=$D/tandt/train"; }
bl25 () { for S in chair drums ficus hotdog materials mic ship; do b $S 2.5; done; }
bl4 () { for S in lego chair drums ficus hotdog materials mic ship; do b $S 4; done; }
case $MODE in
  all) mip5; bl25; mip10; bl4; mip20; N=43;;
  mip) mip5; mip10; mip20; N=28;;
  blender) bl25; bl4; N=15;;
esac
echo "submitted $k of $N sweep jobs (mode $MODE) $(date)"
