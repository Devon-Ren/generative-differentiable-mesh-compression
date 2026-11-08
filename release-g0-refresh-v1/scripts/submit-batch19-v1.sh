AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
for f in $AN/adapt-v9d.sbatch $AN/rd-prune-v3c.sbatch $AN/rdgrowth/train_rd_adapt_v9d.py $AN/rdgrowth/rd_prune_v2.py $AN/rdgrowth/train_rd_v3.py $AN/coarse_finetune.py $AN/vis_collapse-v12/coarse_vis.py $AN/vis_collapse-v12/scripts/run_garden.sh $AN/vis_collapse-v12/deliver_coarse.sh; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
grep -q "VERSION v9d-r1" $AN/adapt-v9d.sbatch && grep -q "VERSION v3c" $AN/rd-prune-v3c.sbatch && grep -q "VERSION g0r-v2" $AN/budget-sweep-v2.sbatch || { echo "CHECK FAILED: sbatch versions"; exit 1; }
bash -n $AN/budget-sweep-v2.sbatch && bash -n $AN/adapt-v9d.sbatch && bash -n $AN/rd-prune-v3c.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
ls $AN/vis_collapse-v12/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "CHECK FAILED: native qem module missing (coarse_vis needs it)"; exit 1; }
for S in garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in room kitchen counter bonsai; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in drjohnson playroom; do [ -f $D/db/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for db/$S"; exit 1; }; done
ok=""; for S in truck train flowers treehill garden bicycle stump; do O=/scratch/zr2245/$S-vis-collapse
  if [ -f $O/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then ok="$ok $S"; [ -f $O/footprint-sens.npz ] || echo "note: $S original has no footprint-sens.npz yet (the job computes it, ~5 min)"; else echo "SKIP $S: no FP32 original at $O (G0 job not submitted for it)"; fi; done
echo "G0 scenes with an original:$ok"
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt9|g0r|rd3i|origi|gate1"); [ "$busy" = 0 ] || echo "note: $busy related jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0
sb () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})"; k=$((k+1)); } || echo "FAILED: sbatch $*"; }
# 1) G0 refresh: 7 outdoor / T&T scenes x 3 budgets (= V of the lambda 1.85e-3 / 6.2e-3 / 1.85e-2 assets / 0.9), train-at-budget + post-hoc-collapse control each; ~3.5 h per job
g0 () { S=$1; X=$2; BUD=$3; EXTRA=$4; case " $ok " in *" $S "*) ;; *) return;; esac
  IM=images_4; [ "$S" = truck ] || [ "$S" = train ] && IM=images
  sb --time=05:00:00 --export=ALL,SCENE=$S,IMAGES=$IM,XYZ=$X,INDOOR=0,BUDGETS="$BUD"$EXTRA --job-name=$S-g0r $AN/budget-sweep-v2.sbatch; }
g0 truck q18 "1934000 853000 354000" ",DATA=$D/tandt/truck"
g0 train q18 "2546000 1270000 553000" ",DATA=$D/tandt/train"
g0 flowers q15 "2639000 1227000 522000" ""
g0 treehill q16 "3053000 1245000 482000" ""
g0 garden q15 "2346000 919000 414000" ""
g0 bicycle q16 "2886000 1201000 474000" ""
g0 stump q16 "2292000 888000 344000" ""
# 2) Deep Blending: the two remaining fixed-lambda points (6.2e-4, 1.85e-2) -> full 4-point hull on the third dataset; ~2 h per job
for S in drjohnson playroom; do sb --time=03:30:00 --export=ALL,SCENE=$S,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/$S,LAMS="0.00062 0.0185" --job-name=$S-rd3i2 $AN/rd-prune-v3c.sbatch; done
# 3) adaptive v9d at 2.5 MB on the 11 Mip-360 / T&T scenes (the low end of the controller's range); ~35 min per job
m () { sb --time=01:30:00 --export=ALL,SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=2.5,SMOKE=0,SELFCHECK=0$5 --job-name=$1-2.5-adapt9d $AN/adapt-v9d.sbatch; }
m room 1 images_2 q14; m kitchen 1 images_2 q16; m counter 1 images_2 q15; m bonsai 1 images_2 q15; m garden 0 images_4 q15; m bicycle 0 images_4 q16; m stump 0 images_4 q16; m flowers 0 images_4 q15; m treehill 0 images_4 q16
m truck 0 images q18 ",DATA=$D/tandt/truck"; m train 0 images q18 ",DATA=$D/tandt/train"
echo "submitted $k of up to 20 jobs (7 G0 ~3.5 h each first, then 2 DB ~2 h, then 11 x 2.5 MB ~35 min; ~35 GPU-h, ~8.5-9 h on 4 GPUs)"
