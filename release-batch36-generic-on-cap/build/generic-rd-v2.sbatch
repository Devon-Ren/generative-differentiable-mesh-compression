#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=genrd
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/genrd-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/genrd-%x-%j.err
# generic-rd-v1.sbatch (2026-11-13, VERSION genrd-v2: meshes trained with a vertex cap; text below is v1's): rate-distortion points of the generic codecs (quantised PLY + zstd, Draco, MPEG TFAN) on ONE scene.
# Meshes: the FP32 model, and the same model simplified after training and fine-tuned at three vertex counts (the POST-HOC-COLLAPSE controls of the
# cap jobs, $ORIG/axisB/g0ctrl-<B>/finetune/best.pt).  For each mesh: SH precision 8 / 6 / 4 bits -> sizes of the three codecs (CPU) and one render of
# the decoded mesh (GPU; the codecs are lossless after quantisation, so they share it), plus one render of the mesh before quantisation.
# Rows -> $AN/generic-rd.txt:  SCENE | mesh | V F | shB | zstd MB | draco MB | tfan MB | test PSNR SSIM LPIPS | unquantised PSNR
#   sbatch --export=ALL,SCENE=garden --job-name=garden-genrd-b33 .../generic-rd-v1.sbatch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; BITS=${BITS:-"8 6 4"}
AN=/scratch/zr2245/analysis; G=$AN/generic-v1; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; D=/scratch/zr2245/datasets; TAB=$AN/generic-rd.txt; DRV=$AN/o3dgc-v1/o3dgc_driver
case $SCENE in
  room)     ORIG=/scratch/zr2245/$SCENE-indoor; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=14; BUD="2198000 1020000 477000";;
  kitchen)  ORIG=/scratch/zr2245/$SCENE-indoor; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=16; BUD="1895000 928000 409000";;
  counter)  ORIG=/scratch/zr2245/$SCENE-indoor; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=15; BUD="2040000 978000 449000";;
  bonsai)   ORIG=/scratch/zr2245/$SCENE-indoor; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=15; BUD="2090000 1005000 473000";;
  garden)   ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=15; BUD="2015000 936000 405000";;
  bicycle)  ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=16; BUD="2029000 986000 440000";;
  stump)    ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=16; BUD="2015000 983000 440000";;
  flowers)  ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=15; BUD="2068000 992000 447000";;
  treehill) ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=16; BUD="2054000 1002000 449000";;
  truck)    ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/tandt/$SCENE; IMAGES=images; QP=18; BUD="1853000 929000 421000";;
  train)    ORIG=/scratch/zr2245/$SCENE-vis-collapse; DATA=$D/tandt/$SCENE; IMAGES=images; QP=18; BUD="2040000 975000 436000";;
  *) echo "FAILED: no settings for scene $SCENE"; exit 1;;
esac
case $SCENE in room) MIDCAP=1078000;; kitchen) MIDCAP=1086000;; counter) MIDCAP=1083000;; bonsai) MIDCAP=1073000;; garden) MIDCAP=1073000;; bicycle) MIDCAP=1080000;; stump) MIDCAP=1069000;; flowers) MIDCAP=1071000;; treehill) MIDCAP=1079000;; truck) MIDCAP=1078000;; train) MIDCAP=1085000;; esac; CAPS=${CAPS:-"600000 $MIDCAP 3000000"}
RUN=$ORIG/original; CK=$RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt; OUT=$AN/generic-rd/$SCENE; mkdir -p $OUT
for f in $CK $DATA/$IMAGES/. $G/generic_codec_v1.py $G/eval_ckpt_v1.py $AN/generic_rd_v1.py; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
python -c "import DracoPy, zstandard" 2>/dev/null || pip install DracoPy zstandard > $OUT/pip.log 2>&1 || { echo "FAILED: pip install DracoPy zstandard"; tail -5 $OUT/pip.log; exit 1; }
DOPT=""; [ -x $DRV ] && DOPT="--driver $DRV" || echo "note: no Open3DGC driver at $DRV, the TFAN column stays empty"
echo "generic-rd-v1 | scene $SCENE | original $ORIG | qp $QP | bits $BITS | cap-trained meshes $CAPS | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ev () { # $1 label  $2 ckpt  $3 out json
  [ -f $3 ] && return 0
  python $G/eval_ckpt_v1.py --lab $LAB --model $RUN --dataset $DATA --images $IMAGES --ckpt $2 --out $3 --label "$SCENE $1" > ${3%.json}.log 2>&1 && grep '^EVAL' ${3%.json}.log || { echo "FAILED: eval $1"; tail -6 ${3%.json}.log; return 1; }; }
nf=0
for M in $CAPS; do
  MK=/scratch/zr2245/$SCENE-budget-$M/axisB/v1.0/finetune/best.pt; [ -f $MK ] || MK=/scratch/zr2245/$SCENE-budget-$M/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  echo "mesh cap $M: $MK"
  [ -f $MK ] || { echo "note: $SCENE mesh $M missing ($MK), skipped"; continue; }
  O=$OUT/cap$M; mkdir -p $O
  echo "=== $SCENE mesh $M: sizes ($(date)) ==="
  python $AN/generic_rd_v1.py --ckpt $MK --qp $QP --out $O --bits $BITS $DOPT --generic $G --label "$SCENE $M" > $O/codec.log 2>&1 || { echo "FAILED: generic_rd_v1 $SCENE $M"; tail -8 $O/codec.log; nf=$((nf+1)); continue; }
  grep -h "ROW\|WARNING" $O/codec.log
  echo "=== $SCENE mesh $M: renders ($(date)) ==="
  ev "$M unquantised" $MK $O/ref.eval.json || nf=$((nf+1))
  for B in $BITS; do ev "$M sh$B" $O/sh$B/point_cloud_state_dict.pt $O/sh$B.eval.json || nf=$((nf+1)); done
  python - "$SCENE" "cap$M" "$O" <<'PY' | tee -a $TAB
import json, os, sys
S, M, O = sys.argv[1:4]; rows = json.load(open(O + '/rows.json')); ref = json.load(open(O + '/ref.eval.json')) if os.path.exists(O + '/ref.eval.json') else None
for k, r in sorted(rows.items(), key=lambda kv: -kv[1]['bits']):
    e = json.load(open(f'{O}/{k}.eval.json')) if os.path.exists(f'{O}/{k}.eval.json') else None
    q = (f"{e['psnr']:.3f} {e['ssim']:.4f} {e['lpips']:.4f}" if e else 'n/a')
    print(f"{S} GENERIC-RD | mesh {M} | V {r['vertices']} F {r['faces']} | sh{r['bits']} | zstd {r.get('zstd_MB', 'n/a')} MB | draco {r.get('draco_MB', 'n/a')} MB | tfan {r.get('tfan_MB', 'n/a')} MB | test {q} | unquantised {ref['psnr']:.3f}" if ref else f"{S} GENERIC-RD | mesh {M} | sh{r['bits']} | no reference render")
PY
  rm -f $O/sh*/point_cloud_state_dict.pt   # the decoded meshes are large and no longer needed
done
echo "generic-rd-v1 $SCENE exit 0 ($nf failures)  $(date)"
