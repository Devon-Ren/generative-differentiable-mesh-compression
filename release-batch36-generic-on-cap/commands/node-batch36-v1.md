# node-batch36-v1 (2026-11-13): generic codecs on cap-trained meshes (11 GPU jobs, no training)

Why: in batch33 the generic codecs were given the FP32 model simplified after training with OUR simplifier. Yao: reviewers will ask why not
vary the vertex cap and compress those meshes. Here zstd / Draco / TFAN encode meshes trained with a cap of 600,000, about 1.08 million and
3,000,000 (the FP32 model at the default cap is already in batch33). Our codec on the same cap runs exists (cap rows).

Reading, fixed before the runs: per scene and codec the points (MB, PSNR); means over the 11 scenes; the size each generic codec needs for
the PSNR of (a) our codec on the same cap meshes (codec alone) and (b) BudgetMesh. Whatever the ratios are, they replace the batch33 ones.

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/make_batch36_files_v1.py <<'PYGEN36'
#!/usr/bin/env python3
"""make_batch36_files_v1.py (2026-11-13): the generic codecs (zstd, Draco, MPEG TFAN) on meshes TRAINED with a vertex cap, instead of the FP32
model simplified after training (batch33).  Writes generic-rd-v2.sbatch from generic-rd-v1.sbatch: per scene three cap-trained meshes
(600,000 / the scene's cap near 1.08 million / 3,000,000), the fine-tuned mesh that our codec also encodes when it exists, else the trained one.
Rows go to the same table with 'mesh cap<N>'.   python3 make_batch36_files_v1.py --sbatch <generic-rd-v1.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--sbatch', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); s = pathlib.Path(a.sbatch).read_text()
if md5(s) != 'a9ad66a9e7bd9a740b82f5b7bff42681': sys.exit(f'FAILED: {a.sbatch} md5 {md5(s)}')
MID = {'room': 1078000, 'kitchen': 1086000, 'counter': 1083000, 'bonsai': 1073000, 'garden': 1073000, 'bicycle': 1080000, 'stump': 1069000, 'flowers': 1071000, 'treehill': 1079000, 'truck': 1078000, 'train': 1085000}
CASE = 'case $SCENE in ' + ' '.join('%s) MIDCAP=%d;;' % kv for kv in MID.items()) + ' esac; CAPS=${CAPS:-"600000 $MIDCAP 3000000"}\n'
s = rep(s, "RUN=$ORIG/original; CK=", CASE + "RUN=$ORIG/original; CK=", 'caps')
s = rep(s, "simplified at $BUD |", "cap-trained meshes $CAPS |", 'banner')
s = rep(s, "for M in full $BUD; do\n", "for M in $CAPS; do\n", 'loop')
s = rep(s, "  if [ $M = full ]; then MK=$CK; else MK=$ORIG/axisB/g0ctrl-$M/finetune/best.pt; fi\n",
        "  MK=/scratch/zr2245/$SCENE-budget-$M/axisB/v1.0/finetune/best.pt; [ -f $MK ] || MK=/scratch/zr2245/$SCENE-budget-$M/original/point_cloud/iteration_30000/point_cloud_state_dict.pt\n"
        "  echo \"mesh cap $M: $MK\"\n", 'mesh path')
s = rep(s, "  O=$OUT/$M; mkdir -p $O\n", "  O=$OUT/cap$M; mkdir -p $O\n", 'out dir')
s = rep(s, "  python - \"$SCENE\" \"$M\" \"$O\" <<'PY' | tee -a $TAB\n", "  python - \"$SCENE\" \"cap$M\" \"$O\" <<'PY' | tee -a $TAB\n", 'row tag')
s = rep(s, "VERSION genrd-v1)", "VERSION genrd-v2: meshes trained with a vertex cap; text below is v1's)", 'version')
(out / 'generic-rd-v2.sbatch').write_text(s); print(f'generic-rd-v2.sbatch: {len(s.splitlines())} lines md5 {md5(s)}')
PYGEN36
python3 $AN/make_batch36_files_v1.py --sbatch $AN/generic-rd-v1.sbatch --out $AN/batch36-build && cp $AN/batch36-build/generic-rd-v2.sbatch $AN/ || echo "GENERATOR FAILED: stop here"
cat > $AN/submit-batch36-v1.sh <<'SUB36'
#!/usr/bin/env bash
# submit-batch36-v1.sh (2026-11-13): zstd / Draco / MPEG TFAN on meshes trained with a vertex cap (600,000, about 1.08 million, 3,000,000),
# 11 Mip-NeRF 360 / T&T scenes, SH precision 8 / 6 / 4 bits.  No training: the cap runs exist.  11 GPU jobs (encoding on CPU, one render per point).
AN=/scratch/zr2245/analysis
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/generic_rd_v1.py 46 17c568d6da5cfaf473cfe7067bf38b60
chk $AN/generic-rd-v2.sbatch 70 e7e3bd69c56855efda4dbd86d7c4de7d
bash -n $AN/generic-rd-v2.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for f in $AN/generic-v1/generic_codec_v1.py $AN/generic-v1/eval_ckpt_v1.py; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
[ -x $AN/o3dgc-v1/o3dgc_driver ] || echo "note: no Open3DGC driver, the TFAN column will be empty"
mid () { case $1 in room) echo 1078000;; kitchen) echo 1086000;; counter) echo 1083000;; bonsai|garden) echo 1073000;; bicycle) echo 1080000;; stump) echo 1069000;; flowers) echo 1071000;; treehill) echo 1079000;; truck) echo 1078000;; train) echo 1085000;; esac; }
SC="garden room truck kitchen counter bonsai bicycle stump flowers treehill train"; ft=0; raw=0; miss=0
for S in $SC; do for B in 600000 $(mid $S) 3000000; do G=/scratch/zr2245/$S-budget-$B
  if [ -f $G/axisB/v1.0/finetune/best.pt ]; then ft=$((ft+1)); elif [ -f $G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then raw=$((raw+1)); echo "note: $S cap $B has no fine-tuned mesh, the trained one will be used"; else miss=$((miss+1)); echo "note: $S cap $B: no mesh found under $G"; fi; done; done
echo "CHECK OK | meshes: $ft fine-tuned, $raw trained only, $miss missing (of 33)"; [ $miss -gt 6 ] && { echo "STOPPED: too many meshes missing, nothing submitted"; exit 1; }
n=0
for S in $SC; do
  j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=8 --export=ALL,SCENE=$S --job-name=$S-gencap-b36 $AN/generic-rd-v2.sbatch) && { n=$((n+1)); echo "submitted $j  $S-gencap-b36"; } || echo "FAILED: sbatch $S"; done
echo "$n GPU jobs submitted"
SUB36
bash $AN/submit-batch36-v1.sh
```

## Collect

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch36-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== generic-rd rows (cap-trained meshes)"; grep -h "GENERIC-RD | mesh cap" $AN/generic-rd.txt
  echo; echo "===== which mesh was encoded"; grep -h "^mesh cap\|^note:\|FAILED\|exit 0" $AN/genrd-*-gencap-b36-*.out 2>/dev/null
  echo; echo "===== sacct"; sacct -X --starttime=now-2days --format=JobID,JobName%30,State,Elapsed,ExitCode | grep -E "b36"
  echo; echo "===== queue"; squeue -u $USER -h -o "%.10i %.28j %.2t %.10M %R" | grep -E "b36"; } > $O 2>&1; wc -l $O; echo $O
```
