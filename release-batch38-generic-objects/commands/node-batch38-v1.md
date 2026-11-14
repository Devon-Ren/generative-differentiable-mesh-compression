# node-batch38-v1 (2026-11-14): generic codecs on the 8 NeRF-Synthetic objects (appendix figure), 8 GPU jobs, no training

Not yet run on a GPU for this dataset: the render helper assumes a black background (as our object protocol) and the split 'train'; a job that fails does so at its first render, within minutes.

## One paste

```bash
AN=/scratch/zr2245/analysis
cat > $AN/make_batch38_files_v1.py <<'PYGEN38'
#!/usr/bin/env python3
"""make_batch38_files_v1.py (2026-11-14): the generic codecs on the NeRF-Synthetic objects.  Writes generic-rd-v3.sbatch from generic-rd-v2.sbatch:
adds the eight objects (black background, split 'train', positions q14, cap runs under <scene>-budgetb-<cap>-best) and lets a job also take
the model trained at the default cap ('full').  The captured scenes behave as in v2.
  python3 make_batch38_files_v1.py --sbatch <generic-rd-v2.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--sbatch', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); s = pathlib.Path(a.sbatch).read_text()
if md5(s) != 'e7e3bd69c56855efda4dbd86d7c4de7d': sys.exit(f'FAILED: {a.sbatch} md5 {md5(s)}')
OBJ = "  lego|chair|drums|ficus|hotdog|materials|mic|ship) ORIG=/scratch/zr2245/$SCENE-blender4-best; DATA=$D/nerf_synthetic/$SCENE; IMAGES=train; QP=14; BUD=\"\"; CDIR=budgetb; CSUF=-best;;\n"
s = rep(s, "  *) echo \"FAILED: no settings for scene $SCENE\"; exit 1;;\n", OBJ + "  *) echo \"FAILED: no settings for scene $SCENE\"; exit 1;;\n", 'object cases')
s = rep(s, "SCENE=${SCENE:?set SCENE via --export}; BITS=${BITS:-\"8 6 4\"}\n", "SCENE=${SCENE:?set SCENE via --export}; BITS=${BITS:-\"8 6 4\"}; CDIR=budget; CSUF=\"\"\n", 'defaults')
s = rep(s, "  MK=/scratch/zr2245/$SCENE-budget-$M/axisB/v1.0/finetune/best.pt; [ -f $MK ] || MK=/scratch/zr2245/$SCENE-budget-$M/original/point_cloud/iteration_30000/point_cloud_state_dict.pt\n",
        "  CG=/scratch/zr2245/$SCENE-$CDIR-$M$CSUF; MK=$CG/axisB/v1.0/finetune/best.pt; [ -f $MK ] || MK=$CG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; TAG=cap$M\n"
        "  if [ $M = full ]; then MK=$CK; TAG=full; fi\n", 'mesh path')
s = rep(s, "  O=$OUT/cap$M; mkdir -p $O\n", "  O=$OUT/$TAG; mkdir -p $O\n", 'out dir')
s = rep(s, "  python - \"$SCENE\" \"cap$M\" \"$O\" <<'PY' | tee -a $TAB\n", "  python - \"$SCENE\" \"$TAG\" \"$O\" <<'PY' | tee -a $TAB\n", 'row tag')
s = rep(s, "VERSION genrd-v2:", "VERSION genrd-v3 (adds the objects and 'full'); v2:", 'version')
(out / 'generic-rd-v3.sbatch').write_text(s); print(f'generic-rd-v3.sbatch: {len(s.splitlines())} lines md5 {md5(s)}')
PYGEN38
python3 $AN/make_batch38_files_v1.py --sbatch $AN/generic-rd-v2.sbatch --out $AN/batch38-build && cp $AN/batch38-build/generic-rd-v3.sbatch $AN/ || echo "GENERATOR FAILED: stop here"
cat > $AN/submit-batch38-v1.sh <<'SUB38'
#!/usr/bin/env bash
# submit-batch38-v1.sh (2026-11-14): zstd / Draco / MPEG TFAN on the 8 NeRF-Synthetic objects: the model trained at the default cap and three
# cap-trained meshes per object (existing cap runs), SH precision 8 / 6 / 4 bits.  No training.  8 GPU jobs of a few minutes.
AN=/scratch/zr2245/analysis
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/generic_rd_v1.py 46 17c568d6da5cfaf473cfe7067bf38b60
chk $AN/generic-rd-v3.sbatch 72 88b8342f718d39217f76889717b6a6e9
bash -n $AN/generic-rd-v3.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
caps () { case $1 in lego) echo "78124 183000 372025";; chair) echo "66755 178000 238956";; drums) echo "68000 176000 258000";; ficus) echo "35557 67614 171000";;
  hotdog) echo "47578 177000 222000";; materials) echo "68793 176000 259000";; mic) echo "46659 83900 174000";; ship) echo "74000 177000 294000";; esac; }
SC="lego chair drums ficus hotdog materials mic ship"; ft=0; raw=0; miss=0; full=0
for S in $SC; do [ -f /scratch/zr2245/$S-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && full=$((full+1)) || echo "note: $S: no default-cap model under /scratch/zr2245/$S-blender4-best"
  for B in $(caps $S); do G=/scratch/zr2245/$S-budgetb-$B-best
  if [ -f $G/axisB/v1.0/finetune/best.pt ]; then ft=$((ft+1)); elif [ -f $G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then raw=$((raw+1)); echo "note: $S cap $B has no fine-tuned mesh, the trained one will be used"; else miss=$((miss+1)); echo "note: $S cap $B: no mesh found under $G"; fi; done; done
echo "CHECK OK | default-cap models $full of 8 | cap meshes: $ft fine-tuned, $raw trained only, $miss missing (of 24)"; [ $miss -gt 6 ] || [ $full -lt 8 ] && { echo "STOPPED: meshes missing, nothing submitted"; exit 1; }
n=0
for S in $SC; do
  j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=8 --time=00:30:00 --export=ALL,SCENE=$S,CAPS="full $(caps $S)" --job-name=$S-genobj-b38 $AN/generic-rd-v3.sbatch) && { n=$((n+1)); echo "submitted $j  $S-genobj-b38"; } || echo "FAILED: sbatch $S"; done
echo "$n GPU jobs submitted"
SUB38
bash $AN/submit-batch38-v1.sh
```

## Collect

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch38-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== generic-rd rows (objects)"; grep -hE "^(lego|chair|drums|ficus|hotdog|materials|mic|ship) GENERIC-RD" $AN/generic-rd.txt
  echo; echo "===== which mesh was encoded"; grep -h "^mesh cap\|^note:\|FAILED\|exit 0" $AN/genrd-*-genobj-b38-*.out 2>/dev/null; for f in $(grep -l FAILED $AN/genrd-*-genobj-b38-*.out 2>/dev/null | head -2); do echo "--- $f"; grep -h -A 8 FAILED $f | head -24; done
  echo; echo "===== sacct"; sacct -X --starttime=now-4days --format=JobID,JobName%30,State,Elapsed,ExitCode | grep -E "b38"
  echo; echo "===== queue"; squeue -u $USER -h -o "%.10i %.28j %.2t %.10M %R" | grep -E "b38"; } > $O 2>&1; wc -l $O; echo $O
```
