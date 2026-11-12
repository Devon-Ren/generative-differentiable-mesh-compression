# node-batch27-v1 (2026-11-12): the three intuitive size-control baselines, one scene per dataset (fairshare, about 10 GPU-h)

Scenes and targets: garden (Mip-NeRF 360), truck (T&T), playroom (Deep Blending) at 10 MB; lego (NeRF-Synthetic) at 2.5 MB.

| baseline | what it is | runs |
|---|---|---|
| cap1 | MeshSplatting with a vertex cap chosen from a prior (file MB = a x cap + c, fitted on the Mip-360 / T&T cap runs with the test scene left out): garden 1,105,000, truck 1,078,000, playroom 1,081,000, lego 53,000 | 4 |
| cap2 | measure the cap1 file, rescale the cap (cap1 x target / file size), train and encode again; submitted automatically when cap1 ends | 4 |
| adaptcap | "update the cap as you go": our byte probe and growth plan set how many triangles split at each step, MeshSplatting's importance sampling chooses which; no collapse | 4 |

BudgetMesh's own rows for these four scene-targets already exist (garden 10.21 MB, truck 10.12, playroom 10.83, lego 2.50).

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/make_batch27_files_v1.py <<'PYGEN27'
#!/usr/bin/env python3
"""make_batch27_files_v1.py (2026-11-12): adds ONE variant to the batch24 round-2 sbatch files:
  adaptcap : the byte probe and growth plan decide HOW MANY triangles are split at each step (as in BudgetMesh), but WHICH triangles
             is MeshSplatting's own importance sampling (--dd_mode importance); no collapse rounds (they need the per-triangle error).
             = "measure the file as you go and update the vertex cap".  Run dir $SCENE-abl24-adaptcap-t${T}mb, rows tagged abl24-adaptcap.
  python3 make_batch27_files_v1.py --sbatch <ablation-b24-r2.sbatch> --sbatch_blender <ablation-b24-r2-blender.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'sbatch': '2adf57a024da31f1eb41781b174cb49b', 'blender': 'b4315505e93cc9137141d81eb8a37f60'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--sbatch', required=True); ap.add_argument('--sbatch_blender', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
for key, src, name in [('sbatch', a.sbatch, 'ablation-b24-r3.sbatch'), ('blender', a.sbatch_blender, 'ablation-b24-r3-blender.sbatch')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]} (not the batch24 round-2 file)')
    s = rep(s, "constlam|repeat via --export}", "constlam|repeat|adaptcap via --export}", 'ABL prompt')
    s = rep(s, "  constants) PAE=0 ;;\n", "  constants) PAE=0 ;;\n  adaptcap) PRUNE=\"\"; EXTRA=\"\"; DDMODE=importance ;;\n", 'adaptcap case')
    s = s.replace("VERSION abl24-r2", "VERSION abl24-r3").replace("VERSION abl24b-r2", "VERSION abl24b-r3")
    (out / name).write_text(s); print(f'{name}: {len(s.splitlines())} lines md5 {md5(s)}')
PYGEN27
python3 $AN/make_batch27_files_v1.py --sbatch $AN/ablation-b24-r2.sbatch --sbatch_blender $AN/ablation-b24-r2-blender.sbatch --out $AN/batch27-build && cp $AN/batch27-build/ablation-b24-r3.sbatch $AN/batch27-build/ablation-b24-r3-blender.sbatch $AN/ || echo "GENERATOR FAILED: stop here"
cat > $AN/cap-pass2-v1.sbatch <<'SBPASS2'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=cappass2
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/zr2245/analysis/cappass2-%x-%j.out
# cap-pass2-v1.sbatch (2026-11-12, VERSION cappass2-v1): second pass of the "measure, then retrain" vertex-cap baseline.  Runs after the first cap
# job of scene S: reads the delivered file size of the cap-CAP1 run from the result table, rescales the cap to the target (CAP2 = CAP1 * T / MB1,
# rounded to 1000) and submits the same cap job with CAP2.  Needs via --export: S, CAP1, T.  Prints one line 'PASS2 S cap1 mb1 -> cap2'.
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
S=${S:?}; CAP1=${CAP1:?}; T=${T:?}
case $S in lego) TAB=$AN/g0-refresh-blender-$S.txt; TAG="TRAIN-AT-BUDGET-BLENDER";; *) TAB=$AN/g0-refresh-$S.txt; TAG="TRAIN-AT-BUDGET";; esac
MB1=$(grep "^$S budget $CAP1 $TAG " $TAB | tail -1 | sed -n 's/.*| delivered \([0-9.]*\) MB.*/\1/p')
[ -n "$MB1" ] || { echo "FAILED: no '$S budget $CAP1 $TAG' row in $TAB"; exit 1; }
CAP2=$(python3 -c "import sys; c=int(sys.argv[1]); mb=float(sys.argv[2]); t=float(sys.argv[3]); print(int(round(c*t/mb/1000.0))*1000)" $CAP1 $MB1 $T)
echo "PASS2 $S cap1 $CAP1 delivered $MB1 MB (target $T MB, error $(python3 -c "print('%+.1f%%' % (100*($MB1/$T-1)))")) -> cap2 $CAP2"
[ "$CAP2" != "$CAP1" ] || { echo "cap2 equals cap1: no second pass needed"; exit 0; }
case $S in
  garden)   sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=02:00:00 --export=ALL,SCENE=garden,IMAGES=images_4,XYZ=q15,INDOOR=0,BUDGETS=$CAP2 --job-name=garden-cap2-b27 $AN/budget-sweep-v2.sbatch;;
  truck)    sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=02:00:00 --export=ALL,SCENE=truck,IMAGES=images,XYZ=q18,INDOOR=0,DATA=$D/tandt/truck,BUDGETS=$CAP2 --job-name=truck-cap2-b27 $AN/budget-sweep-v2.sbatch;;
  playroom) sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS=$CAP2 --job-name=playroom-cap2-b27 $AN/budget-sweep-v3.sbatch;;
  lego)     sbatch --account=torch_pr_38_general --cpus-per-task=6 --time=02:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=$CAP2 --job-name=lego-cap2-b27 $AN/budget-sweep-blender-v1.sbatch;;
  *) echo "FAILED: unknown scene $S"; exit 1;;
esac
SBPASS2
cat > $AN/submit-batch27-v1.sh <<'SUB27'
#!/usr/bin/env bash
# submit-batch27-v1.sh (2026-11-12): the three intuitive size-control baselines on ONE scene per dataset (garden, truck, playroom at T = 10 MB;
# lego at T = 2.5 MB), public partition (fairshare), 12 GPU runs, about 10 GPU-h.
#   cap1  : MeshSplatting with a vertex cap chosen from a prior (file MB = a * cap + c fitted on the Mip-360 / T&T cap runs, the test scene left out)
#   cap2  : measure the cap1 file, rescale the cap, train and encode again (submitted automatically by cap-pass2 after cap1)
#   adapt : "update the cap as you go": the byte probe + growth plan set the number of splits per step, MeshSplatting's importance sampling
#           picks the triangles, no collapse (ablation-b24-r3 sbatch, ABL=adaptcap)
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/ablation-b24-r3.sbatch 204 949703fb20c0b3c5ba58a8d3e53eb21c
chk $AN/ablation-b24-r3-blender.sbatch 206 69614021a3aaf74cc7f999697ed8be54
chk $AN/rdgrowth/train_rd_adapt_v9b_abl2.py 938 75e92ffc397018fe472fc97801d0b9f9
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/cap-pass2-v1.sbatch 26 63398b4a79453d26734ccfe0bcf2fb32
bash -n $AN/ablation-b24-r3.sbatch && bash -n $AN/ablation-b24-r3-blender.sbatch && bash -n $AN/cap-pass2-v1.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
[ -f /scratch/zr2245/playroom-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no playroom --indoor original (cap job needs it)"; exit 1; }
echo "CHECK OK"; k=0
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})" >&2; k=$((k+1)); echo $j; } || { echo "FAILED: sbatch $*" >&2; echo ""; }; }
p2 () { [ -n "$1" ] && sbatch --parsable --dependency=afterok:$1 --export=ALL,S=$2,CAP1=$3,T=$4 --job-name=$2-pass2 $AN/cap-pass2-v1.sbatch >/dev/null && echo "  pass-2 launcher queued for $2 (after $1)"; }
# cap1 (prior-chosen caps: garden 1105000, truck 1078000, playroom 1081000, lego 53000) + pass-2 launchers
j=$(sb --time=02:00:00 --export=ALL,SCENE=garden,IMAGES=images_4,XYZ=q15,INDOOR=0,BUDGETS=1105000 --job-name=garden-cap1-b27 $AN/budget-sweep-v2.sbatch); p2 "$j" garden 1105000 10
j=$(sb --time=02:00:00 --export=ALL,SCENE=truck,IMAGES=images,XYZ=q18,INDOOR=0,DATA=$D/tandt/truck,BUDGETS=1078000 --job-name=truck-cap1-b27 $AN/budget-sweep-v2.sbatch); p2 "$j" truck 1078000 10
j=$(sb --time=02:30:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS=1081000 --job-name=playroom-cap1-b27 $AN/budget-sweep-v3.sbatch); p2 "$j" playroom 1081000 10
j=$(sb --time=02:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=53000 --job-name=lego-cap1-b27 $AN/budget-sweep-blender-v1.sbatch); p2 "$j" lego 53000 2.5
# adaptive cap
sb --time=02:00:00 --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,SMOKE=0,SELFCHECK=0,ABL=adaptcap,TARGETS=10 --job-name=garden-adapt-b27 $AN/ablation-b24-r3.sbatch >/dev/null
sb --time=02:00:00 --export=ALL,SCENE=truck,INDOOR=0,IMAGES=images,XYZ=q18,SMOKE=0,SELFCHECK=0,DATA=$D/tandt/truck,ABL=adaptcap,TARGETS=10 --job-name=truck-adapt-b27 $AN/ablation-b24-r3.sbatch >/dev/null
sb --time=02:30:00 --export=ALL,SCENE=playroom,INDOOR=1,IMAGES=images,XYZ=q15,SMOKE=0,SELFCHECK=0,DATA=$D/db/playroom,ABL=adaptcap,TARGETS=10 --job-name=playroom-adapt-b27 $AN/ablation-b24-r3.sbatch >/dev/null
sb --time=02:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=adaptcap,TARGETS=2.5 --job-name=lego-adapt-b27 $AN/ablation-b24-r3-blender.sbatch >/dev/null
echo "8 GPU jobs submitted now (4 cap1 + 4 adaptcap); 4 cap2 jobs are submitted by the pass-2 launchers when each cap1 finishes"
squeue -u $USER -o "%.10i %.22j %.2t %.10M %R" | head -20
SUB27
bash $AN/submit-batch27-v1.sh
```

Expected: two generator lines (`ablation-b24-r3.sbatch: 204 lines md5 949703fb20c0b3c5ba58a8d3e53eb21c`, `ablation-b24-r3-blender.sbatch: 206 lines md5 69614021a3aaf74cc7f999697ed8be54`), then `CHECK OK`, 8 `submitted` lines and 4 `pass-2 launcher queued` lines.

## Collect (when `squeue --me | grep -c "b27\|pass2"` is 0)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch27-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== pass-2 launchers"; grep -h "PASS2\|FAILED" $AN/cappass2-*.out; echo; echo "===== cap rows"; for S in garden truck playroom; do grep -h "TRAIN-AT-BUDGET" $AN/g0-refresh-$S.txt | tail -4; done; grep -h "TRAIN-AT-BUDGET-BLENDER" $AN/g0-refresh-blender-lego.txt | tail -4; echo; echo "===== adaptcap rows"; grep -h "abl24-adaptcap" $AN/ablation-b24.txt $AN/ablation-b24-blender.txt; echo; echo "===== sacct"; sacct -X --starttime=now-3days --format=JobID,JobName%26,State,Elapsed,ExitCode | grep -E "b27|pass2"; } > $O 2>&1; wc -l $O; echo $O
```
