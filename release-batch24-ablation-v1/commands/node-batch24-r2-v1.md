# node-batch24-r2-v1 (2026-11-11): round 2 on the remaining reservation hours (4 GPUs until 11/12 09:35 EST)

27 jobs, about 65 GPU-hours, all on the reservation (about 17 h on 4 GPUs; the reservation has about 23 h). Two purposes:

| jobs | what | why |
|---|---|---|
| `<scene>-clam3-abl24` (11) | third fixed price per Mip-360 / T&T scene, lam 0.00062 (files 20-36 MB) | the reference curve gets 3 points per scene: the 20 MB quality column stops being an extrapolation, and the interpolation error can be stated (reviewer W4) |
| `<object>-clam3-abl24b` (8) | third fixed price per NeRF-Synthetic object, lam 0.00185 (about 4 MB) | same for the 4 MB column |
| `<scene>-rep1/2-abl24`, `lego-rep1/2-abl24b` (8) | the full method re-run with seeds 1 and 2 (seed 0 = the paper's run): room, garden, truck at 10 MB, lego at 2.5 MB | run-to-run variance of file size and PSNR (reviewer Q6) |

Note on the reviewer's W1 ("error-only ranking without the byte term"): not run, because it is the same ordering by construction. Within a growth
interval every split costs the same number of bytes (three new vertices and faces), so ranking by error per byte and ranking by error pick the same
triangles; the byte term acts through the number of splits (the price set from the target) and through the collapse, where the bytes freed differ per edge.
The paper will say this in one sentence.

## §1 generate the round-2 files (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/make_batch24r2_files_v1.py <<'PYGEN24R2'
#!/usr/bin/env python3
"""make_batch24r2_files_v1.py (2026-11-11): round 2 of batch24 on the remaining reservation hours.
  python3 make_batch24r2_files_v1.py --abl_train <train_rd_adapt_v9b_abl.py> --abl_sbatch <ablation-b24.sbatch> --abl_sbatch_blender <ablation-b24-blender.sbatch> --out <dir>
Writes train_rd_adapt_v9b_abl2.py (= abl + an optional --seed that reseeds python/numpy/torch after upstream's safe_state; without the flag
nothing changes) and ablation-b24-r2.sbatch / ablation-b24-r2-blender.sbatch (= the batch24 sbatch files + the variant 'repeat': the full
method re-run with --seed SEED into $SCENE-abl24-repeat$SEED-t${T}mb, rows tagged abl24-repeat-s$SEED).  Inputs are md5-checked."""
import argparse, hashlib, pathlib, sys
IN = {'train': '2ae0a46b27aa135a02c1b56194e52e6d', 'sbatch': 'a823daddfa8374bffd8537e0ba896d75', 'blender': '741bd58b89a9e4b1e410b70a440e8896'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--abl_train', required=True); ap.add_argument('--abl_sbatch', required=True); ap.add_argument('--abl_sbatch_blender', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
t = pathlib.Path(a.abl_train).read_text()
if md5(t) != IN['train']: sys.exit(f'FAILED: {a.abl_train} md5 {md5(t)} != {IN["train"]} (not the batch24 file)')
t = rep(t, "    parser.add_argument('--hold_lambda', action='store_true',", "    parser.add_argument('--seed', type=int, default=None, help='batch24 round 2: reseed python / numpy / torch after upstream safe_state (which fixes seed 0); None = unchanged')\n    parser.add_argument('--hold_lambda', action='store_true',", 'seed arg')
t = rep(t, "    safe_state(args.quiet)\n", "    safe_state(args.quiet)\n    if args.seed is not None:\n        import random as _random, numpy as _np\n        _random.seed(args.seed); _np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); print('[RD] reseeded with', args.seed)\n", 'seed call')
(out / 'train_rd_adapt_v9b_abl2.py').write_text(t)
for key, src, name, dirtag in [('sbatch', a.abl_sbatch, 'ablation-b24-r2.sbatch', 'abl24'), ('blender', a.abl_sbatch_blender, 'ablation-b24-r2-blender.sbatch', 'abl24b')]:
    s = pathlib.Path(src).read_text()
    if md5(s) != IN[key]: sys.exit(f'FAILED: {src} md5 {md5(s)} != {IN[key]} (not the batch24 file)')
    s = rep(s, "ABL=${ABL:?set ABL=nocollapse|matched|constants|constlam via --export}", "ABL=${ABL:?set ABL=nocollapse|matched|constants|constlam|repeat via --export}", 'ABL prompt')
    s = rep(s, "  constants) PAE=0 ;;\n", "  constants) PAE=0 ;;\n  repeat) SEED=${SEED:?set SEED=<int> via --export}; XARGS=\"--seed $SEED\" ;;\n", 'repeat case')
    s = rep(s, "train_rd_adapt_v9b_abl.py;", "train_rd_adapt_v9b_abl2.py;", 'script path')
    s = rep(s, "elif [ \"$ABL\" = matched ]; then T=$RUN; L0=0; G=/scratch/zr2245/$SCENE-" + dirtag + "-matched-t${T}mb;", "elif [ \"$ABL\" = matched ]; then T=$RUN; L0=0; G=/scratch/zr2245/$SCENE-" + dirtag + "-matched-t${T}mb; elif [ \"$ABL\" = repeat ]; then T=$RUN; L0=$(lam0_of $T); G=/scratch/zr2245/$SCENE-" + dirtag + "-repeat$SEED-t${T}mb;", 'repeat dir')
    s = rep(s, "{('-lam' + L0) if os.environ['ABL'] == 'constlam' else ''}", "{('-lam' + L0) if os.environ['ABL'] == 'constlam' else ('-s' + os.environ.get('SEED', '')) if os.environ['ABL'] == 'repeat' else ''}", 'row tag')
    s = rep(s, "export ABL;", "export ABL; export SEED;", 'export seed')
    s = s.replace("VERSION abl24-r1", "VERSION abl24-r2").replace("VERSION abl24b-r1", "VERSION abl24b-r2")
    (out / name).write_text(s)
for f in ('train_rd_adapt_v9b_abl2.py', 'ablation-b24-r2.sbatch', 'ablation-b24-r2-blender.sbatch'):
    txt = (out / f).read_text(); print(f'{f}: {len(txt.splitlines())} lines md5 {md5(txt)}')
PYGEN24R2
python3 $AN/make_batch24r2_files_v1.py --abl_train $AN/rdgrowth/train_rd_adapt_v9b_abl.py --abl_sbatch $AN/ablation-b24.sbatch --abl_sbatch_blender $AN/ablation-b24-blender.sbatch --out $AN/batch24-build-r2 \
  && cp $AN/batch24-build-r2/train_rd_adapt_v9b_abl2.py $AN/rdgrowth/ && cp $AN/batch24-build-r2/ablation-b24-r2.sbatch $AN/batch24-build-r2/ablation-b24-r2-blender.sbatch $AN/ \
  && md5sum $AN/rdgrowth/train_rd_adapt_v9b_abl2.py $AN/ablation-b24-r2.sbatch $AN/ablation-b24-r2-blender.sbatch
```

Expected: `train_rd_adapt_v9b_abl2.py: 938 lines md5 75e92ffc397018fe472fc97801d0b9f9`, `ablation-b24-r2.sbatch: 203 lines md5 2adf57a024da31f1eb41781b174cb49b`, `ablation-b24-r2-blender.sbatch: 205 lines md5 b4315505e93cc9137141d81eb8a37f60`.

## §2 submit

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-batch24-r2-v1.sh <<'SUB24R2'
#!/usr/bin/env bash
# submit-batch24-r2-v1.sh (2026-11-11): round 2 on the remaining reservation hours (4 L40S until 11/12 09:35 EST), 27 jobs, ~65 GPU-h.
#  (a) third fixed price per scene for the reference curve: Mip/T&T lam 0.00062 (covers 20 MB), NeRF-Synthetic lam 0.00185 (covers 4 MB)
#      -> bounds the interpolation and gives the 20 MB / 4 MB quality columns (reviewer W4 / Q1)
#  (b) run-to-run variance of the full method: room, garden, truck at 10 MB and lego at 2.5 MB, seeds 1 and 2 (seed 0 = the paper's run) (Q6)
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl2.py 938 75e92ffc397018fe472fc97801d0b9f9
chk $AN/ablation-b24-r2.sbatch 203 2adf57a024da31f1eb41781b174cb49b
chk $AN/ablation-b24-r2-blender.sbatch 205 b4315505e93cc9137141d81eb8a37f60
bash -n $AN/ablation-b24-r2.sbatch && bash -n $AN/ablation-b24-r2-blender.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
grep -q "VERSION abl24-r2" $AN/ablation-b24-r2.sbatch && grep -q "VERSION abl24b-r2" $AN/ablation-b24-r2-blender.sbatch || { echo "CHECK FAILED: sbatch versions"; exit 1; }
for S in room garden truck; do [ -d /scratch/zr2245/$S-abl24-repeat1-t10mb ] && echo "note: $S repeat1 dir exists (restartable)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "abl24|g0r24"); [ "$busy" = 0 ] || echo "note: $busy batch24 jobs still in the queue"
scontrol show reservation $RES | grep -o "EndTime=[^ ]*\|NodeCnt=[^ ]*"
echo "CHECK OK"; k=0
sb () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})" >&2; echo $j; } || { echo "FAILED: sbatch $*" >&2; echo ""; }; }
acc () { [ -n "$1" ] && k=$((k+1)); }
cfg () { case $1 in
  room)     echo "1 images_2 q14 ''";;     kitchen)  echo "1 images_2 q16 ''";;   counter)  echo "1 images_2 q15 ''";;  bonsai)   echo "1 images_2 q15 ''";;
  garden)   echo "0 images_4 q15 ''";;     bicycle)  echo "0 images_4 q16 ''";;   stump)    echo "0 images_4 q16 ''";;  flowers)  echo "0 images_4 q15 ''";;
  treehill) echo "0 images_4 q16 ''";;     truck)    echo "0 images q18 ',DATA=$D/tandt/truck'";;  train)    echo "0 images q18 ',DATA=$D/tandt/train'";; esac; }
# (a) Mip/T&T third price, 11 jobs (the longest: 20-36 MB meshes), submitted first
for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train; do
  eval set -- $(cfg $S); B="SCENE=$S,INDOOR=$1,IMAGES=$2,XYZ=$3,SMOKE=0,SELFCHECK=0,SMOKE2=0$4"
  j=$(sb --time=04:30:00 --export=ALL,$B,ABL=constlam,LAMS="0.00062" --job-name=$S-clam3-abl24 $AN/ablation-b24-r2.sbatch); acc $j
done
# (b) repeats of the full method, seeds 1 and 2
for S in room garden truck; do eval set -- $(cfg $S); B="SCENE=$S,INDOOR=$1,IMAGES=$2,XYZ=$3,SMOKE=0,SELFCHECK=0$4"
  for SEED in 1 2; do j=$(sb --time=03:30:00 --export=ALL,$B,ABL=repeat,TARGETS=10,SEED=$SEED --job-name=$S-rep$SEED-abl24 $AN/ablation-b24-r2.sbatch); acc $j; done; done
for SEED in 1 2; do j=$(sb --time=02:30:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=repeat,TARGETS=2.5,SEED=$SEED --job-name=lego-rep$SEED-abl24b $AN/ablation-b24-r2-blender.sbatch); acc $j; done
# (a) NeRF-Synthetic third price, 8 jobs
for S in lego chair drums ficus hotdog materials mic ship; do
  j=$(sb --time=03:00:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,SMOKE2=0,ABL=constlam,LAMS="0.00185" --job-name=$S-clam3-abl24b $AN/ablation-b24-r2-blender.sbatch); acc $j
done
echo "submitted $k jobs (expected 27: 11 + 6 + 2 + 8), all on the reservation"
squeue -u $USER -o "%.10i %.22j %.2t %.10M %R" | head -40
SUB24R2
bash $AN/submit-batch24-r2-v1.sh
```

Expected: `CHECK OK`, 27 `submitted` lines.

## §3 collect (when the queue is empty)

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch24-r2-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== ablation-b24.txt (all rows)"; cat $AN/ablation-b24.txt; echo; echo "===== ablation-b24-blender.txt (all rows)"; cat $AN/ablation-b24-blender.txt; echo; echo "===== exits"; grep -h "exit 0\|FAILED" $AN/abl24-*clam3*.out $AN/abl24-*rep*.out $AN/abl24b-*clam3*.out $AN/abl24b-*rep*.out 2>/dev/null | sort | uniq -c; echo; echo "===== reseed lines"; grep -h "reseeded with" /scratch/zr2245/*-abl24*-repeat*-t*mb/train.log 2>/dev/null; echo; echo "===== sacct"; sacct -X --starttime=now-2days --format=JobID,JobName%28,State,Elapsed,ExitCode | grep -E "clam3|rep[12]-abl24"; } > $O 2>&1; wc -l $O; echo $O
```
