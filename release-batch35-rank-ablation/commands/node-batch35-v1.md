# node-batch35-v1 (2026-11-13): our two selection rules against MeshSplatting's importance score, inside the hold controller (4 GPU jobs)

## Question

Does it matter WHICH triangles are split and WHICH edges are collapsed, when the count and the file size are controlled the same way?

| variant | split: which triangles | collapse: which edges | everything else |
|---|---|---|---|
| hold (batch31, reference) | largest error reduction per byte | smallest error per byte freed | hold controller, 10 MB |
| holdis | upstream importance sampling | as hold | as hold |
| holdic | as hold | upstream importance score in place of the error | as hold |

Scenes: garden (outdoor), room (indoor). MeshSplatting has no edge collapse, so holdic puts its importance score into our collapse step.
Already measured on 18 scenes with the earlier controller (batch24, same split counts, no collapse): rate-aware ranking against importance sampling
+0.55 dB on captured scenes (11/11) and +0.39 dB on objects (7/7), with a larger file; against the full method at the same file size the
importance variant is -0.42 dB (captured) and -0.24 dB (objects).

## How the result will be read (fixed before the runs)

For each scene and variant: file size, PSNR / SSIM / LPIPS, and PSNR minus the hold run of the same scene (files are held at 10 MB, so the
comparison is at the same size; a variant that lands more than 2 % away is also compared on the hold curve of the scene at its own size).
One seed per run. A difference below 0.1 dB is reported as no measurable effect.

## Code under test

The two options are new and have not run on a GPU. The first densification interval exercises --split_rank within minutes, the first
collapse round (iteration 12,500) exercises --trim_score. If a job fails there, the others are unaffected.

## One paste (login node)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/make_batch35_files_v1.py <<'PYGEN35'
#!/usr/bin/env python3
"""make_batch35_files_v1.py (2026-11-13): ablation of the two selection rules of the hold controller against MeshSplatting's importance score.
  --split_rank importance : the triangles to split are drawn by upstream's importance-weighted sampling (the code path of the batch24 'matched'
                            runs); how MANY are split still follows the same plan, and the residual is still accumulated for the collapse rounds.
  --trim_score importance : in the collapse rounds upstream's per-triangle importance (largest blending weight seen) takes the place of the
                            residual in the cost of an edge; the byte target, the fold-over test and the rounds are unchanged.
Outputs train_rd_adapt_v9b_abl8.py and ablation-b24-r9.sbatch with ABL = holdis (split by importance) and holdic (collapse cost by importance);
ABL = hold is unchanged, and with both options at their default 'rd' the training script behaves exactly as abl7.
  python3 make_batch35_files_v1.py --train <train_rd_adapt_v9b_abl7.py> --sbatch <ablation-b24-r8.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = {'train': '1eeb64ab5032e903f69ad1a21466d99f', 'sbatch': '82259b73557bfcdb760648f42fad08cc'}
def md5(s): return hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--sbatch', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
s = pathlib.Path(a.train).read_text()
if md5(s) != IN['train']: sys.exit(f'FAILED: {a.train} md5 {md5(s)} != {IN["train"]}')
SEL = ("                    idx, st = rd_select(acc, triangles, lam_now, bytes_per_split, max_split,\n"
       "                                        triangles.size_probs_zero, triangles.size_probs_zero_image_space)\n")
s = rep(s, SEL, "                    acc_mode_keep = acc.mode\n"
        "                    if rd.get('split_rank') == 'importance': acc.mode = 'importance'  # batch35 ablation: upstream importance sampling picks the triangles; the planned count and the residual accumulation stay\n"
        + SEL + "                    acc.mode = acc_mode_keep\n", 'split rank')
DD = ("                dD = acc.distortion()\n"
      "                if dD.numel() != T0:\n"
      "                    raise RuntimeError(f'accumulator/face mismatch at prune {iteration}: {dD.numel()} vs {T0}')\n")
s = rep(s, DD, DD + "                if rd.get('trim_score') == 'importance':  # batch35 ablation: upstream's importance score takes the place of the residual in the collapse cost\n"
        "                    dD = triangles.importance_score.detach().reshape(-1).to(device=dD.device, dtype=dD.dtype)\n"
        "                    if dD.numel() != T0:\n"
        "                        raise RuntimeError(f'importance/face mismatch at prune {iteration}: {dD.numel()} vs {T0}')\n", 'trim score')
s = rep(s, "    parser.add_argument('--trim_margin', type=float, default=0.0,",
        "    parser.add_argument('--split_rank', choices=['rd', 'importance'], default='rd', help='batch35 ablation: importance = upstream importance sampling picks the triangles to split (count plan unchanged)')\n"
        "    parser.add_argument('--trim_score', choices=['rd', 'importance'], default='rd', help='batch35 ablation: importance = upstream importance score replaces the residual in the collapse cost')\n"
        "    parser.add_argument('--trim_margin', type=float, default=0.0,", 'arguments')
s = rep(s, "rd=dict(trim_hold=args.trim_hold,", "rd=dict(split_rank=args.split_rank, trim_score=args.trim_score, trim_hold=args.trim_hold,", 'rd dict')
(out / 'train_rd_adapt_v9b_abl8.py').write_text(s); print(f'train_rd_adapt_v9b_abl8.py: {len(s.splitlines())} lines md5 {md5(s)}')
HOLD = "  hold) XARGS=\"--trim_exact 5 --trim_hold --allow_any_prune_iter\"; OVER=1.35; PRUNE=\"$(seq -s \" \" 12500 500 29500)\"; EXTRA=\"\" ;;\n"
NEW = HOLD + HOLD.replace("  hold) ", "  holdis) ").replace("--allow_any_prune_iter\"", "--allow_any_prune_iter --split_rank importance\"") \
           + HOLD.replace("  hold) ", "  holdic) ").replace("--allow_any_prune_iter\"", "--allow_any_prune_iter --trim_score importance\"")
s = pathlib.Path(a.sbatch).read_text()
if md5(s) != IN['sbatch']: sys.exit(f'FAILED: {a.sbatch} md5 {md5(s)} != {IN["sbatch"]}')
s = rep(s, "|ratio|ratio25|ratio35|final|hold via --export}", "|ratio|ratio25|ratio35|final|hold|holdis|holdic via --export}", 'ABL prompt')
s = rep(s, HOLD, NEW, 'hold cases')
s = rep(s, "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl7.py;", "TR=$AN/rdgrowth/train_rd_adapt_v9b_abl8.py;", 'TR')
s = s.replace("VERSION abl24-r8", "VERSION abl24-r9")
(out / 'ablation-b24-r9.sbatch').write_text(s); print(f'ablation-b24-r9.sbatch: {len(s.splitlines())} lines md5 {md5(s)}')
PYGEN35
python3 $AN/make_batch35_files_v1.py --train $AN/rdgrowth/train_rd_adapt_v9b_abl7.py --sbatch $AN/ablation-b24-r8.sbatch --out $AN/batch35-build && cp $AN/batch35-build/train_rd_adapt_v9b_abl8.py $AN/rdgrowth/ && cp $AN/batch35-build/ablation-b24-r9.sbatch $AN/ || echo "GENERATOR FAILED: stop here"
cat > $AN/submit-batch35-v1.sh <<'SUB35'
#!/usr/bin/env bash
# submit-batch35-v1.sh (2026-11-13): the hold controller with MeshSplatting's importance score in place of one of our two selection rules.
#   holdis : the triangles to split are drawn by upstream importance sampling (how many are split: unchanged plan)
#   holdic : upstream's importance score replaces the residual in the cost of an edge collapse (byte target and fold-over test unchanged)
#   Scenes garden (outdoor) and room (indoor), target 10 MB.  4 GPU jobs.  Reference: the hold runs of batch31 (garden 9.997 MB, room 9.992 MB).
# Usage: bash submit-batch35-v1.sh [VARIANTS]   (default "holdis holdic")
AN=/scratch/zr2245/analysis; VAR=${1:-"holdis holdic"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl8.py 1092 0f6ca00cec9da3f1cf3da0f6d9f59bc5
chk $AN/ablation-b24-r9.sbatch 220 86899591acad64fe4783baab438a0093
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl8.py && bash -n $AN/ablation-b24-r9.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
echo "CHECK OK (variants: $VAR)"; n=0
cfg () { case $1 in room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; garden) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; esac; }
for V in $VAR; do
  case $V in holdis|holdic) ;; *) echo "FAILED: unknown variant $V"; continue;; esac
  for S in garden room; do
    [ -e /scratch/zr2245/$S-abl24-$V-t10mb ] && { echo "exists, skipped: $S $V"; continue; }
    j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS="10" --job-name=$S-$V-b35 $AN/ablation-b24-r9.sbatch) && { n=$((n+1)); echo "submitted $j  $S-$V-b35"; } || echo "FAILED: sbatch $S $V"
  done
done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
SUB35
bash $AN/submit-batch35-v1.sh
```

Expected: two lines from the generator (`train_rd_adapt_v9b_abl8.py: 1092 lines md5 0f6ca00c...`, `ablation-b24-r9.sbatch: 220 lines md5 86899591...`),
`CHECK OK (variants: holdis holdic)`, 4 `submitted` lines, `4 GPU jobs submitted`.

## Collect

```bash
AN=/scratch/zr2245/analysis; O=$AN/batch35-collect-$(date +%Y-%m-%d-%H%M).txt
{ echo "===== controller rows"; grep -h "^garden abl24-hold\|^room abl24-hold" $AN/ablation-b24.txt | grep "target 10 MB" | sed -E 's/, val_select kept faces.*//; s/ \| phase-A probes.*//; s/, rounds \(it, probe MB.*//; s/, extra probes.*//'
  echo; echo "===== failures"; grep -l "FAILED\|Traceback" $AN/abl24*-*-b35-*.out $AN/abl24*-*-b35-*.err 2>/dev/null; for f in $(grep -l "FAILED\|Traceback" $AN/abl24*-*-b35-*.out 2>/dev/null | head -4); do echo "--- $f"; grep -h -A 14 "FAILED\|Traceback" $f | head -40; done
  echo; echo "===== sacct"; sacct -X --starttime=now-2days --format=JobID,JobName%30,State,Elapsed,ExitCode | grep -E "b35"
  echo; echo "===== queue"; squeue -u $USER -h -o "%.10i %.28j %.2t %.10M %R" | grep -E "b35"; } > $O 2>&1; wc -l $O; echo $O
```
