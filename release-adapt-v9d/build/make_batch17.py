"""make_batch17.py: commands/node-batch17-v1.md = one paste: adapt-v9e.sbatch (heredoc) + submit script (md5 checks of the v9d trainer / rd_prune_v2 /
v9e sbatch, 33 Mip-360 / T&T jobs with the calibrated aim per class) + morning collection block."""
import hashlib, os, sys
R = sys.argv[1]; out = sys.argv[2]


def read(p):
    s = open(p).read(); assert s.endswith('\n'), p
    return s, len(s.splitlines()), hashlib.md5(s.encode()).hexdigest()


tr, tr_n, tr_md5 = read(R + '/code/rdgrowth/train_rd_adapt_v9d.py'); pr, pr_n, pr_md5 = read(R + '/code/rdgrowth/rd_prune_v2.py'); se, se_n, se_md5 = read(R + '/sbatch/adapt-v9e.sbatch')
IN, OUT_, TT = '1.16', '1.27', '1.23'
SUBMIT = f'''AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }}; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || {{ echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }}; }}
chk $AN/rdgrowth/train_rd_adapt_v9d.py {tr_n} {tr_md5}
chk $AN/rdgrowth/rd_prune_v2.py {pr_n} {pr_md5}
chk $AN/adapt-v9e.sbatch {se_n} {se_md5}
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
grep -q "VERSION v9e-r1" $AN/adapt-v9e.sbatch && grep -q 'TAB=$AN/adapt-v9e.txt' $AN/adapt-v9e.sbatch && grep -q 'adapt9e-t${{T}}mb' $AN/adapt-v9e.sbatch || {{ echo "CHECK FAILED: adapt-v9e.sbatch names"; exit 1; }}
grep -q "version='adapt-v9d'" $AN/rdgrowth/train_rd_adapt_v9d.py && grep -q "from rd_prune_v2 import" $AN/rdgrowth/train_rd_adapt_v9d.py || {{ echo "CHECK FAILED: train_rd_adapt_v9d.py is not adapt-v9d"; exit 1; }}
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9d.py $AN/rdgrowth/rd_prune_v2.py || {{ echo "CHECK FAILED: python files do not compile"; exit 1; }}
bash -n $AN/adapt-v9e.sbatch || {{ echo "CHECK FAILED: sbatch syntax"; exit 1; }}
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
ls -d /scratch/zr2245/*-adapt9e-t*mb 2>/dev/null | head -3 | sed 's/^/note: exists (restart skips finished stages): /'
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt9|gate1"); [ "$busy" = 0 ] || echo "note: $busy adapt/gate jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5,OVER=$6$7"; [ "$5" = 5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9e $AN/adapt-v9e.sbatch) && {{ echo "submitted $1 $5 MB OVER $6 $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $5"; }}
# aim per capture class: indoor {IN}, outdoor {OUT_}, T&T {TT} (calibrated on the v7b / v9 / v9b runs, not on these)
tier () {{ T=$1
  m room 1 images_2 q14 $T {IN}; m kitchen 1 images_2 q16 $T {IN}; m counter 1 images_2 q15 $T {IN}; m bonsai 1 images_2 q15 $T {IN}
  m garden 0 images_4 q15 $T {OUT_}; m bicycle 0 images_4 q16 $T {OUT_}; m stump 0 images_4 q16 $T {OUT_}; m flowers 0 images_4 q15 $T {OUT_}; m treehill 0 images_4 q16 $T {OUT_}
  m truck 0 images q18 $T {TT} ",DATA=$D/tandt/truck"; m train 0 images q18 $T {TT} ",DATA=$D/tandt/train"; }}
tier 5; tier 10; tier 20
echo "submitted $k of 33 (4 GPUs, ~30 GPU-h, ~7.5 h wall; 5 MB jobs run smoke + self-check)"
'''
for body, tag in ((se, 'SBEOF_ADAPT9E'), (SUBMIT, 'SUB17')):
    assert tag not in body and '```' not in body and 'SBEOF_ADAPT9E' not in SUBMIT
md = f'''# node-batch17-v1 (2026-11-08 evening): the calibrated-aim sweep, 33 Mip-360 / T&T jobs with v9d + per-class aim (no gate, no code change)

**What batch16 showed** (v9d test, 5 rows + lego): the priced collapse works as designed, both-kept share 1.00 on every row and 8.6-10.4 B freed per pair (v9c: 0.6-2.4), kitchen 10 removed 0.78 MB of real bytes. The rows still landed 1.047 / 1.046 / 1.130 / 1.070 / 1.088, because the kept set keeps un-stacking during phase B: (final probe + bytes removed) / first probe = 1.10 (room 5), 1.13 (counter 5), 1.24 (counter 10), 1.28 (kitchen 10), 1.49 (kitchen 20). Thirty rounds at ~0.4 % of V per round (independent set + foldover) cannot trim that. Phase B is not the lever for a 10-30 % surplus; the aim is.

**The aim, measured two independent ways.** From the 59 earlier runs (kappa-analysis-v1: cleaning share x delivered topology price x soup over-read) the aim that lands on target is indoor 1.11-1.16, outdoor 1.23-1.28, T&T 1.17-1.22. From the v9b sweep's own pre-trim landing with the uniform 1.25: indoor 1.076 -> 1.16, outdoor 0.988 -> 1.27, T&T 1.020 -> 1.23. The uniform 1.25 was my choice to avoid class constants; the data say the class constant is real (indoor meshes un-stack more and carry a higher topology price). This sweep uses **indoor 1.16, outdoor 1.27, T&T 1.23**, calibrated on earlier runs, not on these; they are reported in the paper as three calibration numbers next to B_a / B_f.

**Files**: `adapt-v9e.sbatch` ({se_n} lines, md5 {se_md5}, VERSION v9e-r1) = the v9d sbatch with new names only (jobs `adapt9e`, run dirs `$SCENE-adapt9e-t$T mb`, table `adapt-v9e.txt`); trainer `train_rd_adapt_v9d.py` ({tr_n} lines, md5 {tr_md5}) and `rd_prune_v2.py` ({pr_n} lines, md5 {pr_md5}) must be the files batch16 wrote (md5-checked). OVER is passed per job. The Blender table is the batch16 v9d sweep (16 rows, OVER 1.1) and is not rerun.

**Expected**: 33 rows within +-5 % on ~28, +-10 % on 33; indoor rows 0.97-1.05 with phase B active and trimming 2-5 %; outdoor 20 MB rows up from 0.93-0.96 to ~0.98-1.02.

## 1. Write the sbatch + submit 33 jobs (one paste)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/adapt-v9e.sbatch <<'SBEOF_ADAPT9E'
{se}SBEOF_ADAPT9E
cat > $AN/submit-batch17-v1.sh <<'SUB17'
{SUBMIT}SUB17
bash $AN/submit-batch17-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R" | head -40
```

## 2. After 10-15 min: smoke of the four 5 MB jobs in the first wave

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R" | head -40
for f in $AN/adapt9e-*-adapt9e-*.out; do echo "=== $f"; grep -h "phase A aims at\\|self-check\\|smoke prune\\|smoke final\\|FAILED" $f | tail -4 | cut -c1-300; done
```

Expect `phase A aims at 5.80` (indoor 5 MB), `6.35` (outdoor), `6.15` (T&T).

## 3. Morning: everything to one file, upload it

```bash
AN=/scratch/zr2245/analysis; OUT=$AN/batch17-results-$(date +%m%d-%H%M).txt
{{
echo "===== sacct"; sacct -X -u $USER -S now-16hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt9e|JobID"
echo "===== squeue"; squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
echo "===== adapt-v9e.txt"; cat $AN/adapt-v9e.txt
echo "===== kappa (v9e rows)"; python3 $AN/rdgrowth/kappa_analysis_v1.py '/scratch/zr2245/*-adapt9e-t*mb' 2>&1
echo "===== train logs"; for d in /scratch/zr2245/*-adapt9e-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\\[RD 9500\\]\\|phase A ended\\|RD adapt 1[2-9][0-9]*\\] probe\\|^\\[RD prune [0-9]*\\] V\\|no collapse this round\\|^Training is done" $d/train.log | sed -E 's/\\(lambda-prune held; [0-9]+ edges below lam [0-9.e+-]+\\)//; s/ \\| score q10.*//; s/\\[0[0-9]\\/11 [0-9:]+\\]//' | cut -c1-330; done
echo "===== FAILED lines in job outputs"; grep -l "FAILED" $AN/adapt9e-*.out 2>/dev/null | while read f; do echo "--- $f"; grep -h "FAILED" $f | head -3; tail -6 $f | cut -c1-300; done
}} > $OUT 2>&1
echo "wrote $OUT ($(grep -c '' $OUT) lines, $(du -h $OUT | cut -f1))"
```
'''
open(out, 'w').write(md); open(R + '/scripts/submit-batch17-v1.sh', 'w').write(SUBMIT)
print(f'wrote {out}: {len(md.splitlines())} lines | v9e sbatch {se_n} {se_md5}')
