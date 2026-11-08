"""make_batch15.py: write commands/node-batch15-v1.md = one paste (train_rd_adapt_v9c.py + two v9c sbatch + gate script + gate sbatch + sweep submitter +
test submitter) + follow-up blocks.  Usage: make_batch15.py <release-adapt-v9c dir> <out.md>.  Expects release-adapt-v9b next to it (md5 of the v9b Blender sbatch
for the mic / ship resubmit)."""
import hashlib, os, sys
R = sys.argv[1]; out = sys.argv[2]; R9B = os.path.join(os.path.dirname(os.path.abspath(R)), 'release-adapt-v9b')


def read(p):
    s = open(p).read(); assert s.endswith('\n'), p
    return s, len(s.splitlines()), hashlib.md5(s.encode()).hexdigest()


tr, tr_n, tr_md5 = read(R + '/code/rdgrowth/train_rd_adapt_v9c.py')
sb, sb_n, sb_md5 = read(R + '/sbatch/adapt-v9c-blender.sbatch')
sm, sm_n, sm_md5 = read(R + '/sbatch/adapt-v9c.sbatch')
gp, gp_n, gp_md5 = read(R + '/scripts/gate_batch15_v1.py')
_, b9_n, b9_md5 = read(R9B + '/sbatch/adapt-v9b-blender.sbatch'); _, t9_n, t9_md5 = read(R9B + '/code/rdgrowth/train_rd_adapt_v9b.py')

GATE_SB = '''#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=gate15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/zr2245/analysis/gate15-%j.out
# gate-batch15.sbatch (2026-11-08): CPU job after the six v9c test jobs (--dependency=afterany); applies gate_batch15_v1.py and submits the v9c sweep part(s) that passed.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
AN=/scratch/zr2245/analysis
echo "gate-batch15 $(date) on $(hostname) job $SLURM_JOB_ID"
sacct -X -u $USER -S now-8hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt9c|adapt9xb|JobID"
for f in $AN/adapt9c-*-adapt9c-*.out $AN/adapt9cb-*-adapt9cb-*.out; do [ -f $f ] || continue; grep -q "FAILED" $f && { echo "--- $f FAILED:"; grep -h "FAILED" $f | head -3; tail -5 $f | cut -c1-300; }; done
python3 $AN/gate_batch15_v1.py $AN/adapt-v9c.txt $AN/adapt-v9c-blender.txt 2>&1 | tee $AN/gate-batch15.log || { echo "GATE FAILED: gate script crashed"; exit 1; }
mode=$(grep "^GATE SUBMIT" $AN/gate-batch15.log | awk '{print $3}')
case "$mode" in
  all|mip|blender) echo "GATE: submitting the v9c sweep ($mode) $(date)" | tee -a $AN/gate-batch15.log; bash $AN/submit-batch15-sweep-v1.sh $mode 2>&1 | tee -a $AN/gate-batch15.log;;
  *) echo "GATE: nothing submitted (mode '$mode'); read $AN/gate-batch15.log" | tee -a $AN/gate-batch15.log;;
esac
echo "gate-batch15 done $(date)"
'''

SWEEP = f'''AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets; MODE=${{1:-all}}
case $MODE in all|mip|blender) ;; *) echo "FAILED: mode must be all|mip|blender, got '$MODE'"; exit 1;; esac
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }}; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || {{ echo "CHECK FAILED: $1 md5 $m != $3 (not the file the test batch used?)"; exit 1; }}; }}
chk $AN/rdgrowth/train_rd_adapt_v9c.py {tr_n} {tr_md5}
chk $AN/adapt-v9c-blender.sbatch {sb_n} {sb_md5}
chk $AN/adapt-v9c.sbatch {sm_n} {sm_md5}
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9c.py || {{ echo "CHECK FAILED: train_rd_adapt_v9c.py does not compile"; exit 1; }}
bash -n $AN/adapt-v9c.sbatch && bash -n $AN/adapt-v9c-blender.sbatch || {{ echo "CHECK FAILED: sbatch syntax"; exit 1; }}
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in lego chair drums ficus hotdog materials mic ship; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"
echo "SWEEP CHECK OK (mode $MODE)"; k=0
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5$6"; [ "$5" = 5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9c $AN/adapt-v9c.sbatch) && {{ echo "submitted $1 $5 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $5"; }}
b () {{ EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"; case $1 in mic|ship) EX="$EX,SMOKE=0,SELFCHECK=0";; esac   # mic / ship: the 1550-it smoke mesh has < 4096 kept vertices (K-means fails); the real runs are fine
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9cb $AN/adapt-v9c-blender.sbatch) && {{ echo "submitted $1 $2 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $2"; }}
# the six test rows (room 5, counter 5, kitchen 10, counter 10, kitchen 20, lego 2.5) are not resubmitted
mip5 () {{ m kitchen 1 images_2 q16 5; m bonsai 1 images_2 q15 5; m garden 0 images_4 q15 5; m bicycle 0 images_4 q16 5; m stump 0 images_4 q16 5; m flowers 0 images_4 q15 5; m treehill 0 images_4 q16 5; m truck 0 images q18 5 ",DATA=$D/tandt/truck"; m train 0 images q18 5 ",DATA=$D/tandt/train"; }}
mip10 () {{ m room 1 images_2 q14 10; m bonsai 1 images_2 q15 10; m garden 0 images_4 q15 10; m bicycle 0 images_4 q16 10; m stump 0 images_4 q16 10; m flowers 0 images_4 q15 10; m treehill 0 images_4 q16 10; m truck 0 images q18 10 ",DATA=$D/tandt/truck"; m train 0 images q18 10 ",DATA=$D/tandt/train"; }}
mip20 () {{ m room 1 images_2 q14 20; m counter 1 images_2 q15 20; m bonsai 1 images_2 q15 20; m garden 0 images_4 q15 20; m bicycle 0 images_4 q16 20; m stump 0 images_4 q16 20; m flowers 0 images_4 q15 20; m treehill 0 images_4 q16 20; m truck 0 images q18 20 ",DATA=$D/tandt/truck"; m train 0 images q18 20 ",DATA=$D/tandt/train"; }}
bl25 () {{ for S in chair drums ficus hotdog materials mic ship; do b $S 2.5; done; }}
bl4 () {{ for S in lego chair drums ficus hotdog materials mic ship; do b $S 4; done; }}
case $MODE in
  all) mip5; bl25; mip10; bl4; mip20; N=43;;
  mip) mip5; mip10; mip20; N=28;;
  blender) bl25; bl4; N=15;;
esac
echo "submitted $k of $N sweep jobs (mode $MODE) $(date)"
'''

SUBMIT = f'''AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }}; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || {{ echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }}; }}
chk $AN/rdgrowth/train_rd_adapt_v9c.py {tr_n} {tr_md5}
chk $AN/adapt-v9c-blender.sbatch {sb_n} {sb_md5}
chk $AN/adapt-v9c.sbatch {sm_n} {sm_md5}
chk $AN/gate_batch15_v1.py {gp_n} {gp_md5}
chk $AN/rdgrowth/train_rd_adapt_v9b.py {t9_n} {t9_md5}
chk $AN/adapt-v9b-blender.sbatch {b9_n} {b9_md5}
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
grep -q "version='adapt-v9c'" $AN/rdgrowth/train_rd_adapt_v9c.py && grep -q "max(0.0, slope) \\* (horizon_iter - iteration)" $AN/rdgrowth/train_rd_adapt_v9c.py && grep -q "if k_asked > 0 else lam_used" $AN/rdgrowth/train_rd_adapt_v9c.py || {{ echo "CHECK FAILED: train_rd_adapt_v9c.py is not adapt-v9c"; exit 1; }}
grep -q "VERSION v9cb-r1" $AN/adapt-v9c-blender.sbatch && grep -q "VERSION v9c-r1" $AN/adapt-v9c.sbatch || {{ echo "CHECK FAILED: sbatch files are not v9c-r1 / v9cb-r1"; exit 1; }}
grep -q '12500 250 19750' $AN/adapt-v9c.sbatch && grep -q '12500 250 19750' $AN/adapt-v9c-blender.sbatch && grep -q 'OVER=${{OVER:-1.25}}' $AN/adapt-v9c.sbatch && grep -q 'OVER=${{OVER:-1.1}}' $AN/adapt-v9c-blender.sbatch || {{ echo "CHECK FAILED: sbatch defaults (PRUNE every 250, OVER 1.25 / 1.1) missing"; exit 1; }}
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9c.py $AN/gate_batch15_v1.py || {{ echo "CHECK FAILED: python files do not compile"; exit 1; }}
bash -n $AN/adapt-v9c-blender.sbatch && bash -n $AN/adapt-v9c.sbatch && bash -n $AN/gate-batch15.sbatch && bash -n $AN/submit-batch15-sweep-v1.sh || {{ echo "CHECK FAILED: bash syntax"; exit 1; }}
for S in lego mic ship; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in room kitchen counter; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for d in /scratch/zr2245/room-adapt9c-t5mb /scratch/zr2245/counter-adapt9c-t5mb /scratch/zr2245/kitchen-adapt9c-t10mb /scratch/zr2245/counter-adapt9c-t10mb /scratch/zr2245/kitchen-adapt9c-t20mb /scratch/zr2245/lego-adapt9cb-t2.5mb /scratch/zr2245/mic-adapt9xb-t2.5mb /scratch/zr2245/ship-adapt9xb-t2.5mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt9|gate1"); [ "$busy" = 0 ] || echo "note: $busy adapt/gate jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0; IDS=""
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5"; [ "$6" = smoke ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9c $AN/adapt-v9c.sbatch) && {{ echo "submitted $1 $5 MB $j"; k=$((k+1)); IDS="$IDS:$j"; }} || echo "FAILED: sbatch $1 $5"; }}
b () {{ EX="SCENE=$1,XYZ=q14,TARGETS=$2"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9cb $AN/adapt-v9c-blender.sbatch) && {{ echo "submitted $1 $2 MB $j"; k=$((k+1)); IDS="$IDS:$j"; }} || echo "FAILED: sbatch $1 $2"; }}
m kitchen 1 images_2 q16 20 nosmoke; m kitchen 1 images_2 q16 10 smoke; m counter 1 images_2 q15 10 nosmoke; m room 1 images_2 q14 5 smoke; m counter 1 images_2 q15 5 nosmoke; b lego 2.5
echo "submitted $k of 6 v9c test jobs (kitchen 20 / kitchen 10 / counter 10 / room 5 first; counter 5 + lego 2.5 as GPUs free up; ~1.7 h wall)"
[ "$k" = 6 ] || {{ echo "FAILED: only $k test jobs submitted; NOT submitting the gate (run section 4 by hand)"; exit 1; }}
g=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --dependency=afterany${{IDS}} $AN/gate-batch15.sbatch) && echo "gate job $g (after ${{IDS#:}}; on PASS it submits the 43-job v9c sweep; log $AN/gate-batch15.log)" || echo "FAILED: gate sbatch not accepted -> run section 4 by hand when the 6 jobs are done"
# v9b table completion: mic / ship 2.5 MB with v9b, SMOKE=0 (the smoke stage is what failed); not part of the gate
for S in mic ship; do j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,SCENE=$S,XYZ=q14,TARGETS=2.5,SMOKE=0,SELFCHECK=0 --job-name=$S-2.5-adapt9xb $AN/adapt-v9b-blender.sbatch) && echo "submitted v9b $S 2.5 MB $j (completes adapt-v9b-blender.txt)" || echo "FAILED: sbatch v9b $S 2.5"; done
'''

for body, tag in ((tr, 'PYEOF_TRAIN9C'), (sb, 'SBEOF_ADAPT9CB'), (sm, 'SBEOF_ADAPT9C'), (gp, 'PYEOF_GATE15'), (GATE_SB, 'SBEOF_GATE15'), (SWEEP, 'SWEEP15'), (SUBMIT, 'SUB15')):
    assert tag not in body and '```' not in body, tag
    for other in ('PYEOF_TRAIN9C', 'SBEOF_ADAPT9CB', 'SBEOF_ADAPT9C', 'PYEOF_GATE15', 'SBEOF_GATE15', 'SWEEP15', 'SUB15'): assert other not in body, (tag, other)

head = f'''# node-batch15-v1 (2026-11-08): adapt-v9c test (6 jobs) + mic / ship v9b resubmit (2 jobs) -> automatic gate -> v9c sweep (43 jobs)

**v9b result** (node-batch14, 47/49 jobs): Mip-360 / T&T hit 1.006 / 1.020 / 0.995 at 5 / 10 / 20 MB, 22/33 within +-5 %, 32/33 within +-10 % (v7b 0.888 / 0.831 / 0.767, 8/33), equal-bytes dPSNR vs the fixed-lambda hull -0.05 mean (v7b -0.10), phase B active on 26/33 rows. Residual: seven indoor rows 5-13 % OVER target although phase B ran 7-11 rounds, because the landing estimate read UNDER target for the first 3-9 rounds (kitchen 10: probe 9.43 at 12500, estimates 8.51 / 9.24 / 9.79 at 13000-14000, final 11.34): `extrapolate_probe` projects a least-squares slope 11 000 iterations ahead, and with 2-3 probes a small negative sample slope becomes -0.8 MB. The kept set only grows under the opacity ramp (kappa-analysis-v1: final / first probe 1.00-1.39), so a negative slope is noise.

**v9c** = v9b + ONE line: the slope used for the estimate is clamped at zero (raw slope still logged); `train_rd_adapt_v9c.py` ({tr_n} lines, md5 {tr_md5}). Sbatch (`adapt-v9c.sbatch` {sm_n} lines md5 {sm_md5}, VERSION v9c-r1; `adapt-v9c-blender.sbatch` {sb_n} lines md5 {sb_md5}, VERSION v9cb-r1): prune rounds every **250** from 12500 to 19750 (30 rounds; the over-target rows collapsed only 0.8-1 % of V per round, the foldover / independent-set limit, and ran out of rounds; +15 probes of ~30 s per Mip run), OVER unchanged (1.25 / 1.1), names `adapt9c` / `adapt9cb`, tables `adapt-v9c.txt` / `adapt-v9c-blender.txt`.

**Test jobs**: the five worst v9b over-target rows, room 5 (1.061), counter 5 (1.092), kitchen 10 (1.135), counter 10 (1.079), kitchen 20 (1.100), plus lego 2.5 (Blender regression check; v9b 0.999 / 32.383). Also submitted, outside the gate: mic 2.5 and ship 2.5 with **v9b** and SMOKE=0 (their smoke failed on the 1550-it mesh having < 4096 kept vertices), which completes the v9b Blender table. About 7 GPU-h, ~1.7 h wall; the sweep (43 jobs, ~38 GPU-h) follows automatically if the gate passes.

**Pass bar (`gate_batch15_v1.py`, fixed now)**: Mip: >= 4 of 5 rows, >= 4 within 0.95-1.05, none outside 0.90-1.08, test PSNR >= the fixed-lambda hull at the delivered bytes - 0.15 dB. Blender: lego within 0.95-1.05, PSNR >= 32.383 - 0.15.

**Expected**: first active round 2-4 rounds earlier (counter 5: 13000 instead of 15000), 15-25 active rounds, the five rows at 0.97-1.05; lego unchanged within 0.1 dB.

## 1. Write six files + submit (one paste)

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/train_rd_adapt_v9c.py <<'PYEOF_TRAIN9C'
'''
mid1 = '''PYEOF_TRAIN9C
cat > $AN/adapt-v9c-blender.sbatch <<'SBEOF_ADAPT9CB'
'''
mid2 = '''SBEOF_ADAPT9CB
cat > $AN/adapt-v9c.sbatch <<'SBEOF_ADAPT9C'
'''
mid3 = '''SBEOF_ADAPT9C
cat > $AN/gate_batch15_v1.py <<'PYEOF_GATE15'
'''
mid4 = '''PYEOF_GATE15
cat > $AN/gate-batch15.sbatch <<'SBEOF_GATE15'
'''
mid5 = '''SBEOF_GATE15
cat > $AN/submit-batch15-sweep-v1.sh <<'SWEEP15'
'''
mid6 = '''SWEEP15
cat > $AN/submit-batch15-v1.sh <<'SUB15'
'''
tail = '''SUB15
bash $AN/submit-batch15-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %.12E %R"
```

## 2. After 10-15 min: smoke of room 5 / kitchen 10 / lego 2.5, first densify lines

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %.12E %R"
for f in $AN/adapt9c-*-adapt9c-*.out $AN/adapt9cb-*-adapt9cb-*.out; do echo "=== $f"; grep -h "self-check\\|smoke v9 plan\\|smoke prune\\|smoke final\\|note:\\|FAILED\\|=== .* target" $f | tail -6 | cut -c1-400; done
```

## 3. Later (test done ~1.7 h; sweep done next morning): write everything to a file and upload it

```bash
AN=/scratch/zr2245/analysis; OUT=$AN/batch15-results-$(date +%m%d-%H%M).txt
{
echo "===== sacct"; sacct -X -u $USER -S now-20hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt9c|adapt9xb|gate15|JobID"
echo "===== squeue"; squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
echo "===== gate"; cat $AN/gate-batch15.log 2>/dev/null || echo "no gate log"
echo "===== adapt-v9c.txt"; cat $AN/adapt-v9c.txt
echo "===== adapt-v9c-blender.txt"; cat $AN/adapt-v9c-blender.txt
echo "===== adapt-v9b-blender.txt (mic / ship rows)"; grep -E "^(mic|ship) " $AN/adapt-v9b-blender.txt
echo "===== kappa (v9c rows)"; python3 $AN/rdgrowth/kappa_analysis_v1.py '/scratch/zr2245/*-adapt9c-t*mb' '/scratch/zr2245/*-adapt9cb-t*mb' 2>&1
echo "===== train logs"; for d in /scratch/zr2245/*-adapt9c-t*mb /scratch/zr2245/*-adapt9cb-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\\[RD 9500\\]\\|phase A ended\\|RD adapt 1[2-9][0-9]*\\] probe\\|^\\[RD prune\\] *1[2-9][0-9]*.*collapsed\\|no collapse this round\\|^Training is done" $d/train.log | sed -E 's/\\(lambda-prune held; [0-9]+ edges below lam [0-9.e+-]+\\)//; s/\\[0[0-9]\\/11 [0-9:]+\\]//' | cut -c1-330; done
echo "===== FAILED lines in job outputs"; grep -l "FAILED" $AN/adapt9c-*.out $AN/adapt9cb-*.out $AN/adapt9xb-mic-*.out $AN/adapt9xb-ship-*.out 2>/dev/null | while read f; do echo "--- $f"; grep -h "FAILED" $f | head -3; tail -6 $f | cut -c1-300; done
} > $OUT 2>&1
echo "wrote $OUT ($(grep -c '' $OUT) lines, $(du -h $OUT | cut -f1))"
```

## 4. Only if the gate job did not run

```bash
AN=/scratch/zr2245/analysis
python3 $AN/gate_batch15_v1.py $AN/adapt-v9c.txt $AN/adapt-v9c-blender.txt | tee $AN/gate-batch15.log
# then, if GATE SUBMIT says all / mip / blender:   bash $AN/submit-batch15-sweep-v1.sh all
```
'''
md = head + tr + mid1 + sb + mid2 + sm + mid3 + gp + mid4 + GATE_SB + mid5 + SWEEP + mid6 + SUBMIT + tail
open(out, 'w').write(md)
open(R + '/scripts/gate-batch15.sbatch', 'w').write(GATE_SB); open(R + '/scripts/submit-batch15-sweep-v1.sh', 'w').write(SWEEP); open(R + '/scripts/submit-batch15-v1.sh', 'w').write(SUBMIT)
print(f'wrote {out}: {len(md.splitlines())} lines | train {tr_n} {tr_md5} | blender sbatch {sb_n} {sb_md5} | mip sbatch {sm_n} {sm_md5} | gate {gp_n} {gp_md5}')
