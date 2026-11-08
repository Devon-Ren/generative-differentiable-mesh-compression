"""make_batch16.py: write commands/node-batch16-v1.md = one paste (train_rd_adapt_v9d.py + two v9c sbatch + gate script + gate sbatch + sweep submitter +
test submitter) + follow-up blocks.  Usage: make_batch16.py <release-adapt-v9d dir> <out.md>.  Expects release-adapt-v9b next to it (md5 of the v9b Blender sbatch
for the mic / ship resubmit)."""
import hashlib, os, sys
R = sys.argv[1]; out = sys.argv[2]; 


def read(p):
    s = open(p).read(); assert s.endswith('\n'), p
    return s, len(s.splitlines()), hashlib.md5(s.encode()).hexdigest()


tr, tr_n, tr_md5 = read(R + '/code/rdgrowth/train_rd_adapt_v9d.py')
sb, sb_n, sb_md5 = read(R + '/sbatch/adapt-v9d-blender.sbatch')
sm, sm_n, sm_md5 = read(R + '/sbatch/adapt-v9d.sbatch')
gp, gp_n, gp_md5 = read(R + '/scripts/gate_batch16_v1.py')
pr, pr_n, pr_md5 = read(R + '/code/rdgrowth/rd_prune_v2.py')

GATE_SB = '''#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=gate16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/zr2245/analysis/gate16-%j.out
# gate-batch16.sbatch (2026-11-08): CPU job after the six v9c test jobs (--dependency=afterany); applies gate_batch16_v1.py and submits the v9c sweep part(s) that passed.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
AN=/scratch/zr2245/analysis
echo "gate-batch16 $(date) on $(hostname) job $SLURM_JOB_ID"
sacct -X -u $USER -S now-8hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt9d|adapt9xb|JobID"
for f in $AN/adapt9d-*-adapt9d-*.out $AN/adapt9db-*-adapt9db-*.out; do [ -f $f ] || continue; grep -q "FAILED" $f && { echo "--- $f FAILED:"; grep -h "FAILED" $f | head -3; tail -5 $f | cut -c1-300; }; done
python3 $AN/gate_batch16_v1.py $AN/adapt-v9d.txt $AN/adapt-v9d-blender.txt 2>&1 | tee $AN/gate-batch16.log || { echo "GATE FAILED: gate script crashed"; exit 1; }
mode=$(grep "^GATE SUBMIT" $AN/gate-batch16.log | awk '{print $3}')
case "$mode" in
  all|mip|blender) echo "GATE: submitting the v9c sweep ($mode) $(date)" | tee -a $AN/gate-batch16.log; bash $AN/submit-batch16-sweep-v1.sh $mode 2>&1 | tee -a $AN/gate-batch16.log;;
  *) echo "GATE: nothing submitted (mode '$mode'); read $AN/gate-batch16.log" | tee -a $AN/gate-batch16.log;;
esac
echo "gate-batch16 done $(date)"
'''

SWEEP = f'''AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets; MODE=${{1:-all}}
case $MODE in all|mip|blender) ;; *) echo "FAILED: mode must be all|mip|blender, got '$MODE'"; exit 1;; esac
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }}; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || {{ echo "CHECK FAILED: $1 md5 $m != $3 (not the file the test batch used?)"; exit 1; }}; }}
chk $AN/rdgrowth/train_rd_adapt_v9d.py {tr_n} {tr_md5}
chk $AN/rdgrowth/rd_prune_v2.py {pr_n} {pr_md5}
chk $AN/adapt-v9d-blender.sbatch {sb_n} {sb_md5}
chk $AN/adapt-v9d.sbatch {sm_n} {sm_md5}
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9d.py || {{ echo "CHECK FAILED: train_rd_adapt_v9d.py does not compile"; exit 1; }}
bash -n $AN/adapt-v9d.sbatch && bash -n $AN/adapt-v9d-blender.sbatch || {{ echo "CHECK FAILED: sbatch syntax"; exit 1; }}
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in lego chair drums ficus hotdog materials mic ship; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"
echo "SWEEP CHECK OK (mode $MODE)"; k=0
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5$6"; [ "$5" = 5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9d $AN/adapt-v9d.sbatch) && {{ echo "submitted $1 $5 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $5"; }}
b () {{ EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"; case $1 in mic|ship) EX="$EX,SMOKE=0,SELFCHECK=0";; esac   # mic / ship: the 1550-it smoke mesh has < 4096 kept vertices (K-means fails); the real runs are fine
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9db $AN/adapt-v9d-blender.sbatch) && {{ echo "submitted $1 $2 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $2"; }}
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
chk $AN/rdgrowth/train_rd_adapt_v9d.py {tr_n} {tr_md5}
chk $AN/adapt-v9d-blender.sbatch {sb_n} {sb_md5}
chk $AN/adapt-v9d.sbatch {sm_n} {sm_md5}
chk $AN/gate_batch16_v1.py {gp_n} {gp_md5}
chk $AN/rdgrowth/rd_prune_v2.py {pr_n} {pr_md5}
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
grep -q "version='adapt-v9d'" $AN/rdgrowth/train_rd_adapt_v9d.py && grep -q "from rd_prune_v2 import" $AN/rdgrowth/train_rd_adapt_v9d.py && grep -q "kept_v=kept_v, kept_f=kept_f" $AN/rdgrowth/train_rd_adapt_v9d.py && grep -q "max(0.0, slope) \\* (horizon_iter - iteration)" $AN/rdgrowth/train_rd_adapt_v9d.py || {{ echo "CHECK FAILED: train_rd_adapt_v9d.py is not adapt-v9d"; exit 1; }}
grep -q "PRUNE_VERSION = 'rd_prune_v2'" $AN/rdgrowth/rd_prune_v2.py && grep -q "kept_v=None, kept_f=None" $AN/rdgrowth/rd_prune_v2.py || {{ echo "CHECK FAILED: rd_prune_v2.py missing / wrong"; exit 1; }}
grep -q "VERSION v9db-r1" $AN/adapt-v9d-blender.sbatch && grep -q "VERSION v9d-r1" $AN/adapt-v9d.sbatch || {{ echo "CHECK FAILED: sbatch files are not v9d-r1 / v9db-r1"; exit 1; }}
grep -q '12500 250 19750' $AN/adapt-v9d.sbatch && grep -q '12500 250 19750' $AN/adapt-v9d-blender.sbatch && grep -q 'OVER=${{OVER:-1.25}}' $AN/adapt-v9d.sbatch && grep -q 'OVER=${{OVER:-1.1}}' $AN/adapt-v9d-blender.sbatch || {{ echo "CHECK FAILED: sbatch defaults (PRUNE every 250, OVER 1.25 / 1.1) missing"; exit 1; }}
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9d.py $AN/rdgrowth/rd_prune_v2.py $AN/gate_batch16_v1.py || {{ echo "CHECK FAILED: python files do not compile"; exit 1; }}
bash -n $AN/adapt-v9d-blender.sbatch && bash -n $AN/adapt-v9d.sbatch && bash -n $AN/gate-batch16.sbatch && bash -n $AN/submit-batch16-sweep-v1.sh || {{ echo "CHECK FAILED: bash syntax"; exit 1; }}
for S in lego; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in room kitchen counter; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for d in /scratch/zr2245/room-adapt9d-t5mb /scratch/zr2245/counter-adapt9d-t5mb /scratch/zr2245/kitchen-adapt9d-t10mb /scratch/zr2245/counter-adapt9d-t10mb /scratch/zr2245/kitchen-adapt9d-t20mb /scratch/zr2245/lego-adapt9db-t2.5mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt9|gate1"); [ "$busy" = 0 ] || echo "note: $busy adapt/gate jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0; IDS=""
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5"; [ "$6" = smoke ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9d $AN/adapt-v9d.sbatch) && {{ echo "submitted $1 $5 MB $j"; k=$((k+1)); IDS="$IDS:$j"; }} || echo "FAILED: sbatch $1 $5"; }}
b () {{ EX="SCENE=$1,XYZ=q14,TARGETS=$2"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9db $AN/adapt-v9d-blender.sbatch) && {{ echo "submitted $1 $2 MB $j"; k=$((k+1)); IDS="$IDS:$j"; }} || echo "FAILED: sbatch $1 $2"; }}
m kitchen 1 images_2 q16 20 nosmoke; m kitchen 1 images_2 q16 10 smoke; m counter 1 images_2 q15 10 nosmoke; m room 1 images_2 q14 5 smoke; m counter 1 images_2 q15 5 nosmoke; b lego 2.5
echo "submitted $k of 6 v9d test jobs (kitchen 20 / kitchen 10 / counter 10 / room 5 first; counter 5 + lego 2.5 as GPUs free up; ~1.7 h wall)"
[ "$k" = 6 ] || {{ echo "FAILED: only $k test jobs submitted; NOT submitting the gate (run section 4 by hand)"; exit 1; }}
g=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --dependency=afterany${{IDS}} $AN/gate-batch16.sbatch) && echo "gate job $g (after ${{IDS#:}}; on PASS it submits the 43-job v9d sweep; log $AN/gate-batch16.log)" || echo "FAILED: gate sbatch not accepted -> run section 4 by hand when the 6 jobs are done"
'''

for body, tag in ((tr, 'PYEOF_TRAIN9D'), (pr, 'PYEOF_PRUNE2'), (sb, 'SBEOF_ADAPT9DB'), (sm, 'SBEOF_ADAPT9D'), (gp, 'PYEOF_GATE16'), (GATE_SB, 'SBEOF_GATE16'), (SWEEP, 'SWEEP16'), (SUBMIT, 'SUB16')):
    assert tag not in body and '```' not in body, tag
    for other in ('PYEOF_TRAIN9D', 'PYEOF_PRUNE2', 'SBEOF_ADAPT9DB', 'SBEOF_ADAPT9D', 'PYEOF_GATE16', 'SBEOF_GATE16', 'SWEEP16', 'SUB16'): assert other not in body, (tag, other)

head = f'''# node-batch16-v1 (2026-11-08): adapt-v9d test (6 jobs) -> automatic gate -> v9d sweep (43 jobs)

**v9c result** (node-batch15): the slope clamp worked (the estimate never read below the probe; kitchen 10's estimate rose 9.4 -> 11.3 MB as the kept set un-stacked, final 11.15) and phase B ran 15-24 of 30 rounds, yet the five over-target rows only moved 1.061 / 1.092 / 1.135 / 1.079 / 1.100 -> 1.028 / 1.051 / 1.116 / 1.067 / 1.075 (gate FAIL; the Blender sweep passed and ran, 15/15). The rounds freed almost nothing: 4-8k pairs each, 0.002-0.03 MB each; kitchen 10 collapsed 109 724 pairs for 0.5 MB of its 1.3 MB excess; both-kept share 0.35-0.57 on Mip, measured bytes per pair 0.6-2.4 against the 6.3 prior. **The cheapest dD/dR edges are the zero-residual ones: the triangles nobody renders, which the final cleaning drops anyway and the probe already excludes. Phase B was trimming geometry that costs no bytes.**

**v9d** = v9c + the phase-B collapse priced on the DELIVERED set. `rd_prune_v2.py` (new file, {pr_n} lines, md5 {pr_md5}; `rd_prune.py` untouched): `select_collapses` takes the probe's kept-vertex / kept-face masks and prices dR = b_attr x [both endpoints kept] + b_face x (kept faces on the edge); unpriced edges are not candidates. `train_rd_adapt_v9d.py` ({tr_n} lines, md5 {tr_md5}): the same price in `edge_scores` (n_lam, rank threshold, shadow price), bytes-per-collapse prior b_v + 2 b_f, the prune line prints the priced edge count. An edge that frees no delivered bytes has no price and no reason to collapse under a rate objective. Sbatch (`adapt-v9d.sbatch` {sm_n} lines md5 {sm_md5}, VERSION v9d-r1; `adapt-v9d-blender.sbatch` {sb_n} lines md5 {sb_md5}, VERSION v9db-r1): PRUNE / OVER as v9c (30 rounds; 1.25 / 1.1), names `adapt9d` / `adapt9db`, tables `adapt-v9d.txt` / `adapt-v9d-blender.txt`.

**Test jobs**: the same five rows (room 5, counter 5, kitchen 10, counter 10, kitchen 20) + lego 2.5, then the gate (same pass bar as batch15: >= 4 of 5 within 0.95-1.05, none outside 0.90-1.08, PSNR >= hull - 0.15; lego within 0.95-1.05 and >= 32.233) submits the 43-job v9d sweep. About 7 GPU-h test (~1.7 h wall), ~38 GPU-h sweep (~10 h).

**Expected**: 5-7 B freed per pair instead of 0.6-2.4, so the same rounds trim 3-5x more bytes and the five rows land 0.97-1.05; lego unchanged within 0.1 dB. In the train logs: `lam-admitted N of M priced edges` with M well below the edge count, `kept-removed` close to `pairs`.

## 1. Write six files + submit (one paste)

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/train_rd_adapt_v9d.py <<'PYEOF_TRAIN9D'
'''
mid1 = '''PYEOF_TRAIN9D
cat > $AN/rdgrowth/rd_prune_v2.py <<'PYEOF_PRUNE2'
'''
mid1b = '''PYEOF_PRUNE2
cat > $AN/adapt-v9d-blender.sbatch <<'SBEOF_ADAPT9DB'
'''
mid2 = '''SBEOF_ADAPT9DB
cat > $AN/adapt-v9d.sbatch <<'SBEOF_ADAPT9D'
'''
mid3 = '''SBEOF_ADAPT9D
cat > $AN/gate_batch16_v1.py <<'PYEOF_GATE16'
'''
mid4 = '''PYEOF_GATE16
cat > $AN/gate-batch16.sbatch <<'SBEOF_GATE16'
'''
mid5 = '''SBEOF_GATE16
cat > $AN/submit-batch16-sweep-v1.sh <<'SWEEP16'
'''
mid6 = '''SWEEP16
cat > $AN/submit-batch16-v1.sh <<'SUB16'
'''
tail = '''SUB16
bash $AN/submit-batch16-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %.12E %R"
```

## 2. After 10-15 min: smoke of room 5 / kitchen 10 / lego 2.5, first densify lines

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %.12E %R"
for f in $AN/adapt9d-*-adapt9d-*.out $AN/adapt9db-*-adapt9db-*.out; do echo "=== $f"; grep -h "self-check\\|smoke v9 plan\\|smoke prune\\|smoke final\\|note:\\|FAILED\\|=== .* target" $f | tail -6 | cut -c1-400; done
```

## 3. Later (test done ~1.7 h; sweep done next morning): write everything to a file and upload it

```bash
AN=/scratch/zr2245/analysis; OUT=$AN/batch16-results-$(date +%m%d-%H%M).txt
{
echo "===== sacct"; sacct -X -u $USER -S now-20hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt9d|gate16|JobID"
echo "===== squeue"; squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
echo "===== gate"; cat $AN/gate-batch16.log 2>/dev/null || echo "no gate log"
echo "===== adapt-v9d.txt"; cat $AN/adapt-v9d.txt
echo "===== adapt-v9d-blender.txt"; cat $AN/adapt-v9d-blender.txt
echo "===== kappa (v9c rows)"; python3 $AN/rdgrowth/kappa_analysis_v1.py '/scratch/zr2245/*-adapt9d-t*mb' '/scratch/zr2245/*-adapt9db-t*mb' 2>&1
echo "===== train logs"; for d in /scratch/zr2245/*-adapt9d-t*mb /scratch/zr2245/*-adapt9db-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\\[RD 9500\\]\\|phase A ended\\|RD adapt 1[2-9][0-9]*\\] probe\\|^\\[RD prune\\] *1[2-9][0-9]*.*collapsed\\|no collapse this round\\|^Training is done" $d/train.log | sed -E 's/\\(lambda-prune held; [0-9]+ edges below lam [0-9.e+-]+\\)//; s/\\[0[0-9]\\/11 [0-9:]+\\]//' | cut -c1-330; done
echo "===== FAILED lines in job outputs"; grep -l "FAILED" $AN/adapt9d-*.out $AN/adapt9db-*.out 2>/dev/null | while read f; do echo "--- $f"; grep -h "FAILED" $f | head -3; tail -6 $f | cut -c1-300; done
} > $OUT 2>&1
echo "wrote $OUT ($(grep -c '' $OUT) lines, $(du -h $OUT | cut -f1))"
```

## 4. Only if the gate job did not run

```bash
AN=/scratch/zr2245/analysis
python3 $AN/gate_batch16_v1.py $AN/adapt-v9d.txt $AN/adapt-v9d-blender.txt | tee $AN/gate-batch16.log
# then, if GATE SUBMIT says all / mip / blender:   bash $AN/submit-batch16-sweep-v1.sh all
```
'''
md = head + tr + mid1 + pr + mid1b + sb + mid2 + sm + mid3 + gp + mid4 + GATE_SB + mid5 + SWEEP + mid6 + SUBMIT + tail
open(out, 'w').write(md)
open(R + '/scripts/gate-batch16.sbatch', 'w').write(GATE_SB); open(R + '/scripts/submit-batch16-sweep-v1.sh', 'w').write(SWEEP); open(R + '/scripts/submit-batch16-v1.sh', 'w').write(SUBMIT)
print(f'wrote {out}: {len(md.splitlines())} lines | train {tr_n} {tr_md5} | blender sbatch {sb_n} {sb_md5} | mip sbatch {sm_n} {sm_md5} | gate {gp_n} {gp_md5}')
