"""make_batch14.py: write commands/node-batch14-v1.md = one paste (train_rd_adapt_v9b.py + two sbatch + gate script + gate sbatch + sweep
submitter + test submitter with checks) and the follow-up blocks.  Usage: make_batch14.py <release-adapt-v9b dir> <out.md>"""
import hashlib, sys
R = sys.argv[1]; out = sys.argv[2]


def read(p):
    s = open(p).read(); assert s.endswith('\n'), p
    return s, len(s.splitlines()), hashlib.md5(s.encode()).hexdigest()


tr, tr_n, tr_md5 = read(R + '/code/rdgrowth/train_rd_adapt_v9b.py')
sb, sb_n, sb_md5 = read(R + '/sbatch/adapt-v9b-blender.sbatch')
sm, sm_n, sm_md5 = read(R + '/sbatch/adapt-v9b.sbatch')
gp, gp_n, gp_md5 = read(R + '/scripts/gate_batch14_v1.py')

GATE_SB = '''#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=gate14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/zr2245/analysis/gate14-%j.out
# gate-batch14.sbatch (2026-11-07): CPU job that runs after the six v9b test jobs (--dependency=afterany), applies the pass bar
# (gate_batch14_v1.py) to the rows they wrote, and submits the v9b sweep part(s) that passed (submit-batch14-sweep-v1.sh all|mip|blender).
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
AN=/scratch/zr2245/analysis
echo "gate-batch14 $(date) on $(hostname) job $SLURM_JOB_ID"
sacct -X -u $USER -S now-8hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt9x|JobID"
for f in $AN/adapt9x-*-adapt9x-*.out $AN/adapt9xb-*-adapt9xb-*.out; do [ -f $f ] || continue; grep -q "FAILED" $f && { echo "--- $f FAILED:"; grep -h "FAILED" $f | head -3; tail -5 $f | cut -c1-300; }; done
python3 $AN/gate_batch14_v1.py $AN/adapt-v9b.txt $AN/adapt-v9b-blender.txt 2>&1 | tee $AN/gate-batch14.log || { echo "GATE FAILED: gate script crashed"; exit 1; }
mode=$(grep "^GATE SUBMIT" $AN/gate-batch14.log | awk '{print $3}')
case "$mode" in
  all|mip|blender) echo "GATE: submitting the v9b sweep ($mode) $(date)" | tee -a $AN/gate-batch14.log; bash $AN/submit-batch14-sweep-v1.sh $mode 2>&1 | tee -a $AN/gate-batch14.log;;
  *) echo "GATE: nothing submitted (mode '$mode'); read $AN/gate-batch14.log" | tee -a $AN/gate-batch14.log;;
esac
echo "gate-batch14 done $(date)"
'''

SWEEP = f'''AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets; MODE=${{1:-all}}
case $MODE in all|mip|blender) ;; *) echo "FAILED: mode must be all|mip|blender, got '$MODE'"; exit 1;; esac
chk () {{ n=$(wc -l < $1); [ "$n" = $2 ] || {{ echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }}; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || {{ echo "CHECK FAILED: $1 md5 $m != $3 (not the file the test batch used?)"; exit 1; }}; }}
chk $AN/rdgrowth/train_rd_adapt_v9b.py {tr_n} {tr_md5}
chk $AN/adapt-v9b-blender.sbatch {sb_n} {sb_md5}
chk $AN/adapt-v9b.sbatch {sm_n} {sm_md5}
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b.py || {{ echo "CHECK FAILED: train_rd_adapt_v9b.py does not compile"; exit 1; }}
bash -n $AN/adapt-v9b.sbatch && bash -n $AN/adapt-v9b-blender.sbatch || {{ echo "CHECK FAILED: sbatch syntax"; exit 1; }}
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in lego chair drums ficus hotdog materials mic ship; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"
echo "SWEEP CHECK OK (mode $MODE)"; k=0
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5$6"; [ "$5" = 5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9x $AN/adapt-v9b.sbatch) && {{ echo "submitted $1 $5 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $5"; }}
b () {{ EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9xb $AN/adapt-v9b-blender.sbatch) && {{ echo "submitted $1 $2 MB $j"; k=$((k+1)); }} || echo "FAILED: sbatch $1 $2"; }}
# the six test rows (room 10, bicycle 10, room 20, garden 5, truck 10, lego 2.5) are not resubmitted
mip5 () {{ for S in room kitchen counter bonsai; do case $S in room) X=q14;; kitchen) X=q16;; *) X=q15;; esac; m $S 1 images_2 $X 5; done; m bicycle 0 images_4 q16 5; m stump 0 images_4 q16 5; m flowers 0 images_4 q15 5; m treehill 0 images_4 q16 5; m truck 0 images q18 5 ",DATA=$D/tandt/truck"; m train 0 images q18 5 ",DATA=$D/tandt/train"; }}
mip10 () {{ m kitchen 1 images_2 q16 10; m counter 1 images_2 q15 10; m bonsai 1 images_2 q15 10; m garden 0 images_4 q15 10; m stump 0 images_4 q16 10; m flowers 0 images_4 q15 10; m treehill 0 images_4 q16 10; m train 0 images q18 10 ",DATA=$D/tandt/train"; }}
mip20 () {{ m kitchen 1 images_2 q16 20; m counter 1 images_2 q15 20; m bonsai 1 images_2 q15 20; m garden 0 images_4 q15 20; m bicycle 0 images_4 q16 20; m stump 0 images_4 q16 20; m flowers 0 images_4 q15 20; m treehill 0 images_4 q16 20; m truck 0 images q18 20 ",DATA=$D/tandt/truck"; m train 0 images q18 20 ",DATA=$D/tandt/train"; }}
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
chk $AN/rdgrowth/train_rd_adapt_v9b.py {tr_n} {tr_md5}
chk $AN/adapt-v9b-blender.sbatch {sb_n} {sb_md5}
chk $AN/adapt-v9b.sbatch {sm_n} {sm_md5}
chk $AN/gate_batch14_v1.py {gp_n} {gp_md5}
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
[ -f $AN/rdgrowth/train_rd_adapt_v9.py ] && [ "$(md5sum < $AN/rdgrowth/train_rd_adapt_v9.py | cut -c1-32)" = a120351f64c9472e0353cef29d8ddb04 ] || echo "note: train_rd_adapt_v9.py is not the batch12 file (v9b does not import it; informational)"
grep -q "version='adapt-v9b'" $AN/rdgrowth/train_rd_adapt_v9b.py && grep -q "if k_asked > 0 else lam_used" $AN/rdgrowth/train_rd_adapt_v9b.py || {{ echo "CHECK FAILED: train_rd_adapt_v9b.py is not adapt-v9b"; exit 1; }}
grep -q "max_split_up = max_split" $AN/rdgrowth/train_rd_adapt_v9b.py && grep -q "remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))" $AN/rdgrowth/train_rd_adapt_v9b.py || {{ echo "CHECK FAILED: train_rd_adapt_v9b.py lacks the v9 plan fixes"; exit 1; }}
grep -q "VERSION v9xb-r1" $AN/adapt-v9b-blender.sbatch && grep -q "VERSION v9x-r1" $AN/adapt-v9b.sbatch || {{ echo "CHECK FAILED: sbatch files are not v9x-r1 / v9xb-r1"; exit 1; }}
grep -q 'OVER=${{OVER:-1.25}}' $AN/adapt-v9b.sbatch && grep -q 'OVER=${{OVER:-1.1}}' $AN/adapt-v9b-blender.sbatch && grep -q '12500 500 19500' $AN/adapt-v9b.sbatch || {{ echo "CHECK FAILED: sbatch defaults (OVER 1.25 / 1.1, PRUNE from 12500) missing"; exit 1; }}
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b.py $AN/gate_batch14_v1.py $AN/blender/val_select_v3.py $AN/rdgrowth/probe_bytes_v1.py || {{ echo "CHECK FAILED: python files do not compile"; exit 1; }}
bash -n $AN/adapt-v9b-blender.sbatch && bash -n $AN/adapt-v9b.sbatch && bash -n $AN/gate-batch14.sbatch && bash -n $AN/submit-batch14-sweep-v1.sh || {{ echo "CHECK FAILED: bash syntax"; exit 1; }}
for f in $AN/vis_collapse-v12/stage2_run.sh $AN/vis_collapse-v12/scripts/run_garden.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py $AN/blender/train_ckpt_v2.py $AN/coarse_finetune.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || {{ echo "CHECK FAILED: MISSING $f"; exit 1; }}; done
for S in lego; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
for S in room bicycle garden; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for $S"; exit 1; }}; done
[ -f $D/tandt/truck/sparse/0/cameras.bin ] || {{ echo "CHECK FAILED: no dataset for truck"; exit 1; }}
for d in /scratch/zr2245/room-adapt9x-t10mb /scratch/zr2245/bicycle-adapt9x-t10mb /scratch/zr2245/room-adapt9x-t20mb /scratch/zr2245/garden-adapt9x-t5mb /scratch/zr2245/truck-adapt9x-t10mb /scratch/zr2245/lego-adapt9xb-t2.5mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
[ -f $AN/adapt-v9b.txt ] && echo "note: $AN/adapt-v9b.txt already has $(grep -c '' $AN/adapt-v9b.txt) rows (the gate reads the LAST row per scene/target; fine on a restart)"
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt7|adapt8|adapt9|gate14"); [ "$busy" = 0 ] || echo "note: $busy adapt/gate jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show node gl030 | grep -o "CfgTRES=[^ ]*"
scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0; IDS=""
m () {{ EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5$6"; [ "$7" = smoke ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9x $AN/adapt-v9b.sbatch) && {{ echo "submitted $1 $5 MB $j"; k=$((k+1)); IDS="$IDS:$j"; }} || echo "FAILED: sbatch $1 $5"; }}
b () {{ EX="SCENE=$1,XYZ=q14,TARGETS=$2"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9xb $AN/adapt-v9b-blender.sbatch) && {{ echo "submitted $1 $2 MB $j"; k=$((k+1)); IDS="$IDS:$j"; }} || echo "FAILED: sbatch $1 $2"; }}
m room 1 images_2 q14 20 "" nosmoke; m room 1 images_2 q14 10 "" smoke; m bicycle 0 images_4 q16 10 "" nosmoke; m garden 0 images_4 q15 5 "" smoke; m truck 0 images q18 10 ",DATA=$D/tandt/truck" nosmoke; b lego 2.5
echo "submitted $k of 6 test jobs (4 GPUs: room 20 / room 10 / bicycle 10 / garden 5 first; truck 10 + lego 2.5 start as GPUs free up; ~1.5 h wall)"
[ "$k" = 6 ] || {{ echo "FAILED: only $k test jobs submitted; NOT submitting the gate (run it by hand, see section 4)"; exit 1; }}
g=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --dependency=afterany${{IDS}} $AN/gate-batch14.sbatch) && echo "gate job $g (runs after ${{IDS#:}}; if it passes it submits the 43-job sweep itself; log $AN/gate-batch14.log)" || echo "FAILED: gate sbatch not accepted (CPU job on the reservation refused?) -> run section 4 by hand when the 6 jobs are done"
'''

for body, tag in ((tr, 'PYEOF_TRAIN9B'), (sb, 'SBEOF_ADAPT9XB'), (sm, 'SBEOF_ADAPT9X'), (gp, 'PYEOF_GATE'), (GATE_SB, 'SBEOF_GATE'), (SWEEP, 'SWEEP14'), (SUBMIT, 'SUB14')):
    assert tag not in body and '```' not in body, tag
    for other in ('PYEOF_TRAIN9B', 'SBEOF_ADAPT9XB', 'SBEOF_ADAPT9X', 'PYEOF_GATE', 'SBEOF_GATE', 'SWEEP14', 'SUB14'):
        assert other not in body, (tag, other)

head = f'''# node-batch14-v1 (2026-11-07 evening): adapt-v9b test (6 jobs) -> automatic gate -> v9b sweep (43 jobs). One paste, the node runs all night.

**What v9b is** (`release-adapt-v9b/`): `train_rd_adapt_v9b.py` ({tr_n} lines, md5 {tr_md5}) = v9 + ONE code line: at the last densify interval the plan asks for zero splits, `rank_lambda(k=0)` returned lam_max = 0.1, and every run entered phase B with lambda 0.1, which admits every edge so phase B's first active round cut the whole excess at once (one-shot, not the progressive lambda-prune). v9b keeps the price just used (`lam_now = ... if k_asked > 0 else lam_used`). Everything else (phase-A plan, phase-A probes, phase B) is v9 verbatim.

**What the sbatch files change** (`adapt-v9b.sbatch` {sm_n} lines md5 {sm_md5}, VERSION v9x-r1; `adapt-v9b-blender.sbatch` {sb_n} lines md5 {sb_md5}, VERSION v9xb-r1): phase A aims at target x **1.25** on Mip-360 / T&T and x **1.1** on NeRF-Synthetic (`OVER` default), prune rounds **12500 ... 19500** (15; the 12000 probe right after the Delaunay reads high), job names `adapt9x` / `adapt9xb`, run dirs `$SCENE-adapt9x-t$T mb` / `$SCENE-adapt9xb-t$T mb`, tables `adapt-v9b.txt` / `adapt-v9b-blender.txt`. Why 1.25: batch12's Mip miss decomposes exactly as hit = kappa_end x (b_v,mesh + topology) / (b_v,soup + 5) = 0.880 / 0.807 / 0.829 (room 10 / bicycle 10 / room 20; observed 0.877 / 0.790 / 0.829): the cleaning keeps 0.92-0.94 of the vertices, the delivered topology costs 3.7-4.4 B/V against the prior 5, the soup over-reads b_v by 3-9 %. None of it is observable before the 11k Delaunay and nothing adds bytes after 9500, so the error goes on the side phase B can trim: 1/0.80 = 1.25, one value for every scene of the class.

**The six test jobs**: room 10 / bicycle 10 / room 20 (direct before/after against batch12), garden 5 + truck 10 (two scene classes not in batch12), lego 2.5 (does a 10 % surplus plus trimming cost dB?). Then a CPU gate job (`--dependency=afterany`) applies the pass bar below and, if it passes, submits the 43-job sweep by itself (`submit-batch14-sweep-v1.sh all|mip|blender`), so the 4 GPUs run until the morning (test ~5 GPU-h, sweep ~36 GPU-h, ~10.5 h wall).

**Pass bar (written before the runs; `gate_batch14_v1.py`)**: Mip/T&T: >= 4 of 5 rows present, >= 4 of the present rows within 0.93-1.07 x target, none outside 0.88-1.12, every row's test PSNR >= the v9/v7b row of the same scene and target - 0.08 dB (those were delivered at fewer bytes). Blender: lego 2.5 within 0.93-1.07, test PSNR >= 32.173 - 0.10. Mip pass alone submits the 28 Mip/T&T sweep jobs; Blender pass alone the 15 Blender jobs.

**Expected**: Mip rows land 0.99-1.09 before phase B and 0.95-1.05 after; `prune rounds 15 active N` with N >= 1 on every Mip row (v9: 0); `lam_final` a real price (not 1e-1); lego ~1.0 with more pairs collapsed than v9's 4321 and PSNR within 0.1 dB of 32.17.

## 1. Write six files + submit the six test jobs + the gate (one paste; all checks are inside the script, nothing touches your terminal)

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/train_rd_adapt_v9b.py <<'PYEOF_TRAIN9B'
'''
mid1 = '''PYEOF_TRAIN9B
cat > $AN/adapt-v9b-blender.sbatch <<'SBEOF_ADAPT9XB'
'''
mid2 = '''SBEOF_ADAPT9XB
cat > $AN/adapt-v9b.sbatch <<'SBEOF_ADAPT9X'
'''
mid3 = '''SBEOF_ADAPT9X
cat > $AN/gate_batch14_v1.py <<'PYEOF_GATE'
'''
mid4 = '''PYEOF_GATE
cat > $AN/gate-batch14.sbatch <<'SBEOF_GATE'
'''
mid5 = '''SBEOF_GATE
cat > $AN/submit-batch14-sweep-v1.sh <<'SWEEP14'
'''
mid6 = '''SWEEP14
cat > $AN/submit-batch14-v1.sh <<'SUB14'
'''
tail = '''SUB14
bash $AN/submit-batch14-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %.12E %R"
```

## 2. After 10-15 min: self-check + smoke of the three smoke jobs (room 10, garden 5, lego 2.5) and the first densify lines

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %.12E %R"
for f in $AN/adapt9x-*-adapt9x-*.out $AN/adapt9xb-*-adapt9xb-*.out; do echo "=== $f"; grep -h "self-check\\|smoke v9 plan\\|smoke probe-a\\|smoke prune\\|smoke final\\|note:\\|FAILED\\|=== .* target" $f | tail -8 | cut -c1-600; done
for d in /scratch/zr2245/*-adapt9x-t*mb /scratch/zr2245/*-adapt9xb-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\\[RD adapt\\] target\\|^\\[RD 3500\\]\\|^\\[RD 4500\\]" $d/train.log | cut -c1-300; done
```

Expect `phase A aims at 12.50` (room 10), `25.00` (room 20), `6.25` (garden 5), `2.75` (lego 2.5) in the `[RD adapt] target` line, and no FAILED. A FAILED job: paste me the last 40 lines of its `.out`; do not resubmit. The gate ignores failed jobs (it needs 4 of 5 Mip rows).

## 3. In the morning: everything in one paste

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-16hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "adapt9x|gate14|JobID"
echo "===== gate"; cat $AN/gate-batch14.log 2>/dev/null || echo "no gate log (gate job did not run: section 4)"
echo "===== adapt-v9b.txt"; cat $AN/adapt-v9b.txt
echo "===== adapt-v9b-blender.txt"; cat $AN/adapt-v9b-blender.txt
echo "===== 9500 plan + first probe + phase-B summary + done"; for d in /scratch/zr2245/*-adapt9x-t*mb /scratch/zr2245/*-adapt9xb-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\\[RD 9500\\]\\|phase A ended\\|RD adapt 12500\\] probe\\|^\\[RD prune\\] *1[5-9][05]00.*collapsed\\|^Training is done" $d/train.log | cut -c1-330 | head -8; done
```

## 4. Only if the gate job did not run (not accepted, or you want to judge by hand): the gate on the login/cpu node, then the sweep

```bash
AN=/scratch/zr2245/analysis
python3 $AN/gate_batch14_v1.py $AN/adapt-v9b.txt $AN/adapt-v9b-blender.txt | tee $AN/gate-batch14.log
# then, if GATE SUBMIT says all / mip / blender:
# bash $AN/submit-batch14-sweep-v1.sh all
```
'''
md = head + tr + mid1 + sb + mid2 + sm + mid3 + gp + mid4 + GATE_SB + mid5 + SWEEP + mid6 + SUBMIT + tail
open(out, 'w').write(md)
open(R + '/scripts/gate-batch14.sbatch', 'w').write(GATE_SB); open(R + '/scripts/submit-batch14-sweep-v1.sh', 'w').write(SWEEP); open(R + '/scripts/submit-batch14-v1.sh', 'w').write(SUBMIT)
print(f'wrote {out}: {len(md.splitlines())} lines | train {tr_n} {tr_md5} | blender sbatch {sb_n} {sb_md5} | mip sbatch {sm_n} {sm_md5} | gate {gp_n} {gp_md5}')
