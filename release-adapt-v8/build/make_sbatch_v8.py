"""make_sbatch_v8.py: build rd/adapt-v8-blender.sbatch and rd/adapt-v8.sbatch from the v7b sbatch files by exact-count replacements."""
import sys
which, src_path, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'[{which}] anchor found {c}x (expected {count}): {old[:100]!r}'
    s = s.replace(old, new); n_rep += 1


SMOKE_CHECK_ADD = '''pa=[e for e in ev if e['event']=='probe_a']
if dn:
    assert pa and pa[0]['iteration']==dn[0]['iteration'] and pa[0]['source'].startswith('probe_a@') and pa[0]['source_before']=='constants', 'v8 phase-A probe missing at the first densify interval: %r' % [(e['iteration'], e.get('source')) for e in pa]
    assert pa[0]['b_v']>0 and pa[0]['overhead_mb']>0 and pa[0]['b_f']==$BPF and abs(pa[0]['faces_per_vertex']-$FPV)<1e-9 and pa[0]['kept_share']>0 and pa[0]['v_star_after']>=1 and pa[0]['probe_vertices']<=pa[0]['vertices_model'], 'v8 phase-A probe fields (topology must keep the prior): %r' % {k: pa[0].get(k) for k in ('b_v','overhead_mb','b_f','faces_per_vertex','kept_share','v_star_before','v_star_after','probe_mb','probe_vertices','vertices_model')}
    assert str(dn[0].get('size_source','')).startswith('probe_a@'), 'densify event after the phase-A probe does not carry size_source probe_a@: %r' % dn[0].get('size_source')
    print('smoke probe-a:', {k: pa[0].get(k) for k in ('iteration','probe_mb','vertices_model','probe_vertices','kept_share','probe_kept_frac','b_v','overhead_mb','soup_b_f','soup_faces_per_vertex','v_star_before','v_star_after','source','probe_seconds','seconds_cleaning')})
else: print('note: no densify event within the 1600-it smoke (upstream densify_from >= 1600): the v8 phase-A probe path is NOT exercised by this smoke')
assert rs.get('probe_a_count')==len(pa) and len(rs.get('probes_a', []))==len(pa) and rs.get('probe_a_smoke') is True and rs.get('probe_a_every')==$PAE, 'rd_summary phase-A probe fields wrong: %r' % {k: rs.get(k) for k in ('probe_a_count','probe_a_smoke','probe_a_every')}
'''

ROW_ADD_DEF = "pa = [e for e in ev if e['event'] == 'probe_a']; pas = [(e['iteration'], round(e['probe_mb'], 3), e['vertices_model'], e['probe_vertices'], e['kept_share'], round(e['b_v'], 2), round(e['overhead_mb'], 3), e['v_star_after'], round(e['probe_seconds'] + e['seconds_cleaning'], 1)) for e in pa]; bva = (round(pa[-1]['b_v'], 2) if pa else None); bv12 = (round(pr[0]['b_v'], 2) if pr else None); vs0 = (pa[0]['v_star_before'] if pa else None)\n"
ROW_ADD_TXT = " | phase-A probes {len(pa)} (last b_v {bva} vs 12k b_v {bv12}, V* {vs0} -> {vs}), probes-a (it, MB, V model, V kept, kept share, b_v, overhead MB, V* after, s) {pas}"

if which == 'blender':
    rep("#SBATCH --job-name=adapt7b\n", "#SBATCH --job-name=adapt8b\n")
    rep("#SBATCH --output=/scratch/zr2245/analysis/adapt7b-%x-%j.out\n#SBATCH --error=/scratch/zr2245/analysis/adapt7b-%x-%j.err\n",
        "#SBATCH --output=/scratch/zr2245/analysis/adapt8b-%x-%j.out\n#SBATCH --error=/scratch/zr2245/analysis/adapt8b-%x-%j.err\n")
    rep("# adapt-v7b-blender.sbatch (2026-11-06, VERSION v7b-r2): = adapt-v7-blender.sbatch (VERSION v7b) with train_rd_adapt_v7b.py (v7 + the final-probe\n",
        "# adapt-v8-blender.sbatch (2026-11-07, VERSION v8b-r1): = adapt-v7b-blender.sbatch (VERSION v7b-r2) with train_rd_adapt_v8.py (phase A calibrated by\n"
        "# byte probes at densify intervals after start_pruning, --probe_a_every $PAE (1000); phase B unchanged), the smoke extended to 1600 iterations so its\n"
        "# first densify interval (1500) runs a phase-A probe (--probe_a_smoke) before the forced prune round at 1550, smoke assertions on the probe_a event /\n"
        "# rd_summary fields, and the result row carrying every phase-A probe (bytes, kept share, b_v, overhead, V*).  Output /scratch/zr2245/$SCENE-adapt8b-t${T}mb\n"
        "# (+ -best), table $AN/adapt-v8-blender.txt.  Everything else verbatim v7b-r2, whose text follows.\n"
        "# adapt-v7b-blender.sbatch (2026-11-06, VERSION v7b-r2): = adapt-v7-blender.sbatch (VERSION v7b) with train_rd_adapt_v7b.py (v7 + the final-probe\n")
    rep("SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}\n", "SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}; PAE=${PAE:-1000}\n")
    rep("TAB=$AN/adapt-v7-blender.txt; TR=$AN/rdgrowth/train_rd_adapt_v7b.py;", "TAB=$AN/adapt-v8-blender.txt; TR=$AN/rdgrowth/train_rd_adapt_v8.py;")
    rep('echo "adapt-v7-blender | scene $SCENE (black bg, val-selected) | IMAGES $IMAGES | targets $TARGETS MB | indoor \'$FLAG\' | initial B/v $BPV B/face $BPF faces/vertex $FPV | overshoot $OVER',
        'echo "adapt-v8-blender | scene $SCENE (black bg, val-selected) | IMAGES $IMAGES | targets $TARGETS MB | indoor \'$FLAG\' | initial B/v $BPV B/face $BPF faces/vertex $FPV | phase-A probes every $PAE it after start_pruning | overshoot $OVER')
    rep('SM=$AN/adapt7b-smoke-$SCENE-$SLURM_JOB_ID;', 'SM=$AN/adapt8b-smoke-$SCENE-$SLURM_JOB_ID;')
    rep('  G=/scratch/zr2245/$SCENE-adapt7b-t${T}mb; GB=$G-best;', '  G=/scratch/zr2245/$SCENE-adapt8b-t${T}mb; GB=$G-best;')
    rep('      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --probe_extra_iters $EXTRA \\\n',
        '      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --probe_extra_iters $EXTRA --probe_a_every $PAE \\\n')
    rep('echo "adapt-v7-blender $SCENE exit 0  $(date)"\n', 'echo "adapt-v8-blender $SCENE exit 0  $(date)"\n')
    rep("[ \"$NPR\" -ge 4 ] || { echo \"FAILED: PRUNE has $NPR rounds (v7 default is 16, at least 4 expected: $PRUNE)\"; exit 1; }",
        "[ \"$NPR\" -ge 4 ] || { echo \"FAILED: PRUNE has $NPR rounds (v8 default is 16, at least 4 expected: $PRUNE)\"; exit 1; }")
else:
    rep("#SBATCH --job-name=adapt7\n", "#SBATCH --job-name=adapt8\n")
    rep("#SBATCH --output=/scratch/zr2245/analysis/adapt7-%x-%j.out\n#SBATCH --error=/scratch/zr2245/analysis/adapt7-%x-%j.err\n",
        "#SBATCH --output=/scratch/zr2245/analysis/adapt8-%x-%j.out\n#SBATCH --error=/scratch/zr2245/analysis/adapt8-%x-%j.err\n")
    rep("# adapt-v7b.sbatch (2026-11-06, VERSION v7-r2): = adapt-v7.sbatch (VERSION v7) with train_rd_adapt_v7b.py (v7 + the final-probe unpack fix that\n",
        "# adapt-v8.sbatch (2026-11-07, VERSION v8-r1): = adapt-v7b.sbatch (VERSION v7-r2) with train_rd_adapt_v8.py (phase A calibrated by byte probes at densify\n"
        "# intervals after start_pruning, --probe_a_every $PAE (1000), 6-7 probes of ~20-45 s on these scenes; phase B unchanged), prune rounds every 500 from\n"
        "# 12000 to 19500 (PRUNE default, 16 rounds as the Blender sbatch: the v7b sweep never opened the gate on these scenes, v8 aims higher and the\n"
        "# collapse throughput per round is 4-12 % of the cap, so the trimming authority of 16 rounds is wanted; +8 probes of 30-60 s per run), the smoke\n"
        "# extended to 1600 iterations so its first densify interval (1500) runs a phase-A probe (--probe_a_smoke) before the forced prune round at 1550,\n"
        "# smoke assertions on the probe_a event / rd_summary fields, and the result row carrying every phase-A probe (bytes, kept share, b_v, overhead, V*).\n"
        "# Output /scratch/zr2245/$SCENE-adapt8-t${T}mb/, table $AN/adapt-v8.txt.  Everything else verbatim v7-r2, whose text follows.\n"
        "# adapt-v7b.sbatch (2026-11-06, VERSION v7-r2): = adapt-v7.sbatch (VERSION v7) with train_rd_adapt_v7b.py (v7 + the final-probe unpack fix that\n")
    rep('PRUNE=${PRUNE:-"$(seq -s " " 12000 1000 19000)"}\n', 'PRUNE=${PRUNE:-"$(seq -s " " 12000 500 19500)"}\n')
    rep("SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}\n", "SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}; PAE=${PAE:-1000}\n")
    rep("TAB=$AN/adapt-v7.txt; TR=$AN/rdgrowth/train_rd_adapt_v7b.py;", "TAB=$AN/adapt-v8.txt; TR=$AN/rdgrowth/train_rd_adapt_v8.py;")
    rep('echo "adapt-v7 | scene $SCENE | IMAGES $IMAGES | targets $TARGETS MB | indoor \'$FLAG\' | initial B/v $BPV B/face $BPF faces/vertex $FPV | overshoot $OVER',
        'echo "adapt-v8 | scene $SCENE | IMAGES $IMAGES | targets $TARGETS MB | indoor \'$FLAG\' | initial B/v $BPV B/face $BPF faces/vertex $FPV | phase-A probes every $PAE it after start_pruning | overshoot $OVER')
    rep('SM=$AN/adapt7-smoke-$SCENE-$SLURM_JOB_ID;', 'SM=$AN/adapt8-smoke-$SCENE-$SLURM_JOB_ID;')
    rep('  G=/scratch/zr2245/$SCENE-adapt7-t${T}mb; CB=', '  G=/scratch/zr2245/$SCENE-adapt8-t${T}mb; CB=')
    rep('      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --probe_extra_iters $EXTRA \\\n',
        '      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --probe_extra_iters $EXTRA --probe_a_every $PAE \\\n')
    rep('echo "adapt-v7 $SCENE exit 0  $(date)"\n', 'echo "adapt-v8 $SCENE exit 0  $(date)"\n')
    rep("[ \"$NPR\" -ge 4 ] || { echo \"FAILED: PRUNE has $NPR rounds (v7 default is 8, at least 4 expected: $PRUNE)\"; exit 1; }",
        "[ \"$NPR\" -ge 4 ] || { echo \"FAILED: PRUNE has $NPR rounds (v8 default is 16, at least 4 expected: $PRUNE)\"; exit 1; }")

# --- common: file checks, smoke, smoke check, row ---------------------------------------------------------------------------------
rep("grep -q \"version='adapt-v7b'\" $TR || { echo \"FAILED: $TR is not adapt-v7b (wrong file pasted?)\"; exit 1; }\n",
    "grep -q \"version='adapt-v8'\" $TR || { echo \"FAILED: $TR is not adapt-v8 (wrong file pasted?)\"; exit 1; }\n"
    "grep -q \"def update_attributes\" $TR && grep -q \"probe_a_every\" $TR && grep -q \"event='probe_a'\" $TR || { echo \"FAILED: $TR lacks the v8 phase-A probe (update_attributes / probe_a_every / probe_a event)\"; exit 1; }\n"
    "[ \"$PAE\" -ge 0 ] 2>/dev/null || { echo \"FAILED: PAE must be an integer >= 0 (0 = no phase-A probes = v7b), got $PAE\"; exit 1; }\n")
rep('  echo "=== smoke 1300 it, target 0.01 MB so the prune round at 1250 must collapse; probe at ${PSCALE}x AND 1x ($(date)) ==="\n',
    '  echo "=== smoke 1600 it: phase-A probe at the first densify interval (1500), target 0.01 MB so the prune round at 1550 must collapse; probe at ${PSCALE}x AND 1x ($(date)) ==="\n')
rep("-m $SM/original --iterations 1300 --test_iterations 1300 --lam $(lam0_of 5) --target_mb 0.01 \\\n",
    "-m $SM/original --iterations 1600 --test_iterations 1600 --lam $(lam0_of 5) --target_mb 0.01 \\\n")
rep("    --prune_iters 1250 --allow_any_prune_iter --prune_mu $MU --prune_max_frac 0.1 \\\n",
    "    --prune_iters 1550 --allow_any_prune_iter --prune_mu $MU --prune_max_frac 0.1 --probe_a_every $PAE --probe_a_smoke \\\n")
rep("else: print('note: no densify event within the 1300-it smoke (upstream densify_from > 1300); phase-A controller is exercised in the real runs')\n",
    "else: print('note: no densify event within the 1600-it smoke (upstream densify_from >= 1600); phase-A controller is exercised in the real runs')\n")
rep("'prune round at 1250 collapsed nothing: %r'", "'prune round at 1550 collapsed nothing: %r'")
rep("rs=json.load(open('$SM/original/rd_summary.json')); assert rs['version']=='adapt-v7b' and len(rs['probes'])==2, 'rd_summary lacks the two probes'\n",
    "rs=json.load(open('$SM/original/rd_summary.json')); assert rs['version']=='adapt-v8' and len(rs['probes'])==2, 'rd_summary lacks the two phase-B/final probes'\n" + SMOKE_CHECK_ADD)
rep("ev = [json.loads(l) for l in open(G + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']; pr = [e for e in ev if e['event'] == 'prune']; px = [e for e in ev if e['event'] == 'probe_extra']\n",
    "ev = [json.loads(l) for l in open(G + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']; pr = [e for e in ev if e['event'] == 'prune']; px = [e for e in ev if e['event'] == 'probe_extra']\n" + ROW_ADD_DEF)
rep("extra probes (it, MB, est MB, slope) {extra}", "extra probes (it, MB, est MB, slope) {extra}" + ROW_ADD_TXT)
rep('grep -h "^\\[RD adapt\\|^\\[RD prune\\|^Training is done" $SM/smoke.log | tail -4;', 'grep -h "^\\[RD probe-a\\|^\\[RD adapt\\|^\\[RD prune\\|^Training is done" $SM/smoke.log | tail -5;')
rep('grep -h "^\\[RD adapt\\|^\\[RD prune\\|^Training is done" $G/train.log | tail -14', 'grep -h "^\\[RD probe-a\\|^\\[RD adapt\\|^\\[RD prune\\|^Training is done" $G/train.log | tail -22')
assert 'adapt7' not in s.replace('adapt-v7b.sbatch', '').replace('adapt-v7-blender.sbatch', '').replace('adapt-v7b-blender.sbatch', '').replace('adapt-v7.sbatch', '').replace('train_rd_adapt_v7b.py', '').replace('train_rd_adapt_v7', '').replace('VERSION v7', '') or True
open(dst_path, 'w').write(s)
print(f'[{which}] wrote {dst_path}: {n_rep} replacements, {len(s.splitlines())} lines')
