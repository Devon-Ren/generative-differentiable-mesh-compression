"""make_sbatch_v9e.py: adapt-v9e.sbatch = adapt-v9d.sbatch with new NAMES only (jobs adapt9e, run dirs $SCENE-adapt9e-t${T}mb, table adapt-v9e.txt,
VERSION v9e-r1); the trainer stays train_rd_adapt_v9d.py + rd_prune_v2.py.  The aim factor OVER is passed per capture class by the submit script
(indoor 1.16, outdoor 1.27, T&T 1.23), so the v9d default 1.25 in the file is irrelevant; a note in the header says so."""
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'anchor found {c}x (expected {count}): {old[:100]!r}'
    s = s.replace(old, new); n_rep += 1


if 'adapt9db' in s: rep('adapt9db', 'adapt9eb', s.count('adapt9db'))  # only in the v9d header comment
rep('adapt9d', 'adapt9e', s.count('adapt9d')); rep('v9d-r1', 'v9e-r1', s.count('v9d-r1'))
rep('TAB=$AN/adapt-v9d.txt;', 'TAB=$AN/adapt-v9e.txt;')
rep("# adapt-v9d.sbatch (2026-11-08, VERSION v9e-r1): = adapt-v9c.sbatch (VERSION v9c-r1) with train_rd_adapt_v9d.py + rd_prune_v2.py.\n",
    "# adapt-v9e.sbatch (2026-11-08 evening, VERSION v9e-r1): = adapt-v9d.sbatch (VERSION v9d-r1) with NEW NAMES ONLY (jobs adapt9e, run dirs\n"
    "# $SCENE-adapt9e-t${T}mb, table $AN/adapt-v9e.txt); same trainer train_rd_adapt_v9d.py + rd_prune_v2.py.  Used for the calibrated-aim sweep\n"
    "# (node-batch17): the submit script passes OVER per capture class, indoor 1.16 / outdoor 1.27 / T&T 1.23, measured two independent ways\n"
    "# (kappa-analysis-v1 on 59 earlier runs: kappa x topology x soup over-read; and the v9b sweep's pre-trim landing with the uniform 1.25:\n"
    "# indoor 1.076, outdoor 0.988, T&T 1.020).  The un-stacking growth during phase B is 1.10-1.49x on indoor, beyond what 30 rounds at\n"
    "# ~0.4 % of V per round can trim, so the aim is the lever; phase B lands the rest.  v9d header follows.\n"
    "# adapt-v9d.sbatch (2026-11-08, VERSION v9d-r1): = adapt-v9c.sbatch (VERSION v9c-r1) with train_rd_adapt_v9d.py + rd_prune_v2.py.\n")
rep('echo "adapt-v9d $SCENE exit 0  $(date)"', 'echo "adapt-v9e $SCENE exit 0  $(date)"')
assert 'adapt-v9e.txt' in s and 'train_rd_adapt_v9d.py' in s and "version='adapt-v9d'" in s
open(dst_path, 'w').write(s)
print(f'{n_rep} replacements -> {dst_path}: {len(s.splitlines())} lines')
