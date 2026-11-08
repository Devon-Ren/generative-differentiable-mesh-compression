"""make_v9c.py: build train_rd_adapt_v9c.py from train_rd_adapt_v9b.py by exact-count replacements (v9b untouched).
Code diff: ONE line (the landing extrapolation never uses a negative slope) + version strings."""
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'anchor found {c}x (expected {count}): {old[:90]!r}'
    s = s.replace(old, new); n_rep += 1


HEADER = '''"""train_rd_adapt_v9c.py (adapt-v9c, 2026-11-08 = adapt-v9b with ONE code fix; everything else verbatim v9b; v1-v9b kept as they are).
What the v9b sweep (node-batch14, 47 runs) showed: landing 1.006 / 1.020 / 0.995 x target at 5 / 10 / 20 MB (32 of 33 Mip-360 / T&T rows
within +-10 %), but seven indoor rows ended 5-13 % OVER target although phase B ran 7-11 active rounds: the gate stayed closed for the first
3-9 rounds because the landing ESTIMATE read under target (kitchen 10: probe 9.43 MB at 12500, estimates 8.51 / 9.24 / 9.79 at 13000-14000,
final 11.34).  Cause: extrapolate_probe fits a least-squares slope over the probes so far and projects it 11 000 iterations ahead; with two or
three probes a small negative sample slope (the probe fluctuates by a few percent under the opacity ramp) becomes -0.8 MB at the horizon.
The kept set only GROWS under the opacity ramp (kappa-analysis-v1: final / first probe 1.00-1.39 on indoor, 1.0-1.1 elsewhere; the virtual
size adds the bytes already collapsed, so collapses do not make it shrink either), so a negative slope is never information: v9c clamps the
slope used for the estimate at zero (the raw slope is still logged).  The gate then opens as soon as the probe itself exceeds the target
and phase B gets its first rounds back.  Expected on the over-target rows: the first active round 2-4 rounds earlier; together with the
sbatch's 30 prune rounds (every 250 from 12500; +15 probes of ~30 s) the trimming authority doubles.  version strings 'adapt-v9c'.

adapt-v9b header follows.
'''
rep('"""train_rd_adapt_v9b.py (adapt-v9b, 2026-11-07 evening = ', HEADER + 'train_rd_adapt_v9b.py (adapt-v9b, 2026-11-07 evening = ')
rep("    return r_meas + slope * (horizon_iter - iteration), slope * 1000.0\n",
    "    return r_meas + max(0.0, slope) * (horizon_iter - iteration), slope * 1000.0  # v9c: the kept set only grows under the opacity ramp, so a negative sample slope is noise, never information; the raw slope is still returned for the log\n")
rep("version='adapt-v9b'", "version='adapt-v9c'", 2)
rep("| adapt-v9b: target {rd['target_mb']} MB, final probe", "| adapt-v9c: target {rd['target_mb']} MB, final probe")
rep("'| RD growth+prune adapt-v9b target_mb'", "'| RD growth+prune adapt-v9c target_mb'")
assert "version='adapt-v9b'" not in s and s.count("version='adapt-v9c'") == 2 and 'max(0.0, slope)' in s and 'else lam_used' in s
open(dst_path, 'w').write(s)
print(f'{n_rep} replacements -> {dst_path}: {len(s.splitlines())} lines')
