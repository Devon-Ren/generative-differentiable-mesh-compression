"""make_v9b.py: build train_rd_adapt_v9b.py from train_rd_adapt_v9.py by exact-count text replacements (v9 untouched).
Code diff: ONE line (the phase-A price is kept when the plan asks for zero splits) + version strings."""
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'anchor found {c}x (expected {count}): {old[:90]!r}'
    s = s.replace(old, new); n_rep += 1


HEADER = '''"""train_rd_adapt_v9b.py (adapt-v9b, 2026-11-07 evening = adapt-v9 with ONE code fix; phase-A plan, phase-A probes and phase B otherwise
verbatim v9; v1-v9 kept as they are).  Two things the v9 batch12 logs showed:
 1. CONTINUOUS PRICE.  At the last densify interval (9500) the plan has nothing left to ask for (remaining == 1 -> desired_next = 0 ->
    k_asked = 0) and rank_lambda(k=0) returns lam_max = 0.1, so every run entered phase B with lambda = 0.1.  Phase B then admits every
    edge under lambda x mu (logs: '663k-7.9M edges below lam 1e-1'), wanted = min(max(desired, n_lam), excess_pairs) becomes the WHOLE
    excess in the first round whose landing estimate is above target, and the progressive lambda-prune (the regulariser v6 measured at
    +0.3/+0.5 dB on Blender 4 MB against a closed gate) degenerates into one capped cut.  v9b: when the plan asks for zero splits the
    price of the interval just run is kept (lam_now = lam_used), so phase B starts from the last real rank price of phase A, which is
    what 'the same price in reverse' means.  One line.  The 'lam_final (shadow) 1e-1' of untrimmed rows disappears with it.
 2. ASYMMETRIC AIM (no code; the sbatch sets --phase_a_overshoot).  batch12's remaining miss on Mip-360 (hit room 10 0.877, bicycle 10
    0.790, room 20 0.829) decomposes as hit = kappa_end x (b_v,mesh + b_f x fpv) / (b_v,soup + 5): the final cleaning keeps 0.92-0.94
    of the model vertices (Blender 0.995), the delivered Delaunay topology costs 3.7-4.4 B/vertex against the prior 5, and the soup
    over-reads b_v by 3-9 %; predicted 0.880 / 0.807 / 0.829 vs observed 0.877 / 0.790 / 0.829.  None of the three is observable on
    the soup before the 11k Delaunay rebuild, and nothing adds bytes after 9500, so the error must be put on the side phase B can
    correct: phase A aims at target x 1.25 (= 1 / 0.80) on Mip-360 / T&T and x 1.1 on NeRF-Synthetic, one value per dataset class
    for every scene, and the 15 prune rounds (12500 ... 19500; the 12000 probe right after the Delaunay reads high and poisoned the
    first slope) trim with the real encoder in the loop.
 3. version strings 'adapt-v9b'.
Expected: Mip-360 / T&T land 0.99-1.09 before phase B and within +-5 % after (the last-estimate-vs-final error is -1.4..+4.4 %, 20 MB
indoor the worst); Blender 1.03-1.13 before, 0.97-1.03 after; every run now exercises phase B.

adapt-v9 header follows.
'''
rep('"""train_rd_adapt_v9.py (adapt-v9, 2026-11-07 = ', HEADER + 'train_rd_adapt_v9.py (adapt-v9, 2026-11-07 = ')
rep("                    lam_now = rank_lambda(ratio_pre, n_allowed, k_asked, rd['lam_min'], rd['lam_max'])  # price of the next planned count, corrected by the measured yield\n",
    "                    lam_now = rank_lambda(ratio_pre, n_allowed, k_asked, rd['lam_min'], rd['lam_max']) if k_asked > 0 else lam_used  # price of the next planned count, corrected by the measured yield; v9b: nothing left to plan (last interval) -> keep the price just used, so phase B starts from the last real phase-A price instead of lam_max\n")
rep("version='adapt-v9'", "version='adapt-v9b'", 2)
rep("| adapt-v9: target {rd['target_mb']} MB, final probe", "| adapt-v9b: target {rd['target_mb']} MB, final probe")
rep("'| RD growth+prune adapt-v9 target_mb'", "'| RD growth+prune adapt-v9b target_mb'")
assert "version='adapt-v9'" not in s and s.count("version='adapt-v9b'") == 2
assert 'adapt-v9 header follows' in s and 'max_split_up = max_split' in s and 'else lam_used' in s
open(dst_path, 'w').write(s)
print(f'{n_rep} replacements -> {dst_path}: {len(s.splitlines())} lines')
