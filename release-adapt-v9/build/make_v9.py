"""make_v9.py: build analysis/rdgrowth/train_rd_adapt_v9.py from train_rd_adapt_v8.py by exact-count text replacements (v8 untouched)."""
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'anchor found {c}x (expected {count}): {old[:90]!r}'
    s = s.replace(old, new); n_rep += 1


HEADER = '''"""train_rd_adapt_v9.py (adapt-v9, 2026-11-07 = adapt-v8 with ONE phase-A control fix; the v8 phase-A probes and the v7b phase B are untouched;
v1-v8 kept as they are).  What node-batch11 (v8: lego/chair x 2.5/4, room/bicycle x 10 MB) showed: the soup probe measures the attribute price
right (b_v at 9500 within 1-8 % of the 12k mesh probe, overhead within 1 %), yet every run landed BELOW v7b (hit 0.71-0.92 vs 0.78-1.01), and the
densify logs explain why: at every interval the plan counted ONE MORE densify interval than upstream runs (remaining = (until - it) // interval + 1
= 2 at 9500, but upstream densifies only while iteration < densify_until, so 9500 is the LAST interval: 1500 ... 9500 are 17 intervals, the plan
believed 18).  The geometric plan therefore spread the growth still needed over one interval too many and, at the last one, aimed at
sqrt(V x V*) instead of V*: the model after the last upstream prune was 0.81-0.92 of V* on all six runs (predicted sqrt(V_9500 / V*) 0.83-0.92),
and the same shortfall sits in every v4-v7b run (the sweep's 'phase A reaches 0.83-0.97 of V*').  v7b's constant prior put V* 10-30 % too HIGH
on Blender, which hid the shortfall there; v8's measured V* removed the mask.  v9:
 1. remaining = ceil((densify_until - iteration) / interval): the intervals upstream will still run, this one included (iteration + k * interval
    < densify_until).  Nothing else in plan_splits changes; sigmas and desired_next follow the corrected count.
 2. The densify line prints the plan cap and upstream's limiter separately ('cap N limiter M'; v4-v8 printed min(cap, limiter) as 'limit', which
    read as limiter-bound when it was the plan's own cap) and the densify event carries limiter_max.
 3. plan_splits' limiter model: upstream's add_new_gs splits at most add_pct * V - V TRIANGLES per interval and rd_select's max_split is that
    same count; v4-v8 divided it by 3 inside the plan's forward simulation (as if it were vertices), so the plan believed the limiter 3x
    tighter than it is, front-loaded too early and, when it did, asked for only a third of the allowance.  v9 uses the count as applied.
 4. version strings 'adapt-v9'; --probe_a_every etc. as in v8.
Expected (batch11 numbers, V* reached, v8 size prior): room 10 -> ~0.91, bicycle 10 -> ~0.87 (the remaining miss is the topology prior, 5 B/V vs
3.7-4.5 measured at delivery on Mip-360), lego/chair 2.5 -> 1.00-1.10 before phase B (prior 3-10 % high on Blender: phase B prunes the surplus),
lego/chair 4 limiter-bound as before.

adapt-v8 header follows.
train_rd_adapt_v8.py (adapt-v8, 2026-11-07 = '''
rep('"""train_rd_adapt_v8.py (adapt-v8, 2026-11-07 = ', HEADER)
rep("                    remaining = max(1, (opt.densify_until_iter - iteration) // opt.densification_interval + 1)\n",
    "                    remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))  # v9: intervals upstream will still run, this one included (iteration + k * interval < densify_until); v4-v8 counted one too many, so the plan always left one interval's growth undone\n")
rep("                    limiter_bound = bool(front or max_split < cap); max_split = min(max_split, cap)  # the plan caps the count; lambda is the price of that count\n",
    "                    max_split_up = max_split; limiter_bound = bool(front or max_split_up < cap); max_split = min(max_split_up, cap)  # the plan caps the count; lambda is the price of that count; max_split_up = what upstream's limiter allows\n")
rep("limiter_bound=limiter_bound, front_loaded=front,", "limiter_bound=limiter_bound, limiter_max=int(max_split_up), front_loaded=front,")
rep('f"(limit {max_split}) ratio q50/q90/q99', 'f"(cap {cap} limiter {max_split_up}) ratio q50/q90/q99')
rep("        d = min(geo(v, remaining - j, s), max(0.0, add_pct * v - v) / 3.0)\n", "        d = min(geo(v, remaining - j, s), max(0.0, add_pct * v - v))  # v9: upstream's allowance is add_pct * V - V TRIANGLES split per interval (add_new_gs), the same unit as d\n")
rep("    if v < 0.995 * V_star: return max(want, max(0.0, add_pct * V - V) / 3.0), True\n", "    if v < 0.995 * V_star: return max(want, max(0.0, add_pct * V - V)), True\n")
rep("    With add_pct (upstream growth limiter: at most add_pct * V - V new vertices per interval) the plan is simulated forward\n", "    With add_pct (upstream growth limiter: at most add_pct * V - V triangles split per interval, upstream add_new_gs; v4-v8 divided this\n    by 3 as if it were vertices, understating the allowance 3x and front-loading too early) the plan is simulated forward\n")
rep("version='adapt-v8'", "version='adapt-v9'", 2)
rep("| adapt-v8: target {rd['target_mb']} MB, final probe", "| adapt-v9: target {rd['target_mb']} MB, final probe")
rep("'| RD growth+prune adapt-v8 target_mb'", "'| RD growth+prune adapt-v9 target_mb'")
assert "version='adapt-v8'" not in s and s.count("version='adapt-v9'") == 2
assert 'adapt-v8 header follows' in s and 'phase A CALIBRATED BY THE BYTE PROBE' in s
open(dst_path, 'w').write(s)
print(f'{n_rep} replacements -> {dst_path}: {len(s.splitlines())} lines')
