"""train_rd_adapt_v9b.py (adapt-v9b, 2026-11-07 evening = adapt-v9 with ONE code fix; phase-A plan, phase-A probes and phase B otherwise
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
train_rd_adapt_v9.py (adapt-v9, 2026-11-07 = adapt-v8 with ONE phase-A control fix; the v8 phase-A probes and the v7b phase B are untouched;
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
train_rd_adapt_v8.py (adapt-v8, 2026-11-07 = adapt-v7b with phase A CALIBRATED BY THE BYTE PROBE; phase B of v7b untouched; v1-v7b kept as
they are).  What the 47-run v7b sweep measured (analysis/rd/adapt-v7b-sweep-analysis.txt): the delivery is exact (probe / delivered 0.999-1.001 on
all 49 runs) and phase B lands within 3 % of its estimate, but every Mip-360 / T&T run ended UNDER target (5 / 10 / 20 MB: 0.89 / 0.83 / 0.77 x,
within +-10 % on 8 of 33) because phase A plans its vertex aim V* from the CONSTANT size prior (10 B per vertex + 2 B per face x 2.5 faces per vertex
+ 0.6 MB = 11 B per vertex + 0.6 MB) and nothing can add bytes after densification ends at 10000: the measured bytes per vertex at 12000 were 11-12.8
at 5 MB but 9-11 at 10 MB and 7.7-10 at 20 MB (the fixed overhead is spread over more vertices and the attribute streams code denser point sets
cheaper), so V* was 10-30 % too low on the large targets and phase B had nothing to trim.  v8 measures the size model DURING phase A:
 1. Phase-A probes.  At every densify interval after upstream has started pruning (iteration > start_pruning: before that the model is the
    untrained initial soup, 70-95 % of which the first prune removes) and at least --probe_a_every (1000) iterations after the previous one, the
    byte probe of v2-v7 (upstream final cleaning at the final opacity floor and supersampling simulated on a copy, real delivery encoder in a
    subprocess) runs on the current model: 7 probes on Blender (3500 ... 9500), ~15 s each; 6-7 on Mip-360 / T&T, ~20-45 s each.  A probe whose
    cleaned copy keeps fewer vertices than the dictionary size max(K, R) is skipped (event probe_a_skipped; the encoder's K-means needs K samples)
    and tried again at the next interval; the model keeps its last values.
 2. What is taken from a phase-A probe (SizeModel.update_attributes): the attribute price b_v = (file bytes - topology streams - banks - header)
    per kept vertex and the fixed overhead (banks + header).  NOT the topology price: in phase A the model is a triangle SOUP (every face owns its
    three vertices, no shared connectivity), so its connectivity streams say nothing about the restricted-Delaunay mesh that is delivered; b_f and
    faces per vertex keep their initial values until the first mesh probe at 12000 replaces them (as in v2-v7b).  Topology is ~35-45 % of the
    bytes (Mip-360 2.2-2.9 faces/vertex x 1.75 B = 3.9-5.1 B, Blender 3.4 x 1.76 = 6 B, prior 2.5 x 2 = 5 B), so the remaining prior error of V* is
    within ~+-10 %; the attribute part (b_v 3-9 B, the one that varies with V and dataset) and the overhead (0.67 MB, 1-13 % of the target) are
    measured.  The kept vertex share of the probe (probe vertices / model vertices) is logged, not applied: at 12000 it is 0.8-1.0 and rises to
    ~0.98 by the end of training (the un-stacking of v6/v7), so the delivered count is the model count, as v7b assumes.
 3. V* = vertices_for(target_a) is re-derived from the updated model at every interval (it already was; now the model moves), and the geometric
    plan / limiter simulation / front-loading of v4 act on the measured V*.  A low prior (V* too high) made the plan front-load against the
    limiter; a high prior (V* too low) made it stop growing: both are now the measured value's call.  Phase-A probe events ('probe_a') carry the
    probe bytes, kept share, b_v, overhead and V* before / after the update; the densify events carry size_source.  rd_summary.json carries
    probes_a (every phase-A probe dict) next to probes (phase B + final).
 4. --probe_a_every 0 disables the phase-A probes (= adapt-v7b).  --probe_a_smoke (smoke only) probes at the first densify interval even before
    start_pruning, so the 1600-iteration smoke of the sbatch exercises the new path on the initial soup.
 Not changed: bytes_per_split stays the initial unit (it scales dD/dR and lambda together; lambda is a rank price of a planned count, so the
 unit cancels), phase B (gate, exact accounting, shadow price, growth_capped) and the diagnostic probes are v7b verbatim.  The v7b header follows.)
train_rd_adapt_v7b.py (adapt-v7b, 2026-11-06 = adapt-v7 with ONE fix: the final probe after the upstream cleaning unpacked two values,
probe_bytes returns three since v7 (probe, kept-vertex mask, kept-face mask) -> ValueError at the end of every batch8 smoke; no other
line changed.  The v7 header follows.)
train_rd_adapt_v7.py (adapt-v7, 2026-11-06 = adapt-v6 with the three measured faults of ITS Blender testbed (lego, chair x {2.5, 4, 6} MB,
node-batch7) fixed; phase A untouched; v1-v6 kept as they are):
 1. Byte accounting of a collapse round is exact w.r.t. the probe: removed_mb = (collapsed pairs whose BOTH endpoints were in the probe's kept
    vertex set) x b_v + (removed faces that were in the probe's kept face set) x b_f.  v6 charged pairs x (b_v + b_f x ALL faces removed per
    pair) and over-counted the removal x1.5-1.9 (first rounds), which inflated the virtual-size slope: the second-round extrapolation read
    1.20-1.44 x the final probe and the first 10 rounds pruned against it; at the last round still 1.04-1.06.  Replayed on the v6 logs the
    exact accounting gives 1.006-1.030 at the last round (analysis/rd/adapt-v7-replay-on-v6-logs.txt).  probe_bytes returns the kept face
    mask as well; faces_removed_by() reproduces collapse_edges' degenerate/duplicate rule before the collapse.
 2. The prune is the byte lever, so it only acts while there are bytes to remove: a round collapses only if the extrapolated landing is above
    target (gate), and the count is capped at the pairs that remove the whole excess (wanted = min(max(desired, lambda-admitted), excess /
    bytes per pair)).  Under target -> mode 'none', nothing collapsed, the lambda-admitted count is logged.  v6 ran the fixed-lambda prune in
    all 16 rounds whatever the estimate said and removed 14-19 % of V (4-14 % of it in rounds where the controller wanted MORE bytes); every
    run landed 0.67-0.95 x target.  When the cap or the rank binds, lambda := shadow price of the pairs taken (mode 'capped' / 'count').
    The bytes-per-pair prior before the first collapse is b_v x (kept V / model V)^2 + 2 b_f x kept face share; measured from the first round on.
 3. growth_capped is reported (final event, rd_summary.json, the 'Training is done' line): the last round's extrapolated landing is under target
    AND the upstream add_percentage limiter bound a majority of the densify intervals.  Phase B has no lever that adds bytes; chair 4 / 6 MB
    landed at 0.88 / 0.76 x target with no collapse at all (replay) because phase A reached 0.68 / 0.61 of its vertex aim under the limiter.
 4. --probe_extra_iters (default none): diagnostic probes after the last round (the Blender sbatch asks 22000 26000) that only log the probe
    and its extrapolation, to see where the un-stacking growth ends (the horizon is final_opacity_iter = 24000; v6 saw +0.1-0.5 MB between
    19.5k and the final probe at 30k with no probe in between).  No collapse, no decision.
 Known and left as is: phase A's size prior (bytes_per_vertex 10, bytes_per_face 2, faces/vertex 2.5 = 11 B per vertex + 0.6 MB) is right on
 Mip-360 (measured 7.0 + 2.2-2.9 x 1.75 = 11-12 B) and ~25 % low on Blender (8.2 + 3.4 x 1.8 = 14.3 B), so V* is ~30 % high there; on the
 testbed the limiter masked it.  A dataset-dependent constant is not tuned; the gate and cap of (2) absorb an overshoot of that size.

adapt-v6 (2026-11-06 = adapt-v5 with the findings of the Blender testbed (lego, chair x {2.5, 4, 6} MB) applied; v1-v5 untouched):
 a. Phase-A overshoot default 1.0 (was 1.15).  The margin assumed phase B removes more than the kept set grows after 12k; measured the
    opposite on both datasets: the kept faces un-stack +9-40 % (Blender) / +6-23 % (Mip-360 indoor) while the fixed-lambda prune
    removes 7-10 % of V.  Phase B can only remove bytes, so the aim is the target itself and the trimming is left to the rounds.
 b. More rounds, same code: --prune_iters is meant to be every 500 from 12000 to 19500 (16 rounds; the sbatch sets it).  One round
    collapses only ~3 % of V (of ~22k cap-admitted pairs 12-16k are foldover-rejected), 4 rounds gave 7-10 %; 16 rounds give the
    authority to trim what the extrapolated probe says.  The decision rule is unchanged (lambda prune + count-driven on the estimate).
 c. extrapolate_probe fits the slope by least squares over ALL probes so far ('virtual' size = probe + bytes collapsed before it),
    instead of the last two: with 500-it rounds a two-point slope error of 0.01 MB becomes 0.2 MB at the horizon.  Two probes -> v5.
 d. Diagnostic: how many of a round's collapsed vertices were in the probe's kept set (kept_removed, kept_removed_frac in the prune
    event); the accounting of removed_mb is v5's (est@18k / final 1.00-1.04 on the testbed).
 Everything else (phase A plan, yield-corrected rank price, probe at final floor + 4x, phase B of v3) is adapt-v5 verbatim.

adapt-v5 (2026-11-06 = adapt-v4 with the three findings of ITS 12 runs applied; v1-v4 kept untouched):
 1. The phase-B decision uses the probe EXTRAPOLATED to final_opacity_iter (24000).  Indoors the probe rose 12k->18k with V constant
    (kitchen 20 MB: 14.3 -> 16.7 MB, final 20.5): the kept face set keeps growing while the opacity floor ramps, which no probe at
    <= 18000 can see.  The slope of the last two probes (corrected for what the previous round collapsed) extrapolated to 24000
    lands within ~5 % of the final on 12/12 adapt-v4 runs (outdoor slopes ~0).  The first round has no slope and uses the probe as is.
 2. Phase B is adapt-v3's again: the fixed-lambda prune (edges with dD/dR < lambda_now * mu) every round, plus count-driven
    collapses when the extrapolated probe is above target (threshold = max(rank, lambda * mu), lambda := shadow price when the
    count binds).  adapt-v4 ran count-only and lost 0.07 dB at equal bytes (mean -0.070 vs v3 -0.007): the prune is what keeps
    the run on the fixed-lambda curve; the phase-A margin (x1.15) is what it consumes.
 3. Phase A corrects the rank price by its measured yield: rank_lambda admitted k triangles among the CURRENT ratios, but the next
    interval's ratios differ (the split triangles are gone), and bicycle delivered pass / k = 0.66-0.85 every interval.  The next
    k is desired_next / mean yield of the last three intervals (same window as sigma; 1.0 until measured).
 4. --probe_check_scaling is applied to the mid-training probes only (the final probe runs after upstream's own cleaning, where
    '4x keeps more than 1x' is not meaningful; adapt-v4's smoke failed there on room/kitchen).

adapt-v4 (2026-11-07 = adapt-v3 with the three faults of ITS 12 runs fixed):
 A. The probe's cleaning pass renders at the FINAL supersampling (triangles.scaling = --probe_scaling 4, what upstream uses from
    start_upsampling+5000 on and therefore in the final cleaning) as well as the final opacity floor.  adapt-v3's 1x probes kept
    0.30-0.59 of the faces at 18000 while the final cleaning keeps far more (faces that get a pixel only at 4x), so the probe still
    read 1.03-1.39x below the final size.  --probe_check_scaling (smoke) renders the pass at 1x too and fails unless 4x keeps more.
 B. Phase B is count-driven only: collapses happen only when the probe is ABOVE target (the cheapest independent edges by dD/dR,
    lambda := their shadow price).  adapt-v3 also ran the fixed-lambda prune every round and removed 11-16 % of the vertices while
    the controller wanted more bytes.  Under target -> no collapse (mode 'none').
 C. Phase A plans the vertex count geometrically: V* from the size model at target * overshoot, per-interval growth factor
    f = (V*/V)^(1/remaining), desired splits = V (f - 1) / (3 sigma).  The bytes-linear plan of v3 asked for little early and
    for more than the upstream growth limiter (add_percentage: at most 23-27 % new vertices per interval) allows late, so the
    20 MB runs ended 18-26 % short.  The plan is simulated forward under that limiter (sigma = 1 until start_pruning, the
    measured sigma after, --sigma_prior 0.5 before the first measurement); when the geometric path cannot reach V* the
    interval asks for everything the limiter allows (growth compounds).  limiter_bound / front_loaded are logged.  The next
    interval's plan uses the expected post-prune count V + sigma * (V_after - V).

adapt-v3 (2026-11-06 = adapt-v2 with the three faults of its 12 runs fixed):
 a. Phase A planned with the MEASURED survival of new vertices.  Upstream prunes every 500 iterations, and 30-70 % of the
    vertices created by the previous interval's splits die in that prune (adapt-v2 dump: pruned / newly created = 0.4-0.7 on
    every interval of every run).  sigma = 1 - pruned / new is measured each interval (mean of the last three) and the split
    budget is desired = headroom / (sigma * bytes_per_split * remaining): the gross cost 3 vertices x 11 B per split that
    adapt-v2 assumed was 2-3x too optimistic, so its lambda never fell far enough on the large targets.
 b. lambda is no longer stepped by a gain (ctrl_gamma / deadband removed): after each split round the next lambda is the
    dD/dR ratio of rank desired_next among the current triangles, i.e. the shadow price of the planned count, exactly as
    phase B reports it.  Phase A lands on target_mb * phase_a_overshoot (1.15: a margin phase B trims back to the probe,
    not a calibration -- the result does not depend on its value as long as it covers the final cleaning, 5-8 %).
 c. The byte probe renders the cleaning pass with the FINAL opacity floor (0.9999, what upstream reaches at final_opacity_iter)
    instead of the current one.  adapt-v2's probes at 12k-18k read 1.1-1.6x below the final size because the floor was still
    ramping and faces that end opaque (kept) were dropped; with the final floor the probe sees the final kept set up to the
    geometry that still moves.  faces_kept / faces_model is logged per probe.
 d. Phase B keeps the fixed-lambda prune of train_rd_v3 (edges with dD/dR < lambda_now * mu are collapsed every round, the
    +0.1 dB at equal bytes adapt-v2 lost by collapsing nothing) and adds the count-driven collapses when the probe is above
    target: threshold = max(rank threshold of the wanted count, lambda_now * mu); lambda_now := shadow price when the count
    was binding.  Config/summary version adapt-v3.

adapt-v2 (2026-11-06 = adapt-v1 with the size model measured instead of assumed):
MeshSplatting training with rate-distortion controlled growth AND pruning where lambda is NOT a user constant but is
adapted during training so that the DELIVERED size hits --target_mb.

What changed from adapt-v1 (whose 12 runs landed at 52-77 % of target, all on the fixed-lambda hull):
 1. Size model.  R_hat = (b_v * V + b_f * F + overhead), where b_v = attribute bytes per vertex, b_f = topology bytes per face.
    adapt-v1 used V*B_v + F*B_f with B_v = B_a + 2*B_f, i.e. counted the connectivity twice (x1.30).  The constants
    (--bytes_per_vertex 10, --bytes_per_face 2, --overhead_mb 0.6, --faces_per_vertex 2.5) are now only the INITIAL values.
 2. Byte probe (probe_bytes_v1.py, run through stage2_run.sh in a subprocess): before every prune round and once at the end,
    the upstream final cleaning (faces whose max blending weight over the training views is <= 0.5, unused vertices) is
    simulated on a copy of the mesh and the REAL delivery encoder (k-means K x R dictionaries, XYZ/DC grids, constant opacity,
    context index coder, edge-arithmetic connectivity) is run on that copy.  Its stream sizes replace b_v, b_f, overhead and
    faces_per_vertex, so from the first prune round on every price and the projection are measured, not assumed, and the
    post-control shrink that adapt-v1 could not see (x0.85) is inside the measurement.
 3. Phase B is count-driven: excess = probe_MB - target; the bytes one collapse frees (measured) and the rounds left give the
    number of collapses wanted now; the cheapest independent edges (dD/dR ranking of rd_prune, same foldover filter, same
    prune_max_frac cap) are collapsed and lambda is REPORTED as the shadow price of that budget (the largest dD/dR collapsed).
    Under target -> no collapse this round.  Phase A is adapt-v1's multiplicative controller on the corrected projection,
    aimed at target * --phase_a_overshoot (1.3) because phase B can only remove bytes (prune rounds end at 18000, before
    the upstream supersampling).
rd_log.jsonl: densify events (lam_used, lam_next, rhat_mb, desired), prune events (probe_mb, probe_vertices/faces, b_v, b_f,
overhead_mb, desired_pairs, rounds_left, lam_now = shadow price, skipped), final (probe_final_mb, target_mb).  rd_summary.json:
version adapt-v2, lam_final, probe_final_mb, probes (every probe dict).  Probe artefacts: <model>/probes/probe-<it>.{json,log}.

adapt-v1 (2026-11-05): closed-loop rate controller on lambda. Rate controller (the same idea as rate control in a video encoder: the Lagrange multiplier is searched online so that
the bit budget is met).  Projected delivered size  R_hat = (V * bytes_per_vertex + F * bytes_per_face) / 1e6 + overhead_mb.
Phase A (growth, every densification interval): the byte headroom  target - R_hat  is spread over the remaining
densification intervals, giving the number of splits the schedule allows now,  desired = headroom / (bytes_per_split *
remaining).  The split rule runs with the current lambda and reports n_pass (triangles with dD > lambda * dR); then
    lambda <- clip(lambda * (n_pass / desired)^gamma, lam_min, lam_max)   unless |n_pass/desired - 1| < deadband,
so the price rises when the rule lets more through than the budget affords and falls when it lets fewer.  The selection
keeps the RD ranking (largest dD/dR first) and the per-interval split count is capped at desired, so phase A lands on
the target at densify_until and lambda converges to the shadow price of that budget (the value phase B then prunes with).  Phase B (prune rounds): lambda <- clip(lambda * (R_hat/target)^gamma_prune),
so an asset above target (projection error) is pruned harder.  Everything else (split rule, prune rule, units, upstream
schedule) is train_rd_v3 verbatim.  rd_log.jsonl densify/prune events carry lam_now, rhat_mb, desired, n_pass;
rd_summary.json carries lam_final / target_mb / rhat_final.  Answers "how is lambda chosen for a target rate": by this loop.

train_rd_v3.py (v3, 2026-10-19 = v2m + phase-B rate-controlled edge-collapse pruning; v2m kept as train_rd_v2m.py): MeshSplatting training with rate-distortion controlled growth AND pruning.

v3 adds, at each iteration in --prune_iters (default 12000 14000 16000 18000: after the restricted Delaunay rebuild, before
supersampling), one round of rd_prune.select_collapses / collapse_edges: collapse edge (u,v) iff the residual mass of the
faces touching u or v, per byte the collapse saves (one vertex + the 1/2/3+ faces that disappear), is below lam * prune_mu.
Same lam and same per-view per-megapixel residual units as the split rule; the residual keeps being attributed through
rend_ids in phase B.  At most prune_max_frac of the vertices go per round (pruning twin of the growth limiter).
--prune_iters with no values disables pruning (then v3 == v2m).

v2m adds ONE option, --split_schedule <rd_log.jsonl of a reference run>: at every densification interval the number of
triangles chosen by the selection rule is capped at the number the reference run chose at the same iteration (its
"n_split" field).  With --dd_mode importance this gives the upstream importance-sampled selection on exactly the
growth schedule of a lambda run, so the two runs differ only in WHICH triangles split (ranking), not in how many or
when (the stop).  Without --split_schedule v2m behaves exactly like v2.

A copy of upstream `train.training` with ONE change: the densification call `add_new_gs(cap)` is replaced by the
Lagrangian split rule of rd_growth.py (dD_t > lam * dR_t).  Everything else (losses, pruning, opacity schedule,
restricted Delaunay at densify_until+1000, final importance cleaning) is upstream code, imported, not edited.

Run from anywhere:
  python train_rd.py --lab /scratch/zr2245/meshsplatting-original-lossless-v1 -s <scene> -i images_2 -r 1 --eval \
      --data_device cpu -m <out>/original --iterations 30000 --lam 1e-3 --bytes_per_vertex 10.0 --dd_mode error
Outputs the usual <out>/original/point_cloud/iteration_30000/point_cloud_state_dict.pt plus <out>/original/rd_log.jsonl
(one line per densification interval: vertices, triangles, candidates passing, splits, ratio quantiles).
"""
import argparse
import math
import json
import os
import shlex
import subprocess
import sys
import time
from random import randint

import torch
import torch.nn.functional as F

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--lab', required=True, help='root containing upstream/ (train.py)')
_lab, _rest = _pre.parse_known_args()
UP = os.path.join(os.path.abspath(_lab.lab), 'upstream')
if not os.path.isdir(UP):
    sys.exit(f'upstream dir not found: {UP}')
sys.path.insert(0, UP)
os.chdir(UP)  # upstream resolves relative paths (output/, submodules) from its own dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train as up  # noqa: E402  (upstream train.py; its __main__ block is guarded)
from arguments import ModelParams, PipelineParams, OptimizationParams, update_indoor  # noqa: E402
from scene import Scene, TriangleModel  # noqa: E402
from triangle_renderer import render  # noqa: E402
from utils.loss_utils import l1_loss, ssim, vertex_depth_loss_hr  # noqa: E402
from utils.general_utils import safe_state, get_expon_lr_func  # noqa: E402
from tqdm import tqdm  # noqa: E402
import lpips  # noqa: E402
from rd_growth_v2 import RDAccumulator, RDLog, rd_select, rd_split  # noqa: E402
from rd_prune import select_collapses, collapse_edges, edge_table, _face_normals  # noqa: E402


TOPO_KEYS = ('top_stream', '_triangle_indices', 'local_faces', 'local_rotations')   # connectivity streams of the delivery codec
BANK_KEYS = ('app_bank', 'app_residual_bank')                                           # the two SH dictionaries (fixed cost)


class SizeModel:
    """Delivered size R_hat = b_v * V + b_f * F + overhead (bytes).  Starts from the calibrated constants (attribute bytes per vertex
    = bytes_per_vertex - 2 * bytes_per_face, topology bytes per face, overhead, faces per vertex of the delivered mesh) and is
    replaced by the encoder's own stream sizes at every probe."""
    def __init__(self, rd):
        self.b_v = float(rd['bytes_per_vertex']) - 2.0 * float(rd['bytes_per_face']); self.b_f = float(rd['bytes_per_face'])
        self.overhead = float(rd['overhead_mb']) * 1e6; self.faces_per_vertex = float(rd['faces_per_vertex']); self.source = 'constants'; self.probes = []; self.probes_a = []
        if self.b_v <= 0: raise SystemExit(f"bytes_per_vertex {rd['bytes_per_vertex']} must exceed 2 * bytes_per_face {rd['bytes_per_face']}")
    def mb(self, V, F=None):
        if F is None: F = self.faces_per_vertex * V  # phase A: the soup's own face count (V/3) is not the delivered mesh's
        return (self.b_v * V + self.b_f * F + self.overhead) / 1e6
    def bytes_per_split(self): return 3.0 * (self.b_v + self.faces_per_vertex * self.b_f)  # three midpoints + their faces (upper bound)
    def vertices_for(self, mb): return max(1.0, (mb * 1e6 - self.overhead) / (self.b_v + self.faces_per_vertex * self.b_f))  # inverse of mb(V)
    def bytes_per_collapse(self, faces_removed=2.0): return self.b_v + faces_removed * self.b_f
    def update(self, probe):
        pb = probe['payload_bytes']; V, F = int(probe['vertices']), int(probe['faces'])
        topo = sum(b for k, b in pb.items() if k in TOPO_KEYS); banks = sum(b for k, b in pb.items() if k in BANK_KEYS)
        header = max(0, int(probe['file_bytes']) - sum(pb.values()))
        self.b_f = topo / max(F, 1); self.overhead = banks + header; self.b_v = (int(probe['file_bytes']) - topo - self.overhead) / max(V, 1)
        self.faces_per_vertex = F / max(V, 1); self.source = f"probe@{probe['iteration']}"; self.probes.append(probe)
        probe.update(b_v=round(self.b_v, 3), b_f=round(self.b_f, 3), overhead_mb=round(self.overhead / 1e6, 4), topo_keys_found=[k for k in pb if k in TOPO_KEYS])
    def update_attributes(self, probe):
        """Phase A (v8): the probed model is a triangle SOUP (three own vertices per face), so only the attribute price per kept vertex and the fixed
        overhead are taken from it; the topology price (b_f, faces per vertex) keeps its current value until the first mesh probe (update)."""
        pb = probe['payload_bytes']; V, F = int(probe['vertices']), int(probe['faces'])
        topo = sum(b for k, b in pb.items() if k in TOPO_KEYS); banks = sum(b for k, b in pb.items() if k in BANK_KEYS)
        header = max(0, int(probe['file_bytes']) - sum(pb.values()))
        self.overhead = banks + header; self.b_v = (int(probe['file_bytes']) - topo - self.overhead) / max(V, 1)
        self.source = f"probe_a@{probe['iteration']}"; self.probes_a.append(probe)
        probe.update(b_v=round(self.b_v, 3), overhead_mb=round(self.overhead / 1e6, 4), soup_b_f=round(topo / max(F, 1), 3), soup_faces_per_vertex=round(F / max(V, 1), 4), topo_keys_found=[k for k in pb if k in TOPO_KEYS])
    def state(self): return dict(b_v=round(self.b_v, 3), b_f=round(self.b_f, 3), overhead_mb=round(self.overhead / 1e6, 4), faces_per_vertex=round(self.faces_per_vertex, 3), source=self.source)


def probe_bytes(triangles, scene, pipe, bg, rd, iteration, model_path, check=True, min_vertices=0):
    """What the encoder would deliver if training stopped now.  The upstream final cleaning (max blending weight over the training
    views <= 0.5 -> face dropped, unused vertices dropped) is simulated on a COPY of the mesh, the copy is saved with the keys of
    coarse_finetune's best.pt, and probe_bytes_v1.py runs the real delivery encoder on it in a subprocess (rd['probe_cmd']).
    The training model is not touched.  Returns the probe dict (file_bytes, payload_bytes, vertices, faces, ...)."""
    t1 = time.time()
    floor0 = float(triangles.opacity_floor); scale0 = getattr(triangles, 'scaling', 1)
    def importance(scaling):
        imp = torch.zeros(int(triangles._triangle_indices.shape[0]), dtype=torch.float32, device='cuda')
        try:
            triangles.opacity_floor = float(rd['probe_floor']); triangles.scaling = scaling  # render as at the end: final opacity floor + final supersampling; logits untouched
            for cam in scene.getTrainCameras():
                pk = render(cam, triangles, pipe, bg)
                imp = torch.maximum(imp, pk['max_blending'].detach().reshape(-1).float())
        finally:
            triangles.opacity_floor = floor0; triangles.scaling = scale0
        return imp
    kept_1x = None
    with torch.no_grad():
        T = int(triangles._triangle_indices.shape[0])
        imp = importance(int(rd['probe_scaling']))
        keep = imp > 0.5
        if check and rd.get('probe_check_scaling'):
            kept_1x = int((importance(1) > 0.5).sum())
            if not int(keep.sum()) > kept_1x:
                raise RuntimeError(f'probe scaling has no effect: kept {int(keep.sum())} faces at {rd["probe_scaling"]}x vs {kept_1x} at 1x (triangles.scaling not read by the renderer?)')
        faces = triangles._triangle_indices[keep].long()
        used = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=faces.device)
        if faces.numel(): used[faces.reshape(-1)] = True
        if min_vertices and int(used.sum()) < int(min_vertices):  # v8 phase A: the K x R dictionaries need at least max(K, R) vertices; the caller skips this probe
            return None, used.cpu(), keep.cpu()
        remap = torch.cumsum(used.long(), 0) - 1
        state = dict(triangles_points=triangles.vertices.detach()[used].cpu().clone(), features_dc=triangles._features_dc.detach()[used].cpu().clone(),
                     features_rest=triangles._features_rest.detach()[used].cpu().clone(), _triangle_indices=remap[faces].to(torch.int32).cpu(),
                     vertex_weight=triangles.vertex_weight.detach()[used].cpu().clone(), sigma=triangles._sigma, active_sh_degree=int(triangles.active_sh_degree),
                     opacity_floor=floor0)
    pdir = os.path.join(model_path, 'probes'); os.makedirs(pdir, exist_ok=True)
    pt = os.path.join(pdir, f'probe-{iteration}.pt'); js = os.path.join(pdir, f'probe-{iteration}.json'); lg = os.path.join(pdir, f'probe-{iteration}.log')
    torch.save(state, pt)
    cmd = shlex.split(rd['probe_cmd']) + ['--coarse', pt, '--out', js, '--k', str(rd['probe_k']), '--r', str(rd['probe_r']), '--xyz', rd['probe_xyz'], '--dc', rd['probe_dc'],
                                          '--dictionary-rounds', str(rd['probe_dictionary_rounds'])]
    t2 = time.time()
    with open(lg, 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.exists(js):
        raise RuntimeError(f'byte probe failed at iteration {iteration} (rc {rc}); see {lg}')
    probe = json.load(open(js))
    probe.update(iteration=iteration, vertices_model=int(triangles.vertices.shape[0]), faces_model=T, faces_kept=int(keep.sum()), kept_frac=round(int(keep.sum()) / max(T, 1), 4), probe_floor=float(rd['probe_floor']), probe_scaling=int(rd['probe_scaling']), faces_kept_1x=kept_1x,
                 seconds_cleaning=round(t2 - t1, 1), seconds_probe=round(time.time() - t2, 1))
    os.remove(pt)
    return probe, used.cpu(), keep.cpu()  # kept-vertex / kept-face masks of the model at probe time (the collapse accounting of phase B)


def faces_removed_by(faces, pairs, n_vertices):
    """Boolean mask over the current faces: which ones collapse_edges(triangles, pairs) will delete (degenerate after v -> u, or a duplicate
    of an earlier surviving face).  Same rule as rd_prune.collapse_edges, evaluated BEFORE the collapse so the probe's face mask still applies."""
    dev = faces.device
    f = faces.long()
    if pairs.numel() == 0: return torch.zeros(f.shape[0], dtype=torch.bool, device=dev)
    new_id = torch.arange(n_vertices, device=dev); new_id[pairs[:, 1].to(dev).long()] = pairs[:, 0].to(dev).long()
    f2 = new_id[f]
    degenerate = (f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2])
    key, _ = torch.sort(f2, dim=1)
    _, first = torch.unique(key, dim=0, return_inverse=True)
    order = torch.arange(f2.shape[0], device=dev)
    firstpos = torch.full((int(first.max()) + 1,), f2.shape[0], dtype=torch.long, device=dev)
    firstpos.scatter_reduce_(0, first, order, reduce='amin')
    dup = (firstpos[first] != order) & ~degenerate
    return degenerate | dup


def edge_scores(vertices, faces, face_err, b_attr, b_face):
    """Per-edge collapse score dD/dR with the formula of rd_prune.select_collapses (float64).  Returns (edges, cnt, score)."""
    f = faces.long(); V = vertices.shape[0]
    edges, inv, cnt = edge_table(f); err = face_err.double()
    D_v = torch.zeros(V, dtype=torch.float64, device=vertices.device)
    for k in range(3): D_v.index_add_(0, f[:, k], err)
    D_uv = torch.zeros(edges.shape[0], dtype=torch.float64, device=vertices.device); D_uv.index_add_(0, inv, err.repeat(3))
    dR = float(b_attr) + cnt.double() * float(b_face)
    score = (D_v[edges[:, 0]] + D_v[edges[:, 1]] - D_uv) / dR
    return edges, cnt, score


# >>> select_feasible
@torch.no_grad()
def _match_cheapest(e, s, V, passes=24):
    """Vertex-disjoint subset of the candidate edges e [C,2] (scores s [C], distinct): repeatedly take the edges that are the cheapest at both endpoints."""
    dev = e.device; C = e.shape[0]; picked = torch.zeros(C, dtype=torch.bool, device=dev); avail = torch.ones(V, dtype=torch.bool, device=dev)
    inf = torch.full_like(s, float('inf'))
    for _ in range(passes):
        live = ~picked & avail[e[:, 0]] & avail[e[:, 1]]
        if not bool(live.any()): break
        sl = torch.where(live, s, inf)
        vmin = torch.full((V,), float('inf'), dtype=s.dtype, device=dev)
        vmin.scatter_reduce_(0, e[:, 0], sl, reduce='amin'); vmin.scatter_reduce_(0, e[:, 1], sl, reduce='amin')
        idx = torch.nonzero(live & (sl == vmin[e[:, 0]]) & (sl == vmin[e[:, 1]]), as_tuple=True)[0]
        if idx.numel() == 0: break
        use = torch.zeros(V, dtype=torch.long, device=dev)
        use.index_add_(0, e[idx, 0], torch.ones_like(idx)); use.index_add_(0, e[idx, 1], torch.ones_like(idx))
        idx = idx[(use[e[idx, 0]] == 1) & (use[e[idx, 1]] == 1)]
        if idx.numel() == 0: break
        picked[idx] = True; avail[e[idx].reshape(-1)] = False
    return picked


@torch.no_grad()
def select_feasible(pos, faces, face_err, kept_v, kept_f, b_attr, b_face, want_bytes, max_pairs, max_iters=40):
    """Collapses that free `want_bytes` of the delivered file, cheapest error per freed byte first, as a set that is vertex-disjoint and
    free of fold-overs when applied together.  Freed bytes of edge (u,v): b_attr if both endpoints are in the delivered set + b_face per
    delivered face on the edge; edges that free nothing are not candidates.  The set is built in batches: each batch is matched, tested
    together with everything accepted so far, the pairs that flip a face are banned, the rest is accepted.
    Returns (edges [E,2], sel [P] edge indices, stats)."""
    dev = pos.device; f = faces.long(); V = pos.shape[0]
    edges, inv, cnt = edge_table(f); E = edges.shape[0]; err = face_err.double().to(dev)
    D_v = torch.zeros(V, dtype=torch.float64, device=dev)
    for k in range(3): D_v.index_add_(0, f[:, k], err)
    D_uv = torch.zeros(E, dtype=torch.float64, device=dev); D_uv.index_add_(0, inv, err.repeat(3))
    kv = kept_v.to(dev).bool(); kf = kept_f.to(dev).bool()
    cnt_k = torch.zeros(E, dtype=torch.float64, device=dev); cnt_k.index_add_(0, inv, kf.double().repeat(3))
    dR = float(b_attr) * (kv[edges[:, 0]] & kv[edges[:, 1]]).double() + float(b_face) * cnt_k
    priced = dR > 0
    score = torch.where(priced, (D_v[edges[:, 0]] + D_v[edges[:, 1]] - D_uv) / torch.where(priced, dR, torch.ones_like(dR)), torch.full_like(dR, float('inf')))
    gen = torch.Generator(device=dev); gen.manual_seed(4321 + E)
    fin = score[priced]; scale = float(fin.mean()) if fin.numel() else 1.0
    s = score + torch.rand(E, dtype=torch.float64, device=dev, generator=gen) * 1e-9 * max(scale, 1e-30)  # distinct scores (ties: faces with no error)
    order = torch.argsort(s); order = order[priced[order]]
    n0 = _face_normals(pos, f)
    avail = torch.ones(V, dtype=torch.bool, device=dev); banned = torch.zeros(E, dtype=torch.bool, device=dev)
    pos2 = pos.clone(); new_id = torch.arange(V, device=dev)
    acc = []; n_acc = 0; acc_bytes = 0.0; n_banned = 0; it = 0; n_dropped_batches = 0
    while it < max_iters and acc_bytes < want_bytes and n_acc < max_pairs:
        it += 1
        cand = order[~banned[order] & avail[edges[order, 0]] & avail[edges[order, 1]]]
        if cand.numel() == 0: break
        mean_b = (acc_bytes / n_acc) if n_acc else float(dR[cand[:2000]].mean())
        need = max(1, min(max_pairs - n_acc, int(0.7 * (want_bytes - acc_bytes) / max(mean_b, 1e-6)) + 1))  # 70 % of what is left per batch: the last batches are small, so the total ends within a few pairs of the wanted bytes
        cand = cand[:4 * need + 256]
        new = cand[_match_cheapest(edges[cand], s[cand], V)][:need]
        converged = False
        for _ in range(10):
            if new.numel() == 0: break
            u, v = edges[new, 0], edges[new, 1]
            p2 = pos2.clone(); mid = 0.5 * (pos[u] + pos[v]); p2[u] = mid; p2[v] = mid
            nid = new_id.clone(); nid[v] = u; f2 = nid[f]
            degenerate = (f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2])
            flipped = ((n0 * _face_normals(p2, f)).sum(1) < 0) & ~degenerate
            if not bool(flipped.any()): converged = True; break
            pair_of_v = torch.full((V,), -1, dtype=torch.long, device=dev); kk = torch.arange(new.numel(), device=dev); pair_of_v[u] = kk; pair_of_v[v] = kk
            bad = pair_of_v[f[flipped].reshape(-1)]; bad = torch.unique(bad[bad >= 0])
            if bad.numel() == 0: break
            keep = torch.ones(new.numel(), dtype=torch.bool, device=dev); keep[bad] = False
            banned[new[~keep]] = True; n_banned += int(bad.numel()); new = new[keep]
        if not converged or new.numel() == 0:
            if new.numel(): banned[new] = True; n_dropped_batches += 1
            continue
        u, v = edges[new, 0], edges[new, 1]; mid = 0.5 * (pos[u] + pos[v]); pos2[u] = mid; pos2[v] = mid; new_id[v] = u
        avail[u] = False; avail[v] = False
        acc.append(new); n_acc += int(new.numel()); acc_bytes += float(dR[new].sum())
    sel = torch.cat(acc) if acc else torch.zeros(0, dtype=torch.long, device=dev)
    stats = dict(n_pairs=int(sel.numel()), planned_bytes=acc_bytes, want_bytes=float(want_bytes), n_banned=n_banned, iters=it, n_priced=int(priced.sum()), n_edges=E, dropped_batches=n_dropped_batches,
                 score_max=(float(score[sel].max()) if sel.numel() else None))
    return edges, sel, stats
# <<< select_feasible


def rank_threshold(score, n_wanted, margin=4.0):
    """Score threshold admitting (about) margin * n_wanted cheapest edges as collapse candidates: the k-th smallest score (k capped at
    E - 1), scaled so the k-th edge itself passes the strict '<' test of select_collapses.  Returns (thr, k)."""
    fin = score[torch.isfinite(score)]
    k = max(1, min(int(fin.numel()) - 1, int(margin * n_wanted) + 1))
    kth = float(torch.kthvalue(fin, k).values)
    return max(kth * (1.0 + 1e-6), 1e-30), k


def split_ratios(acc, triangles, bytes_per_split, size_probs_zero, size_probs_zero_image_space):
    """Per-triangle dD/dR of the split rule, the tensor rd_select ranks (disallowed triangles -> 0)."""
    dD = acc.distortion(); areas = triangles.triangle_areas().squeeze()
    ratio = dD / float(bytes_per_split)
    allowed = (areas >= size_probs_zero) & (triangles.image_size >= size_probs_zero_image_space)
    return torch.where(allowed, ratio, torch.zeros_like(ratio)), int(allowed.sum())


def plan_splits(V, V_star, sigmas, add_pct=None):
    """Splits wanted now so that V grows geometrically to V_star over the intervals left.  sigmas[j] = survival of the vertices
    created j intervals from now (1.0 while upstream does not prune yet, the measured value after, the prior before the first
    measurement); len(sigmas) = intervals left.  Geometric plan: f = (V_star / V)^(1 / remaining), desired = V (f - 1) / (3 sigma_0).
    With add_pct (upstream growth limiter: at most add_pct * V - V triangles split per interval, upstream add_new_gs; v4-v8 divided this
    by 3 as if it were vertices, understating the allowance 3x and front-loading too early) the plan is simulated forward
    under the limiter; when it cannot reach V_star this interval asks for everything the limiter allows, because growth
    compounds and a split now is worth more than the same split later.  Returns (desired, front_loaded)."""
    remaining = len(sigmas)
    if remaining < 1 or V_star <= V or V <= 0: return 0.0, False
    def geo(v, r, s): return v * ((V_star / v) ** (1.0 / r) - 1.0) / (3.0 * max(s, 1e-3)) if (r >= 1 and V_star > v) else 0.0
    want = geo(V, remaining, sigmas[0])
    if add_pct is None: return want, False
    v = V
    for j, s in enumerate(sigmas):
        d = min(geo(v, remaining - j, s), max(0.0, add_pct * v - v))  # v9: upstream's allowance is add_pct * V - V TRIANGLES split per interval (add_new_gs), the same unit as d
        v += 3.0 * d * s
    if v < 0.995 * V_star: return max(want, max(0.0, add_pct * V - V)), True
    return want, False


# >>> extrapolate_ratio
def extrapolate_ratio(trend, iteration, r_meas, horizon_iter, window=6):
    """Prediction of the final size when a removal scales with the growth that follows it.  trend = list of (iteration, probe_mb,
    mb removed by that round).  Natural series (the file without any collapse): n_0 = first probe, n_k = n_{k-1} * probe_k /
    (probe_{k-1} - removed_{k-1}).  A least-squares line through the last `window` points (iteration, n_k), anchored at the current point, gives the growth factor
    G = n(horizon) / n(now); the prediction is r_meas * G.  Returns (est, slope of the natural series per 1000 it, G)."""
    if not trend or iteration >= horizon_iter: return r_meas, 0.0, 1.0
    xs, ns, n, prev = [], [], None, None
    for it0, p0, rem0 in trend:
        n = float(p0) if n is None else n * float(p0) / max(prev, 1e-9)
        xs.append(float(it0)); ns.append(n); prev = float(p0) - float(rem0)
    n = n * float(r_meas) / max(prev, 1e-9); xs.append(float(iteration)); ns.append(n)
    xs, ns = xs[-window:], ns[-window:]  # the last probes only: growth right after the stitch is not representative (146 logged runs: 6 probes give the smallest error at the last round)
    k = len(xs); mx = sum(xs) / k; my = sum(ns) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ns)) / sxx if sxx > 0 else 0.0
    g = max(0.8, (n + slope * (horizon_iter - iteration)) / max(n, 1e-9))
    return r_meas * g, slope * 1000.0, g
# <<< extrapolate_ratio


def extrapolate_probe(trend, iteration, r_meas, horizon_iter):
    """trend = list of (iteration, probe_mb, mb_removed_by_that_round's_collapses).  Estimate of the probe at horizon_iter: least-squares
    slope of the 'virtual' probe (probe + everything collapsed before it = what the delivered set would weigh without any collapse)
    over all probes so far, this one included; no previous probe -> r_meas.  With two probes this is v5's two-point slope.
    Returns (est, slope per 1000 it)."""
    if not trend or iteration >= horizon_iter: return r_meas, 0.0
    xs, ys, cum = [], [], 0.0
    for it0, p0, rem0 in trend:
        xs.append(float(it0)); ys.append(p0 + cum); cum += rem0
    xs.append(float(iteration)); ys.append(r_meas + cum)
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
    return r_meas + slope * (horizon_iter - iteration), slope * 1000.0


def rank_lambda(ratio, n_allowed, k, lo, hi):
    """lambda at which exactly k triangles pass (the k-th largest ratio), clipped to [lo, hi]; k >= n_allowed -> lo, k < 1 -> hi."""
    if k < 1: return hi
    if k >= n_allowed or n_allowed < 1: return lo
    kth = float(torch.topk(ratio, int(k), largest=True, sorted=True).values[-1])
    return min(hi, max(lo, kth))


class SurvivalMeter:
    """sigma = fraction of the vertices created by the previous split round that survive the next upstream prune (per interval,
    mean of the last `window` measurements, floored at 0.2).  1.0 until a cohort prune has been observed; a prune larger than the
    last cohort (old vertices going too) is skipped."""
    def __init__(self, window=3): self.window = window; self.hist = []; self.prev_after = None; self.prev_new = 0
    def observe(self, V_before):
        """called at the start of a densify interval with the vertex count after this interval's prune."""
        s = None
        if self.prev_after is not None and self.prev_new > 0 and V_before < self.prev_after:
            pruned = self.prev_after - V_before
            if pruned < self.prev_new:  # a prune larger than the last cohort (the first prune after start_pruning removes old vertices too) is not a survival measurement
                s = max(0.2, 1.0 - pruned / self.prev_new); self.hist.append(round(s, 4))
        return s
    def record(self, V_before, V_after): self.prev_after = V_after; self.prev_new = max(0, V_after - V_before)
    @property
    def sigma(self):
        h = self.hist[-self.window:]
        return sum(h) / len(h) if h else 1.0


def shadow_price(edges, score, pairs, mu):
    """lambda implied by a count-driven collapse: the largest dD/dR among the collapsed edges, divided by prune_mu (the threshold
    select_collapses would have needed to admit exactly these)."""
    if pairs.numel() == 0: return None
    V = int(edges.max()) + 1
    key = edges[:, 0] * V + edges[:, 1]; order = torch.argsort(key); skey = key[order]
    u = torch.minimum(pairs[:, 0], pairs[:, 1]).long(); v = torch.maximum(pairs[:, 0], pairs[:, 1]).long(); pk = u * V + v
    pos = torch.searchsorted(skey, pk).clamp_max(skey.numel() - 1); e = order[pos]
    if not bool((skey[pos] == pk).all()): return None
    return float(score[e].max()) / float(mu)


def load_split_schedule(path):
    """iteration -> n_split (triangles chosen by the rule, before the largest-area top-k) of a reference rd_log.jsonl."""
    sched = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e.get('event') == 'densify':
                sched[int(e['iteration'])] = int(e['n_split'])
    if not sched:
        raise RuntimeError(f'no densify events in split schedule {path}')
    return sched


def training(dataset, opt, pipe, testing_iterations, checkpoint, debug_from, scene_name, rd):
    tb_writer = up.prepare_output_and_logger(dataset)
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    triangles.add_percentage = opt.add_percentage
    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_training_views = len(viewpoint_stack)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc='Training progress (RD growth)')
    first_iter += 1

    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until
    init_opacity = 0.1
    final_opacity = .9999
    total_iters_opacity = opt.final_opacity_iter
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space
    need_delaunay = False
    run_restricted_delaunay = opt.densify_until_iter + 1000
    depth_l1_weight = get_expon_lr_func(opt.depth_lambda_init, opt.depth_lambda_final, max_steps=opt.iterations)

    # --- RD growth state ---------------------------------------------------------------------------------------
    acc = RDAccumulator(triangles._triangle_indices.shape[0], mode=rd['dd_mode'], norm=rd['dd_norm'])
    log = RDLog(os.path.join(dataset.model_path, 'rd_log.jsonl'))
    size = SizeModel(rd); bytes_per_split = size.bytes_per_split()  # initial values; re-derived from the probe in phase B
    target_a = rd['target_mb'] * rd['phase_a_overshoot']
    sched = load_split_schedule(rd['split_schedule']) if rd['split_schedule'] else None
    prune_set = sorted(set(int(i) for i in rd['prune_iters']))
    if prune_set:
        if rd['dd_mode'] != 'error':
            raise SystemExit('pruning needs --dd_mode error (residual attribution); got ' + rd['dd_mode'])
        bad = [i for i in prune_set if not (run_restricted_delaunay + 1 < i < start_upsampling)]
        if bad and not rd['allow_any_prune_iter']:
            raise SystemExit(f'prune_iters {bad} must lie after the Delaunay rebuild ({run_restricted_delaunay}) and before supersampling ({start_upsampling}); --allow_any_prune_iter overrides (smoke only)')
    last_prune = max(prune_set) if prune_set else -1
    lam_now = float(rd['lam']); lam_hist = []; phase_b_started = False; last_faces_removed = None; surv = SurvivalMeter()
    last_bytes_per_pair = None; n_densify = 0; n_bound = 0; last_r_est = None  # v7: measured bytes per collapsed pair; limiter statistics; last extrapolated landing
    extra_set = sorted(set(int(i) for i in rd['probe_extra_iters']))
    bad_x = [i for i in extra_set if i in prune_set or i <= last_prune or i >= opt.iterations]
    if bad_x: raise SystemExit(f'probe_extra_iters {bad_x} must lie after the last prune round ({last_prune}) and before the end ({opt.iterations}), and not be prune rounds')
    trend = []; k_asked = None; yield_hist = []  # probe trend for the extrapolation; rank-price yield (n_pass / k asked) of the last intervals
    last_probe_a = None; n_probe_a = 0  # v8: iteration of the last phase-A probe; how many ran
    if rd['probe_a_every'] < 0: raise SystemExit(f"--probe_a_every must be >= 0 (0 disables the phase-A probes), got {rd['probe_a_every']}")
    log.write(event='config', version='adapt-v9b', probe_a_every=rd['probe_a_every'], probe_a_smoke=rd['probe_a_smoke'], start_pruning=opt.start_pruning, probe_extra_iters=extra_set, probe_scaling=rd['probe_scaling'], sigma_prior=rd['sigma_prior'], target_mb=rd['target_mb'], target_a_mb=target_a, phase_a_overshoot=rd['phase_a_overshoot'], probe_floor=rd['probe_floor'],
              size_model=size.state(), probe_cmd=rd['probe_cmd'], probe=dict(k=rd['probe_k'], r=rd['probe_r'], xyz=rd['probe_xyz'], dc=rd['probe_dc'], dictionary_rounds=rd['probe_dictionary_rounds']),
              lam_min=rd['lam_min'], lam_max=rd['lam_max'], overhead_mb=rd['overhead_mb'], lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'],
              bytes_per_split=bytes_per_split, max_points=opt.max_points, add_percentage=opt.add_percentage,
              densify_from=opt.densify_from_iter, densify_until=opt.densify_until_iter, interval=opt.densification_interval,
              split_schedule=rd['split_schedule'], schedule_total=(sum(sched.values()) if sched else None),
              prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'])
    print(f"[RD adapt] target {rd['target_mb']} MB (phase A aims at {target_a:.2f}) | lam0 {lam_now:.3e} | lambda = rank price of the planned count | phase-A probes every {rd['probe_a_every']} it after start_pruning {opt.start_pruning}{' (smoke: from the first interval)' if rd['probe_a_smoke'] else ''} | probe floor {rd['probe_floor']} | lam in [{rd['lam_min']:.1e}, {rd['lam_max']:.1e}] | size model {size.state()} | probe {rd['probe_k']}x{rd['probe_r']} {rd['probe_xyz']}/{rd['probe_dc']}")
    if prune_set:
        print('[RD] phase-B lambda-prune at', prune_set, '| mu', rd['prune_mu'], 'max_frac/round', rd['prune_max_frac'], 'B/face', rd['bytes_per_face'])
    if sched:
        print('[RD] matched-count control: schedule from', rd['split_schedule'], '|', len(sched), 'intervals, total chosen', sum(sched.values()))
    t0 = time.time()

    for iteration in range(first_iter, opt.iterations + 1):
        if need_delaunay:
            with torch.no_grad():
                triangles.run_restricted_delaunay()
            need_delaunay = False
            log.write(event='delaunay', iteration=iteration, vertices=int(triangles.vertices.shape[0]),
                      triangles=int(triangles._triangle_indices.shape[0]), elapsed_s=round(time.time() - t0))
            acc.reset(triangles._triangle_indices.shape[0])  # new face set after the rebuild
        if iteration == start_upsampling:
            triangles.scaling = opt.upscaling_factor
        if iteration == start_upsampling + 5000:
            triangles.scaling = 4

        iter_start.record()
        triangles.update_learning_rate(iteration)
        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = min((iteration - sigma_start) / (total_iters - sigma_start), 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device='cuda') if opt.random_background else background

        if not viewpoint_stack or len(scene.getTrainCameras()) + iteration == opt.iterations:
            viewpoint_stack = scene.getTrainCameras().copy()
            if len(scene.getTrainCameras()) + iteration == opt.iterations:
                triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device='cuda')
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg['render']
        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(viewpoint_cam, 'normal_map', None) is not None:
            gt_normal = viewpoint_cam.normal_map.cuda()
            gt_normal = F.interpolate(gt_normal.unsqueeze(0), size=(gt_image.shape[1], gt_image.shape[2]), mode='area').squeeze(0)
        else:
            gt_normal = None

        pixel_loss = l1_loss(image, gt_image)

        image_size = render_pkg['scaling'].detach()
        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        importance_score = render_pkg['max_blending'].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]
        pixel_count = render_pkg['triangle_was_rendered'].detach()
        mask = pixel_count > triangles.pixel_count
        triangles.pixel_count[mask] = pixel_count[mask]

        if up.FUSED_SSIM_AVAILABLE:
            ssim_value = up.fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim_value)

        lambda_weight = opt.lambda_weight if iteration < opt.start_opacity_floor else 0
        if lambda_weight > 0:
            mask_out = triangles.vertices.shape[0]
            vertex_weights = triangles.get_vertex_weight[:mask_out][triangles._triangle_indices]
            loss += lambda_weight * vertex_weights.mean()

        lambda_vertex = opt.lambda_vertex if iteration > opt.start_vertex_opt else 0
        if lambda_vertex > 0:
            loss += lambda_vertex * vertex_depth_loss_hr(render_pkg['vertex_depth_out'], render_pkg['image_2D'], render_pkg['vertex_rendered'],
                                                         render_pkg['surf_depth'], max_diff_threshold=opt.max_diff_threshold)

        if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, 'invdepthmap', None) is not None:
            invDepth = 1.0 / (render_pkg['expected_depth'] + 1e-6)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            loss += depth_l1_weight(iteration) * torch.abs((invDepth - mono_invdepth) * depth_mask).mean()

        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0
        if lambda_normal > 0:
            loss += lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0))[None].mean()
        if gt_normal is not None:
            lambda_normals_super = opt.lambda_normals_super if iteration > opt.iteration_mesh else 0
            loss += lambda_normals_super * (1 - (rend_normal * gt_normal).sum(dim=0))[None].mean()

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # RD: attribute this view's residual to triangles (phase A for growth; phase B up to the last prune round)
            if iteration < run_restricted_delaunay or iteration <= last_prune:
                acc.accumulate(render_pkg, image, gt_image, triangles._triangle_indices,
                               vertex_grad=triangles.vertices.grad if rd['dd_mode'] == 'grad' else None)

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({'Loss': f'{ema_loss_for_log:.5f}', 'V': int(triangles.vertices.shape[0])})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            up.training_report(tb_writer, scene_name, iteration, pixel_loss, loss, l1_loss, iter_start.elapsed_time(iter_end),
                               testing_iterations, scene, render, (pipe, background))

            if iteration % 500 == 0 and iteration < run_restricted_delaunay:
                triangle_vertex_weights = triangles.opacity_activation(triangles.vertex_weight[triangles._triangle_indices])
                min_weights = triangle_vertex_weights.min(dim=1).values
                mask_opacity = (min_weights <= prune_triangles).squeeze()
                mask_importance = (triangles.importance_score <= prune_triangles).squeeze()
                mask_size = (triangles.image_size > prune_size).squeeze()
                delete_mask = mask_opacity | mask_size
                if number_of_training_views < 500:
                    delete_mask = delete_mask | mask_importance
                keep_mask = ~delete_mask
                if iteration > opt.start_pruning:
                    triangles.prune_triangles(keep_mask)
                    acc.apply_mask(keep_mask)

                device = triangles.vertices.device
                used_vertex_mask = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=device)
                if triangles._triangle_indices.numel() > 0:
                    used_vertex_mask[triangles._triangle_indices.flatten()] = True
                weight_mask = (triangles.get_vertex_weight.squeeze() >= prune_triangles)
                mask_out = triangles.vertices.shape[0]
                vertex_mask = weight_mask[:mask_out] | used_vertex_mask
                valid_tris = vertex_mask[triangles._triangle_indices.long()].all(dim=1) if triangles._triangle_indices.numel() > 0 else None
                triangles._prune_vertices(vertex_mask)
                if valid_tris is not None:
                    acc.apply_mask(valid_tris)

                needs_densification = (iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0
                                       and iteration > opt.densify_from_iter)
                if needs_densification:
                    V = int(triangles.vertices.shape[0])
                    max_split = max(0, min(opt.max_points, int(opt.add_percentage * V)) - V)  # upstream growth limiter
                    k_ref = None
                    if sched is not None:  # matched-count control: never choose more than the reference run did here
                        k_ref = sched.get(iteration, 0)
                        max_split = min(max_split, k_ref)
                    s_obs = surv.observe(V); sigma = surv.sigma  # survival of the previous round's new vertices in this interval's upstream prune
                    if rd['probe_a_every'] > 0 and (iteration > opt.start_pruning or rd['probe_a_smoke']) and (last_probe_a is None or iteration - last_probe_a >= rd['probe_a_every']):
                        v_star_before = size.vertices_for(target_a); src_before = size.source; min_v = max(int(rd['probe_k']), int(rd['probe_r']))
                        pa, used_a, _ = probe_bytes(triangles, scene, pipe, bg, rd, iteration, dataset.model_path, check=False, min_vertices=min_v)  # v8: the real encoder on the cleaned copy of the current soup
                        if pa is None:  # fewer kept vertices than the dictionary size: the encoder cannot run (initialize raises 'K exceeds sample count'); try again at the next interval
                            log.write(event='probe_a_skipped', iteration=iteration, elapsed_s=round(time.time() - t0), vertices_model=V, kept_vertices=int(used_a.sum()), min_vertices=min_v, **size.state())
                            print(f"\n[RD probe-a {iteration}] skipped: {int(used_a.sum())} kept vertices of {V} < dictionary size {min_v}; size model stays {size.source}")
                        else:
                            size.update_attributes(pa); last_probe_a = iteration; n_probe_a += 1
                            r_pa = pa['file_bytes'] / 1e6; v_star_after = size.vertices_for(target_a); kappa = pa['vertices'] / max(V, 1)
                            log.write(event='probe_a', iteration=iteration, elapsed_s=round(time.time() - t0), probe_mb=round(r_pa, 4), probe_ratio=round(r_pa / rd['target_mb'], 4), probe_vertices=pa['vertices'], probe_faces=pa['faces'],
                                      vertices_model=V, faces_model=pa['faces_model'], kept_share=round(kappa, 4), probe_kept_frac=pa['kept_frac'], probe_seconds=pa['seconds_probe'], seconds_cleaning=pa['seconds_cleaning'],
                                      soup_b_f=pa['soup_b_f'], soup_faces_per_vertex=pa['soup_faces_per_vertex'], v_star_before=int(v_star_before), v_star_after=int(v_star_after), source_before=src_before, **size.state())
                            print(f"\n[RD probe-a {iteration}] soup {V} V -> cleaned {pa['vertices']} V {pa['faces']} F (kept share {kappa:.3f}) -> {r_pa:.3f} MB in {pa['seconds_cleaning']} + {pa['seconds_probe']} s | b_v {size.b_v:.2f} B (attributes per kept vertex) overhead {size.overhead / 1e6:.3f} MB | topology prior b_f {size.b_f:.2f} x {size.faces_per_vertex:.2f} faces/vertex (soup read {pa['soup_b_f']:.2f} x {pa['soup_faces_per_vertex']:.2f}, not used) | V* {int(v_star_before)} -> {int(v_star_after)} ({src_before} -> {size.source})")
                    rhat = size.mb(V)  # corrected projection of the delivered mesh (faces_per_vertex * V faces, not the soup's)
                    remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))  # v9: intervals upstream will still run, this one included (iteration + k * interval < densify_until); v4-v8 counted one too many, so the plan always left one interval's growth undone
                    V_star = size.vertices_for(target_a)
                    sig_meas = sigma if surv.hist else float(rd['sigma_prior'])  # measured once a cohort prune has been seen, else the prior
                    sigmas = [(sig_meas if iteration + opt.densification_interval * (j + 1) > opt.start_pruning else 1.0) for j in range(remaining)]  # cohort j is pruned at its next interval only after start_pruning
                    desired, front = plan_splits(V, V_star, sigmas, opt.add_percentage); cap = int(math.ceil(desired))
                    max_split_up = max_split; limiter_bound = (False if rd.get('hold_lambda') else bool(front or max_split_up < cap)); max_split = (max_split_up if rd.get('hold_lambda') else min(max_split_up, cap))  # the plan caps the count; lambda is the price of that count; max_split_up = what upstream's limiter allows
                    n_densify += 1; n_bound += int(limiter_bound)
                    lam_used = lam_now; lam_hist.append(lam_now)
                    ratio_pre, n_allowed = split_ratios(acc, triangles, bytes_per_split, triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    acc_mode_keep = acc.mode
                    if rd.get('split_rank') == 'importance': acc.mode = 'importance'  # batch35 ablation: upstream importance sampling picks the triangles; the planned count and the residual accumulation stay
                    idx, st = rd_select(acc, triangles, lam_now, bytes_per_split, max_split,
                                        triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    acc.mode = acc_mode_keep
                    n_split = rd_split(triangles, idx, iteration, splitt_large_triangles)
                    V_after = int(triangles.vertices.shape[0]); surv.record(V, V_after)
                    V_exp = V + sigmas[0] * (V_after - V)  # expected count after the next upstream prune
                    desired_next = plan_splits(V_exp, V_star, sigmas[1:], opt.add_percentage)[0] if remaining > 1 else 0.0
                    if k_asked: yield_hist.append(min(5.0, max(0.2, float(st['n_pass']) / max(k_asked, 1))))  # what the last rank price actually admitted
                    yl = yield_hist[-3:]; yld = sum(yl) / len(yl) if yl else 1.0
                    k_asked = int(math.ceil(desired_next / yld)) if desired_next > 0 else 0
                    lam_now = lam_used if rd.get('hold_lambda') else (rank_lambda(ratio_pre, n_allowed, k_asked, rd['lam_min'], rd['lam_max']) if k_asked > 0 else lam_used)  # price of the next planned count, corrected by the measured yield; v9b: nothing left to plan (last interval) -> keep the price just used, so phase B starts from the last real phase-A price instead of lam_max
                    ratio = float(st['n_pass']) / max(desired, 1.0)
                    st.update(event='densify', hold_lambda=bool(rd.get('hold_lambda')), iteration=iteration, vertices_before=V, vertices_after=V_after,
                              triangles_after=int(triangles._triangle_indices.shape[0]), n_split_total=n_split, k_ref=k_ref, elapsed_s=round(time.time() - t0),
                              lam_used=lam_used, lam_next=lam_now, rhat_mb=round(rhat, 4), desired=round(desired, 1), desired_next=round(desired_next, 1), remaining=remaining, cap=cap,
                              pass_ratio=round(ratio, 4), sigma=round(sigma, 4), sigma_obs=(round(s_obs, 4) if s_obs is not None else None), sigma_plan=round(sigmas[0], 4), n_allowed=n_allowed, v_star=int(V_star), limiter_bound=limiter_bound, limiter_max=int(max_split_up), front_loaded=front, rank_yield=round(yld, 4), k_asked=k_asked, size_source=size.source)
                    log.write(**st)
                    print(f"\n[RD {iteration}] V {V}->{V_after} pass {st['n_pass']}/{st['n_tri']} split {n_split} "
                          f"(cap {cap} limiter {max_split_up}) ratio q50/q90/q99 {st['ratio_q50']:.2e}/{st['ratio_q90']:.2e}/{st['ratio_q99']:.2e} "
                          f"lam {lam_used:.2e} -> {lam_now:.2e} (projected {rhat:.2f} / phase-A aim {target_a:.2f} MB = {V_star} vertices, desired {desired:.0f} -> next {desired_next:.0f}{' LIMITER' if limiter_bound else ''}{' FRONT-LOAD' if front else ''}, pass/desired {ratio:.2f}, yield {yld:.2f} -> k {k_asked}, sigma {sigma:.2f} (obs {s_obs}, plan {sigmas[0]:.2f}), {remaining} intervals left) valid {st['valid_frac']}")
                acc.reset(triangles._triangle_indices.shape[0])

                if iteration > opt.start_opacity_floor:
                    a = min(1.0, max(0.0, (iteration - opt.start_opacity_floor) / max(1, total_iters_opacity - opt.start_opacity_floor)))
                    triangles.update_min_weight(min(init_opacity + (final_opacity - init_opacity) * a, final_opacity))
                    prune_triangles += 0.01
            elif iteration == run_restricted_delaunay:
                need_delaunay = True
            elif iteration % 500 == 0 and iteration > run_restricted_delaunay + 1000:
                if iteration > opt.start_opacity_floor:
                    a = min(1.0, max(0.0, (iteration - opt.start_opacity_floor) / max(1, total_iters_opacity - opt.start_opacity_floor)))
                    triangles.update_min_weight(min(init_opacity + (final_opacity - init_opacity) * a, final_opacity))
                    prune_triangles += 0.01

            if iteration in prune_set:
                V0, T0 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
                dD = acc.distortion()
                if dD.numel() != T0:
                    raise RuntimeError(f'accumulator/face mismatch at prune {iteration}: {dD.numel()} vs {T0}')
                if rd.get('trim_score') == 'importance':  # batch35 ablation: upstream's importance score takes the place of the residual in the collapse cost
                    dD = triangles.importance_score.detach().reshape(-1).to(device=dD.device, dtype=dD.dtype)
                    if dD.numel() != T0:
                        raise RuntimeError(f'importance/face mismatch at prune {iteration}: {dD.numel()} vs {T0}')
                n_views = acc.views
                probe, kept_v, kept_f = probe_bytes(triangles, scene, pipe, bg, rd, iteration, dataset.model_path)  # real encoder on the cleaned copy
                size.update(probe); r_meas = probe['file_bytes'] / 1e6
                rounds_left = len([i for i in prune_set if i >= iteration])
                te5 = (not rd.get('hold_lambda')) and int(rd.get('trim_exact') or 0) == 5; g_fac = 1.0  # batch29 v3
                predictive = iteration < int(total_iters_opacity)  # batch30: before the ramp ends the final size is predicted, afterwards it is measured
                if te5: rounds_left = len([i for i in prune_set if i >= iteration and ((i < int(total_iters_opacity)) == predictive)])  # batch30: each phase spreads over its own rounds
                if te5: r_est, slope_k, g_fac = extrapolate_ratio(trend, iteration, r_meas, int(total_iters_opacity)); last_r_est = r_est
                else: r_est, slope_k = extrapolate_probe(trend, iteration, r_meas, int(total_iters_opacity)); last_r_est = r_est  # what this probe will read once the opacity ramp has ended
                if te5 and rd.get('trim_hold'): r_est, slope_k, g_fac = r_meas, 0.0, 1.0; last_r_est = r_est; rounds_left = 1; predictive = False  # batch31: no prediction, the measured excess goes now
                excess = (r_est - rd['target_mb'] * (1.0 + (float(rd.get('trim_margin') or 0.0) if (te5 and predictive) else 0.0))) * 1e6  # batch30: predictive rounds stop above the target
                if last_bytes_per_pair: b_coll = last_bytes_per_pair  # measured on the last collapse round: bytes the probe lost per pair
                else: b_coll = size.b_v * (probe['vertices'] / max(probe['vertices_model'], 1)) ** 2 + 2.0 * size.b_f * float(probe['kept_frac'])  # prior: both endpoints kept, two kept faces
                excess_pairs = excess / max(b_coll, 1e-6)  # pairs that remove the whole excess now
                desired = excess_pairs / rounds_left
                if not phase_b_started:
                    phase_b_started = True
                    if lam_hist:
                        tail = lam_hist[-20:]; lam_a = math.exp(sum(math.log(x) for x in tail) / len(tail))
                        print(f"\n[RD adapt {iteration}] phase A ended at lam {lam_now:.2e} (geo-mean of last {len(tail)} intervals {lam_a:.2e})")
                print(f"\n[RD adapt {iteration}] probe: {probe['vertices']} V {probe['faces']} F after cleaning -> {r_meas:.3f} MB ({r_meas / rd['target_mb']:.3f} x target), extrapolated to {int(total_iters_opacity)}: {r_est:.3f} MB ({r_est / rd['target_mb']:.3f} x, slope {slope_k:+.3f} MB/1k it) in {probe['seconds_probe']} s"
                      f" | b_v {size.b_v:.2f} b_f {size.b_f:.2f} overhead {size.overhead / 1e6:.3f} MB | {b_coll:.1f} B per collapse -> {desired:.0f} collapses wanted ({rounds_left} rounds left)")
                ev = dict(event='prune', iteration=iteration, views=n_views, elapsed_s=round(time.time() - t0), probe_mb=round(r_meas, 4), probe_ratio=round(r_meas / rd['target_mb'], 4),
                          probe_est_mb=round(r_est, 4), probe_est_ratio=round(r_est / rd['target_mb'], 4), probe_slope_mb_per_1k=round(slope_k, 4),
                          probe_vertices=probe['vertices'], probe_faces=probe['faces'], probe_kept_frac=probe['kept_frac'], probe_seconds=probe['seconds_probe'], desired_pairs=round(desired, 1), rounds_left=rounds_left,
                          bytes_per_collapse=round(b_coll, 2), **size.state())
                b_v_full = size.b_v + 2.0 * size.b_f  # select_collapses takes bytes_per_vertex = attributes + 2 faces, bytes_per_face
                edges, cnt, score = edge_scores(triangles.vertices, triangles._triangle_indices, dD, size.b_v, size.b_f)
                thr_lam = float(lam_now) * float(rd['prune_mu']); n_lam = int((score < thr_lam).sum())  # the fixed-lambda prune of train_rd_v3 at the current price
                thr_rank, k_rank = (rank_threshold(score, desired) if desired >= 1.0 else (0.0, 0))
                count_driven = thr_rank > thr_lam
                thr = max(thr_rank, thr_lam)
                wanted = min(max(desired, float(n_lam)), excess_pairs) if excess > 0 else 0.0  # gate: nothing to remove -> nothing collapsed; cap: never past the target
                cap_bound = bool(excess > 0 and excess_pairs < max(desired, float(n_lam)))
                te = 0 if rd.get('hold_lambda') else int(rd.get('trim_exact') or 0)  # batch29
                if te >= 3: wanted = (max(1.0, min(desired, excess_pairs)) if excess > 0 else 0.0); cap_bound = False  # batch29 v2: any excess opens the round
                if te == 5 and iteration < int(rd.get('trim_start') or 0): wanted = 0.0  # batch29 v3: measure only before trim_start
                if te == 1: wanted = (min(desired, excess_pairs) if excess > 0 else 0.0); cap_bound = False  # batch29: the per-round plan, nothing else
                if rd.get('hold_lambda'): wanted = float(n_lam); count_driven = False; cap_bound = False; thr = thr_lam  # ablation (batch24): constant price -> only the fixed-lambda prune of train_rd_v3, no target gate
                removed_mb = 0.0; n_kept_faces_removed = 0
                if wanted < 1.0:
                    acc.reset(T0)
                    ev.update(skipped=True, mode='none', lam_now=lam_now, n_lam=n_lam, excess_pairs=round(excess_pairs, 1), vertices_before=V0, vertices_after=V0, faces_before=T0, faces_after=T0, n_pairs=0, kept_removed=0, kept_faces_removed=0, removed_mb=0.0)
                    log.write(**ev)
                    print(f"[RD prune {iteration}] landing {r_est:.3f} MB {'under' if excess <= 0 else 'at'} target {rd['target_mb']} MB: no collapse this round (lambda-prune held; {n_lam} edges below lam {lam_now:.2e})")
                else:
                    frac = min(rd['prune_max_frac'], wanted / V0)
                    pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac)
                    if te in (1, 2):  # batch29: deliver the wanted pairs (the matching and the fold-over filter both lose pairs)
                        goal = int(frac * V0); margin = 4.0; frac_eff = frac; tries = 1; first_pairs = int(ps['n_pairs']); n_fin = int(torch.isfinite(score).sum())
                        while ps['n_pairs'] < 0.97 * goal and tries < 14:
                            tries += 1
                            if ps['n_pass_pairs'] < int(frac_eff * V0):  # the matching ran out of candidates: admit more edges
                                margin *= 2.0; thr_m, k_m = rank_threshold(score, wanted, margin=margin)
                                if thr_m <= thr:
                                    if k_m >= n_fin - 1: break
                                    continue
                                thr = thr_m; count_driven = True
                            else:  # enough matched pairs, the fold-over filter removed too many: ask the matching for more
                                if frac_eff >= 0.45: break
                                frac_eff = min(0.45, frac_eff * max(1.25, goal / max(ps['n_pairs'], 1)))
                            pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac_eff)
                        if int(pairs.shape[0]) > goal:  # keep the cheapest `goal` pairs
                            Vn = int(edges.max()) + 1; key = edges[:, 0] * Vn + edges[:, 1]; order_e = torch.argsort(key); skey = key[order_e]
                            pu = torch.minimum(pairs[:, 0], pairs[:, 1]).long(); pv = torch.maximum(pairs[:, 0], pairs[:, 1]).long(); pk = pu * Vn + pv
                            pos = torch.searchsorted(skey, pk).clamp_max(skey.numel() - 1); sc = score[order_e[pos]]
                            pairs = pairs[torch.argsort(sc)[:goal]]; ps['n_pairs'] = int(pairs.shape[0])
                        ev.update(trim_exact=te, trim_goal=goal, trim_margin=margin, trim_frac_eff=round(frac_eff, 4), trim_tries=tries, trim_first_pairs=first_pairs)
                        print(f"[RD prune {iteration}] exact trim (mode {te}): goal {goal} pairs, first pass {first_pairs}, after {tries} tries (margin {margin:g}, pair cap {frac_eff:.3f} V) {ps['n_pairs']} pairs (matching found {ps['n_pass_pairs']}, flip-rejected {ps['n_flip_rejected']})")
                    if te >= 3:  # batch29 v2: byte-targeted, fold-over-safe selection on the delivered set
                        first_pairs = int(ps['n_pairs']); want_bytes = (excess / g_fac / rounds_left) if te == 5 else excess / (rounds_left if te == 3 else min(rounds_left, 3)); cap_pairs = int(rd['prune_max_frac'] * V0)
                        edges_f, sel_f, st_f = select_feasible(triangles.vertices.detach(), triangles._triangle_indices, dD, kept_v, kept_f, size.b_v, size.b_f, want_bytes, cap_pairs)
                        if sel_f.numel() > 0:
                            pairs = edges_f[sel_f]; ps['n_pairs'] = int(pairs.shape[0]); ps['n_flip_rejected'] = st_f['n_banned']; ps['max_pairs'] = cap_pairs; count_driven = True
                        ev.update(trim_exact=te, trim_predictive=bool(predictive), trim_rounds_left=rounds_left, trim_growth=round(g_fac, 4), trim_want_mb=round(want_bytes / 1e6, 4), trim_planned_mb=round(st_f['planned_bytes'] / 1e6, 4), trim_batches=st_f['iters'], trim_banned=st_f['n_banned'],
                                  trim_first_pairs=first_pairs, trim_goal=int(st_f['n_pairs']), trim_priced=st_f['n_priced'], trim_dropped_batches=st_f['dropped_batches'])
                        print(f"[RD prune {iteration}] feasible trim (mode {te}): wanted {want_bytes / 1e6:.3f} MB, selected {st_f['n_pairs']} pairs freeing {st_f['planned_bytes'] / 1e6:.3f} MB in {st_f['iters']} batches ({st_f['n_banned']} pairs banned by the fold-over test, {st_f['n_priced']} of {st_f['n_edges']} edges free bytes); the original rule gave {first_pairs} pairs")
                    lam_sp = shadow_price(edges, score, pairs, rd['prune_mu'])
                    if (count_driven or cap_bound) and lam_sp is not None: lam_now = lam_sp  # the count was binding (rank or excess cap): lambda := its shadow price
                    rm_f = faces_removed_by(triangles._triangle_indices, pairs, V0).cpu()  # faces the collapse deletes, before it happens
                    n_kept_faces_removed = int((rm_f & kept_f).sum())
                    pc = pairs.cpu().long(); n_kept_removed = int((kept_v[pc[:, 0]] & kept_v[pc[:, 1]]).sum()) if pairs.numel() else 0  # one kept vertex fewer only when both endpoints were kept
                    cs = collapse_edges(triangles, pairs)
                    if int(rm_f.sum()) != T0 - cs['faces_after']:
                        raise RuntimeError(f'collapse accounting mismatch at {iteration}: predicted {int(rm_f.sum())} faces removed, collapse_edges removed {T0 - cs["faces_after"]}')
                    if ps['n_pairs'] > 0: last_faces_removed = (T0 - cs['faces_after']) / ps['n_pairs']
                    removed_mb = (n_kept_removed * size.b_v + n_kept_faces_removed * size.b_f) / 1e6  # exact w.r.t. the probe's kept sets and the probed stream costs
                    if ps['n_pairs'] > 0: last_bytes_per_pair = removed_mb * 1e6 / ps['n_pairs']
                    acc.reset(triangles._triangle_indices.shape[0])
                    mode = 'count' if count_driven else ('capped' if cap_bound else 'lambda')
                    ev.update(skipped=False, mode=mode, lam_now=lam_now, shadow_lam=lam_sp, n_lam=n_lam, excess_pairs=round(excess_pairs, 1), cap_frac=round(frac, 4), rank_k=k_rank, removed_mb=round(removed_mb, 4), kept_removed=n_kept_removed, kept_removed_frac=round(n_kept_removed / max(ps['n_pairs'], 1), 4),
                              kept_faces_removed=n_kept_faces_removed, bytes_per_pair_measured=(round(last_bytes_per_pair, 2) if last_bytes_per_pair is not None else None), faces_removed_per_pair=(round(last_faces_removed, 3) if last_faces_removed else None)); ev.update(ps); ev.update(cs)
                    log.write(**ev)
                    print(f"\n[RD prune {iteration}] V {V0}->{cs['vertices_after']} T {T0}->{cs['faces_after']} | {mode}-driven: wanted {wanted:.0f} pairs (per-round plan {desired:.0f}, whole excess {excess_pairs:.0f}), lam-admitted {n_lam} edges, cap {ps['max_pairs']} -> {ps['n_pairs']} collapsed ({removed_mb:.3f} MB exact: {n_kept_removed} kept vertices + {n_kept_faces_removed} kept faces; {(last_bytes_per_pair if last_bytes_per_pair else float('nan')):.1f} B per pair) "
                          f"(flip-rejected {ps['n_flip_rejected']}, passes {ps['passes']}) | lam {lam_now:.2e} (shadow {lam_sp}) | score q10/q50/q90 {ps['score_q10']:.2e}/{ps['score_q50']:.2e}/{ps['score_q90']:.2e} "
                          f"| faces-removed share sel {ps['faces_removed_share_sel']} all {ps['faces_removed_share_all']} | views {n_views}")
                trend.append((iteration, r_meas, removed_mb))
            elif iteration in extra_set:  # v7 diagnostic probes after the last round: measure, extrapolate, log; no collapse
                probe, _, _ = probe_bytes(triangles, scene, pipe, bg, rd, iteration, dataset.model_path)
                size.update(probe); r_meas = probe['file_bytes'] / 1e6
                r_est, slope_k = extrapolate_probe(trend, iteration, r_meas, int(total_iters_opacity))
                print(f"\n[RD adapt {iteration}] diagnostic probe: {probe['vertices']} V {probe['faces']} F after cleaning -> {r_meas:.3f} MB ({r_meas / rd['target_mb']:.3f} x target), extrapolated to {int(total_iters_opacity)}: {r_est:.3f} MB ({r_est / rd['target_mb']:.3f} x, slope {slope_k:+.3f} MB/1k it) in {probe['seconds_probe']} s | floor now {float(triangles.opacity_floor):.4f} scaling {getattr(triangles, 'scaling', 1)}")
                log.write(event='probe_extra', iteration=iteration, elapsed_s=round(time.time() - t0), probe_mb=round(r_meas, 4), probe_ratio=round(r_meas / rd['target_mb'], 4), probe_est_mb=round(r_est, 4), probe_slope_mb_per_1k=round(slope_k, 4),
                          probe_vertices=probe['vertices'], probe_faces=probe['faces'], probe_kept_frac=probe['kept_frac'], probe_seconds=probe['seconds_probe'], floor_now=float(triangles.opacity_floor), scaling_now=int(getattr(triangles, 'scaling', 1)), **size.state())
                trend.append((iteration, r_meas, 0.0))
            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)

    # final cleaning, verbatim upstream
    viewpoint_stack = scene.getTrainCameras().copy()
    triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), dtype=torch.float, device='cuda')
    while viewpoint_stack:
        viewpoint_cam = viewpoint_stack.pop(0)
        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        importance_score = render_pkg['max_blending'].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]
    mask_importance = (triangles.importance_score <= 0.5).squeeze()
    triangles.prune_triangles(~mask_importance)
    used_vertex_mask = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=triangles.vertices.device)
    if triangles._triangle_indices.numel() > 0:
        used_vertex_mask[triangles._triangle_indices.flatten()] = True
    triangles._prune_vertices(used_vertex_mask)
    scene.save(iteration)
    rhat_final = size.mb(int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0]))
    final_probe, _, _ = probe_bytes(triangles, scene, pipe, bg, rd, iteration, dataset.model_path, check=False); size.probes.append(final_probe)
    probe_final = final_probe['file_bytes'] / 1e6
    growth_capped = bool(prune_set) and last_r_est is not None and last_r_est < rd['target_mb'] and 2 * n_bound > n_densify  # phase B had nothing to remove and phase A could not grow to its aim
    log.write(event='final', growth_capped=growth_capped, probe_a_count=n_probe_a, limiter_bound_intervals=n_bound, densify_intervals=n_densify, last_probe_est_mb=(round(last_r_est, 4) if last_r_est is not None else None), vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0]),
              elapsed_s=round(time.time() - t0), lam_final=lam_now, rhat_mb=round(rhat_final, 4), probe_final_mb=round(probe_final, 4), probe_final_ratio=round(probe_final / rd['target_mb'], 4),
              target_mb=rd['target_mb'], **size.state())
    with open(os.path.join(dataset.model_path, 'rd_summary.json'), 'w') as f:
        json.dump(dict(version='adapt-v9b', growth_capped=growth_capped, probe_a_every=rd['probe_a_every'], probe_a_smoke=rd['probe_a_smoke'], probe_a_count=n_probe_a, probes_a=size.probes_a, start_pruning=opt.start_pruning, limiter_bound_intervals=n_bound, densify_intervals=n_densify, last_probe_est_mb=(round(last_r_est, 4) if last_r_est is not None else None), probe_extra_iters=extra_set, probe_scaling=rd['probe_scaling'], sigma_hist=surv.hist, probe_trend=trend, yield_hist=yield_hist, extrapolate_to=int(total_iters_opacity), probe_floor=rd['probe_floor'], target_mb=rd['target_mb'], lam_final=lam_now, rhat_final_mb=round(rhat_final, 4), probe_final_mb=round(probe_final, 4), probes=size.probes, size_model=size.state(),
                       phase_a_overshoot=rd['phase_a_overshoot'], probe_cmd=rd['probe_cmd'], lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'], split_schedule=rd['split_schedule'],
                       prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'],
                       vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0])), f, indent=1)
    print(f"Training is done; vertices {int(triangles.vertices.shape[0])} triangles {int(triangles._triangle_indices.shape[0])} | adapt-v9b: target {rd['target_mb']} MB, final probe {probe_final:.3f} MB ({probe_final / rd['target_mb']:.3f} x), model {rhat_final:.2f} MB, lam0 {rd['lam']:.2e} -> lam_final {lam_now:.2e}"
          f" | growth-capped {growth_capped} (limiter bound {n_bound} of {n_densify} intervals, last landing estimate {last_r_est}) | phase-A probes {n_probe_a}"
          f"{(' (last b_v %.2f overhead %.3f MB at %s)' % (size.probes_a[-1]['b_v'], size.probes_a[-1]['overhead_mb'], size.probes_a[-1]['iteration'])) if size.probes_a else ''}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MeshSplatting training with RD-controlled growth', parents=[_pre])
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[7_000, 30_000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[7_000, 30_000])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    parser.add_argument('--scene_name', default='Garden', type=str)
    parser.add_argument('--indoor', action='store_true', default=False)
    parser.add_argument('--lam', type=float, required=True, help='price of a byte in per-view pixel-L1 units (0 = error-ranked growth, no price)')
    parser.add_argument('--bytes_per_vertex', type=float, default=10.0, help='measured delivered bytes per vertex for this scene/precision')
    parser.add_argument('--dd_mode', choices=['error', 'grad', 'importance'], default='error')
    parser.add_argument('--dd_norm', choices=['mpx', 'none'], default='mpx', help='mpx: dD per megapixel of the loss image (resolution-independent lambda); none: v1 raw pixel units')
    parser.add_argument('--split_schedule', type=str, default=None, help='rd_log.jsonl of a reference run: cap the per-interval split count at its n_split (matched-count control)')
    parser.add_argument('--prune_iters', nargs='*', type=int, default=[12000, 14000, 16000, 18000], help='phase-B prune rounds (no values = no pruning); v6 is meant to run every 500 from 12000 to 19500, the sbatch sets it')
    parser.add_argument('--prune_mu', type=float, default=1.0, help='collapse iff ring-residual / bytes-saved < lam * prune_mu')
    parser.add_argument('--prune_max_frac', type=float, default=0.15, help='at most this fraction of vertices collapsed per round')
    parser.add_argument('--probe_extra_iters', nargs='*', type=int, default=[], help='v7 diagnostic probes after the last prune round (e.g. 22000 26000): measured and logged, no collapse')
    parser.add_argument('--probe_a_every', type=int, default=1000, help='v8: phase-A byte probes at densify intervals after start_pruning, at least this many iterations apart (0 = none, i.e. adapt-v7b); they replace the attribute price and the overhead of the size model')
    parser.add_argument('--probe_a_smoke', action='store_true', help='smoke only: phase-A probe at the first densify interval even before start_pruning (the initial soup)')
    parser.add_argument('--bytes_per_face', type=float, default=2.0, help='topology bytes per face in the delivered file (bytes_per_vertex - 2*bytes_per_face = attribute bytes per vertex)')
    parser.add_argument('--allow_any_prune_iter', action='store_true', help='smoke only: allow prune rounds outside (Delaunay, supersampling)')
    parser.add_argument('--target_mb', type=float, required=True, help='target delivered size in MB; lambda is adapted so the projected size V*B_v + F*B_f + overhead hits it')
    parser.add_argument('--seed', type=int, default=None, help='batch24 round 2: reseed python / numpy / torch after upstream safe_state (which fixes seed 0); None = unchanged')
    parser.add_argument('--trim_hold', action='store_true', help='batch31: with --trim_exact 5, no prediction: every round removes what the measured file has above the target')
    parser.add_argument('--split_rank', choices=['rd', 'importance'], default='rd', help='batch35 ablation: importance = upstream importance sampling picks the triangles to split (count plan unchanged)')
    parser.add_argument('--trim_score', choices=['rd', 'importance'], default='rd', help='batch35 ablation: importance = upstream importance score replaces the residual in the collapse cost')
    parser.add_argument('--trim_margin', type=float, default=0.0, help='batch30: with --trim_exact 5, rounds before the end of the opacity ramp steer to target x (1 + margin); later rounds steer to the target')
    parser.add_argument('--trim_start', type=int, default=0, help='batch29 v3: with --trim_exact 5, collapse rounds before this iteration only measure')
    parser.add_argument('--trim_exact', type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help='batch29: 1 = collapse the per-round plan exactly (candidate threshold raised until the matching delivers it); 2 = the original wanted count, delivered exactly; 0 = unchanged')
    parser.add_argument('--hold_lambda', action='store_true', help='ablation (batch24): constant price. The controller never changes lambda: phase A splits every triangle above --lam (upstream limiter only, no plan cap); phase B keeps only the fixed-lambda prune (edges below lam x mu), no target gate; --target_mb is then a label only')
    parser.add_argument('--probe_scaling', type=int, default=4, help='supersampling factor used when the probe renders the cleaning pass (= upstream final scaling)')
    parser.add_argument('--probe_check_scaling', action='store_true', help='smoke: also render the cleaning pass at 1x and fail unless the probe scaling keeps more faces')
    parser.add_argument('--sigma_prior', type=float, default=0.5, help='survival of new vertices assumed for the forward plan until the first cohort prune is measured (adapt-v2/v3 runs: 0.3-0.7); replaced by the measurement')
    parser.add_argument('--probe_floor', type=float, default=0.9999, help='opacity floor used when the probe renders the cleaning pass (= upstream final_opacity)')
    parser.add_argument('--lam_min', type=float, default=1e-4); parser.add_argument('--lam_max', type=float, default=1e-1)
    parser.add_argument('--overhead_mb', type=float, default=0.6, help='initial fixed bytes of the delivered file (dictionaries + header); measured from the first probe on')
    parser.add_argument('--faces_per_vertex', type=float, default=2.5, help='initial faces per vertex of the delivered mesh (phase-A projection); measured from the first probe on')
    parser.add_argument('--phase_a_overshoot', type=float, default=1.0, help='phase A aims at target * this (v6 default 1.0: the kept set grows after phase A on every scene measured, so no margin is needed; phase B can only remove bytes)')
    parser.add_argument('--probe_cmd', type=str, required=True, help='command prefix that runs probe_bytes_v1.py in the stage2 env, e.g. "bash <PKG>/stage2_run.sh <AN>/rdgrowth/probe_bytes_v1.py"')
    parser.add_argument('--probe_k', type=int, default=4096); parser.add_argument('--probe_r', type=int, default=256)
    parser.add_argument('--probe_xyz', type=str, required=True, help='XYZ precision of the delivery (q13..q18, the scene rule; never auto)')
    parser.add_argument('--probe_dc', type=str, default='q7'); parser.add_argument('--probe_dictionary_rounds', type=int, default=2)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.probe_a_every < 0: sys.exit(f'--probe_a_every must be >= 0, got {args.probe_a_every}')
    if args.target_mb <= 0 or not (0 < args.lam_min < args.lam_max) or args.phase_a_overshoot < 1.0 or not (0.5 <= args.probe_floor < 1.0) or args.probe_scaling < 1 or not (0 < args.sigma_prior <= 1):
        sys.exit(f'bad controller settings: target_mb {args.target_mb} lam range [{args.lam_min}, {args.lam_max}] overshoot {args.phase_a_overshoot} probe_floor {args.probe_floor}')
    if not args.probe_xyz.startswith('q') or not args.probe_xyz[1:].isdigit() or not (12 <= int(args.probe_xyz[1:]) <= 18):
        sys.exit(f'--probe_xyz must be q12..q18, got {args.probe_xyz}')
    _probe_script = [t for t in shlex.split(args.probe_cmd) if t.endswith('.py')]
    if not _probe_script or not os.path.isfile(_probe_script[-1]):
        sys.exit(f'--probe_cmd must name an existing probe script: {args.probe_cmd}')
    print('Optimizing ' + args.model_path, '| RD growth+prune adapt-v9b target_mb', args.target_mb, '| hold_lambda (ablation)', args.hold_lambda, 'phase-A probes every', args.probe_a_every, 'overshoot', args.phase_a_overshoot, 'probe', args.probe_cmd, 'lam0', args.lam, 'mode', args.dd_mode, 'norm', args.dd_norm, 'B/v', args.bytes_per_vertex,
          'schedule', args.split_schedule, '| prune', args.prune_iters, 'mu', args.prune_mu, 'max_frac', args.prune_max_frac, 'B/face', args.bytes_per_face)
    if args.split_schedule and not os.path.isfile(args.split_schedule):
        sys.exit(f'split schedule not found: {args.split_schedule}')
    up.lpips_fn = lpips.LPIPS(net='vgg').to(device='cuda')  # training_report reads this module global
    safe_state(args.quiet)
    if args.seed is not None:
        import random as _random, numpy as _np
        _random.seed(args.seed); _np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); print('[RD] reseeded with', args.seed)
    lps, ops, pps = lp.extract(args), op.extract(args), pp.extract(args)
    if args.indoor:
        ops = update_indoor(ops)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps, ops, pps, args.test_iterations, args.start_checkpoint, args.debug_from, args.scene_name,
             rd=dict(split_rank=args.split_rank, trim_score=args.trim_score, trim_hold=args.trim_hold, trim_margin=args.trim_margin, trim_start=args.trim_start, trim_exact=args.trim_exact, hold_lambda=args.hold_lambda, lam=args.lam, bytes_per_vertex=args.bytes_per_vertex, dd_mode=args.dd_mode, dd_norm=args.dd_norm, split_schedule=args.split_schedule,
                     prune_iters=args.prune_iters, prune_mu=args.prune_mu, prune_max_frac=args.prune_max_frac, bytes_per_face=args.bytes_per_face, probe_extra_iters=args.probe_extra_iters, probe_a_every=args.probe_a_every, probe_a_smoke=args.probe_a_smoke,
                     allow_any_prune_iter=args.allow_any_prune_iter, target_mb=args.target_mb, probe_floor=args.probe_floor, probe_scaling=args.probe_scaling, probe_check_scaling=args.probe_check_scaling, sigma_prior=args.sigma_prior,
                     lam_min=args.lam_min, lam_max=args.lam_max, overhead_mb=args.overhead_mb, faces_per_vertex=args.faces_per_vertex, phase_a_overshoot=args.phase_a_overshoot,
                     probe_cmd=args.probe_cmd, probe_k=args.probe_k, probe_r=args.probe_r, probe_xyz=args.probe_xyz, probe_dc=args.probe_dc, probe_dictionary_rounds=args.probe_dictionary_rounds))
