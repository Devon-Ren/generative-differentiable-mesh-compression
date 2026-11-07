# node-batch12-v1（2026-11-07）：adapt-v9 = v8 + phase-A 计划的两处计数修正 —— lego / chair 2.5 + chair 4 + room / bicycle 10 + room 20，6 个 job 一波（≈6.5 GPU-h，4 卡 ≈2.3 h 墙钟）

**batch11 判定**：(a)(b)(e) 过——soup 上量的 b_v 和 12k 网格的 b_v 相差 1–8 %，overhead 相差 < 1 %，probe 总开销 2–3 min；(c)(d) 不过——6 行全比 v7b 低（hit 0.71–0.92 vs 0.78–1.01）。**原因在 densify 日志里，不在 b_v**：

1. **计划多数了一个 interval**。`remaining = (until − it) // interval + 1` 在 9500 给 2，但 upstream 只在 `iteration < densify_until`（10000）时 densify，9500 就是最后一个（1500…9500 共 17 个，计划以为 18 个）。几何计划把还差的增长摊到"两个"interval，最后一个只长到 sqrt(V·V*)：6 个 run 最后一次 upstream prune 后的模型 = 0.81–0.92 V*，和 sqrt(V_9500/V*) = 0.83–0.92 一致（CPU 重放 6 条轨迹，旧计划预测误差 0.3–5 %）。v4–v7b 全部带着这个 bug（sweep 分析里"phase A 只到 V* 的 0.83–0.97"就是它）；v7b 的常数先验在 Blender 上把 V* 抬高 30 % 恰好盖住了，v8 把先验量准了，缺口就露出来。
2. **limiter 单位错了 3 倍**。upstream `add_new_gs` 每个 interval 最多分裂 add_pct·V − V 个**三角形**（rd_select 的 max_split 也是这个数），plan_splits 的前向模拟却把它除以 3 当成顶点数——以为 limiter 紧 3 倍，过早 front-load，而 front-load 时又只要了允许量的 1/3。4 MB Blender 行的"limiter-bound 12–13/17"大半是这个假象。

**v9**（`rdgrowth/train_rd_adapt_v9.py`，910 行 = v8 887 行 + 头注释，代码 diff 4 行）：`remaining = ceil((until − it)/interval)`；plan_splits 的 limiter 用三角形数；densify 行打印 `cap N limiter M`（以前把 min(cap, limiter) 打成 limit，看起来像 limiter 卡住，其实是计划自己的上限），事件带 `limiter_max`。phase-A probe、phase B 一字不动。CPU 检查：生成脚本精确替换（11 处）、py_compile/pyflakes、intervals 公式对 5 种 schedule 穷举比对、6 条 batch11 轨迹重放（旧计划复现实测缺口，v9 在 9500 全部到 V*；从 3500 整段重放 6/6 到 1.000 V*、limiter 不卡）、smoke 断言块在假日志上正反各跑一次。

**sbatch**：`adapt-v9-blender.sbatch`（162 行，VERSION v9b-r1）、`adapt-v9.sbatch`（154 行，VERSION v9-r1）= v8 的两个 + 文件名 / 目录名 adapt9 / 表 adapt-v9*.txt / smoke 断言 densify 事件的 `remaining` 等于按 config 算的值且带 `limiter_max`。

**这 6 个 job**：lego/chair 2.5（机制 + 和 v7b/v8 直接比；预期 phase A 到 V*，先验在 Blender 偏高 3–10 % → 落点 1.00–1.10，phase B 这次要真砍）、chair 4（v8 0.797 capped，看 limiter 修正后是否还 capped）、room 10 / bicycle 10（v8 0.795 / 0.710；预期 ≈0.91 / ≈0.87，剩下的差是拓扑先验 5 B/V vs Mip 实测 3.7–4.5）、room 20（v7b 0.75，看 20 MB 档）。

**通过标准**：
- (a) 6 个 job exit 0，smoke 里有 `smoke v9 plan: {... 'remaining': 17 ...} | expected remaining 17`；
- (b) **计划到位**：每行 `prune rounds 16 ... (V M -> ...)` 的 M（最后一次 upstream prune 后的模型顶点数）≥ 0.95 × 9500 那行的 V*（`[RD 9500] ... = V* vertices`），limiter-bound 的行除外；
- (c) room 10 ≥ 0.88、bicycle 10 ≥ 0.85（v8 0.795 / 0.710），等字节 PSNR 比 v7b 不低超过 0.05 dB；
- (d) lego/chair 2.5 落点 0.95–1.05（phase B 砍过之后），PSNR 比 v7b 同目标行不低超过 0.05 dB；chair 4 ≥ 0.85；
- (e) room 20 ≥ 0.80（v7b 0.75）。
(b)(c) 过 → batch13 = 33 个 Mip/T&T 重跑（`adapt-v9.sbatch` 不再改，场景参数同 batch9-v2）+ Blender 16 行重跑（v9 的 phase B 在 Blender 上会动，v7b 那批不等价）。

## 1. 写三个文件 + 提交 6 个 job（一次粘贴；检查和提交都在脚本里，不碰你的终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/train_rd_adapt_v9.py <<'PYEOF_TRAIN9'
"""train_rd_adapt_v9.py (adapt-v9, 2026-11-07 = adapt-v8 with ONE phase-A control fix; the v8 phase-A probes and the v7b phase B are untouched;
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
from rd_prune import select_collapses, collapse_edges, edge_table  # noqa: E402


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
    log.write(event='config', version='adapt-v9', probe_a_every=rd['probe_a_every'], probe_a_smoke=rd['probe_a_smoke'], start_pruning=opt.start_pruning, probe_extra_iters=extra_set, probe_scaling=rd['probe_scaling'], sigma_prior=rd['sigma_prior'], target_mb=rd['target_mb'], target_a_mb=target_a, phase_a_overshoot=rd['phase_a_overshoot'], probe_floor=rd['probe_floor'],
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
                    max_split_up = max_split; limiter_bound = bool(front or max_split_up < cap); max_split = min(max_split_up, cap)  # the plan caps the count; lambda is the price of that count; max_split_up = what upstream's limiter allows
                    n_densify += 1; n_bound += int(limiter_bound)
                    lam_used = lam_now; lam_hist.append(lam_now)
                    ratio_pre, n_allowed = split_ratios(acc, triangles, bytes_per_split, triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    idx, st = rd_select(acc, triangles, lam_now, bytes_per_split, max_split,
                                        triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    n_split = rd_split(triangles, idx, iteration, splitt_large_triangles)
                    V_after = int(triangles.vertices.shape[0]); surv.record(V, V_after)
                    V_exp = V + sigmas[0] * (V_after - V)  # expected count after the next upstream prune
                    desired_next = plan_splits(V_exp, V_star, sigmas[1:], opt.add_percentage)[0] if remaining > 1 else 0.0
                    if k_asked: yield_hist.append(min(5.0, max(0.2, float(st['n_pass']) / max(k_asked, 1))))  # what the last rank price actually admitted
                    yl = yield_hist[-3:]; yld = sum(yl) / len(yl) if yl else 1.0
                    k_asked = int(math.ceil(desired_next / yld)) if desired_next > 0 else 0
                    lam_now = rank_lambda(ratio_pre, n_allowed, k_asked, rd['lam_min'], rd['lam_max'])  # price of the next planned count, corrected by the measured yield
                    ratio = float(st['n_pass']) / max(desired, 1.0)
                    st.update(event='densify', iteration=iteration, vertices_before=V, vertices_after=V_after,
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
                n_views = acc.views
                probe, kept_v, kept_f = probe_bytes(triangles, scene, pipe, bg, rd, iteration, dataset.model_path)  # real encoder on the cleaned copy
                size.update(probe); r_meas = probe['file_bytes'] / 1e6
                rounds_left = len([i for i in prune_set if i >= iteration])
                r_est, slope_k = extrapolate_probe(trend, iteration, r_meas, int(total_iters_opacity)); last_r_est = r_est  # what this probe will read once the opacity ramp has ended
                excess = (r_est - rd['target_mb']) * 1e6
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
                removed_mb = 0.0; n_kept_faces_removed = 0
                if wanted < 1.0:
                    acc.reset(T0)
                    ev.update(skipped=True, mode='none', lam_now=lam_now, n_lam=n_lam, excess_pairs=round(excess_pairs, 1), vertices_before=V0, vertices_after=V0, faces_before=T0, faces_after=T0, n_pairs=0, kept_removed=0, kept_faces_removed=0, removed_mb=0.0)
                    log.write(**ev)
                    print(f"[RD prune {iteration}] landing {r_est:.3f} MB {'under' if excess <= 0 else 'at'} target {rd['target_mb']} MB: no collapse this round (lambda-prune held; {n_lam} edges below lam {lam_now:.2e})")
                else:
                    frac = min(rd['prune_max_frac'], wanted / V0)
                    pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac)
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
        json.dump(dict(version='adapt-v9', growth_capped=growth_capped, probe_a_every=rd['probe_a_every'], probe_a_smoke=rd['probe_a_smoke'], probe_a_count=n_probe_a, probes_a=size.probes_a, start_pruning=opt.start_pruning, limiter_bound_intervals=n_bound, densify_intervals=n_densify, last_probe_est_mb=(round(last_r_est, 4) if last_r_est is not None else None), probe_extra_iters=extra_set, probe_scaling=rd['probe_scaling'], sigma_hist=surv.hist, probe_trend=trend, yield_hist=yield_hist, extrapolate_to=int(total_iters_opacity), probe_floor=rd['probe_floor'], target_mb=rd['target_mb'], lam_final=lam_now, rhat_final_mb=round(rhat_final, 4), probe_final_mb=round(probe_final, 4), probes=size.probes, size_model=size.state(),
                       phase_a_overshoot=rd['phase_a_overshoot'], probe_cmd=rd['probe_cmd'], lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'], split_schedule=rd['split_schedule'],
                       prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'],
                       vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0])), f, indent=1)
    print(f"Training is done; vertices {int(triangles.vertices.shape[0])} triangles {int(triangles._triangle_indices.shape[0])} | adapt-v9: target {rd['target_mb']} MB, final probe {probe_final:.3f} MB ({probe_final / rd['target_mb']:.3f} x), model {rhat_final:.2f} MB, lam0 {rd['lam']:.2e} -> lam_final {lam_now:.2e}"
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
    print('Optimizing ' + args.model_path, '| RD growth+prune adapt-v9 target_mb', args.target_mb, 'phase-A probes every', args.probe_a_every, 'overshoot', args.phase_a_overshoot, 'probe', args.probe_cmd, 'lam0', args.lam, 'mode', args.dd_mode, 'norm', args.dd_norm, 'B/v', args.bytes_per_vertex,
          'schedule', args.split_schedule, '| prune', args.prune_iters, 'mu', args.prune_mu, 'max_frac', args.prune_max_frac, 'B/face', args.bytes_per_face)
    if args.split_schedule and not os.path.isfile(args.split_schedule):
        sys.exit(f'split schedule not found: {args.split_schedule}')
    up.lpips_fn = lpips.LPIPS(net='vgg').to(device='cuda')  # training_report reads this module global
    safe_state(args.quiet)
    lps, ops, pps = lp.extract(args), op.extract(args), pp.extract(args)
    if args.indoor:
        ops = update_indoor(ops)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps, ops, pps, args.test_iterations, args.start_checkpoint, args.debug_from, args.scene_name,
             rd=dict(lam=args.lam, bytes_per_vertex=args.bytes_per_vertex, dd_mode=args.dd_mode, dd_norm=args.dd_norm, split_schedule=args.split_schedule,
                     prune_iters=args.prune_iters, prune_mu=args.prune_mu, prune_max_frac=args.prune_max_frac, bytes_per_face=args.bytes_per_face, probe_extra_iters=args.probe_extra_iters, probe_a_every=args.probe_a_every, probe_a_smoke=args.probe_a_smoke,
                     allow_any_prune_iter=args.allow_any_prune_iter, target_mb=args.target_mb, probe_floor=args.probe_floor, probe_scaling=args.probe_scaling, probe_check_scaling=args.probe_check_scaling, sigma_prior=args.sigma_prior,
                     lam_min=args.lam_min, lam_max=args.lam_max, overhead_mb=args.overhead_mb, faces_per_vertex=args.faces_per_vertex, phase_a_overshoot=args.phase_a_overshoot,
                     probe_cmd=args.probe_cmd, probe_k=args.probe_k, probe_r=args.probe_r, probe_xyz=args.probe_xyz, probe_dc=args.probe_dc, probe_dictionary_rounds=args.probe_dictionary_rounds))
PYEOF_TRAIN9
cat > $AN/adapt-v9-blender.sbatch <<'SBEOF_ADAPT9B'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=adapt9b
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/zr2245/analysis/adapt9b-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/adapt9b-%x-%j.err
# adapt-v9-blender.sbatch (2026-11-07, VERSION v9b-r1): = adapt-v8-blender.sbatch (VERSION v8b-r1) with train_rd_adapt_v9.py (v8 + the phase-A plan
# fixes: intervals-left counted as upstream runs them, limiter allowance in triangles as applied; phase-A probes and phase B as in v8), job names
# adapt9b, output /scratch/zr2245/$SCENE-adapt9b-t${T}mb (+ -best), table $AN/adapt-v9-blender.txt, the smoke asserting the densify event's
# 'remaining' against the config schedule and the presence of limiter_max.  Everything else verbatim v8b-r1, whose text follows (names rewritten).
# adapt-v8-blender.sbatch (2026-11-07, VERSION v8b-r1): = adapt-v7b-blender.sbatch (VERSION v7b-r2) with train_rd_adapt_v8.py (phase A calibrated by
# byte probes at densify intervals after start_pruning, --probe_a_every $PAE (1000); phase B unchanged), the smoke extended to 1600 iterations so its
# first densify interval (1500) runs a phase-A probe (--probe_a_smoke) before the forced prune round at 1550, smoke assertions on the probe_a event /
# rd_summary fields, and the result row carrying every phase-A probe (bytes, kept share, b_v, overhead, V*).  Output /scratch/zr2245/$SCENE-adapt9b-t${T}mb
# (+ -best), table $AN/adapt-v9-blender.txt.  Everything else verbatim v7b-r2, whose text follows.
# adapt-v7b-blender.sbatch (2026-11-06, VERSION v7b-r2): = adapt-v7-blender.sbatch (VERSION v7b) with train_rd_adapt_v7b.py (v7 + the final-probe
# unpack fix that crashed every batch8 smoke at its last line); nothing else changed.  v7b text: = the v6b Blender sbatch with train_rd_adapt_v7 (exact kept-aware collapse accounting, prune gated
# on the extrapolated landing and capped at the excess, growth-capped flag, diagnostic probes at EXTRA="22000 26000" after the last round) and the
# matching smoke assertions / result row (growth-capped, limiter counts, extra probes, kept faces removed per round).  Everything else as v6b:
# Blender protocol (--indoor, black background, train_ckpt_v2 checkpoints every 1k from 12k, best OFFICIAL-val iteration, val_select_v3, level-1.0
# delivery on $G-best), prune rounds every 500 from 12000 to 19500, stages (probe self-check on the lam 6.2e-3 delivery, smoke 1300 it with a forced
# prune round at 1250 + --probe_check_scaling, then per target).  One run ~25 min.  Output /scratch/zr2245/$SCENE-adapt7b-t${T}mb (+ -best), table $AN/adapt-v7-blender.txt.  Restartable.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; TARGETS=${TARGETS:-"2.5 4"}; INDOOR=${INDOOR:-1}
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"$(seq -s " " 12000 500 19500)"}
OVER=${OVER:-1.0}; EXTRA=${EXTRA:-"22000 26000"}; FPV=${FPV:-2.5}; PFLOOR=${PFLOOR:-0.9999}; PSCALE=${PSCALE:-4}; SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}; PAE=${PAE:-1000}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; STAGE2=/scratch/zr2245/shared-collapse-stage2-v2; VS=$AN/blender/val_select_v3.py; CK=$AN/blender/train_ckpt_v2.py
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/adapt-v9-blender.txt; TR=$AN/rdgrowth/train_rd_adapt_v9.py; PROBE_PY=$AN/rdgrowth/probe_bytes_v1.py
PROBE="bash $PKG/stage2_run.sh $PROBE_PY"
echo "adapt-v9-blender | scene $SCENE (black bg, val-selected) | IMAGES $IMAGES | targets $TARGETS MB | indoor '$FLAG' | initial B/v $BPV B/face $BPF faces/vertex $FPV | phase-A probes every $PAE it after start_pruning | overshoot $OVER probe floor $PFLOOR probe scaling $PSCALE sigma prior $SPRIOR | prune at $PRUNE mu $MU max_frac $MAXF | extra probes $EXTRA | probe $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $PROBE_PY $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $PKG/stage2_run.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $VS $CK $LAB/upstream/train.py $STAGE2/STAGE2_VERSION $STAGE2/runtime.py $STAGE2/sharing_after.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python -m py_compile $TR || { echo "FAILED: $TR does not compile"; exit 1; }
python -m py_compile $PROBE_PY || { echo "FAILED: $PROBE_PY does not compile"; exit 1; }
grep -q "version='adapt-v9'" $TR || { echo "FAILED: $TR is not adapt-v9 (wrong file pasted?)"; exit 1; }
grep -q "max_split_up = max_split" $TR && grep -q "limiter_max=int(max_split_up)" $TR && grep -q "remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))" $TR || { echo "FAILED: $TR lacks the v9 plan fixes"; exit 1; }
grep -q "def update_attributes" $TR && grep -q "probe_a_every" $TR && grep -q "event='probe_a'" $TR || { echo "FAILED: $TR lacks the v8 phase-A probe (update_attributes / probe_a_every / probe_a event)"; exit 1; }
[ "$PAE" -ge 0 ] 2>/dev/null || { echo "FAILED: PAE must be an integer >= 0 (0 = no phase-A probes = v7b), got $PAE"; exit 1; }
grep -q "PROBE_VERSION = 'probe-v1'" $PROBE_PY || { echo "FAILED: $PROBE_PY is not probe-v1"; exit 1; }
grep -q "def faces_removed_by" $TR || { echo "FAILED: $TR lacks faces_removed_by (v6 pasted instead of v7?)"; exit 1; }
grep -q "growth_capped" $TR && grep -q "probe_extra_iters" $TR || { echo "FAILED: $TR lacks growth_capped / probe_extra_iters"; exit 1; }
grep -q "default=1.0, help='phase A aims at target" $TR || { echo "FAILED: $TR phase_a_overshoot default is not 1.0"; exit 1; }
grep -q "VERSION = 'val_select_v3" $VS || { echo "FAILED: $VS is not val_select_v3"; exit 1; }
grep -q "def recompute_importance" $VS || { echo "FAILED: $VS lacks recompute_importance"; exit 1; }
python -m py_compile $VS || { echo "FAILED: $VS does not compile"; exit 1; }
NPR=$(echo $PRUNE | wc -w); [ "$NPR" -ge 4 ] || { echo "FAILED: PRUNE has $NPR rounds (v8 default is 16, at least 4 expected: $PRUNE)"; exit 1; }
LASTP=$(echo $PRUNE | tr ' ' '\n' | sort -n | tail -1); for x in $EXTRA; do [ "$x" -gt "$LASTP" ] && [ "$x" -lt 30000 ] || { echo "FAILED: EXTRA probe $x must lie after the last prune round $LASTP and before 30000"; exit 1; }; done
grep -q "VERSION = 'train_ckpt_v2" $CK || { echo "FAILED: $CK is not train_ckpt_v2"; exit 1; }
grep -q "triangles.scaling" $LAB/upstream/train.py || { echo "FAILED: $LAB/upstream/train.py never sets triangles.scaling (the probe's 4x pass assumes upstream's supersampling attribute)"; exit 1; }
lam0_of () { python -c "import sys; t=float(sys.argv[1]); print('%.4e' % min(1e-1, max(1e-4, 6.2e-3 * (9.0 / t) ** 1.67)))" $1; }   # initial guess only; adapted from the first interval on
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
SM=$AN/adapt9b-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
if [ "$SELFCHECK" = "1" ]; then
  PV=/scratch/zr2245/$SCENE-rd3b4-lam0.0062-mu1.0-best/axisB/v1.0
  if [ -f $PV/finetune/best.pt ] && [ -f $PV/deliver/summary.json ]; then
    echo "=== probe self-check on $PV ($(date)) ==="
    $PROBE --coarse $PV/finetune/best.pt --out $SM/selfcheck.json --k $K --r $R --xyz $XYZ --dc $DC > $SM/selfcheck.log 2>&1 || { echo "FAILED: PROBE SELF-CHECK (probe crashed)"; tail -30 $SM/selfcheck.log; exit 1; }
    python - "$SM/selfcheck.json" "$PV/deliver/summary.json" <<'PY' || { echo "FAILED: PROBE SELF-CHECK (bytes disagree with the delivered file)"; tail -5 $SM/selfcheck.log; exit 1; }
import json, sys
p = json.load(open(sys.argv[1])); d = json.load(open(sys.argv[2])); pm = p['file_bytes'] / 1e6; dm = d['file_MB']; r = pm / dm
big = {k: round(v / 1e6, 3) for k, v in sorted(p['payload_bytes'].items(), key=lambda kv: -kv[1]) if v > 20000}
print(f"self-check: probe {pm:.3f} MB vs delivered {dm:.3f} MB -> probe/delivered {r:.3f} | V {p['vertices']} F {p['faces']} | {p['seconds']['total']} s | streams {big}")
assert abs(r - 1.0) <= 0.08, f'probe/delivered {r:.3f} is outside 0.92..1.08'
PY
  else echo "note: no lam 6.2e-3 Blender delivery for $SCENE at $PV; self-check skipped (the smoke stage still exercises the probe)"; fi
else echo "self-check skipped (SELFCHECK=0)"; fi
if [ "$SMOKE" = "1" ]; then
  echo "=== smoke 1600 it: phase-A probe at the first densify interval (1500), target 0.01 MB so the prune round at 1550 must collapse; probe at ${PSCALE}x AND 1x ($(date)) ==="
  python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1600 --test_iterations 1600 --lam $(lam0_of 5) --target_mb 0.01 \
    --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --probe_check_scaling --sigma_prior $SPRIOR \
    --prune_iters 1550 --allow_any_prune_iter --prune_mu $MU --prune_max_frac 0.1 --probe_a_every $PAE --probe_a_smoke \
    --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $SM/smoke.log 2>&1 || { echo "FAILED: SMOKE"; tail -40 $SM/smoke.log; for l in $SM/original/probes/*.log; do echo "--- $l"; tail -15 $l; done; exit 1; }
  python - <<PY || { echo "FAILED: SMOKE CHECK"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
if dn: assert all('lam_used' in e and 'desired' in e and 'sigma_plan' in e and 'front_loaded' in e and 'v_star' in e for e in dn), 'densify events lack the v4 controller fields'
else: print('note: no densify event within the 1600-it smoke (upstream densify_from >= 1600); phase-A controller is exercised in the real runs')
assert p and 'probe_mb' in p[0], 'no prune event with probe_mb (probe did not run)'
assert not p[0].get('skipped') and p[0]['n_pairs'] > 0, 'prune round at 1550 collapsed nothing: %r' % {k: p[0].get(k) for k in ('skipped','desired_pairs','n_pairs','n_cand','max_pairs')}
assert p[0]['source'].startswith('probe@'), 'size model not replaced by the probe: %r' % p[0]['source']
assert p[0]['lam_now'] is not None and p[0].get('mode') in ('count', 'lambda', 'capped'), 'no prune mode / lambda: %r' % p[0].get('mode')
assert 'kept_removed' in p[0] and 0 <= p[0]['kept_removed'] <= p[0]['n_pairs'], 'kept-removed count missing / out of range: %r' % {k: p[0].get(k) for k in ('kept_removed','n_pairs')}
assert 'kept_faces_removed' in p[0] and 0 <= p[0]['kept_faces_removed'] <= p[0]['faces_before'] - p[0]['faces_after'], 'kept-faces-removed count missing / out of range: %r' % {k: p[0].get(k) for k in ('kept_faces_removed','faces_before','faces_after')}
assert 'removed_mb' in p[0] and p[0].get('bytes_per_pair_measured') is not None and p[0].get('excess_pairs', 0) > 0, 'v7 accounting fields missing: %r' % {k: p[0].get(k) for k in ('removed_mb','bytes_per_pair_measured','excess_pairs')}
if p[0]['removed_mb'] == 0: print('note: the smoke round collapsed pairs outside the probe kept set (removed_mb 0): accounting exercised, nothing to charge')
assert abs(p[0]['removed_mb'] * 1e6 - (p[0]['kept_removed'] * p[0]['b_v'] + p[0]['kept_faces_removed'] * p[0]['b_f'])) < 2000, 'removed_mb is not kept vertices x b_v + kept faces x b_f: %r' % {k: p[0].get(k) for k in ('removed_mb','kept_removed','kept_faces_removed','b_v','b_f')}
assert 'probe_est_mb' in p[0] and p[0]['probe_est_mb'] == p[0]['probe_mb'], 'first round must use the probe as is (no slope yet): %r' % {k: p[0].get(k) for k in ('probe_mb','probe_est_mb')}
assert 'probe_kept_frac' in p[0] and p[0]['probe_kept_frac'] > 0, 'probe kept fraction missing'
assert fin and 'probe_final_mb' in fin[0], 'no final event with probe_final_mb'
rs=json.load(open('$SM/original/rd_summary.json')); assert rs['version']=='adapt-v9' and len(rs['probes'])==2, 'rd_summary lacks the two phase-B/final probes'
pa=[e for e in ev if e['event']=='probe_a']
if dn:
    assert pa and pa[0]['iteration']==dn[0]['iteration'] and pa[0]['source'].startswith('probe_a@') and pa[0]['source_before']=='constants', 'v8 phase-A probe missing at the first densify interval: %r' % [(e['iteration'], e.get('source')) for e in pa]
    assert pa[0]['b_v']>0 and pa[0]['overhead_mb']>0 and pa[0]['b_f']==$BPF and abs(pa[0]['faces_per_vertex']-$FPV)<1e-9 and pa[0]['kept_share']>0 and pa[0]['v_star_after']>=1 and pa[0]['probe_vertices']<=pa[0]['vertices_model'], 'v8 phase-A probe fields (topology must keep the prior): %r' % {k: pa[0].get(k) for k in ('b_v','overhead_mb','b_f','faces_per_vertex','kept_share','v_star_before','v_star_after','probe_mb','probe_vertices','vertices_model')}
    assert str(dn[0].get('size_source','')).startswith('probe_a@'), 'densify event after the phase-A probe does not carry size_source probe_a@: %r' % dn[0].get('size_source')
    cfg=[e for e in ev if e['event']=='config'][0]; exp_rem=-(-(cfg['densify_until']-dn[0]['iteration'])//cfg['interval']); assert dn[0].get('remaining')==exp_rem and isinstance(dn[0].get('limiter_max'), int) and dn[0]['limiter_max']>=0, 'v9 intervals-left / limiter_max wrong (expected remaining %d): %r' % (exp_rem, {k: dn[0].get(k) for k in ('iteration','remaining','limiter_max','cap','desired','v_star')})
    print('smoke v9 plan:', {k: dn[0].get(k) for k in ('iteration','remaining','cap','limiter_max','desired','desired_next','v_star','limiter_bound','front_loaded')}, '| expected remaining', exp_rem)
    print('smoke probe-a:', {k: pa[0].get(k) for k in ('iteration','probe_mb','vertices_model','probe_vertices','kept_share','probe_kept_frac','b_v','overhead_mb','soup_b_f','soup_faces_per_vertex','v_star_before','v_star_after','source','probe_seconds','seconds_cleaning')})
else: print('note: no densify event within the 1600-it smoke (upstream densify_from >= 1600): the v8 phase-A probe path is NOT exercised by this smoke')
assert rs.get('probe_a_count')==len(pa) and len(rs.get('probes_a', []))==len(pa) and rs.get('probe_a_smoke') is True and rs.get('probe_a_every')==$PAE, 'rd_summary phase-A probe fields wrong: %r' % {k: rs.get(k) for k in ('probe_a_count','probe_a_smoke','probe_a_every')}
assert rs.get('growth_capped') is False and 'limiter_bound_intervals' in rs and rs.get('probe_extra_iters') == [], 'growth-capped / limiter fields wrong in the smoke: %r' % {k: rs.get(k) for k in ('growth_capped','limiter_bound_intervals','densify_intervals','probe_extra_iters')}
assert len(rs['probe_trend'])==1 and rs['extrapolate_to']>18000 and rs['probes'][1].get('faces_kept_1x') is None, 'trend / final-probe check fields wrong: %r' % {k: rs.get(k) for k in ('probe_trend','extrapolate_to')}
pr=rs['probes'][0]; assert pr.get('probe_scaling')==$PSCALE and pr.get('faces_kept_1x') is not None and pr['faces_kept']>pr['faces_kept_1x'], 'probe scaling check: %r' % {k: pr.get(k) for k in ('probe_scaling','faces_kept','faces_kept_1x','faces_model')}
print('smoke probe scaling: kept %d faces at %dx vs %d at 1x (of %d; %.2fx more)' % (pr['faces_kept'], pr['probe_scaling'], pr['faces_kept_1x'], pr['faces_model'], pr['faces_kept']/max(pr['faces_kept_1x'],1)))
print('smoke prune:', {k: p[0].get(k) for k in ('iteration','probe_mb','probe_est_mb','probe_kept_frac','probe_vertices','probe_faces','b_v','b_f','overhead_mb','faces_per_vertex','desired_pairs','excess_pairs','mode','n_lam','rank_k','n_cand','max_pairs','n_pairs','kept_removed','kept_faces_removed','removed_mb','bytes_per_pair_measured','lam_now','shadow_lam','vertices_before','vertices_after','probe_seconds')})
print('smoke final:', {k: fin[0][k] for k in ('vertices','triangles','lam_final','rhat_mb','probe_final_mb','growth_capped','limiter_bound_intervals','densify_intervals')})
PY
  grep -h "^\[RD probe-a\|^\[RD adapt\|^\[RD prune\|^Training is done" $SM/smoke.log | tail -5; for l in $SM/original/probes/*.log; do grep -h "^\[probe\]" $l; done
else echo "smoke skipped (SMOKE=0)"; fi
for T in $TARGETS; do
  G=/scratch/zr2245/$SCENE-adapt9b-t${T}mb; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G; L0=$(lam0_of $T)
  echo "=== $SCENE target $T MB (lam0 $L0) $FLAG: train with closed-loop rate control, checkpoints every 1k from 12k ($(date)) ==="
  if [ ! -f $CB ]; then
    python $CK --saves $SAVES --script $TR -- --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L0 --target_mb $T \
      --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --probe_extra_iters $EXTRA --probe_a_every $PAE \
      --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $G/train.log 2>&1 || { echo "FAILED: TRAIN $SCENE target $T"; tail -12 $G/train.log; for l in $G/original/probes/*.log; do echo "--- $l"; tail -8 $l; done; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE target $T"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "target $T -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD probe-a\|^\[RD adapt\|^\[RD prune\|^Training is done" $G/train.log | tail -22; grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2
  echo "=== $SCENE target $T: val_select ($(date)) ==="
  if [ ! -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then
    rm -rf $GB   # val_select creates the directory before its renders; a failed attempt must not be mistaken for a finished one on resubmit
    python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE target $T"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
  fi
  grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log; grep -h "importance over\|pruned best checkpoint" $G/val_select.log; [ -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt missing (val_select_v3 output)"; grep -h "FAILED\|Traceback" $G/val_select.log | tail -3; exit 1; }
  echo "=== $SCENE target $T: level-1.0 protocol on $GB ==="
  G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE target $T"; tail -8 $GB/level.log; exit 1; }
  NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE target $T"; tail -8 $GB/level.log; exit 1; }
  python - "$SCENE" "$T" "$G" "$NA" "$L0" "$XYZ" "$DC" <<'PY' | tee -a $TAB
import json, sys
S, T, G, NA, L0, xyz, dc = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
rs = json.load(open(G + '/original/rd_summary.json')); vsel = json.load(open(G + '/original/val_select.json')); b = json.load(open(NA + '/report.json')); f = json.load(open(NA + '/finetune/summary.json')); d = json.load(open(NA + '/deliver/summary.json')); t = d['delivery_test']
ev = [json.loads(l) for l in open(G + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']; pr = [e for e in ev if e['event'] == 'prune']; px = [e for e in ev if e['event'] == 'probe_extra']
pa = [e for e in ev if e['event'] == 'probe_a']; pas = [(e['iteration'], round(e['probe_mb'], 3), e['vertices_model'], e['probe_vertices'], e['kept_share'], round(e['b_v'], 2), round(e['overhead_mb'], 3), e['v_star_after'], round(e['probe_seconds'] + e['seconds_cleaning'], 1)) for e in pa]; bva = (round(pa[-1]['b_v'], 2) if pa else None); bv12 = (round(pr[0]['b_v'], 2) if pr else None); vs0 = (pa[0]['v_star_before'] if pa else None)
lams = [e['lam_used'] for e in dn]; q = lambda a, p: sorted(a)[int(p * (len(a) - 1))] if a else float('nan')
rounds = [(e['iteration'], round(e['probe_mb'], 3), round(e.get('probe_est_mb', e['probe_mb']), 3), e.get('probe_kept_frac'), e['vertices_before'], e['vertices_after'], e.get('mode'), ('%.2e' % e['lam_now']) if e.get('lam_now') is not None else None, round(e['b_v'], 2), round(e['b_f'], 2), e.get('n_pairs', 0), e.get('kept_removed'), e.get('kept_faces_removed'), round(e.get('removed_mb') or 0.0, 3)) for e in pr]
extra = [(e['iteration'], round(e['probe_mb'], 3), round(e['probe_est_mb'], 3), round(e['probe_slope_mb_per_1k'], 3)) for e in px]
ncoll = sum(e.get('n_pairs', 0) for e in pr); nkept = sum(e.get('kept_removed') or 0 for e in pr); nact = sum(1 for e in pr if e.get('n_pairs', 0) > 0); vsel_p = vsel.get('pruned') or {}
sig = [e.get('sigma') for e in dn]; sq = lambda p: sorted(x for x in sig if x is not None)[int(p * (len([x for x in sig if x is not None]) - 1))] if any(x is not None for x in sig) else float('nan')
nb = sum(1 for e in dn if e.get('limiter_bound')); nf = sum(1 for e in dn if e.get('front_loaded')); vs = dn[-1]['v_star'] if dn else None
yl = [e.get('rank_yield') for e in dn if e.get('rank_yield') is not None]; yq = lambda p: sorted(yl)[int(p * (len(yl) - 1))] if yl else float('nan')
print(f"{S} | target {T:g} MB | best iter {vsel['best_iteration']} val {vsel['best_val_psnr']:.3f} | lam0 {L0} -> lamA q10/q50/q90 {q(lams,.1):.2e}/{q(lams,.5):.2e}/{q(lams,.9):.2e} -> lam_final (shadow) {rs['lam_final']:.2e} | final probe {rs['probe_final_mb']:.3f} MB"
      f" | trained V {b['vertices']} F {b['faces']} | delivered {d['file_MB']:.3f} MB (hit {d['file_MB'] / T:.3f} x target, probe/delivered {rs['probe_final_mb'] / d['file_MB']:.3f}) | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f}"
      f" | val {f['validation_best']['psnr']:.3f} | K4096 R256 {xyz}/{dc} | growth-capped {rs.get('growth_capped')} (last landing est {rs.get('last_probe_est_mb')} MB) | densify events {len(dn)} (limiter-bound {nb}, front-loaded {nf}, V* {vs}), sigma q10/q50/q90 {sq(.1):.2f}/{sq(.5):.2f}/{sq(.9):.2f}, yield q10/q50/q90 {yq(.1):.2f}/{yq(.5):.2f}/{yq(.9):.2f}, prune rounds {len(pr)} active {nact} collapsed {ncoll} (both-kept share {nkept / max(ncoll, 1):.2f}; V {pr[0]['vertices_before'] if pr else None} -> {pr[-1]['vertices_after'] if pr else None}), val_select kept faces recomputed {vsel_p.get('kept_faces_recomputed')} / final-floor-4x {vsel_p.get('kept_faces_final_floor_4x')} / running-max {vsel_p.get('kept_faces_saved_running_max')} (floor {vsel_p.get('clean_floor')} ss {vsel_p.get('clean_supersample')}), extra probes (it, MB, est MB, slope) {extra} | phase-A probes {len(pa)} (last b_v {bva} vs 12k b_v {bv12}, V* {vs0} -> {vs}), probes-a (it, MB, V model, V kept, kept share, b_v, overhead MB, V* after, s) {pas}, rounds (it, probe MB, est MB, kept, V before, V after, mode, lam, b_v, b_f, pairs, kept-removed, kept-faces-removed, removed MB) {rounds}")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
echo "adapt-v9-blender $SCENE exit 0  $(date)"
SBEOF_ADAPT9B
cat > $AN/adapt-v9.sbatch <<'SBEOF_ADAPT9'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=adapt9
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zr2245/analysis/adapt9-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/adapt9-%x-%j.err
# adapt-v9.sbatch (2026-11-07, VERSION v9-r1): = adapt-v8.sbatch (VERSION v8-r1) with train_rd_adapt_v9.py (v8 + the phase-A plan fixes: intervals-left
# counted as upstream runs them, limiter allowance in triangles as applied; phase-A probes and the 16-round phase B as in v8), job names adapt9,
# output /scratch/zr2245/$SCENE-adapt9-t${T}mb, table $AN/adapt-v9.txt, the smoke asserting the densify event's 'remaining' against the config
# schedule and the presence of limiter_max.  Everything else verbatim v8-r1, whose text follows (names rewritten).
# adapt-v8.sbatch (2026-11-07, VERSION v8-r1): = adapt-v7b.sbatch (VERSION v7-r2) with train_rd_adapt_v8.py (phase A calibrated by byte probes at densify
# intervals after start_pruning, --probe_a_every $PAE (1000), 6-7 probes of ~20-45 s on these scenes; phase B unchanged), prune rounds every 500 from
# 12000 to 19500 (PRUNE default, 16 rounds as the Blender sbatch: the v7b sweep never opened the gate on these scenes, v8 aims higher and the
# collapse throughput per round is 4-12 % of the cap, so the trimming authority of 16 rounds is wanted; +8 probes of 30-60 s per run), the smoke
# extended to 1600 iterations so its first densify interval (1500) runs a phase-A probe (--probe_a_smoke) before the forced prune round at 1550,
# smoke assertions on the probe_a event / rd_summary fields, and the result row carrying every phase-A probe (bytes, kept share, b_v, overhead, V*).
# Output /scratch/zr2245/$SCENE-adapt9-t${T}mb/, table $AN/adapt-v9.txt.  Everything else verbatim v7-r2, whose text follows.
# adapt-v7b.sbatch (2026-11-06, VERSION v7-r2): = adapt-v7.sbatch (VERSION v7) with train_rd_adapt_v7b.py (v7 + the final-probe unpack fix that
# crashed every batch8 smoke at its last line); nothing else changed.  v7 text: Mip-360 / T&T sweep sbatch = adapt-v5.sbatch with train_rd_adapt_v7.py (phase B of v6 + the three fixes
# verified on the Blender testbed: exact kept-aware collapse accounting, prune gated on the extrapolated landing and capped at the excess, growth-capped
# flag; diagnostic probes at EXTRA after the last round), phase-A overshoot 1.0 (OVER default, as v6/v7 Blender), prune rounds every 1000 from 12000 to
# 19000 (PRUNE default; 8 rounds, probe ~30-60 s each on these scenes), the v7b smoke assertions and result row.  Same stages: (0) probe self-check on
# the adapt-v1 10 MB mesh where it exists, (1) smoke 1300 it with a forced prune round at 1250 and --probe_check_scaling, (2) per target T: train 30000 it,
# level-1.0 protocol, one row in $AN/adapt-v7.txt.  Output /scratch/zr2245/$SCENE-adapt7-t${T}mb/ .  Restartable.  INDOOR=1 passes --indoor; T&T scenes
# need DATA=/scratch/zr2245/datasets/tandt/$SCENE and IMAGES=images via --export.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; TARGETS=${TARGETS:-"5 10 20"}; INDOOR=${INDOOR:?set INDOOR=0 or 1 via --export}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"$(seq -s " " 12000 500 19500)"}
OVER=${OVER:-1.0}; EXTRA=${EXTRA:-"22000 26000"}; FPV=${FPV:-2.5}; PFLOOR=${PFLOOR:-0.9999}; PSCALE=${PSCALE:-4}; SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}; PAE=${PAE:-1000}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; STAGE2=/scratch/zr2245/shared-collapse-stage2-v2
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/adapt-v9.txt; TR=$AN/rdgrowth/train_rd_adapt_v9.py; PROBE_PY=$AN/rdgrowth/probe_bytes_v1.py
PROBE="bash $PKG/stage2_run.sh $PROBE_PY"
echo "adapt-v9 | scene $SCENE | IMAGES $IMAGES | targets $TARGETS MB | indoor '$FLAG' | initial B/v $BPV B/face $BPF faces/vertex $FPV | phase-A probes every $PAE it after start_pruning | overshoot $OVER probe floor $PFLOOR probe scaling $PSCALE sigma prior $SPRIOR | prune at $PRUNE mu $MU max_frac $MAXF | extra probes $EXTRA | probe $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $PROBE_PY $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $PKG/stage2_run.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py $STAGE2/STAGE2_VERSION $STAGE2/runtime.py $STAGE2/sharing_after.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python -m py_compile $TR || { echo "FAILED: $TR does not compile"; exit 1; }
python -m py_compile $PROBE_PY || { echo "FAILED: $PROBE_PY does not compile"; exit 1; }
grep -q "version='adapt-v9'" $TR || { echo "FAILED: $TR is not adapt-v9 (wrong file pasted?)"; exit 1; }
grep -q "max_split_up = max_split" $TR && grep -q "limiter_max=int(max_split_up)" $TR && grep -q "remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))" $TR || { echo "FAILED: $TR lacks the v9 plan fixes"; exit 1; }
grep -q "def update_attributes" $TR && grep -q "probe_a_every" $TR && grep -q "event='probe_a'" $TR || { echo "FAILED: $TR lacks the v8 phase-A probe (update_attributes / probe_a_every / probe_a event)"; exit 1; }
[ "$PAE" -ge 0 ] 2>/dev/null || { echo "FAILED: PAE must be an integer >= 0 (0 = no phase-A probes = v7b), got $PAE"; exit 1; }
grep -q "PROBE_VERSION = 'probe-v1'" $PROBE_PY || { echo "FAILED: $PROBE_PY is not probe-v1"; exit 1; }
grep -q "def faces_removed_by" $TR || { echo "FAILED: $TR lacks faces_removed_by (v6 pasted instead of v7?)"; exit 1; }
grep -q "growth_capped" $TR && grep -q "probe_extra_iters" $TR || { echo "FAILED: $TR lacks growth_capped / probe_extra_iters"; exit 1; }
grep -q "default=1.0, help='phase A aims at target" $TR || { echo "FAILED: $TR phase_a_overshoot default is not 1.0"; exit 1; }
NPR=$(echo $PRUNE | wc -w); [ "$NPR" -ge 4 ] || { echo "FAILED: PRUNE has $NPR rounds (v8 default is 16, at least 4 expected: $PRUNE)"; exit 1; }
LASTP=$(echo $PRUNE | tr ' ' '\n' | sort -n | tail -1); for x in $EXTRA; do [ "$x" -gt "$LASTP" ] && [ "$x" -lt 30000 ] || { echo "FAILED: EXTRA probe $x must lie after the last prune round $LASTP and before 30000"; exit 1; }; done
grep -q "triangles.scaling" $LAB/upstream/train.py || { echo "FAILED: $LAB/upstream/train.py never sets triangles.scaling (the probe's 4x pass assumes upstream's supersampling attribute)"; exit 1; }
lam0_of () { python -c "import sys; t=float(sys.argv[1]); print('%.4e' % min(1e-1, max(1e-4, 6.2e-3 * (9.0 / t) ** 1.67)))" $1; }   # initial guess only; adapted from the first interval on
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
SM=$AN/adapt9-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
if [ "$SELFCHECK" = "1" ]; then
  PV=/scratch/zr2245/$SCENE-adapt1-t10mb/axisB/v1.0
  if [ -f $PV/finetune/best.pt ] && [ -f $PV/deliver/summary.json ]; then
    echo "=== probe self-check on $PV ($(date)) ==="
    $PROBE --coarse $PV/finetune/best.pt --out $SM/selfcheck.json --k $K --r $R --xyz $XYZ --dc $DC > $SM/selfcheck.log 2>&1 || { echo "FAILED: PROBE SELF-CHECK (probe crashed)"; tail -30 $SM/selfcheck.log; exit 1; }
    python - "$SM/selfcheck.json" "$PV/deliver/summary.json" <<'PY' || { echo "FAILED: PROBE SELF-CHECK (bytes disagree with the delivered file)"; tail -5 $SM/selfcheck.log; exit 1; }
import json, sys
p = json.load(open(sys.argv[1])); d = json.load(open(sys.argv[2])); pm = p['file_bytes'] / 1e6; dm = d['file_MB']; r = pm / dm
big = {k: round(v / 1e6, 3) for k, v in sorted(p['payload_bytes'].items(), key=lambda kv: -kv[1]) if v > 20000}
print(f"self-check: probe {pm:.3f} MB vs delivered {dm:.3f} MB -> probe/delivered {r:.3f} | V {p['vertices']} F {p['faces']} | {p['seconds']['total']} s | streams {big}")
assert abs(r - 1.0) <= 0.08, f'probe/delivered {r:.3f} is outside 0.92..1.08'
PY
  else echo "note: no adapt-v1 10 MB delivery for $SCENE at $PV; self-check skipped (the smoke stage still exercises the probe)"; fi
else echo "self-check skipped (SELFCHECK=0)"; fi
if [ "$SMOKE" = "1" ]; then
  echo "=== smoke 1600 it: phase-A probe at the first densify interval (1500), target 0.01 MB so the prune round at 1550 must collapse; probe at ${PSCALE}x AND 1x ($(date)) ==="
  python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1600 --test_iterations 1600 --lam $(lam0_of 5) --target_mb 0.01 \
    --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --probe_check_scaling --sigma_prior $SPRIOR \
    --prune_iters 1550 --allow_any_prune_iter --prune_mu $MU --prune_max_frac 0.1 --probe_a_every $PAE --probe_a_smoke \
    --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $SM/smoke.log 2>&1 || { echo "FAILED: SMOKE"; tail -40 $SM/smoke.log; for l in $SM/original/probes/*.log; do echo "--- $l"; tail -15 $l; done; exit 1; }
  python - <<PY || { echo "FAILED: SMOKE CHECK"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
if dn: assert all('lam_used' in e and 'desired' in e and 'sigma_plan' in e and 'front_loaded' in e and 'v_star' in e for e in dn), 'densify events lack the v4 controller fields'
else: print('note: no densify event within the 1600-it smoke (upstream densify_from >= 1600); phase-A controller is exercised in the real runs')
assert p and 'probe_mb' in p[0], 'no prune event with probe_mb (probe did not run)'
assert not p[0].get('skipped') and p[0]['n_pairs'] > 0, 'prune round at 1550 collapsed nothing: %r' % {k: p[0].get(k) for k in ('skipped','desired_pairs','n_pairs','n_cand','max_pairs')}
assert p[0]['source'].startswith('probe@'), 'size model not replaced by the probe: %r' % p[0]['source']
assert p[0]['lam_now'] is not None and p[0].get('mode') in ('count', 'lambda', 'capped'), 'no prune mode / lambda: %r' % p[0].get('mode')
assert 'kept_removed' in p[0] and 0 <= p[0]['kept_removed'] <= p[0]['n_pairs'], 'kept-removed count missing / out of range: %r' % {k: p[0].get(k) for k in ('kept_removed','n_pairs')}
assert 'kept_faces_removed' in p[0] and 0 <= p[0]['kept_faces_removed'] <= p[0]['faces_before'] - p[0]['faces_after'], 'kept-faces-removed count missing / out of range: %r' % {k: p[0].get(k) for k in ('kept_faces_removed','faces_before','faces_after')}
assert 'removed_mb' in p[0] and p[0].get('bytes_per_pair_measured') is not None and p[0].get('excess_pairs', 0) > 0, 'v7 accounting fields missing: %r' % {k: p[0].get(k) for k in ('removed_mb','bytes_per_pair_measured','excess_pairs')}
if p[0]['removed_mb'] == 0: print('note: the smoke round collapsed pairs outside the probe kept set (removed_mb 0): accounting exercised, nothing to charge')
assert abs(p[0]['removed_mb'] * 1e6 - (p[0]['kept_removed'] * p[0]['b_v'] + p[0]['kept_faces_removed'] * p[0]['b_f'])) < 2000, 'removed_mb is not kept vertices x b_v + kept faces x b_f: %r' % {k: p[0].get(k) for k in ('removed_mb','kept_removed','kept_faces_removed','b_v','b_f')}
assert 'probe_est_mb' in p[0] and p[0]['probe_est_mb'] == p[0]['probe_mb'], 'first round must use the probe as is (no slope yet): %r' % {k: p[0].get(k) for k in ('probe_mb','probe_est_mb')}
assert 'probe_kept_frac' in p[0] and p[0]['probe_kept_frac'] > 0, 'probe kept fraction missing'
assert fin and 'probe_final_mb' in fin[0], 'no final event with probe_final_mb'
rs=json.load(open('$SM/original/rd_summary.json')); assert rs['version']=='adapt-v9' and len(rs['probes'])==2, 'rd_summary lacks the two phase-B/final probes'
pa=[e for e in ev if e['event']=='probe_a']
if dn:
    assert pa and pa[0]['iteration']==dn[0]['iteration'] and pa[0]['source'].startswith('probe_a@') and pa[0]['source_before']=='constants', 'v8 phase-A probe missing at the first densify interval: %r' % [(e['iteration'], e.get('source')) for e in pa]
    assert pa[0]['b_v']>0 and pa[0]['overhead_mb']>0 and pa[0]['b_f']==$BPF and abs(pa[0]['faces_per_vertex']-$FPV)<1e-9 and pa[0]['kept_share']>0 and pa[0]['v_star_after']>=1 and pa[0]['probe_vertices']<=pa[0]['vertices_model'], 'v8 phase-A probe fields (topology must keep the prior): %r' % {k: pa[0].get(k) for k in ('b_v','overhead_mb','b_f','faces_per_vertex','kept_share','v_star_before','v_star_after','probe_mb','probe_vertices','vertices_model')}
    assert str(dn[0].get('size_source','')).startswith('probe_a@'), 'densify event after the phase-A probe does not carry size_source probe_a@: %r' % dn[0].get('size_source')
    cfg=[e for e in ev if e['event']=='config'][0]; exp_rem=-(-(cfg['densify_until']-dn[0]['iteration'])//cfg['interval']); assert dn[0].get('remaining')==exp_rem and isinstance(dn[0].get('limiter_max'), int) and dn[0]['limiter_max']>=0, 'v9 intervals-left / limiter_max wrong (expected remaining %d): %r' % (exp_rem, {k: dn[0].get(k) for k in ('iteration','remaining','limiter_max','cap','desired','v_star')})
    print('smoke v9 plan:', {k: dn[0].get(k) for k in ('iteration','remaining','cap','limiter_max','desired','desired_next','v_star','limiter_bound','front_loaded')}, '| expected remaining', exp_rem)
    print('smoke probe-a:', {k: pa[0].get(k) for k in ('iteration','probe_mb','vertices_model','probe_vertices','kept_share','probe_kept_frac','b_v','overhead_mb','soup_b_f','soup_faces_per_vertex','v_star_before','v_star_after','source','probe_seconds','seconds_cleaning')})
else: print('note: no densify event within the 1600-it smoke (upstream densify_from >= 1600): the v8 phase-A probe path is NOT exercised by this smoke')
assert rs.get('probe_a_count')==len(pa) and len(rs.get('probes_a', []))==len(pa) and rs.get('probe_a_smoke') is True and rs.get('probe_a_every')==$PAE, 'rd_summary phase-A probe fields wrong: %r' % {k: rs.get(k) for k in ('probe_a_count','probe_a_smoke','probe_a_every')}
assert rs.get('growth_capped') is False and 'limiter_bound_intervals' in rs and rs.get('probe_extra_iters') == [], 'growth-capped / limiter fields wrong in the smoke: %r' % {k: rs.get(k) for k in ('growth_capped','limiter_bound_intervals','densify_intervals','probe_extra_iters')}
assert len(rs['probe_trend'])==1 and rs['extrapolate_to']>18000 and rs['probes'][1].get('faces_kept_1x') is None, 'trend / final-probe check fields wrong: %r' % {k: rs.get(k) for k in ('probe_trend','extrapolate_to')}
pr=rs['probes'][0]; assert pr.get('probe_scaling')==$PSCALE and pr.get('faces_kept_1x') is not None and pr['faces_kept']>pr['faces_kept_1x'], 'probe scaling check: %r' % {k: pr.get(k) for k in ('probe_scaling','faces_kept','faces_kept_1x','faces_model')}
print('smoke probe scaling: kept %d faces at %dx vs %d at 1x (of %d; %.2fx more)' % (pr['faces_kept'], pr['probe_scaling'], pr['faces_kept_1x'], pr['faces_model'], pr['faces_kept']/max(pr['faces_kept_1x'],1)))
print('smoke prune:', {k: p[0].get(k) for k in ('iteration','probe_mb','probe_est_mb','probe_kept_frac','probe_vertices','probe_faces','b_v','b_f','overhead_mb','faces_per_vertex','desired_pairs','excess_pairs','mode','n_lam','rank_k','n_cand','max_pairs','n_pairs','kept_removed','kept_faces_removed','removed_mb','bytes_per_pair_measured','lam_now','shadow_lam','vertices_before','vertices_after','probe_seconds')})
print('smoke final:', {k: fin[0][k] for k in ('vertices','triangles','lam_final','rhat_mb','probe_final_mb','growth_capped','limiter_bound_intervals','densify_intervals')})
PY
  grep -h "^\[RD probe-a\|^\[RD adapt\|^\[RD prune\|^Training is done" $SM/smoke.log | tail -5; for l in $SM/original/probes/*.log; do grep -h "^\[probe\]" $l; done
else echo "smoke skipped (SMOKE=0)"; fi
for T in $TARGETS; do
  G=/scratch/zr2245/$SCENE-adapt9-t${T}mb; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G; L0=$(lam0_of $T)
  echo "=== $SCENE target $T MB (lam0 $L0) $FLAG: train with closed-loop rate control ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L0 --target_mb $T \
      --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --probe_extra_iters $EXTRA --probe_a_every $PAE \
      --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $G/train.log 2>&1 || { echo "FAILED: TRAIN $SCENE target $T"; tail -12 $G/train.log; for l in $G/original/probes/*.log; do echo "--- $l"; tail -8 $l; done; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE target $T"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "target $T -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD probe-a\|^\[RD adapt\|^\[RD prune\|^Training is done" $G/train.log | tail -22
  echo "=== $SCENE target $T: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  python - "$SCENE" "$T" "$G" "$NA" "$L0" "$XYZ" "$DC" <<'PY' | tee -a $TAB
import json, sys
S, T, G, NA, L0, xyz, dc = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
rs = json.load(open(G + '/original/rd_summary.json')); b = json.load(open(NA + '/report.json')); f = json.load(open(NA + '/finetune/summary.json')); d = json.load(open(NA + '/deliver/summary.json')); t = d['delivery_test']
ev = [json.loads(l) for l in open(G + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']; pr = [e for e in ev if e['event'] == 'prune']; px = [e for e in ev if e['event'] == 'probe_extra']
pa = [e for e in ev if e['event'] == 'probe_a']; pas = [(e['iteration'], round(e['probe_mb'], 3), e['vertices_model'], e['probe_vertices'], e['kept_share'], round(e['b_v'], 2), round(e['overhead_mb'], 3), e['v_star_after'], round(e['probe_seconds'] + e['seconds_cleaning'], 1)) for e in pa]; bva = (round(pa[-1]['b_v'], 2) if pa else None); bv12 = (round(pr[0]['b_v'], 2) if pr else None); vs0 = (pa[0]['v_star_before'] if pa else None)
lams = [e['lam_used'] for e in dn]; q = lambda a, p: sorted(a)[int(p * (len(a) - 1))] if a else float('nan')
rounds = [(e['iteration'], round(e['probe_mb'], 3), round(e.get('probe_est_mb', e['probe_mb']), 3), e.get('probe_kept_frac'), e['vertices_before'], e['vertices_after'], e.get('mode'), ('%.2e' % e['lam_now']) if e.get('lam_now') is not None else None, round(e['b_v'], 2), round(e['b_f'], 2), e.get('n_pairs', 0), e.get('kept_removed'), e.get('kept_faces_removed'), round(e.get('removed_mb') or 0.0, 3)) for e in pr]
extra = [(e['iteration'], round(e['probe_mb'], 3), round(e['probe_est_mb'], 3), round(e['probe_slope_mb_per_1k'], 3)) for e in px]
ncoll = sum(e.get('n_pairs', 0) for e in pr); nkept = sum(e.get('kept_removed') or 0 for e in pr); nact = sum(1 for e in pr if e.get('n_pairs', 0) > 0)
sig = [e.get('sigma') for e in dn]; sq = lambda p: sorted(x for x in sig if x is not None)[int(p * (len([x for x in sig if x is not None]) - 1))] if any(x is not None for x in sig) else float('nan')
nb = sum(1 for e in dn if e.get('limiter_bound')); nf = sum(1 for e in dn if e.get('front_loaded')); vs = dn[-1]['v_star'] if dn else None
yl = [e.get('rank_yield') for e in dn if e.get('rank_yield') is not None]; yq = lambda p: sorted(yl)[int(p * (len(yl) - 1))] if yl else float('nan')
print(f"{S} | target {T:g} MB | lam0 {L0} -> lamA q10/q50/q90 {q(lams,.1):.2e}/{q(lams,.5):.2e}/{q(lams,.9):.2e} -> lam_final (shadow) {rs['lam_final']:.2e} | final probe {rs['probe_final_mb']:.3f} MB"
      f" | trained V {b['vertices']} F {b['faces']} | delivered {d['file_MB']:.3f} MB (hit {d['file_MB'] / T:.3f} x target, probe/delivered {rs['probe_final_mb'] / d['file_MB']:.3f}) | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f}"
      f" | val {f['validation_best']['psnr']:.3f} | K4096 R256 {xyz}/{dc} | growth-capped {rs.get('growth_capped')} (last landing est {rs.get('last_probe_est_mb')} MB) | densify events {len(dn)} (limiter-bound {nb}, front-loaded {nf}, V* {vs}), sigma q10/q50/q90 {sq(.1):.2f}/{sq(.5):.2f}/{sq(.9):.2f}, yield q10/q50/q90 {yq(.1):.2f}/{yq(.5):.2f}/{yq(.9):.2f}, prune rounds {len(pr)} active {nact} collapsed {ncoll} (both-kept share {nkept / max(ncoll, 1):.2f}; V {pr[0]['vertices_before'] if pr else None} -> {pr[-1]['vertices_after'] if pr else None}), extra probes (it, MB, est MB, slope) {extra} | phase-A probes {len(pa)} (last b_v {bva} vs 12k b_v {bv12}, V* {vs0} -> {vs}), probes-a (it, MB, V model, V kept, kept share, b_v, overhead MB, V* after, s) {pas}, rounds (it, probe MB, est MB, kept, V before, V after, mode, lam, b_v, b_f, pairs, kept-removed, kept-faces-removed, removed MB) {rounds}")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
echo "adapt-v9 $SCENE exit 0  $(date)"
SBEOF_ADAPT9
cat > $AN/submit-batch12-v1.sh <<'SUB12'
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9.py 910 a120351f64c9472e0353cef29d8ddb04
chk $AN/adapt-v9-blender.sbatch 162 68a42385114634c478a721a58edf50b6
chk $AN/adapt-v9.sbatch 154 df13ce37898f7dcc34a6c39ccd5d7dba
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
grep -q "version='adapt-v9'" $AN/rdgrowth/train_rd_adapt_v9.py && grep -q "def update_attributes" $AN/rdgrowth/train_rd_adapt_v9.py || { echo "CHECK FAILED: train_rd_adapt_v9.py is not adapt-v9"; exit 1; }
grep -q "max_split_up = max_split" $AN/rdgrowth/train_rd_adapt_v9.py && grep -q "remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))" $AN/rdgrowth/train_rd_adapt_v9.py || { echo "CHECK FAILED: train_rd_adapt_v9.py lacks the v9 plan fixes"; exit 1; }
grep -q "VERSION v9b-r1" $AN/adapt-v9-blender.sbatch && grep -q "VERSION v9-r1" $AN/adapt-v9.sbatch || { echo "CHECK FAILED: sbatch files are not v9-r1"; exit 1; }
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9.py $AN/blender/val_select_v3.py $AN/rdgrowth/probe_bytes_v1.py || { echo "CHECK FAILED: python files do not compile"; exit 1; }
bash -n $AN/adapt-v9-blender.sbatch && bash -n $AN/adapt-v9.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
for f in $AN/vis_collapse-v12/stage2_run.sh $AN/vis_collapse-v12/scripts/run_garden.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py $AN/blender/train_ckpt_v2.py $AN/coarse_finetune.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || { echo "CHECK FAILED: MISSING $f"; exit 1; }; done
for S in lego chair; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in room bicycle; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in lego chair; do [ -f /scratch/zr2245/$S-rd3b4-lam0.0062-mu1.0-best/axisB/v1.0/deliver/summary.json ] || echo "note: no lam 6.2e-3 delivery for $S, its probe self-check will be skipped"; done
for S in room bicycle; do [ -f /scratch/zr2245/$S-adapt1-t10mb/axisB/v1.0/deliver/summary.json ] || echo "note: no adapt-v1 10 MB delivery for $S, its probe self-check will be skipped"; done
for d in /scratch/zr2245/lego-adapt9b-t2.5mb /scratch/zr2245/chair-adapt9b-t2.5mb /scratch/zr2245/chair-adapt9b-t4mb /scratch/zr2245/room-adapt9-t10mb /scratch/zr2245/bicycle-adapt9-t10mb /scratch/zr2245/room-adapt9-t20mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt7|adapt8|adapt9|redeliv3"); [ "$busy" = 0 ] || echo "note: $busy adapt/redeliver jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show node gl030 | grep -o "CfgTRES=[^ ]*"
echo "CHECK OK"; k=0
b () { EX="SCENE=$1,XYZ=q14,TARGETS=$2"; [ "$2" = 2.5 ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9b $AN/adapt-v9-blender.sbatch) && { echo "submitted $1 $2 MB $j"; k=$((k+1)); } || echo "FAILED: sbatch $1 $2"; }
m () { EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9 $AN/adapt-v9.sbatch) && { echo "submitted $1 $5 MB $j"; k=$((k+1)); } || echo "FAILED: sbatch $1 $5"; }
b lego 2.5; b chair 2.5; m room 1 images_2 q14 10; m bicycle 0 images_4 q16 10; b chair 4; m room 1 images_2 q14 20
echo "submitted $k of 6 (4 GPUs: lego/chair 2.5 + room/bicycle 10 first, chair 4 + room 20 start as GPUs free up)"
SUB12
bash $AN/submit-batch12-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
```

## 2. 8–12 分钟后：自检 + smoke（五个带 smoke 的 job：lego 2.5、chair 2.5、room 10、bicycle 10、room 20）

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
for f in $AN/adapt9b-*-adapt9b-*.out $AN/adapt9-*-adapt9-*.out; do echo "=== $f"; grep -h "self-check\|smoke v9 plan\|smoke probe-a\|smoke probe scaling\|smoke prune\|smoke final\|note:\|FAILED\|=== .* target" $f | tail -9 | cut -c1-700; done
```

要看到 `smoke v9 plan: {'iteration': 1500, 'remaining': 17, ...} | expected remaining 17`、`smoke probe-a: {... 'source': 'probe_a@1500' ...}`、`smoke prune: {... 'iteration': 1550 ...}`、`smoke final`，没有 FAILED。有 FAILED 把那个 `.out` 最后 40 行贴我，不要重交；同场景另一个目标的 job 可以 `scancel`。

## 3. ≈40 min 后 lego/chair 三行，≈2.3 h 后 room/bicycle 三行（一次贴全）

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-4hours -o JobID,JobName%22,State,Elapsed,ExitCode | grep -E "adapt9|JobID"
for f in $AN/adapt9b-*-adapt9b-*.out $AN/adapt9-*-adapt9-*.out; do echo "=== $f"; tail -2 $f | cut -c1-200; done
echo "===== adapt-v9-blender.txt"; cat $AN/adapt-v9-blender.txt
echo "===== adapt-v9.txt"; cat $AN/adapt-v9.txt
echo "===== phase A probes / probes / prune"; for d in /scratch/zr2245/*-adapt9b-t*mb /scratch/zr2245/*-adapt9-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\[RD probe-a\|^\[RD adapt\|^\[RD prune\|^Training is done" $d/train.log | cut -c1-420; done
echo "===== densify (V*, limiter) at 3500/6000/9500"; for d in /scratch/zr2245/*-adapt9b-t*mb /scratch/zr2245/*-adapt9-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\[RD 3500\]\|^\[RD 6000\]\|^\[RD 9500\]" $d/train.log | cut -c1-330; done
```

两张表的 6 行 + `phase A probes` 段 + `densify` 段贴回来。我按上面的 (a)–(e) 判：过 → batch13（33 个 Mip/T&T + 16 个 Blender 重跑，sbatch 就是本批的两个，不再改）。
