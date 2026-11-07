# Full-scan assessment for Devon (2026-11-07, evening)

Author: Claude (new thread, after Wenhao's handover v3). Audience: Devon Ren (lead), Prof. Yao Wang.
Everything below is read from this repo plus the T2-Splat paper (arXiv 2611.06974v1, fetched today). Numbers are quoted from the files named; nothing is projected unless marked "estimate".

## 0. Hard dates (checked today on cvpr.thecvf.com)

| | date | days from now |
|---|---|---|
| CVPR 2027 paper registration (title, abstract, authors) | Nov 10, 2026 AoE | 3 |
| Full paper | Nov 16, 2026 AoE | 9 |
| Supplementary | Nov 23, 2026 AoE | 16 |
| GPU reservation `yw523-l40s` (gl030, 4x L40S) ends | Nov 12, 09:35 | 4.7 |

The repo never pinned the deadline ("assume mid-November, unconfirmed"). The GPU window closes 4 days before the paper is due, so every GPU job must be submitted by Nov 11 evening. After that only CPU work (writing, figures from saved assets, tables).

## 1. What we have (paper-grade today)

Fixed-lambda line, complete on 11 scenes (Mip-360 x9 + T&T x2), indoor rerun with `--indoor` (matches the MeshSplatting paper within 0.09 dB):

| lambda | Mip-360 (9) MB / ratio / dPSNR vs own FP32 642 MB | T&T (2) |
|---|---|---|
| 6.2e-4 | 26.6 / 24x / -0.24 | 28.7 / 16x / -0.47 |
| 1.85e-3 | 20.0 / 32x / -0.37 | 22.3 / 21x / -0.21 |
| 6.2e-3 | 9.0 / 72x / -0.92 | 11.4 / 40x / -0.12 |
| 1.85e-2 | 4.3 / 151x / -1.68 | 5.4 / 85x / -0.54 |

(Source: analysis-snapshot/report/latex-v5/main.tex Tables 1-2. Note: paper/technical-report-v1.md and 00-README still quote the pre-rerun 11-scene means -0.22 / -0.59; pick one canonical set. I recommend the latex-v5 set.)

Also complete: lambda x dictionary fan on all 11 scenes (fixed K4096 R256 within 0.16 dB of the joint hull; Yao Q1 answered); indoor top-rate loss decomposition for room/kitchen (Yao Q2: DC q8 recovers only 0.04-0.08 dB, loss is SH sharing); B_a sensitivity 8 points on-curve (Yao Q3 partial); NeRF-Synthetic 8 objects (2.63 MB at FP32 PSNR; black background, val-selected checkpoints, non-standard protocol); deployability (train 0.57-0.87x upstream wall-clock, CPU decode 2-6 s, iPhone 145-350 fps); generic codecs on Room + garden (ours 3.5x below MPEG Open3DGC on the same asset); G0 in-the-loop ablation (7 scenes, 14 points, +0.3..+4.3 dB over post-hoc collapse, but on the OLD parameter set without `--indoor`).

Not paper-grade: the adaptive-lambda controller (latex-v5 still shows v1 at 52-77 % of target; v9 lands 0.79-0.88 on Mip). No intro, no related work in the LaTeX, no qualitative figure, no CVPR template. Yao has not endorsed submission.

## 2. Phase A: the mechanism, the root causes, and what is left

### 2.1 How the controller works (train_rd_adapt_v9.py)

- Growth happens only while `iteration < densify_until` (10k), at 17 densify intervals (1500 ... 9500). The restricted Delaunay rebuild is at 11k, the first *mesh* byte probe at 12k. Phase B (lambda-prune, 16 rounds 12k-19.5k) can only remove. Everything that decides how many vertices exist is therefore decided before 10k, from a size model, open loop.
- Size model: `R = b_v * V + b_f * F + overhead`, so `V* = (target - overhead) / (b_v + fpv * b_f)` (SizeModel.vertices_for). Plan: grow geometrically to V* over the remaining intervals; lambda each interval = the rank price of the planned split count (rank_lambda), corrected by the measured survival sigma and yield.
- v8 made `b_v` and `overhead` measured (byte probe on the cleaned soup every 1000 it after 3000). The topology term `fpv * b_f` stays a prior (2.5 faces/vertex x 2 B = 5 B/V) because the soup's connectivity says nothing about the Delaunay mesh that will be delivered.
- Phase B at every round: real probe, extrapolate to 24k (opacity ramp end), `excess = est - target`; gate: if excess <= 0 nothing is collapsed; otherwise collapse min(max(plan, lambda-admitted), whole excess).

### 2.2 The three bugs that are fixed (v7b -> v8 -> v9) and the evidence they are fixed

1. Constant prior 11 B/V (v7b): delivered B/V falls from 12.8 to 7.7 as targets grow (overhead amortised, attribute streams get cheaper with density) -> V* 10-30 % low. Fixed in v8 by measuring b_v on the soup (soup b_v within 0.8-8 % of the 12k mesh b_v, batch11).
2. Off-by-one in the interval count (v4-v8): `remaining = (until-it)//interval + 1` = 2 at 9500, upstream runs 1 -> the geometric plan only reached sqrt(V * V*) = 0.81-0.92 V*. Fixed in v9 (ceil). batch12: model/V* after the last prune = 0.96-0.99 on 5/6 runs.
3. Limiter simulated in vertices instead of triangles (3x too tight, false "limiter-bound" flags). Fixed in v9.

### 2.3 What remains (batch12, v9): hit = room10 0.877, bicycle10 0.790, room20 0.829; Blender 0.95-0.99

Both residual terms are things the soup cannot show before the 11k Delaunay rebuild:

- kappa_end = delivered vertices / model vertices after the final cleaning: 0.92-0.94 on Mip-360, 0.995 on Blender.
- Topology price of the delivered mesh: fpv * b_f = 2.2-2.4 x 1.7-1.9 = 3.7-4.4 B/V on Mip (prior 5), 6.1 on Blender.

Check: hit = kappa * (b_v + topo) / (b_v + 5) with batch12 b_v: room10 6.0 -> 0.93 x 10.2/11.0 = 0.86 (measured 0.877); bicycle10 6.7 -> 0.93 x 10.4/11.7 = 0.83 (measured 0.79, bicycle kappa is a bit lower); room20 6.1 -> 0.93 x 10.1/11.1 = 0.85 (measured 0.829). The decomposition in release-adapt-v9/README.md holds. **Phase A is no longer "buggy"; it is blind to two dataset-class quantities, and there is no lever after 10k.**

### 2.4 Options

| | what | code change | cost | risk |
|---|---|---|---|---|
| A. Overshoot + let phase B land | `--phase_a_overshoot 1.25` on Mip/T&T (= 1 / 0.80). Phase A aims 25 % above target; the 16 prune rounds remove the surplus with the real probe in the loop. On Blender the current prior is already 3-10 % high, so keep 1.0 there (or 1.1). | none (flag exists, default 1.0) | 6-job test ~4 GPU-h, then the 43-job sweep ~36 GPU-h | Phase B must remove 0-10 % on Mip; at 15 %/round x 16 rounds that is easy. Quality effect of lambda-prune at -8 % bytes was +0.01..0.02 dB (v3), and batch8-v2 showed the gate-closed runs are 0.3-0.5 dB *worse* than progressive pruning on Blender, so an active phase B is likely neutral or positive. Scene spread of kappa x topo is +-4 %, so landings should be 0.99-1.09 before phase B and within +-5 % after. |
| B. v10: second growth window after the 12k probe | Split faces of the Delaunay mesh at 12k-14k with the measured kappa / b_f, re-Delaunay, then prune | method change in upstream's schedule; new interaction with supersampling and opacity ramp | half a day design + CPU review + 6 jobs + sweep | Real risk of a new failure mode 4 days before the GPU window closes. |
| C. Dataset-class priors (kappa 0.93, topo 4.0 on unbounded scenes) | same effect as A but hidden in constants | small | same as A | Wenhao's rule "no dataset constants"; A is the same information expressed as a controller margin, which is standard practice and honest to describe. |

**Recommendation: A, tonight.** It is zero code diff on a file that passed a 6-job test and CPU replay today, it exercises phase B on every run (so the lambda-prune-as-regulariser question is answered by the sweep itself), and it is the only option that can produce a 19-scene table before Nov 12. Describe it in the paper as: phase A aims above the target by the ratio of the prior to the measured delivered-mesh price, because the cleaning share and the Delaunay topology price are unobservable on the soup; phase B lands with the real encoder in the loop.

Test block (6 jobs, same scenes as batch12 so the before/after is one table): room 10, bicycle 10, room 20 at 1.25; garden 5 and truck 10 at 1.25 (new scene classes); lego 2.5 at 1.1 (does a 10 % surplus plus trimming cost dB?). Pass bar, stated now: Mip hit within 0.95-1.05 on 4/5, lego hit 0.97-1.03 and test PSNR within 0.1 dB of batch12. If passed, submit the 43-job sweep immediately (Nov 8 morning), done by Nov 8 night on 4 GPUs.

### 2.5 Findings from an independent line-by-line audit of v9 that change the plan

The audit found no correctness bug in the v9 diff itself (interval formula, limiter unit, phase-B accounting all check out), but four things matter once phase B becomes active on Mip, which option A makes happen:

1. **lam_max at the end of phase A is not cosmetic.** At 9500 `remaining == 1`, so `desired_next = 0`, `k_asked = 0`, and `rank_lambda` returns lam_max = 0.1. Phase B then computes `thr_lam = 0.1 * mu`, which admits every edge (logs: "663k-7.9M edges below lam 1e-1"), so `wanted = min(max(desired, n_lam), excess_pairs)` becomes **the whole excess in the first active round** instead of excess / rounds_left. Phase B is therefore a one-shot capped collapse, not the progressive lambda-prune that v6 showed to be a regulariser on Blender (v6 vs gate-closed v7b: +0.3/+0.5 dB). Likely why v9 lego collapsed 4321 pairs vs v7b 11429 and sits 0.22 dB under the chord. Fix: when `k_asked == 0`, keep the previous interval's price (one line). This must ship with option A, in a new file (v9b), because A makes every Mip run go through phase B.
2. **Transient slope after the Delaunay.** The 12000 probe reads high and 12500 lower (bicycle 8.301 -> 8.224, room20 12.712 -> 12.556), so the two-point slope is strongly negative and the landing estimate dips (room20 est 0.45x at 12500). With a correct V* this would hold the gate closed for a few rounds while the true landing is above target. Cheapest mitigation without code: start `prune_iters` at 12500 (sbatch flag) so the Delaunay settling probe is not in the trend. The 16th round then moves to 20000, which is past start_upsampling and is rejected by the guard, so use 12500..19500 step 500 (15 rounds).
3. **Horizon error.** The estimate extrapolates to 24000 but the kept set keeps changing under supersampling at 20000/25000: last estimate vs final probe is -1.4 % (bicycle) to +4.4 % (room 20). With A the landing on 20 MB indoor scenes may sit +2..+4 % above target. Acceptable for the sweep; fixing it (a late trim round after 25000) is a method change and should wait.
4. **The three biases, measured.** hit = kappa_end x (b_v,mesh + b_f fpv) / (b_v,soup + 5): room10 0.880 (obs 0.877), bicycle 0.807 (0.790), room20 0.829 (0.829). The third factor, soup b_v over-reading the mesh b_v by 3-9 % on Mip, is why 1/0.80 = 1.25 and not 1/0.86 is the right margin.

Audit's own alternative worth keeping for later (not before the sweep): run the restricted Delaunay on the probe *copy* at the 8500/9500 phase-A probes, so the topology price is measured two intervals before growth ends; kappa would still be a prior (1.08). That is a method-neutral code change of maybe half a day; it would let the margin shrink from 1.25 to about 1.08 and is the honest "no constants" version if the reviewers or Yao push on the margin.

**Revised option A = v9b**: (i) keep the previous price when k_asked == 0; (ii) `--phase_a_overshoot 1.25` on Mip/T&T, 1.1 on Blender; (iii) prune rounds 12500..19500. Code diff 1-2 lines plus sbatch flags, same CPU-review discipline as v9 (py_compile, replay of the batch12 trajectories, smoke assertion that the 9500 densify event carries a real lam_next < lam_max).

## 3. The story (for Yao, for reviewers)

### 3.1 What Yao actually wants (from her 10/13, 10/20, 10/23, 11/03, 11/05 feedback)

- RD discipline: joint traversal of lambda x dictionary x precision (done), no self-vs-self baselines (MPEG/Draco rows, partially done), honest about "not end-to-end" (she objected; the honest label is "rate in the loop with a calibrated constant price and a fixed decodable coder").
- Do not frame against 3DGS codecs (she moved them to the appendix herself).
- She doubts the mesh direction after the garden 73x render; she asked for object-centric data (done: 8 objects) and said the real contribution would be a *bounded, compact, physically correct* differentiable mesh.
- She wants a fully adaptive lambda: no tuned constants, a target size as the input.

### 3.2 Proposed line: "size-bounded differentiable meshes"

The adaptive controller is not an appendix. It is the user-facing form of the whole method and the one thing nobody else in mesh-NVS offers (T2-Splat, MiLo, BOG, MeshSplatting all give one asset; T2-Splat's "face budget %" is not bytes). The paper is:

1. **Finding (why).** An opaque differentiable mesh cannot be compressed after training: collapsing a trained mesh to N vertices loses 0.3-4.3 dB against a mesh trained to N (14/14 points). Reason: one opaque triangle per pixel, vertices coupled through connectivity; the optimiser never re-places survivors. So the rate decision must be taken while the mesh grows.
2. **Method (how).** Byte-priced growth and collapse: split iff dD > lambda * dR with dR in bytes of the delivered file; the same price in reverse as an in-training edge collapse that keeps optimiser state; the real encoder as the byte probe; and a closed loop that turns a target size into lambda. One number in, one file out.
3. **System and study (what it buys).** The first complete encode/decode chain for this representation (dictionary SH, grid XYZ, cached-edge connectivity, context coding, isolated decoder), 24-151x on 11 unbounded scenes and 8 objects, engine-ready output (CPU decode seconds, phone 145-350 fps, standard rasteriser), and a compressibility anatomy: where the bytes go, why the codebook does not transfer across scenes, why the connectivity stream is the floor.

Stated in one sentence in the intro, not hidden: opaque meshes are about 3 dB below Gaussian splatting at equal bytes; that gap is the representation's (MeshSplatting FP32 is already 2.6 dB below 3DGS), and this paper is about what the mesh class can do per byte, not about beating Gaussians.

Claims to delete from the current text: "growth stops by itself" (it is schedule-bound at 10k); "one lambda = one exchange rate across scenes" as a contribution (scene slopes differ 5x); "lambda 6.2e-4 matches FP32" (no longer true after the indoor rerun); "48x smaller than MobileNeRF" without the black/white background caveat; any "faster than 3DGS" fps claim; the word "generative" in the title (reviewers will read generative models). Keep "rate in the loop".

### 3.3 T2-Splat (arXiv 2611.06974, HUST, Nov 4 2026): how to position

Not a competitor; a better base model. Their topology adaptation is QEM edge collapse plus gradient-driven vertex split with no byte price; their "compression" is a retained-face percentage; storage is a raw float PLY (garden 62 MB vs MeshSplatting 152 MB); no entropy coding, no quantisation, no bitstream, no rate control. Their limitations section lists "quantization, shared texture bases, asset-size budget" as future work: cite that sentence. They do beat MeshSplatting by +0.71 dB with fewer vertices, so write one limitation sentence: our pricing law and codec are representation-agnostic and would apply to T2-Splat's meshes; no code is released, so this is left to future work. Also: their per-face residual textures are evidence for our "texture-mapped mesh in disguise" reading (8x8 textures on 2.5M faces beat vertex SH on 3M faces).

Wording rule from now on: "compression" means bytes. Face reduction is "simplification" or "topology adaptation".

### 3.4 Honest reviewer attack list (from the repo's own reviews)

1. "Taming-3DGS / RDO-Gaussian moved to meshes" -> necessity (post-hoc fails on meshes, free on Gaussians), real-byte prices, in-training collapse with Adam state, target-size closed loop.
2. Weak post-hoc baseline in G0 (midpoint collapse, 3000-step finetune, no QEM, no long retraining) and G0 run on the old parameter set without `--indoor`, 7 scenes, 2 budgets. This is the biggest scientific hole: the paper's key finding rests on an outdated ablation.
3. Not end-to-end (constant price, fixed coder). Mitigation exists (B_a sensitivity: a wrong constant rescales lambda, the hull is unchanged); state it as a limitation.
4. 3 dB below HAC++ at equal bytes; the qualitative figure will show cracks on vegetation. Choose indoor + object figures, show the 24x and 72x points, never the 150x point as the teaser.
5. No external method re-run; MPEG/Draco rows only on Room + garden.

## 4. GPU plan for the window (Nov 7 evening -> Nov 11 evening), ~4 GPUs

Priority order; GPU-hours are estimates from the documented per-run times.

| # | job | GPU-h | why |
|---|---|---|---|
| 1 | v9b test, 6 jobs (Sec. 2.4-2.5) | 4-5 | gate for the sweep |
| 2 | v9b sweep, 43 jobs (node-batch13 rebuilt for v9b) | 36 | the 19-scene controller table; the headline feature |
| 3 | G0 refresh on current params: 6 scenes (room, kitchen, garden, bicycle, truck, lego) x {budget cap, post-hoc collapse with QEM + 3000 finetune} at one budget | 15-20 | closes attack 2 |
| 4 | Qualitative renders from saved assets (ours vs post-hoc vs Draco-RGB at equal bytes; indoor + object; wireframes) | 1 | the only figure a reviewer looks at first |
| 5 | Unified LPIPS/SSIM pass over the delivered assets (3DGS [0,1] convention) | 3 | removes the dagger footnote |
| 6 | Self-trained 3DGS on NeRF-Synthetic black background (same views) for the object figure | 1-2 | Yao's object chapter needs a fair picture |
| 7 | lambda-prune-always-on ablation (4 jobs) | 3 | only if 1-2 do not already answer it |

Total about 65 GPU-h, roughly 17 h of wall-clock on 4 GPUs if the node is free. Not doing: trainable entropy model, DC predictive coding (bitstream change), v10, anything MeshSplat+++, 3DGS-class codec reruns.

## 5. Writing plan (CPU, parallel to the GPU jobs)

- Nov 8: registration text (title, abstract, author list) for the Nov 10 deadline; intro and related work (paper/related-work-v1.md draft exists); decide the canonical headline set.
- Nov 9-10: method section from latex-v5 Sec. 1 (compress to CVPR length), experiments skeleton with the existing tables; controller section rewritten for v9 + overshoot once the sweep lands.
- Nov 11-13: figures (teaser, qualitative, method), G0 refresh table, limitations.
- Nov 14-16: full read-through by Yao, polish, submit. Supplementary (full fan, per-scene tables, bitstream spec, isolation proof) by Nov 23.

## 6. Decisions I need from Devon

1. Option A (overshoot) vs B (v10) for the controller. I recommend A.
2. Confirm the HPC state is as documented (03-HPC.md Sec. 9 has a read-only verification block) and that `release-adapt-v9` files are on the node unchanged (md5 in release-adapt-v9/README.md).
3. Whether Yao has agreed to submit at all. If not, the Nov 10 registration is the forcing function: a title and abstract she can react to by Nov 9.
4. Headline set: latex-v5 (indoor rerun) or the older 11-scene means. I recommend latex-v5.
