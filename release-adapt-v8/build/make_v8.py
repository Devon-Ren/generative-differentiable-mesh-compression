"""make_v8.py: build analysis/rdgrowth/train_rd_adapt_v8.py from train_rd_adapt_v7b.py by exact-count text replacements (v7b untouched)."""
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
s = open(src_path).read()
n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'anchor found {c}x (expected {count}): {old[:90]!r}'
    s = s.replace(old, new)
    n_rep += 1


HEADER = '''"""train_rd_adapt_v8.py (adapt-v8, 2026-11-07 = adapt-v7b with phase A CALIBRATED BY THE BYTE PROBE; phase B of v7b untouched; v1-v7b kept as
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
'''
rep('"""train_rd_adapt_v7b.py (adapt-v7b, 2026-11-06 = adapt-v7 with ONE fix:', HEADER + 'train_rd_adapt_v7b.py (adapt-v7b, 2026-11-06 = adapt-v7 with ONE fix:')

# --- SizeModel: phase-A probe list + attribute-only update ------------------------------------------------------------------------
rep("        self.overhead = float(rd['overhead_mb']) * 1e6; self.faces_per_vertex = float(rd['faces_per_vertex']); self.source = 'constants'; self.probes = []\n",
    "        self.overhead = float(rd['overhead_mb']) * 1e6; self.faces_per_vertex = float(rd['faces_per_vertex']); self.source = 'constants'; self.probes = []; self.probes_a = []\n")
rep("    def state(self): return dict(b_v=round(self.b_v, 3), b_f=round(self.b_f, 3), overhead_mb=round(self.overhead / 1e6, 4), faces_per_vertex=round(self.faces_per_vertex, 3), source=self.source)\n",
    "    def update_attributes(self, probe):\n"
    "        \"\"\"Phase A (v8): the probed model is a triangle SOUP (three own vertices per face), so only the attribute price per kept vertex and the fixed\n"
    "        overhead are taken from it; the topology price (b_f, faces per vertex) keeps its current value until the first mesh probe (update).\"\"\"\n"
    "        pb = probe['payload_bytes']; V, F = int(probe['vertices']), int(probe['faces'])\n"
    "        topo = sum(b for k, b in pb.items() if k in TOPO_KEYS); banks = sum(b for k, b in pb.items() if k in BANK_KEYS)\n"
    "        header = max(0, int(probe['file_bytes']) - sum(pb.values()))\n"
    "        self.overhead = banks + header; self.b_v = (int(probe['file_bytes']) - topo - self.overhead) / max(V, 1)\n"
    "        self.source = f\"probe_a@{probe['iteration']}\"; self.probes_a.append(probe)\n"
    "        probe.update(b_v=round(self.b_v, 3), overhead_mb=round(self.overhead / 1e6, 4), soup_b_f=round(topo / max(F, 1), 3), soup_faces_per_vertex=round(F / max(V, 1), 4), topo_keys_found=[k for k in pb if k in TOPO_KEYS])\n"
    "    def state(self): return dict(b_v=round(self.b_v, 3), b_f=round(self.b_f, 3), overhead_mb=round(self.overhead / 1e6, 4), faces_per_vertex=round(self.faces_per_vertex, 3), source=self.source)\n")

# --- training state + config ----------------------------------------------------------------------------------------------------
rep("    trend = []; k_asked = None; yield_hist = []  # probe trend for the extrapolation; rank-price yield (n_pass / k asked) of the last intervals\n",
    "    trend = []; k_asked = None; yield_hist = []  # probe trend for the extrapolation; rank-price yield (n_pass / k asked) of the last intervals\n"
    "    last_probe_a = None; n_probe_a = 0  # v8: iteration of the last phase-A probe; how many ran\n"
    "    if rd['probe_a_every'] < 0: raise SystemExit(f\"--probe_a_every must be >= 0 (0 disables the phase-A probes), got {rd['probe_a_every']}\")\n")
rep("    log.write(event='config', version='adapt-v7b', probe_extra_iters=extra_set,",
    "    log.write(event='config', version='adapt-v8', probe_a_every=rd['probe_a_every'], probe_a_smoke=rd['probe_a_smoke'], start_pruning=opt.start_pruning, probe_extra_iters=extra_set,")

# --- densify block: phase-A probe before the plan ---------------------------------------------------------------------------------------
rep("                    s_obs = surv.observe(V); sigma = surv.sigma  # survival of the previous round's new vertices in this interval's upstream prune\n",
    "                    s_obs = surv.observe(V); sigma = surv.sigma  # survival of the previous round's new vertices in this interval's upstream prune\n"
    "                    if rd['probe_a_every'] > 0 and (iteration > opt.start_pruning or rd['probe_a_smoke']) and (last_probe_a is None or iteration - last_probe_a >= rd['probe_a_every']):\n"
    "                        v_star_before = size.vertices_for(target_a); src_before = size.source; min_v = max(int(rd['probe_k']), int(rd['probe_r']))\n"
    "                        pa, used_a, _ = probe_bytes(triangles, scene, pipe, bg, rd, iteration, dataset.model_path, check=False, min_vertices=min_v)  # v8: the real encoder on the cleaned copy of the current soup\n"
    "                        if pa is None:  # fewer kept vertices than the dictionary size: the encoder cannot run (initialize raises 'K exceeds sample count'); try again at the next interval\n"
    "                            log.write(event='probe_a_skipped', iteration=iteration, elapsed_s=round(time.time() - t0), vertices_model=V, kept_vertices=int(used_a.sum()), min_vertices=min_v, **size.state())\n"
    "                            print(f\"\\n[RD probe-a {iteration}] skipped: {int(used_a.sum())} kept vertices of {V} < dictionary size {min_v}; size model stays {size.source}\")\n"
    "                        else:\n"
    "                            size.update_attributes(pa); last_probe_a = iteration; n_probe_a += 1\n"
    "                            r_pa = pa['file_bytes'] / 1e6; v_star_after = size.vertices_for(target_a); kappa = pa['vertices'] / max(V, 1)\n"
    "                            log.write(event='probe_a', iteration=iteration, elapsed_s=round(time.time() - t0), probe_mb=round(r_pa, 4), probe_ratio=round(r_pa / rd['target_mb'], 4), probe_vertices=pa['vertices'], probe_faces=pa['faces'],\n"
    "                                      vertices_model=V, faces_model=pa['faces_model'], kept_share=round(kappa, 4), probe_kept_frac=pa['kept_frac'], probe_seconds=pa['seconds_probe'], seconds_cleaning=pa['seconds_cleaning'],\n"
    "                                      soup_b_f=pa['soup_b_f'], soup_faces_per_vertex=pa['soup_faces_per_vertex'], v_star_before=int(v_star_before), v_star_after=int(v_star_after), source_before=src_before, **size.state())\n"
    "                            print(f\"\\n[RD probe-a {iteration}] soup {V} V -> cleaned {pa['vertices']} V {pa['faces']} F (kept share {kappa:.3f}) -> {r_pa:.3f} MB in {pa['seconds_cleaning']} + {pa['seconds_probe']} s | b_v {size.b_v:.2f} B (attributes per kept vertex) overhead {size.overhead / 1e6:.3f} MB | topology prior b_f {size.b_f:.2f} x {size.faces_per_vertex:.2f} faces/vertex (soup read {pa['soup_b_f']:.2f} x {pa['soup_faces_per_vertex']:.2f}, not used) | V* {int(v_star_before)} -> {int(v_star_after)} ({src_before} -> {size.source})\")\n")
# --- probe_bytes: optional minimum kept-vertex count (phase A only; phase B / final callers unchanged) --------------------------------
rep("def probe_bytes(triangles, scene, pipe, bg, rd, iteration, model_path, check=True):\n",
    "def probe_bytes(triangles, scene, pipe, bg, rd, iteration, model_path, check=True, min_vertices=0):\n")
rep("        if faces.numel(): used[faces.reshape(-1)] = True\n",
    "        if faces.numel(): used[faces.reshape(-1)] = True\n"
    "        if min_vertices and int(used.sum()) < int(min_vertices):  # v8 phase A: the K x R dictionaries need at least max(K, R) vertices; the caller skips this probe\n"
    "            return None, used.cpu(), keep.cpu()\n")
rep("limiter_bound=limiter_bound, front_loaded=front, rank_yield=round(yld, 4), k_asked=k_asked)\n",
    "limiter_bound=limiter_bound, front_loaded=front, rank_yield=round(yld, 4), k_asked=k_asked, size_source=size.source)\n")

# --- final event / summary / done line ------------------------------------------------------------------------------------------
rep("    log.write(event='final', growth_capped=growth_capped, limiter_bound_intervals=n_bound, densify_intervals=n_densify,",
    "    log.write(event='final', growth_capped=growth_capped, probe_a_count=n_probe_a, limiter_bound_intervals=n_bound, densify_intervals=n_densify,")
rep("        json.dump(dict(version='adapt-v7b', growth_capped=growth_capped, limiter_bound_intervals=n_bound,",
    "        json.dump(dict(version='adapt-v8', growth_capped=growth_capped, probe_a_every=rd['probe_a_every'], probe_a_smoke=rd['probe_a_smoke'], probe_a_count=n_probe_a, probes_a=size.probes_a, start_pruning=opt.start_pruning, limiter_bound_intervals=n_bound,")
rep("| adapt-v7b: target {rd['target_mb']} MB, final probe {probe_final:.3f} MB ({probe_final / rd['target_mb']:.3f} x), model {rhat_final:.2f} MB, lam0 {rd['lam']:.2e} -> lam_final {lam_now:.2e}\"\n"
    "          f\" | growth-capped {growth_capped} (limiter bound {n_bound} of {n_densify} intervals, last landing estimate {last_r_est})\")\n",
    "| adapt-v8: target {rd['target_mb']} MB, final probe {probe_final:.3f} MB ({probe_final / rd['target_mb']:.3f} x), model {rhat_final:.2f} MB, lam0 {rd['lam']:.2e} -> lam_final {lam_now:.2e}\"\n"
    "          f\" | growth-capped {growth_capped} (limiter bound {n_bound} of {n_densify} intervals, last landing estimate {last_r_est}) | phase-A probes {n_probe_a}\"\n"
    "          f\"{(' (last b_v %.2f overhead %.3f MB at %s)' % (size.probes_a[-1]['b_v'], size.probes_a[-1]['overhead_mb'], size.probes_a[-1]['iteration'])) if size.probes_a else ''}\")\n")

# --- argparse / rd dict --------------------------------------------------------------------------------------------------------
rep("    parser.add_argument('--probe_extra_iters', nargs='*', type=int, default=[], help='v7 diagnostic probes after the last prune round (e.g. 22000 26000): measured and logged, no collapse')\n",
    "    parser.add_argument('--probe_extra_iters', nargs='*', type=int, default=[], help='v7 diagnostic probes after the last prune round (e.g. 22000 26000): measured and logged, no collapse')\n"
    "    parser.add_argument('--probe_a_every', type=int, default=1000, help='v8: phase-A byte probes at densify intervals after start_pruning, at least this many iterations apart (0 = none, i.e. adapt-v7b); they replace the attribute price and the overhead of the size model')\n"
    "    parser.add_argument('--probe_a_smoke', action='store_true', help='smoke only: phase-A probe at the first densify interval even before start_pruning (the initial soup)')\n")
rep("prune_max_frac=args.prune_max_frac, bytes_per_face=args.bytes_per_face, probe_extra_iters=args.probe_extra_iters,\n",
    "prune_max_frac=args.prune_max_frac, bytes_per_face=args.bytes_per_face, probe_extra_iters=args.probe_extra_iters, probe_a_every=args.probe_a_every, probe_a_smoke=args.probe_a_smoke,\n")
rep("    print('Optimizing ' + args.model_path, '| RD growth+prune adapt-v7b target_mb', args.target_mb,",
    "    print('Optimizing ' + args.model_path, '| RD growth+prune adapt-v8 target_mb', args.target_mb, 'phase-A probes every', args.probe_a_every,")
rep("    if args.target_mb <= 0 or not (0 < args.lam_min < args.lam_max)",
    "    if args.probe_a_every < 0: sys.exit(f'--probe_a_every must be >= 0, got {args.probe_a_every}')\n"
    "    if args.target_mb <= 0 or not (0 < args.lam_min < args.lam_max)")
# the "[RD adapt] target" banner: say the phase-A probing
rep("    print(f\"[RD adapt] target {rd['target_mb']} MB (phase A aims at {target_a:.2f}) | lam0 {lam_now:.3e} | lambda = rank price of the planned count |",
    "    print(f\"[RD adapt] target {rd['target_mb']} MB (phase A aims at {target_a:.2f}) | lam0 {lam_now:.3e} | lambda = rank price of the planned count | phase-A probes every {rd['probe_a_every']} it after start_pruning {opt.start_pruning}{' (smoke: from the first interval)' if rd['probe_a_smoke'] else ''} |")

assert 'adapt-v7b' in s  # the v7b header text is kept inside the docstring
assert s.count("version='adapt-v8'") == 2 and s.count("version='adapt-v7b'") == 0, (s.count("version='adapt-v8'"), s.count("version='adapt-v7b'"))
open(dst_path, 'w').write(s)
print(f'wrote {dst_path}: {n_rep} replacements, {len(s.splitlines())} lines')
