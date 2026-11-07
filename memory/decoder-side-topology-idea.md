---
name: decoder-side-topology-idea
description: Direction A DEAD 10/27 (v4 -1.8 dB, v5 fixed-soup two-event -1.29 dB; trained soup leaves Delaunay-ness within ~6000 steps); snap-only control = free +0.19 dB / -1.5 % bytes on Room (clean+snap+freeze at 26000), user declined follow-up runs (dead = no more experiments)
metadata:
  type: project
  modified: 2026-10-27T05:00:00Z
---

Context: [[meshsplatting-compression-status]], plan/design-space-2026-10-26.md. Picked 10/26 as the one change that could lift the CVPR rating from borderline to accept (B generalisation-aware pricing = support, C scalable stream = no rating effect).

Rule verified in effrdel (rdel/__init__.py): scipy Delaunay of all vertices -> interior faces (face_tet[:,1] >= 0) -> keep a face iff the segment between its two tets' circumcentres hits the soup (lb.build_lbvh + upload_bvh_and_mesh + query_segments_any_hit). Depends only on vertices + soup. Upstream runs it once at 11000, then 19000 geometry steps (final v3 mesh: 2.71 % Delaunay faces).

Canonical decoder-side Delaunay (rdgrowth/delaunay_canon_v1.py): lexsort grid ints, P = q*float32(scale)+float32(low) + 2 %-step splitmix64 jitter, float32, scipy Delaunay; check script delaunay_check_v1.py = coverage + label entropy H0/H1 (+ asset topology bits).

v4 RESULT (Room lam 6.2e-3, 10/26): coverage 100 % ckpt + delivered, H1 7.7 bits/v vs 26.9 now, asset 4.56 MB -> est 3.53 MB (210x) BUT test 25.61 vs v3 27.382 (-1.8 dB) = FAIL. Causes (mine): rd_rebuild_v1 used the previous round's output as the next soup (drift 1.34 M -> 1.04 M faces, 7 rounds, no convergence); one event at 26000 + appearance-only 4000 steps cannot absorb ~30 % new faces.

v5 (delivered 10/26, commands/rd-topo-v5.md; rdgrowth/rd_rebuild_v2.py, train_rd_v5.py, test_rd_rebuild_cpu_v2.py; never edit v4 files): FixedSoupRestriction built once per event from effrdel pieces; --rebuild_iters 20000 26000 (early = float positions, no snap/freeze; final = snap + canonical + freeze, vertex set kept); face-only importance clean after each event; --snap_only control; stats soup_faces_retained / output_faces_new. Job: Room lam 6.2e-3 MODES="v5 snap" ~1 GPU-h -> rows in $AN/rd-topo-v5-room.txt. Pass rule (user): <= 0.1 dB very viable; 0.1-0.3 dB must sit >= 0.15 dB above the v3 RD curve at its est MB (~26.7-26.8 at 3.5 MB); > 0.4 dB dead. If pass: label stream + decoder Delaunay in stage2 (needs user consent, new version), then 11 scenes x 2 lambda (~16 GPU-h).

v5 RESULT (Room lam 6.2e-3, 10/27, rd-topo-v5-room.txt): v5 4.483 MB / 26.094 (-1.29 dB vs v3 27.382; est 3.42 MB with labels is ~0.55 dB below the RD curve) => DEAD. Event stats: soup faces retained 1.5 % at 20000 (post-clean removed 29 % of the 1.2 M new faces), 11 % at 26000 after 6000 geometry steps => restricted Delaunay is not a fixed point of training; the freedom is the quality. Snap-only control (clean + snap q14 + freeze at 26000, no Delaunay): 5.151 MB / 27.57 / .8453 / .4407 vs v3 5.228 / 27.382 / .8388 / .4467, same V (440k) => +0.19 dB, -1.5 % bytes, free. Paper use: A becomes the "connectivity is not re-derivable" ablation. User 10/27: once a direction is judged dead, no follow-up experiments (declined the garden/bonsai snap-only validation). Snap-only gain stays a single-scene observation (Room only); do not propose runs on it unless the user asks.
