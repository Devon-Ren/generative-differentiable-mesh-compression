---
name: evidence-priced-growth-principle
description: 10/29 candidate thesis "grow by paying bytes for multi-view evidence": 11-scene check Spearman(residual level, test/train gain ratio) = -0.73; naive MDL (lambda = noise scale) refuted by 1e6; mechanism = consistency-gated dD (= design-space line B), probe Room+flowers 1-2 GPU-h; 3DGS transfer as the deadline-free shot
metadata:
  type: project
  modified: 2026-10-29T08:00:00Z
---

Doc: plan/evidence-priced-growth-2026-10-29.md (table, derivation, predictions P1-P4, mechanism, run list). Context [[meshsplatting-compression-status]], [[decoder-side-topology-idea]] (A dead), design-space line B.

User asked 10/29 whether anything "general, elegant" is left. Answer given: three measured legs of one principle. (1) bytes in the loop >> post-hoc (+0.3..+4.3 dB, [[scenes7-budget-vs-collapse]]); (2) NEW 10/29 CPU check on all-results-v1.csv: g = dtest/dtrain from lam 6.2e-3 -> 1.85e-3 vs train RMSE at lam_hi: Spearman -0.73 (n=11); lowest g = truck -0.57, flowers -0.30, train -0.14, treehill 0.13 (4 of the 5 highest-residual scenes; bicycle 0.48 is the exception); low-residual scenes g 0.41-0.73; (3) topology is real information (A dead) so "transmit less" is exhausted, only "buy right" remains.
MDL version REFUTED quantitatively: Laplace code-length price lambda_MDL ~ 8 ln2 b/(kappa N 1e6) ~ 2e-9 vs used 1e-3..1e-2 (residual code length GB >> model MB) => our lambda is a generalisation price, not a description-length price. Do not pitch "lambda = noise scale".
Surviving thesis: value signal must separate transferable (multi-view consistent, explainable by one 3D change) from per-view residual; one lambda for all scenes; size becomes a function of evidence => scene-adaptive stop. Mechanism: accumulator adds N_t, n_t, signed per-channel sums; c_t = |sum s|/sum|s|; dD' = m_t * c_t^alpha; same rule for prune. Fix tau/alpha once on Room+flowers, confirm on a third scene (10/13 rule).
Runs: probe Room + flowers lam 6.2e-3 (1-2 GPU-h; +lam 1.85e-3 optional) with pass rule P1 flowers bytes <= v3 and test >= v3-0.05, Room test >= v3-0.05 at equal bytes, gap narrows >= 0.1 dB; then 11 scenes x 1 lam ~8 GPU-h. Deadline-free shot: same gating on 3DGS densification (1 scene 1.5 GPU-h, then 3 scenes ~6). Status: proposed, waiting for the user's go.
