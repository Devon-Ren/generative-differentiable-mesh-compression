---
name: adapt-v6-blender-results-nov2026
description: adapt-v6 Blender testbed RESULT 11/06 (lego/chair x 2.5/4/6 MB, node-batch7): delivery fixed by val_select_v3, quality above hull, every run under target 0.67-0.95; the three measured phase-B faults that v7 fixes, plus the phase-A prior / horizon caveats
metadata:
  type: project
  modified: 2026-11-06T15:01:00.240Z
---

Files: rows rd/adapt-v6-blender-2026-11-06.txt, refs rd/redeliver-v3-blender-2026-11-06.txt, raw paste rd/adapt-v6-blender-2026-11-06-paste.md, analysis rd/adapt_analysis_v6b.py -> rd/adapt-v6-blender-analysis.txt, replay rd/adapt_replay_v7.py -> rd/adapt-v7-replay-on-v6-logs.txt.

Delivery fix CONFIRMED: refs re-delivered with val_select_v3 bytes x0.81-0.90 at +0.00-0.02 dB (lego 7.517 MB/33.516 and 2.763/32.583; chair 4.660/32.212 and 2.391/32.166); adapt probe/delivered 1.006-1.074 (v5 0.83-0.94); recomputed kept faces = final-floor-4x within 0.1 %.

Landing FAILED (delivered hit lego 0.946/0.839/0.884, chair 0.902/0.766/0.666; probe 0.71-0.97). Mechanisms: (1) removed_mb = pairs x (b_v + b_f x ALL faces removed per pair) over-counts the probe's real loss x1.52-1.90 -> LS virtual slope inflated -> est@12.5k/final 1.20-1.44, est@19.5k/final 1.04-1.06; (2) lambda-prune ran in all 16 rounds whatever est said: 14-19 % of V12 removed, 4-14 % in rounds where the controller wanted MORE bytes; (3) no-prune landing (P12 + implied un-stacking growth, kept-aware accounting) lego2.5 1.09, chair2.5 1.03, lego4 0.98, lego6 1.00, chair4 0.88, chair6 0.76 -> chair 4/6 growth-capped (limiter 13/17, V12/V* 0.68/0.61). Replay with exact accounting: est@19.5k/final 1.006-1.030; gate (est > target) would open only on lego 2.5.

Caveats kept: phase-A size prior 11 B/V is right on Mip-360 (b_v 7.0, F/V 2.2-2.9, b_f 1.75) but ~25 % low on Blender (b_v 8.2, F/V 3.4, b_f 1.8 = 14.3 B per kept V) -> V* ~30 % high there, masked by the limiter; dataset-dependent so NOT tuned. Un-stacking growth continues after 19.5k (+0.11-0.51 MB to the 30k final probe) with no probe in between: horizon 24000 unverified -> v7 diagnostic probes at 22000/26000.

Quality: equal-bytes dPSNR vs the 2-point v3 reference chord lego -0.09*/+0.27/+0.28, chair -0.02*/+0.34/+0.51 (* chord extrapolated; a concave hull's chord under-reads, so + values are upper bounds); 16 gradual rounds did not hurt; chair adapt points beat the fixed-lambda chair chord by 0.3-0.5 dB (confirm on the full hull).
See [[adaptive-lambda-nov2026]].
