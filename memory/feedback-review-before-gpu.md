---
name: feedback-review-before-gpu
description: user 11/06 "给我仔细审查，不要不停返工了，我的 GPU 不是这么浪费的" -- after v3->v4->v5 each needed another GPU round; review the whole chain on CPU and ship ONE reviewed version
metadata:
  type: feedback
---

User (Wenhao, 2026-11-06 12:28 UTC) after the adapt-v5 Blender testbed needed yet another fix round: review carefully, stop the endless rework, GPU time is not to be wasted like this.

**Why:** v3, v4 and v5 each fixed what the previous round's runs exposed (probe scaling, missing lambda-prune, extrapolation), each costing 2.5-9 GPU-h, and v5's remaining gap turned out to live in the delivery chain (val_select_v2 running-max importance), not the controller.

**How to apply:** before proposing any GPU batch: (1) read every stage the result passes through (training wrapper, selection, delivery, row script), not just the controller; (2) explain every measured discrepancy with a stated mechanism confirmed in code; (3) replay new logic on CPU against the logs already in hand; (4) have an independent reviewer read new code; (5) ship one versioned batch with startup self-checks that fail before the long part; (6) say the pass bar up front. One batch, one answer; never "run it and see".
