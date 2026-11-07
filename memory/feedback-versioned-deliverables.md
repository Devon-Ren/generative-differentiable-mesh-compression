---
name: feedback-versioned-deliverables
description: Wenhao wants every revised file/zip delivered under a NEW versioned name (never overwrite the same filename); include a VERSION marker inside packages
metadata:
  type: feedback
---

Never deliver a revised package or script under the same filename as an earlier version (e.g. three different `vis_collapse.zip`). Use `name-v2.zip`, `name-v3.zip`, ... and put a VERSION file inside; have scripts print/record the version in their outputs.

**Why:** On 2026-10-12 three different `vis_collapse.zip` versions in one thread left Wenhao unable to tell which one was on HPC (a run silently used the old code), and made rollback impossible. The project attachment card also always opens the latest file under a reused path.

**How to apply:** bump the version in the filename on every change, say in one line what changed, and give a one-command way to check the installed version (e.g. `cat VERSION` or `--help | grep`). Related: [[c2-collapse-experiments]].
