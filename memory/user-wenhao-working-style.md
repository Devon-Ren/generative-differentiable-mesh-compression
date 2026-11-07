---
name: user-wenhao-working-style
description: How Wenhao Su wants to work with Claude — HPC usage habits (one GPU node for everything once held), command style, file delivery, language
metadata:
  type: user
---

Wenhao Su (NYU, MeshSplatting compression, CVPR 2027). Writes Chinese + English technical terms; wants replies the same way.

HPC habits (2026-10-12):
- Once he holds an interactive GPU node, he runs EVERYTHING there (CPU-heavy steps included; GPU nodes have enough RAM). Never tell him to drop the GPU and request a separate large-memory CPU node. Only use the small `cpu-torch` VS Code node (8 GB) for light CPU work when no GPU job is running.
- Wants complete, copy-paste command blocks for every step: full paths, env activation, no "same as before, just change X".
- He often starts a fresh shell between steps, so `$W`/`$C`/... from an earlier block are gone: every command block must re-define its variables. Long loops get mangled by terminal paste: write them to a script via heredoc and `bash` it.
- Never tell him to "upload file X to /scratch" (2026-10-16: "这玩意在哪？你让我手动调吗？"). Every HPC script, sbatch included, is delivered inline as a heredoc block that writes the file and then submits/runs it, in one paste.
- Scripts for him must fail LOUDLY: no `set -e` around `$(...)`/optional lookups (2026-10-13 run_draco.sh exited silently with zero output and he thought nothing ran). Print a line at every stage; on error print the log tail.

- Slurm resources (2026-10-16): default `--mem=32G` for our jobs; do not ask 96G unless a step is known to need it (big requests sit longer in the queue and look like low utilisation). Accounts: `torch_pr_38_general` has the better FairShare (0.26 vs 0.16) -> use it for long sbatch jobs; `torch_pr_38_tandon_advanced` for short / interactive runs. 2026-10-17: BALANCE the two accounts — when general already holds a batch of long jobs (e.g. 4 scenes x 2 h), put the next sbatch on tandon_advanced instead of stacking everything on one account.

Research stance (2026-10-13): Room / fixed train-val-test splits are only a testbed. Design strategies must be general; never tune for specific test views of a specific scene ("绝对不能钻牛角尖"). Select configs by a structural reading of ablations, fix one default, and confirm on another scene.

Delivery: [[feedback-versioned-deliverables]] — never reuse a filename for a revised package; version it.
Code: ask before editing his repos / pushing / opening PRs; new tooling goes under analysis/ as new files.

See [[hpc-layout-nyu]], [[meshsplatting-compression-status]].

- GPU budget (2026-10-18): both accounts' FairShare fell to ~0.18 after the budget/lambda batches; user asked to run only what is necessary. Rule: before each batch give the minimal run list per claim and say the GPU-hour cost; FairShare charges GPU-hours not job count, so cut runs, not jobs. Per lambda run ~35-51 min train (upstream's 2x/4x supersampling after 20k/25k its is the cost, not changeable) + 8-11 min protocol. Ask short walltimes (2-3 h) so pending jobs backfill; `scontrol update JobId=... TimeLimit=` on PD jobs.
- 2026-10-19: tandon_advanced FairShare fell to ~0.1 (job pending 3 h); user moved C to general himself. Until it recovers, put jobs on torch_pr_38_general.
- 2026-10-19: NEVER put `set -e` or a bare `exit 1` in a paste block meant for his interactive terminal: it killed his VS Code terminal (exit code 1, forced an MFA reconnect). Put the logic in a script written by heredoc and run it with `bash file.sh`; the script may exit, the shell survives; print "FAILED: <step>" + log tail.
- 2026-11-04: wasted a training job on an unverified upstream assumption (--save_iterations is a no-op in MeshSplatting's train.py). Rule: before a GPU job that depends on upstream behaviour, have him grep/confirm it first, or make the script self-check and fail BEFORE training; never assume 3DGS behaviour carries over.
