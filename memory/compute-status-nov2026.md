---
name: compute-status-nov2026
description: 11/05 NODE GRANTED - Slurm reservation yw523-l40s on gl030 for zr2245, 2026-11-05 09:35 to 2026-11-12 09:35 (submit with --reservation=yw523-l40s); supersedes the 11/01 compute-exhausted rule and the 11/04 pending status
metadata:
  type: project
  modified: 2026-11-05T15:35:00Z
---

As of 2026-11-05 15:30: NYU HPC (Shenglong) created reservation `yw523-l40s`: Nodes=gl030 (4 x L40S: Gres=gpu:l40s:4, 128 cores), Users=zr2245, StartTime 2026-11-05T09:35 EndTime 2026-11-12T09:35, Flags IGNORE_JOBS,SPEC_NODES,MAGNETIC. Jobs already on the node finish first; preemptible jobs may use idle resources and are preempted within 30 min, so a new job can wait up to ~30 min. HPC asks to request only the resources needed.

**How to apply:** every sbatch for the next week: `--reservation=yw523-l40s` (account still torch_pr_38_general; keep --mem modest, --gres=gpu:1 per job, several jobs in parallel). Big sweeps are affordable now (~7 days x N GPUs). Batch 1 delivered 11/05 16:00 (commands/node-batch1-v1.md: fan v2 on 8 scenes ~12 GPU-h + ba-sens-v1 kitchen/garden BPV {8,13} x 2 lambda ~8 GPU-h). Priority proposed 11/05: fan on the other 8 scenes (~12 GPU-h) + unified LPIPS pass (~3) + K∝V / +1-bit XYZ re-export (~3); B_a sensitivity (~6 GPU-h, Yao's E2E concern); mesh_reg v4 lego; NeRF-Synthetic 3DGS black-bg batch. +++ line parked by the user 11/05. Before the node: general FairShare 0.24, small probes OK. See [[advisor-rd-questions-nov2026]], [[hpc-layout-nyu]].

11/05 lesson (first night on gl030): the node's 128 cores are shared with other users' preemptible CPU jobs (wang af3-pipeline took ~80 cores); with cpu-torch (18) + 8 cores per GPU job only 3 of 4 GPUs could start, squeue showing the misleading reason QOSMaxGRESPerUser (real GPU cap gpu48 QoS = 16/user). Check `scontrol show node gl030 | grep AllocTRES` first; fixes: wait for preemption (<=30 min), `scontrol update JobId=X NumCPUs=6`, or request --cpus-per-task=6 in future sbatch files on this node. `scontrol top` is not permitted for users.
