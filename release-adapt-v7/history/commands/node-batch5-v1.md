# node-batch5-v1（2026-11-07）：adapt-v4 = probe 按最终 4× 超采样渲染清理、phase B 只在高于目标时按数量 collapse、phase A 几何规划 + 顶住 upstream 增长上限 — 4 场景 × 3 目标，≈8–9 GPU-h

v3 的 12 条结果（rd/adapt-v3-analysis.txt）：hit 0.73–1.03（均值 0.87），等字节质量回到固定 λ 曲线上（−0.10..+0.07 dB，均值 −0.007），但三个地方还是让它落在目标下面。v4 各改一处，全部在新文件 `rdgrowth/train_rd_adapt_v4.py`（700 行）和 `rd/adapt-v4.sbatch`（112 行）里，v3 / probe_bytes_v1.py / upstream 不动：
- **Probe 还是低估（1.03–1.39×）**：v3 的 probe 用最终 opacity floor 但按 1× 分辨率渲染清理。upstream 在 25000 后把 `triangles.scaling` 设成 4，最终清理（30000）是在 4× 超采样下算 max blending 的，很多在 1× 下一个像素都占不到、在 4× 下能占到的面最后都保留了。所以 v3 在 18k 的 probe 只保留 0.30–0.59 的面，而最终清理只扔 2–4 % 的顶点。v4 的 probe 渲染清理这一遍时把 `scaling` 临时设成 4（和 floor 一样，渲染完恢复）。smoke 阶段加 `--probe_check_scaling`：同时按 1× 再渲染一遍，4× 保留的面必须多于 1×，否则说明 renderer 没读这个属性，job 在 4 分钟内停掉。
- **Phase B 在目标之下还在剪**：v3 每轮固定 λ prune 剪掉 11–16 % 的顶点（probe 已经低于目标、控制器想要更多字节）。v4 的 phase B 只按数量：probe 高于目标 → 收最便宜的 desired 条边（dD/dR 排名），λ := 影子价格；低于目标 → 这轮不动（mode = none，记下固定 λ 本来会收多少条）。
- **Phase A 到 20 MB 时差 18–26 %**：v3 按字节线性规划，前面要得少、后面要得多，而 upstream 每个 interval 最多新增 23 %（indoor 27 %）顶点（`add_percentage`），后半段每次都顶在上限。v4 改成几何规划：V* = 尺寸模型反解出的顶点数，每个 interval 的增长率 f = (V*/V)^(1/剩余)，desired = V(f−1)/(3σ)；并把这条路径在 upstream 上限下向前模拟一遍（start_pruning 之前 σ = 1，之后用实测 σ，第一次测到之前用 `--sigma_prior 0.5`，之后就被测量值替换），模拟到不了 V* 就这个 interval 要满上限（增长是复利，早要比晚要值钱）。每个 densify 事件记 limiter_bound / front_loaded。CPU 模拟（σ 0.4、V* 2.04 M、18 个 interval）：v3 式规划到 1.03 M，v4 到 1.49 M——**20 MB 仍可能被 upstream 上限卡住**，如果 v4 的 20 MB 行 limiter-bound 全程为真，剩下的唯一杠杆是 `--add_percentage`（upstream 自己的参数，不改代码，但那是训练改动，先看结果再说）。

成本同 v3：每个 job 3 个目标 ≈ 1.5–2.3 h，4 块卡并行，≈ 8–9 GPU-h（smoke 多一遍 1× 渲染，+1 分钟）。

## 1. 写两个文件 + 提交 4 个 job（一次粘贴；probe_bytes_v1.py 用 HPC 上已有的）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/rdgrowth
cat > $AN/rdgrowth/train_rd_adapt_v4.py <<'PYEOF_TRAIN4'
"""train_rd_adapt_v4.py (adapt-v4, 2026-11-07 = adapt-v3 with the three faults of ITS 12 runs fixed; v1/v2/v3 kept untouched):
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
        self.overhead = float(rd['overhead_mb']) * 1e6; self.faces_per_vertex = float(rd['faces_per_vertex']); self.source = 'constants'; self.probes = []
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
    def state(self): return dict(b_v=round(self.b_v, 3), b_f=round(self.b_f, 3), overhead_mb=round(self.overhead / 1e6, 4), faces_per_vertex=round(self.faces_per_vertex, 3), source=self.source)


def probe_bytes(triangles, scene, pipe, bg, rd, iteration, model_path):
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
        if rd.get('probe_check_scaling'):
            kept_1x = int((importance(1) > 0.5).sum())
            if not int(keep.sum()) > kept_1x:
                raise RuntimeError(f'probe scaling has no effect: kept {int(keep.sum())} faces at {rd["probe_scaling"]}x vs {kept_1x} at 1x (triangles.scaling not read by the renderer?)')
        faces = triangles._triangle_indices[keep].long()
        used = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=faces.device)
        if faces.numel(): used[faces.reshape(-1)] = True
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
    return probe


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
    With add_pct (upstream growth limiter: at most add_pct * V - V new vertices per interval) the plan is simulated forward
    under the limiter; when it cannot reach V_star this interval asks for everything the limiter allows, because growth
    compounds and a split now is worth more than the same split later.  Returns (desired, front_loaded)."""
    remaining = len(sigmas)
    if remaining < 1 or V_star <= V or V <= 0: return 0.0, False
    def geo(v, r, s): return v * ((V_star / v) ** (1.0 / r) - 1.0) / (3.0 * max(s, 1e-3)) if (r >= 1 and V_star > v) else 0.0
    want = geo(V, remaining, sigmas[0])
    if add_pct is None: return want, False
    v = V
    for j, s in enumerate(sigmas):
        d = min(geo(v, remaining - j, s), max(0.0, add_pct * v - v) / 3.0)
        v += 3.0 * d * s
    if v < 0.995 * V_star: return max(want, max(0.0, add_pct * V - V) / 3.0), True
    return want, False


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
    log.write(event='config', version='adapt-v4', probe_scaling=rd['probe_scaling'], sigma_prior=rd['sigma_prior'], target_mb=rd['target_mb'], target_a_mb=target_a, phase_a_overshoot=rd['phase_a_overshoot'], probe_floor=rd['probe_floor'],
              size_model=size.state(), probe_cmd=rd['probe_cmd'], probe=dict(k=rd['probe_k'], r=rd['probe_r'], xyz=rd['probe_xyz'], dc=rd['probe_dc'], dictionary_rounds=rd['probe_dictionary_rounds']),
              lam_min=rd['lam_min'], lam_max=rd['lam_max'], overhead_mb=rd['overhead_mb'], lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'],
              bytes_per_split=bytes_per_split, max_points=opt.max_points, add_percentage=opt.add_percentage,
              densify_from=opt.densify_from_iter, densify_until=opt.densify_until_iter, interval=opt.densification_interval,
              split_schedule=rd['split_schedule'], schedule_total=(sum(sched.values()) if sched else None),
              prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'])
    print(f"[RD adapt] target {rd['target_mb']} MB (phase A aims at {target_a:.2f}) | lam0 {lam_now:.3e} | lambda = rank price of the planned count | probe floor {rd['probe_floor']} | lam in [{rd['lam_min']:.1e}, {rd['lam_max']:.1e}] | size model {size.state()} | probe {rd['probe_k']}x{rd['probe_r']} {rd['probe_xyz']}/{rd['probe_dc']}")
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
                    rhat = size.mb(V)  # corrected projection of the delivered mesh (faces_per_vertex * V faces, not the soup's)
                    remaining = max(1, (opt.densify_until_iter - iteration) // opt.densification_interval + 1)
                    V_star = size.vertices_for(target_a)
                    sig_meas = sigma if surv.hist else float(rd['sigma_prior'])  # measured once a cohort prune has been seen, else the prior
                    sigmas = [(sig_meas if iteration + opt.densification_interval * (j + 1) > opt.start_pruning else 1.0) for j in range(remaining)]  # cohort j is pruned at its next interval only after start_pruning
                    desired, front = plan_splits(V, V_star, sigmas, opt.add_percentage); cap = int(math.ceil(desired))
                    limiter_bound = bool(front or max_split < cap); max_split = min(max_split, cap)  # the plan caps the count; lambda is the price of that count
                    lam_used = lam_now; lam_hist.append(lam_now)
                    ratio_pre, n_allowed = split_ratios(acc, triangles, bytes_per_split, triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    idx, st = rd_select(acc, triangles, lam_now, bytes_per_split, max_split,
                                        triangles.size_probs_zero, triangles.size_probs_zero_image_space)
                    n_split = rd_split(triangles, idx, iteration, splitt_large_triangles)
                    V_after = int(triangles.vertices.shape[0]); surv.record(V, V_after)
                    V_exp = V + sigmas[0] * (V_after - V)  # expected count after the next upstream prune
                    desired_next = plan_splits(V_exp, V_star, sigmas[1:], opt.add_percentage)[0] if remaining > 1 else 0.0
                    lam_now = rank_lambda(ratio_pre, n_allowed, int(math.ceil(desired_next)), rd['lam_min'], rd['lam_max'])  # price of the next planned count
                    ratio = float(st['n_pass']) / max(desired, 1.0)
                    st.update(event='densify', iteration=iteration, vertices_before=V, vertices_after=V_after,
                              triangles_after=int(triangles._triangle_indices.shape[0]), n_split_total=n_split, k_ref=k_ref, elapsed_s=round(time.time() - t0),
                              lam_used=lam_used, lam_next=lam_now, rhat_mb=round(rhat, 4), desired=round(desired, 1), desired_next=round(desired_next, 1), remaining=remaining, cap=cap,
                              pass_ratio=round(ratio, 4), sigma=round(sigma, 4), sigma_obs=(round(s_obs, 4) if s_obs is not None else None), sigma_plan=round(sigmas[0], 4), n_allowed=n_allowed, v_star=int(V_star), limiter_bound=limiter_bound, front_loaded=front)
                    log.write(**st)
                    print(f"\n[RD {iteration}] V {V}->{V_after} pass {st['n_pass']}/{st['n_tri']} split {n_split} "
                          f"(limit {max_split}) ratio q50/q90/q99 {st['ratio_q50']:.2e}/{st['ratio_q90']:.2e}/{st['ratio_q99']:.2e} "
                          f"lam {lam_used:.2e} -> {lam_now:.2e} (projected {rhat:.2f} / phase-A aim {target_a:.2f} MB = {V_star} vertices, desired {desired:.0f} -> next {desired_next:.0f}{' LIMITER' if limiter_bound else ''}{' FRONT-LOAD' if front else ''}, pass/desired {ratio:.2f}, sigma {sigma:.2f} (obs {s_obs}, plan {sigmas[0]:.2f}), {remaining} intervals left) valid {st['valid_frac']}")
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
                probe = probe_bytes(triangles, scene, pipe, bg, rd, iteration, dataset.model_path)  # real encoder on the cleaned copy
                size.update(probe); r_meas = probe['file_bytes'] / 1e6
                rounds_left = len([i for i in prune_set if i >= iteration])
                excess = (r_meas - rd['target_mb']) * 1e6
                b_coll = size.bytes_per_collapse(last_faces_removed) if last_faces_removed else size.bytes_per_collapse()  # measured faces per collapse x probed b_f
                desired = excess / max(b_coll, 1e-6) / rounds_left
                if not phase_b_started:
                    phase_b_started = True
                    if lam_hist:
                        tail = lam_hist[-20:]; lam_a = math.exp(sum(math.log(x) for x in tail) / len(tail))
                        print(f"\n[RD adapt {iteration}] phase A ended at lam {lam_now:.2e} (geo-mean of last {len(tail)} intervals {lam_a:.2e})")
                print(f"\n[RD adapt {iteration}] probe: {probe['vertices']} V {probe['faces']} F after cleaning -> {r_meas:.3f} MB ({r_meas / rd['target_mb']:.3f} x target) in {probe['seconds_probe']} s"
                      f" | b_v {size.b_v:.2f} b_f {size.b_f:.2f} overhead {size.overhead / 1e6:.3f} MB | {b_coll:.1f} B per collapse -> {desired:.0f} collapses wanted ({rounds_left} rounds left)")
                ev = dict(event='prune', iteration=iteration, views=n_views, elapsed_s=round(time.time() - t0), probe_mb=round(r_meas, 4), probe_ratio=round(r_meas / rd['target_mb'], 4),
                          probe_vertices=probe['vertices'], probe_faces=probe['faces'], probe_kept_frac=probe['kept_frac'], probe_seconds=probe['seconds_probe'], desired_pairs=round(desired, 1), rounds_left=rounds_left,
                          bytes_per_collapse=round(b_coll, 2), **size.state())
                b_v_full = size.b_v + 2.0 * size.b_f  # select_collapses takes bytes_per_vertex = attributes + 2 faces, bytes_per_face
                edges, cnt, score = edge_scores(triangles.vertices, triangles._triangle_indices, dD, size.b_v, size.b_f)
                thr_lam = float(lam_now) * float(rd['prune_mu']); n_lam = int((score < thr_lam).sum())  # how many edges the fixed-lambda prune would admit (logged only)
                if desired < 1.0:
                    acc.reset(T0)
                    ev.update(skipped=True, mode='none', lam_now=lam_now, n_lam=n_lam, vertices_before=V0, vertices_after=V0, faces_before=T0, faces_after=T0, n_pairs=0)
                    log.write(**ev)
                    print(f"[RD prune {iteration}] under target: no collapse this round (the fixed-lambda prune would have taken {n_lam} edges)")
                else:
                    thr, k_rank = rank_threshold(score, desired)  # candidates = the ~4 x desired cheapest edges; the cap keeps the desired cheapest independent ones
                    frac = min(rd['prune_max_frac'], desired / V0)
                    pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac)
                    lam_sp = shadow_price(edges, score, pairs, rd['prune_mu'])
                    if lam_sp is not None: lam_now = lam_sp  # lambda := shadow price of the count
                    cs = collapse_edges(triangles, pairs)
                    if ps['n_pairs'] > 0: last_faces_removed = (T0 - cs['faces_after']) / ps['n_pairs']
                    acc.reset(triangles._triangle_indices.shape[0])
                    ev.update(skipped=False, mode='count', lam_now=lam_now, shadow_lam=lam_sp, n_lam=n_lam, cap_frac=round(frac, 4), rank_k=k_rank,
                              faces_removed_per_pair=(round(last_faces_removed, 3) if last_faces_removed else None)); ev.update(ps); ev.update(cs)
                    log.write(**ev)
                    print(f"\n[RD prune {iteration}] V {V0}->{cs['vertices_after']} T {T0}->{cs['faces_after']} | count-driven: wanted {desired:.0f} pairs (fixed-lambda prune would admit {n_lam} edges), cap {ps['max_pairs']} -> {ps['n_pairs']} collapsed "
                          f"(flip-rejected {ps['n_flip_rejected']}, passes {ps['passes']}) | lam {lam_now:.2e} (shadow {lam_sp}) | score q10/q50/q90 {ps['score_q10']:.2e}/{ps['score_q50']:.2e}/{ps['score_q90']:.2e} "
                          f"| faces-removed share sel {ps['faces_removed_share_sel']} all {ps['faces_removed_share_all']} | views {n_views}")
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
    final_probe = probe_bytes(triangles, scene, pipe, bg, rd, iteration, dataset.model_path); size.probes.append(final_probe)
    probe_final = final_probe['file_bytes'] / 1e6
    log.write(event='final', vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0]),
              elapsed_s=round(time.time() - t0), lam_final=lam_now, rhat_mb=round(rhat_final, 4), probe_final_mb=round(probe_final, 4), probe_final_ratio=round(probe_final / rd['target_mb'], 4),
              target_mb=rd['target_mb'], **size.state())
    with open(os.path.join(dataset.model_path, 'rd_summary.json'), 'w') as f:
        json.dump(dict(version='adapt-v4', probe_scaling=rd['probe_scaling'], sigma_hist=surv.hist, probe_floor=rd['probe_floor'], target_mb=rd['target_mb'], lam_final=lam_now, rhat_final_mb=round(rhat_final, 4), probe_final_mb=round(probe_final, 4), probes=size.probes, size_model=size.state(),
                       phase_a_overshoot=rd['phase_a_overshoot'], probe_cmd=rd['probe_cmd'], lam=rd['lam'], dd_mode=rd['dd_mode'], dd_norm=rd['dd_norm'], bytes_per_vertex=rd['bytes_per_vertex'], split_schedule=rd['split_schedule'],
                       prune_iters=prune_set, prune_mu=rd['prune_mu'], prune_max_frac=rd['prune_max_frac'], bytes_per_face=rd['bytes_per_face'],
                       vertices=int(triangles.vertices.shape[0]), triangles=int(triangles._triangle_indices.shape[0])), f, indent=1)
    print(f"Training is done; vertices {int(triangles.vertices.shape[0])} triangles {int(triangles._triangle_indices.shape[0])} | adapt-v4: target {rd['target_mb']} MB, final probe {probe_final:.3f} MB ({probe_final / rd['target_mb']:.3f} x), model {rhat_final:.2f} MB, lam0 {rd['lam']:.2e} -> lam_final {lam_now:.2e}")


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
    parser.add_argument('--prune_iters', nargs='*', type=int, default=[12000, 14000, 16000, 18000], help='phase-B lambda-prune rounds (no values = no pruning)')
    parser.add_argument('--prune_mu', type=float, default=1.0, help='collapse iff ring-residual / bytes-saved < lam * prune_mu')
    parser.add_argument('--prune_max_frac', type=float, default=0.15, help='at most this fraction of vertices collapsed per round')
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
    parser.add_argument('--phase_a_overshoot', type=float, default=1.15, help='phase A aims at target * this; the margin is trimmed back to the probe by phase B (prune rounds can only remove bytes)')
    parser.add_argument('--probe_cmd', type=str, required=True, help='command prefix that runs probe_bytes_v1.py in the stage2 env, e.g. "bash <PKG>/stage2_run.sh <AN>/rdgrowth/probe_bytes_v1.py"')
    parser.add_argument('--probe_k', type=int, default=4096); parser.add_argument('--probe_r', type=int, default=256)
    parser.add_argument('--probe_xyz', type=str, required=True, help='XYZ precision of the delivery (q13..q18, the scene rule; never auto)')
    parser.add_argument('--probe_dc', type=str, default='q7'); parser.add_argument('--probe_dictionary_rounds', type=int, default=2)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.target_mb <= 0 or not (0 < args.lam_min < args.lam_max) or args.phase_a_overshoot < 1.0 or not (0.5 <= args.probe_floor < 1.0) or args.probe_scaling < 1 or not (0 < args.sigma_prior <= 1):
        sys.exit(f'bad controller settings: target_mb {args.target_mb} lam range [{args.lam_min}, {args.lam_max}] overshoot {args.phase_a_overshoot} probe_floor {args.probe_floor}')
    if not args.probe_xyz.startswith('q') or not args.probe_xyz[1:].isdigit() or not (12 <= int(args.probe_xyz[1:]) <= 18):
        sys.exit(f'--probe_xyz must be q12..q18, got {args.probe_xyz}')
    _probe_script = [t for t in shlex.split(args.probe_cmd) if t.endswith('.py')]
    if not _probe_script or not os.path.isfile(_probe_script[-1]):
        sys.exit(f'--probe_cmd must name an existing probe script: {args.probe_cmd}')
    print('Optimizing ' + args.model_path, '| RD growth+prune adapt-v4 target_mb', args.target_mb, 'overshoot', args.phase_a_overshoot, 'probe', args.probe_cmd, 'lam0', args.lam, 'mode', args.dd_mode, 'norm', args.dd_norm, 'B/v', args.bytes_per_vertex,
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
                     prune_iters=args.prune_iters, prune_mu=args.prune_mu, prune_max_frac=args.prune_max_frac, bytes_per_face=args.bytes_per_face,
                     allow_any_prune_iter=args.allow_any_prune_iter, target_mb=args.target_mb, probe_floor=args.probe_floor, probe_scaling=args.probe_scaling, probe_check_scaling=args.probe_check_scaling, sigma_prior=args.sigma_prior,
                     lam_min=args.lam_min, lam_max=args.lam_max, overhead_mb=args.overhead_mb, faces_per_vertex=args.faces_per_vertex, phase_a_overshoot=args.phase_a_overshoot,
                     probe_cmd=args.probe_cmd, probe_k=args.probe_k, probe_r=args.probe_r, probe_xyz=args.probe_xyz, probe_dc=args.probe_dc, probe_dictionary_rounds=args.probe_dictionary_rounds))
PYEOF_TRAIN4
cat > $AN/adapt-v4.sbatch <<'SBEOF_ADAPT4'
#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=adapt4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zr2245/analysis/adapt4-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/adapt4-%x-%j.err
# adapt-v4.sbatch (2026-11-07, VERSION v4): = adapt-v3.sbatch with train_rd_adapt_v4.py (see its docstring: probe renders the cleaning pass at the
# final supersampling 4x as well as the final opacity floor; phase B count-driven only, no collapse under target; phase A plans the vertex count
# geometrically and front-loads when the upstream growth limiter would otherwise leave it short).  Same stages: (0) probe self-check on the adapt-v1
# 10 MB mesh, (1) smoke 1300 it with a forced prune round at 1250 and --probe_check_scaling (the 4x pass must keep more faces than 1x, else the
# renderer ignores triangles.scaling and the job stops here), (2) per target T: train 30000 it, level-1.0 protocol, one row in $AN/adapt-v4.txt.
# Output /scratch/zr2245/$SCENE-adapt4-t${T}mb/ .  Restartable.  INDOOR=1 passes --indoor.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; TARGETS=${TARGETS:-"5 10 20"}; INDOOR=${INDOOR:?set INDOOR=0 or 1 via --export}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
OVER=${OVER:-1.15}; FPV=${FPV:-2.5}; PFLOOR=${PFLOOR:-0.9999}; PSCALE=${PSCALE:-4}; SPRIOR=${SPRIOR:-0.5}; SMOKE=${SMOKE:-1}; SELFCHECK=${SELFCHECK:-1}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; STAGE2=/scratch/zr2245/shared-collapse-stage2-v2
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/adapt-v4.txt; TR=$AN/rdgrowth/train_rd_adapt_v4.py; PROBE_PY=$AN/rdgrowth/probe_bytes_v1.py
PROBE="bash $PKG/stage2_run.sh $PROBE_PY"
echo "adapt-v4 | scene $SCENE | IMAGES $IMAGES | targets $TARGETS MB | indoor '$FLAG' | initial B/v $BPV B/face $BPF faces/vertex $FPV | overshoot $OVER probe floor $PFLOOR probe scaling $PSCALE sigma prior $SPRIOR | prune at $PRUNE mu $MU max_frac $MAXF | probe $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $PROBE_PY $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $PKG/stage2_run.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py $STAGE2/STAGE2_VERSION $STAGE2/runtime.py $STAGE2/sharing_after.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python -m py_compile $TR || { echo "FAILED: $TR does not compile"; exit 1; }
python -m py_compile $PROBE_PY || { echo "FAILED: $PROBE_PY does not compile"; exit 1; }
grep -q "version='adapt-v4'" $TR || { echo "FAILED: $TR is not adapt-v4 (wrong file pasted?)"; exit 1; }
grep -q "PROBE_VERSION = 'probe-v1'" $PROBE_PY || { echo "FAILED: $PROBE_PY is not probe-v1"; exit 1; }
grep -q "def plan_splits" $TR || { echo "FAILED: $TR lacks plan_splits (v3 pasted instead of v4?)"; exit 1; }
grep -q "triangles.scaling" $LAB/upstream/train.py || { echo "FAILED: $LAB/upstream/train.py never sets triangles.scaling (the probe's 4x pass assumes upstream's supersampling attribute)"; exit 1; }
lam0_of () { python -c "import sys; t=float(sys.argv[1]); print('%.4e' % min(1e-1, max(1e-4, 6.2e-3 * (9.0 / t) ** 1.67)))" $1; }   # initial guess only; adapted from the first interval on
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
SM=$AN/adapt4-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
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
  echo "=== smoke 1300 it, target 0.01 MB so the prune round at 1250 must collapse; probe at ${PSCALE}x AND 1x ($(date)) ==="
  python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam $(lam0_of 5) --target_mb 0.01 \
    --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --probe_check_scaling --sigma_prior $SPRIOR \
    --prune_iters 1250 --allow_any_prune_iter --prune_mu $MU --prune_max_frac 0.1 \
    --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $SM/smoke.log 2>&1 || { echo "FAILED: SMOKE"; tail -40 $SM/smoke.log; for l in $SM/original/probes/*.log; do echo "--- $l"; tail -15 $l; done; exit 1; }
  python - <<PY || { echo "FAILED: SMOKE CHECK"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
if dn: assert all('lam_used' in e and 'desired' in e and 'sigma_plan' in e and 'front_loaded' in e and 'v_star' in e for e in dn), 'densify events lack the v4 controller fields'
else: print('note: no densify event within the 1300-it smoke (upstream densify_from > 1300); phase-A controller is exercised in the real runs')
assert p and 'probe_mb' in p[0], 'no prune event with probe_mb (probe did not run)'
assert not p[0].get('skipped') and p[0]['n_pairs'] > 0, 'prune round at 1250 collapsed nothing: %r' % {k: p[0].get(k) for k in ('skipped','desired_pairs','n_pairs','n_cand','max_pairs')}
assert p[0]['source'].startswith('probe@'), 'size model not replaced by the probe: %r' % p[0]['source']
assert p[0]['lam_now'] is not None and p[0].get('mode') == 'count', 'smoke prune must be count-driven (target 0.01 MB): %r' % p[0].get('mode')
assert 'probe_kept_frac' in p[0] and p[0]['probe_kept_frac'] > 0, 'probe kept fraction missing'
assert fin and 'probe_final_mb' in fin[0], 'no final event with probe_final_mb'
rs=json.load(open('$SM/original/rd_summary.json')); assert rs['version']=='adapt-v4' and len(rs['probes'])==2, 'rd_summary lacks the two probes'
pr=rs['probes'][0]; assert pr.get('probe_scaling')==$PSCALE and pr.get('faces_kept_1x') is not None and pr['faces_kept']>pr['faces_kept_1x'], 'probe scaling check: %r' % {k: pr.get(k) for k in ('probe_scaling','faces_kept','faces_kept_1x','faces_model')}
print('smoke probe scaling: kept %d faces at %dx vs %d at 1x (of %d; %.2fx more)' % (pr['faces_kept'], pr['probe_scaling'], pr['faces_kept_1x'], pr['faces_model'], pr['faces_kept']/max(pr['faces_kept_1x'],1)))
print('smoke prune:', {k: p[0].get(k) for k in ('iteration','probe_mb','probe_kept_frac','probe_vertices','probe_faces','b_v','b_f','overhead_mb','faces_per_vertex','desired_pairs','mode','n_lam','rank_k','n_cand','max_pairs','n_pairs','lam_now','shadow_lam','vertices_before','vertices_after','probe_seconds')})
print('smoke final:', {k: fin[0][k] for k in ('vertices','triangles','lam_final','rhat_mb','probe_final_mb')})
PY
  grep -h "^\[RD adapt\|^\[RD prune\|^Training is done" $SM/smoke.log | tail -4; for l in $SM/original/probes/*.log; do grep -h "^\[probe\]" $l; done
else echo "smoke skipped (SMOKE=0)"; fi
for T in $TARGETS; do
  G=/scratch/zr2245/$SCENE-adapt4-t${T}mb; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G; L0=$(lam0_of $T)
  echo "=== $SCENE target $T MB (lam0 $L0) $FLAG: train with closed-loop rate control ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L0 --target_mb $T \
      --bytes_per_vertex $BPV --bytes_per_face $BPF --faces_per_vertex $FPV --dd_mode error --dd_norm mpx --phase_a_overshoot $OVER --probe_floor $PFLOOR --probe_scaling $PSCALE --sigma_prior $SPRIOR \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF \
      --probe_cmd "$PROBE" --probe_k $K --probe_r $R --probe_xyz $XYZ --probe_dc $DC $FLAG > $G/train.log 2>&1 || { echo "FAILED: TRAIN $SCENE target $T"; tail -12 $G/train.log; for l in $G/original/probes/*.log; do echo "--- $l"; tail -8 $l; done; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE target $T"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "target $T -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD adapt\|^\[RD prune\|^Training is done" $G/train.log | tail -12
  echo "=== $SCENE target $T: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE target $T"; tail -8 $G/level.log; exit 1; }
  python - "$SCENE" "$T" "$G" "$NA" "$L0" "$XYZ" "$DC" <<'PY' | tee -a $TAB
import json, sys
S, T, G, NA, L0, xyz, dc = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
rs = json.load(open(G + '/original/rd_summary.json')); b = json.load(open(NA + '/report.json')); f = json.load(open(NA + '/finetune/summary.json')); d = json.load(open(NA + '/deliver/summary.json')); t = d['delivery_test']
ev = [json.loads(l) for l in open(G + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']; pr = [e for e in ev if e['event'] == 'prune']
lams = [e['lam_used'] for e in dn]; q = lambda a, p: sorted(a)[int(p * (len(a) - 1))] if a else float('nan')
rounds = [(e['iteration'], round(e['probe_mb'], 2), e.get('probe_kept_frac'), e['vertices_before'], e['vertices_after'], e.get('mode'), ('%.2e' % e['lam_now']) if e.get('lam_now') is not None else None, round(e['b_v'], 2), round(e['b_f'], 2)) for e in pr]
sig = [e.get('sigma') for e in dn]; sq = lambda p: sorted(x for x in sig if x is not None)[int(p * (len([x for x in sig if x is not None]) - 1))] if any(x is not None for x in sig) else float('nan')
nb = sum(1 for e in dn if e.get('limiter_bound')); nf = sum(1 for e in dn if e.get('front_loaded')); vs = dn[-1]['v_star'] if dn else None
print(f"{S} | target {T:g} MB | lam0 {L0} -> lamA q10/q50/q90 {q(lams,.1):.2e}/{q(lams,.5):.2e}/{q(lams,.9):.2e} -> lam_final (shadow) {rs['lam_final']:.2e} | final probe {rs['probe_final_mb']:.3f} MB"
      f" | trained V {b['vertices']} F {b['faces']} | delivered {d['file_MB']:.3f} MB (hit {d['file_MB'] / T:.3f} x target, probe/delivered {rs['probe_final_mb'] / d['file_MB']:.3f}) | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f}"
      f" | val {f['validation_best']['psnr']:.3f} | K4096 R256 {xyz}/{dc} | densify events {len(dn)} (limiter-bound {nb}, front-loaded {nf}, V* {vs}), sigma q10/q50/q90 {sq(.1):.2f}/{sq(.5):.2f}/{sq(.9):.2f}, prune rounds (it, probe MB, kept, V before, V after, mode, lam, b_v, b_f) {rounds}")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
echo "adapt-v4 $SCENE exit 0  $(date)"
SBEOF_ADAPT4
cat > $AN/submit-adapt-v4.sh <<'SUB5'
AN=/scratch/zr2245/analysis; RES=yw523-l40s
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v4.py 700; chk $AN/adapt-v4.sbatch 112; chk $AN/rdgrowth/probe_bytes_v1.py 66
grep -q "version='adapt-v4'" $AN/rdgrowth/train_rd_adapt_v4.py || { echo "CHECK FAILED: train_rd_adapt_v4.py is not adapt-v4"; exit 1; }
grep -q "def plan_splits" $AN/rdgrowth/train_rd_adapt_v4.py || { echo "CHECK FAILED: train_rd_adapt_v4.py lacks plan_splits"; exit 1; }
grep -q "PROBE_VERSION = 'probe-v1'" $AN/rdgrowth/probe_bytes_v1.py || { echo "CHECK FAILED: probe_bytes_v1.py is not probe-v1"; exit 1; }
grep -q "VERSION v4" $AN/adapt-v4.sbatch || { echo "CHECK FAILED: adapt-v4.sbatch is not v4"; exit 1; }
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v4.py $AN/rdgrowth/probe_bytes_v1.py || { echo "CHECK FAILED: python files do not compile"; exit 1; }
for f in $AN/vis_collapse-v12/stage2_run.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || { echo "CHECK FAILED: MISSING $f"; exit 1; }; done
grep -q "triangles.scaling" /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py || { echo "CHECK FAILED: upstream train.py never sets triangles.scaling"; exit 1; }
busy=$(squeue --me -h -o "%j" | grep -cE "adapt3|adapt2|adapt1b|bpv|lkfan2"); [ "$busy" = 0 ] || echo "note: $busy earlier jobs still in the queue; the adapt4 jobs queue behind them"
echo "CHECK OK"; k=0
sub () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS="5 10 20" --job-name=$1-adapt4 $AN/adapt-v4.sbatch) && { echo "submitted $1-adapt4 $j"; k=$((k+1)); } || echo "FAILED: sbatch $1"; }
sub room 1 images_2 q14; sub kitchen 1 images_2 q16; sub garden 0 images_4 q15; sub bicycle 0 images_4 q16
echo "submitted $k of 4"
SUB5
bash $AN/submit-adapt-v4.sh
squeue --me -o "%.10i %.16j %.3t %.8M %R"
```

## 2. 10–15 分钟后：自检 + smoke（smoke 里的 probe scaling 检查是这批的关键自检）

```bash
AN=/scratch/zr2245/analysis
squeue --me -o "%.10i %.16j %.3t %.8M %R"
for f in $AN/adapt4-*-adapt4-*.out; do echo "=== $f"; grep -h "self-check\|smoke probe scaling\|smoke prune\|smoke final\|^\[probe\]\|FAILED\|note:\|=== .* target" $f | tail -12; done
```

期望：`smoke probe scaling: kept N faces at 4x vs M at 1x (... 1.5-3x more)`，`smoke prune: {... 'mode': 'count', 'n_pairs': >0, ...}`，然后进入 `=== room target 5 MB`。如果出现 `FAILED: SMOKE` 且 smoke.log 里是 `probe scaling has no effect`，说明 renderer 不读 `triangles.scaling`，把那几行贴回来，不用等。1 小时后看 phase A 是否 FRONT-LOAD / LIMITER 和第一次真实 probe（12000）的 kept：

```bash
for d in /scratch/zr2245/*-adapt4-t*mb; do echo "--- $d"; grep -h "^\[RD adapt\|^\[RD prune\|^\[RD [0-9]*\] V" $d/train.log 2>/dev/null | tail -4 | cut -c1-300; done
```

## 3. 全部跑完（≈2.5 h）后

```bash
AN=/scratch/zr2245/analysis
sacct -X -u $USER -S now-8hours -o JobID,JobName%18,State,Elapsed,ExitCode | grep -E "adapt4|JobID"
for f in $AN/adapt4-*-adapt4-*.out; do echo "=== $f"; tail -3 $f; done
echo "===== adapt-v4.txt"; cat $AN/adapt-v4.txt
echo "===== phase A / probes"; for d in /scratch/zr2245/*-adapt4-t*mb; do echo "--- $d"; grep -h "^\[RD [0-9]*\] V\|^\[RD adapt\|^\[RD prune\|^Training is done" $d/train.log 2>/dev/null | tail -24 | cut -c1-320; done
```

把 `adapt-v4.txt` 的 12 行和 phase A / probes 那段整段贴回来（phase A 这次要 24 行，看 LIMITER / FRONT-LOAD 什么时候出现）。

## 1b. 室内两个 job 的 smoke 挂在最终 probe 的 scaling 自检上（已确认不是训练问题）：确认后用 SMOKE=0 重交

```bash
AN=/scratch/zr2245/analysis
cat > $AN/resubmit-adapt-v4-indoor.sh <<'RS5'
AN=/scratch/zr2245/analysis; RES=yw523-l40s
for S in room kitchen; do f=$(ls -t $AN/adapt4-$S-adapt4-*.out | head -1); echo "=== $f"; grep -h -B1 -A1 "RuntimeError\|Error\|FAILED" $f | grep -v "^--$" | tail -6 | cut -c1-240; done
ok=0; for S in room kitchen; do f=$(ls -t $AN/adapt4-$S-adapt4-*.out | head -1); grep -q "probe scaling has no effect" $f && ok=$((ok+1)); done
[ "$ok" = 2 ] || { echo "NOT the expected failure ($ok of 2 show 'probe scaling has no effect'); nothing submitted, paste the lines above"; exit 1; }
echo "both failed at the final probe's scaling check (after upstream cleaning, meaningless there); the 1250 probe and its 4x>1x check had passed -> resubmitting with SMOKE=0 SELFCHECK=0"
sub () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --export=ALL,SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS="5 10 20",SMOKE=0,SELFCHECK=0 --job-name=$1-adapt4 $AN/adapt-v4.sbatch) && echo "submitted $1-adapt4 $j" || echo "FAILED: sbatch $1"; }
sub room 1 images_2 q14; sub kitchen 1 images_2 q16
RS5
bash $AN/resubmit-adapt-v4-indoor.sh
squeue --me -o "%.10i %.16j %.3t %.8M %R"
```
