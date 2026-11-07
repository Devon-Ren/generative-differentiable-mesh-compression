# planarity-v1b: 探针 P1b —— 世界在那里是平的吗（顶点级，纯 CPU），2026-10-30

**P1 的结论**（11 场景 λ6.2e-3，commands/planarity-v1.md）：放宽 ε（0.1 → 0.5 %）几乎不改变结果，θ 5° → 10° 才让平面占比翻倍；二面角 < 10° 的相邻面对室内只有 42–52 %、室外 26–29 %，< 30° 室内 70–76 %、室外 54–58 %。即训练出的 mesh 在三角形尺度上是**大角度的瓦片**（相邻三角形互相倾斜 10–30° 以上），不是小幅度的波纹；按面找平面只能找到碎条（内部顶点 ≤ 12 %，轮廓比 1.2–1.5x）。
**P1b 问下一层的问题**：不看连接关系，把顶点当点云。每个顶点用 k = 32 近邻做 PCA 得到**平滑过的法向**和局部 rms（几个三角形尺度的邻域把瓦片平均掉）；kNN 相邻两点若都局部平、法向夹角 ≤ θ、且互相落在对方局部平面 ε 内就连边；连通分量是候选平面；候选用 PCA 平面 + 95 分位距离 ≤ ε 验证，不过的（弯曲的、或在墙角处被串起来的）按法向二分再重新连通，递归到过为止（selftest：带瓦片噪声的盒子六面各自成一面、球被拒或拆成小片、体噪声不成面）。
**报告**：局部平的顶点占比；验证平面（≥ 100 / 1000 / 10000 点）里的顶点占比 —— 这些顶点换成"平面 + 纹理"后只剩轮廓需要保留；前 10 大平面（点数、法向、延展）；k32 邻域半径和局部 rms 中位数（和 ε 同单位），用来判断 ε 是否大于瓦片幅度。ε 取 0.25 / 0.5 / 1.0 % 对角线，θ 10° / 20°；汇总行取 ε 0.5 % / θ 10°。每个 checkpoint 另存一份每顶点平面标签 `<scene>-<tag>-plane-labels.npy`（ε 0.5 % / θ 10°），给后面的 P2 用。
**判据**：ε 0.5 % / θ 10° 下，≥ 1000 点平面里的顶点占比 室内 ≥ 40 %（kitchen/counter/room）→ "世界是平的、mesh 不平"成立，+++ 的轴 4.1 + 4.2 值 ≥ 2x 顶点；若室内也 < 20 %，则场景本身就不平（家具、杂物），平面论据只剩墙和地板，+++ 的重点要落到 cards 和流形光滑。
**同时修正一个 bug**：v1 的 runner 把每个 tag 的 json 都写成 `<scene>-.json`，所以 λ1.85e-3 和 dense 被当作"已存在"跳过了（skipped 22）；v1b 先把那 11 个文件改名，再补跑缺的面级统计和全部顶点级统计。Room 的 dense checkpoint 不在 room-vis-collapse 下，脚本依次找 room-fixed-sharing-0N4OSU/run 和 room-rd-error-lam0。
**成本**：λ 跑每个 checkpoint 约 1.5–3 min（6 组 ε/θ），dense（3 M 顶点）约 6–10 min、5–6 GB；33 个合计约 1.5 h CPU，可中断重跑。

## §1 安装 + selftest（任意 shell，整块粘贴；不会退出你的终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/diag $AN/planarity-v1
cat > $AN/diag/planarity_points_v1.py <<'PYEOF_POINTS'
"""planarity_points_v1.py (2026-10-30): CPU probe P1b of plan/meshsplat-plus-2026-10-29.md, vertex level.

P1 (planarity_v1.py, face level) found that the trained mesh is crumpled at the triangle scale (about half of all
adjacent face pairs tilt by more than 10 degrees even indoors), so planar regions of FACES are shredded.  P1b asks the
question one level down, ignoring connectivity: is the WORLD flat where the mesh is crumpled?
  1. vertices as a point cloud; for every vertex a local plane by PCA over its k nearest neighbours (k = 32 by default,
     a neighbourhood of several triangles, which averages the shingles); local_rms = sqrt(smallest eigenvalue)
  2. two kNN-neighbouring vertices are "locally coplanar" iff both are locally flat (local_rms <= eps), their smoothed
     normals agree within theta, and each lies within eps of the other's local plane
  3. connected components of that graph = candidate planes; a component with >= min_points points is VERIFIED planar
     iff the 95th percentile of its points' distances to its PCA plane is <= eps
  4. statistics per (eps, theta): share of vertices in verified planes with >= 100 / 1000 / 10000 points (every such
     vertex is replaceable by the plane + a texture, up to the plane's outline), share of locally flat vertices, number
     of planes, top planes (points, normal, extent).  eps in % of the robust scene diagonal, as in planarity_v1.
Usage: python planarity_points_v1.py --state <point_cloud_state_dict.pt> --label "room v3 lam0.0062" --out x.json [--save_labels x.npy]
       python planarity_points_v1.py --selftest
"""
import argparse
import json
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

VERSION = 'planarity_points_v1 2026-10-30'
EPS_PCTS = (0.25, 0.5, 1.0)
THETAS = (10.0, 20.0)
K_NORMAL = 32
K_LINK = 12
MIN_POINTS = 100


def load_mesh(path):
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    V = sd.get('triangles_points', sd.get('vertices'))
    if V is None:
        raise SystemExit(f'cannot find vertices in {path}; keys = {keys}')
    V = np.asarray(V.detach().cpu().numpy() if hasattr(V, 'detach') else V, dtype=np.float64)
    return V, keys


def robust_scale(V):
    lo = np.percentile(V, 2, axis=0)
    hi = np.percentile(V, 98, axis=0)
    return float(np.linalg.norm(hi - lo))


def local_planes(V, k=K_NORMAL, chunk=200000):
    """Per-vertex PCA over the k nearest neighbours (self included).  Returns normals (N,3), local_rms (N,), neighbour
    index array (N, k) and the kNN distances (N, k)."""
    tree = cKDTree(V)
    N = len(V)
    normals = np.zeros((N, 3))
    rms = np.zeros(N)
    nbr = np.zeros((N, k), dtype=np.int64)
    dist = np.zeros((N, k), dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        d, idx = tree.query(V[s:e], k=k + 1)
        idx = idx[:, 1:]                       # drop self
        nbr[s:e] = idx
        dist[s:e] = d[:, 1:]
        X = np.concatenate([V[s:e, None, :], V[idx]], axis=1)          # (n, k+1, 3)
        mu = X.mean(1)
        Xc = X - mu[:, None, :]
        cov = np.einsum('nki,nkj->nij', Xc, Xc) / (k + 1)
        evals, evecs = np.linalg.eigh(cov)
        normals[s:e] = evecs[:, :, 0]
        rms[s:e] = np.sqrt(np.clip(evals[:, 0], 0, None))
    return normals, rms, nbr, dist


def group_percentile(values, labels, n_groups, q=0.95):
    order = np.lexsort((values, labels))
    lab_s, val_s = labels[order], values[order]
    cnt = np.bincount(lab_s, minlength=n_groups)
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = first + np.clip(np.floor(q * (cnt - 1)).astype(np.int64), 0, None)
    out = np.full(n_groups, np.inf)
    has = cnt > 0
    out[has] = val_s[pos[has]]
    return out


MAX_DEPTH = 8


def _components(n, i, j):
    g = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)


def _plane_fit(P):
    """PCA plane of a point set: (centre, unit normal, p95 distance, extent ~ 4 sigma along the longest in-plane axis)."""
    mu = P.mean(0)
    D = P - mu
    w, U = np.linalg.eigh(D.T @ D / len(P))
    n = U[:, 0]
    dist = np.abs(D @ n)
    return mu, n, float(np.percentile(dist, 95)), float(4 * np.sqrt(max(w[2], 0.0)))


_S2 = np.sqrt(2.0)


def _embed(nrm):
    """Sign-free embedding of unit normals: ||e(n) - e(m)||^2 = 2 (1 - (n.m)^2)."""
    return np.stack([nrm[:, 0] ** 2, nrm[:, 1] ** 2, nrm[:, 2] ** 2, _S2 * nrm[:, 0] * nrm[:, 1], _S2 * nrm[:, 0] * nrm[:, 2], _S2 * nrm[:, 1] * nrm[:, 2]], 1)


def _bisect(E, iters=12):
    """2-means on the embedded normals, seeded by the two most distant points."""
    m = E.mean(0)
    a = E[np.argmax(((E - m) ** 2).sum(1))]
    b = E[np.argmax(((E - a) ** 2).sum(1))]
    lab = np.zeros(len(E), bool)
    for _ in range(iters):
        new = ((E - b) ** 2).sum(1) < ((E - a) ** 2).sum(1)
        if new.all() or (~new).all():
            break
        if np.array_equal(new, lab):
            break
        lab = new
        a, b = E[~lab].mean(0), E[lab].mean(0)
    return lab


def _refine(gidx, li, lj, V, normals, eps, min_points, depth, out):
    """gidx: global ids of a connected candidate region; li/lj: its pairs in local indices.  Verified planes go to out;
    a rejected (curved or corner-chained) region is bisected by normal direction and re-connected, recursively."""
    n = len(gidx)
    if n < min_points:
        return
    mu, nrm, p95, ext = _plane_fit(V[gidx])
    if p95 <= eps:
        out.append((gidx, nrm, p95, ext))
        return
    if depth >= MAX_DEPTH:
        return
    side = _bisect(_embed(normals[gidx]))
    for s_val in (False, True):
        m = side == s_val
        if m.sum() < min_points:
            continue
        sub = np.nonzero(m)[0]
        remap = np.full(n, -1, dtype=np.int64)
        remap[sub] = np.arange(len(sub))
        keep = m[li] & m[lj]
        si, sj = remap[li[keep]], remap[lj[keep]]
        ncomp, cl = _components(len(sub), si, sj)
        counts = np.bincount(cl, minlength=ncomp)
        pair_c = cl[si]                                   # component of each kept pair (both ends share it)
        order = np.argsort(pair_c, kind='stable')
        bounds = np.searchsorted(pair_c[order], np.arange(ncomp + 1))
        for c in np.nonzero(counts >= min_points)[0]:
            members = np.nonzero(cl == c)[0]
            remap2 = np.full(len(sub), -1, dtype=np.int64)
            remap2[members] = np.arange(len(members))
            sl = order[bounds[c]:bounds[c + 1]]
            _refine(gidx[sub[members]], remap2[si[sl]], remap2[sj[sl]], V, normals, eps, min_points, depth + 1, out)


def analyse(V, normals, rms, nbr, eps, cos_theta, k_link=K_LINK, min_points=MIN_POINTS):
    N = len(V)
    flat = rms <= eps
    i = np.repeat(np.arange(N), k_link)
    j = nbr[:, :k_link].reshape(-1)
    dot = np.abs((normals[i] * normals[j]).sum(1))
    dij = np.abs(((V[j] - V[i]) * normals[i]).sum(1))
    dji = np.abs(((V[i] - V[j]) * normals[j]).sum(1))
    good = flat[i] & flat[j] & (dot >= cos_theta) & (dij <= eps) & (dji <= eps)
    i, j = i[good], j[good]
    n_comp, lab = _components(N, i, j)
    cnt = np.bincount(lab, minlength=n_comp)
    big = np.nonzero(cnt >= min_points)[0]
    pair_c = lab[i]
    order = np.argsort(pair_c, kind='stable')
    bounds = np.searchsorted(pair_c[order], np.arange(n_comp + 1))
    out = []
    n_split = 0
    for c in big:
        gidx = np.nonzero(lab == c)[0]
        remap = np.full(N, -1, dtype=np.int64)        # cheap enough: big comps are few hundred to few thousand
        remap[gidx] = np.arange(len(gidx))
        sl = order[bounds[c]:bounds[c + 1]]
        before = len(out)
        _refine(gidx, remap[i[sl]], remap[j[sl]], V, normals, eps, min_points, 0, out)
        if len(out) - before != 1 or len(out[-1][0]) != len(gidx):
            n_split += 1
    labels = np.full(N, -1, dtype=np.int64)
    sizes = np.array([len(o[0]) for o in out], dtype=np.int64)
    for k, (gidx, nrm, p95, ext) in enumerate(out):
        labels[gidx] = k
    in_plane = labels >= 0
    st = dict(eps=float(eps), cos_theta=float(cos_theta), locally_flat_share=float(flat.mean()), candidate_regions=int(len(big)),
              candidate_vertex_share=float(cnt[big].sum() / N), regions_split_or_rejected=int(n_split), verified_planes=int(len(out)),
              verified_vertex_share=float(in_plane.mean()))
    for kk in (100, 1000, 10000):
        m = sizes >= kk
        st[f'ge{kk}'] = dict(planes=int(m.sum()), vertex_share=float(sizes[m].sum() / N))
    order_p = np.argsort(-sizes)[:10]
    st['top_planes'] = [dict(points=int(sizes[k]), vertex_share=float(sizes[k] / N), normal=[round(float(x), 3) for x in out[k][1]], extent=float(out[k][3]), p95_over_eps=float(out[k][2] / eps))
                        for k in order_p]
    return st, labels


def run(V, label, eps_pcts=EPS_PCTS, thetas=THETAS, k_normal=K_NORMAL, k_link=K_LINK, min_points=MIN_POINTS, F=None):
    t0 = time.time()
    scale = robust_scale(V)
    normals, rms, nbr, dist = local_planes(V, k_normal)
    facts = dict(label=label, version=VERSION, vertices=int(len(V)), scene_diagonal=scale, k_normal=k_normal, k_link=k_link, min_points=min_points,
                 knn_radius_median_pct=float(100 * np.median(dist[:, -1]) / scale), local_rms_median_pct=float(100 * np.median(rms) / scale),
                 local_rms_p90_pct=float(100 * np.percentile(rms, 90) / scale))
    print(f"  k{k_normal} neighbourhood radius median {facts['knn_radius_median_pct']:.3f} % diag | local rms median {facts['local_rms_median_pct']:.3f} % p90 {facts['local_rms_p90_pct']:.3f} %", flush=True)
    rows, best_labels = [], None
    for ep in eps_pcts:
        for th in thetas:
            eps = ep / 100.0 * scale
            st, labels = analyse(V, normals, rms, nbr, eps, np.cos(np.radians(th)), k_link, min_points)
            st.update(eps_pct=ep, theta_deg=th)
            rows.append(st)
            if ep == 0.5 and th == 10.0:
                best_labels = labels
            top3 = ', '.join('%.1f%%' % (100 * t['vertex_share']) for t in st['top_planes'][:3])
            print(f"  eps {ep:.2f} % theta {th:>4.0f} | locally flat {100*st['locally_flat_share']:.1f} % | candidates {st['candidate_regions']} ({100*st['candidate_vertex_share']:.1f} % of vertices), split/rejected {st['regions_split_or_rejected']} "
                  f"| verified planes {st['verified_planes']} = {100*st['verified_vertex_share']:.1f} % of vertices; >=1000 pts {100*st['ge1000']['vertex_share']:.1f} % >=10000 {100*st['ge10000']['vertex_share']:.1f} % | top-3 {top3}", flush=True)
    facts['elapsed_s'] = round(time.time() - t0, 1)
    return dict(facts=facts, rows=rows), best_labels


def summary_line(res):
    f = res['facts']
    r = next(x for x in res['rows'] if x['eps_pct'] == 0.5 and x['theta_deg'] == 10.0)
    top = ', '.join('%.1f%%' % (100 * t['vertex_share']) for t in r['top_planes'][:5])
    return (f"PLANES {f['label']} | V {f['vertices']} | k{f['k_normal']} radius {f['knn_radius_median_pct']:.3f} % diag, local rms median {f['local_rms_median_pct']:.3f} % "
            f"| eps 0.5 % theta 10: locally flat {100*r['locally_flat_share']:.1f} % | planes >=100 pts {r['ge100']['planes']} = {100*r['ge100']['vertex_share']:.1f} % of vertices, >=1000 {100*r['ge1000']['vertex_share']:.1f} %, >=10000 {100*r['ge10000']['vertex_share']:.1f} % "
            f"| top-5 {top} | {f['elapsed_s']} s")


def selftest():
    rng = np.random.default_rng(0)
    s = 10.0
    pts = []
    # 6 box faces, 20k points each, SHINGLE noise: offsets of 0.3 % of the size (bigger than the P1 face tolerance)
    for axis in range(3):
        for side in (0.0, s):
            P = rng.random((20000, 3)) * s
            P[:, axis] = side + rng.normal(0, 0.003 * s, 20000)
            pts.append(P)
    n_box = 6 * 20000
    u = rng.random(20000) * 2 * np.pi; v = np.arccos(2 * rng.random(20000) - 1)
    sphere = np.array([5., 5., 5.]) + 3.0 * np.stack([np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)], 1)
    pts.append(sphere)
    pts.append(rng.random((20000, 3)) * s)                                   # volume noise: not flat anywhere
    V = np.concatenate(pts)
    res, labels = run(V, 'selftest', eps_pcts=(0.25, 0.5, 1.0), thetas=(10.0, 20.0))
    r = next(x for x in res['rows'] if x['eps_pct'] == 0.5 and x['theta_deg'] == 10.0)
    assert r['ge10000']['planes'] == 6, f"expected the 6 box faces, got {r['ge10000']['planes']}: {r['top_planes']}"
    assert r['ge10000']['vertex_share'] >= 0.85 * n_box / len(V), r['ge10000']
    assert all(t['p95_over_eps'] <= 1.0 for t in r['top_planes'])
    assert all(t['points'] < 30000 for t in r['top_planes']), 'box faces must not merge through the corners'
    r2 = next(x for x in res['rows'] if x['eps_pct'] == 0.25 and x['theta_deg'] == 10.0)
    assert r2['ge10000']['planes'] == 0, f"eps 0.25 % is below the shingle amplitude (0.3 % of the box, p95 ~ 0.35 % diag): {r2['ge10000']}"
    print('eps 0.25 (below the shingle amplitude) planes >= 10000:', r2['ge10000'], '| eps 0.5:', r['ge10000'], '| sphere pieces >= 1000 at eps 0.5:', [t['points'] for t in r['top_planes'] if abs(t['normal'][0]) < 0.9 and abs(t['normal'][1]) < 0.9 and abs(t['normal'][2]) < 0.9][:3])
    print('selftest PASS', VERSION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state'); ap.add_argument('--label', default=''); ap.add_argument('--out', default=None); ap.add_argument('--save_labels', default=None)
    ap.add_argument('--eps', type=float, nargs='*', default=list(EPS_PCTS)); ap.add_argument('--theta', type=float, nargs='*', default=list(THETAS))
    ap.add_argument('--k_normal', type=int, default=K_NORMAL); ap.add_argument('--k_link', type=int, default=K_LINK); ap.add_argument('--min_points', type=int, default=MIN_POINTS)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.state:
        ap.error('--state or --selftest')
    V, keys = load_mesh(a.state)
    label = a.label or a.state
    print(f'[{VERSION}] {label}: V {len(V)} | keys {keys[:4]}', flush=True)
    res, labels = run(V, label, tuple(a.eps), tuple(a.theta), a.k_normal, a.k_link, a.min_points)
    res['state'] = a.state
    print(summary_line(res), flush=True)
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1)
    if a.save_labels and labels is not None:
        np.save(a.save_labels, labels.astype(np.int32))


if __name__ == '__main__':
    main()
PYEOF_POINTS
python3 -c "import ast; ast.parse(open('$AN/diag/planarity_points_v1.py').read()); print('compiled OK')" || echo "FAILED: compile"
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
python $AN/diag/planarity_points_v1.py --selftest 2>&1 | tail -3 || echo "FAILED: selftest"
```

## §2 跑全部 checkpoint（整块粘贴；写脚本后用 bash 运行，脚本内部逐个容错）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/run_planarity_v1b.sh <<'SHEOF_PLANRUN1B'
#!/usr/bin/env bash
# run_planarity_v1b.sh (2026-10-30): P1 (face level, planarity_v1.py) where still missing + P1b (vertex level,
# planarity_points_v1.py) on every checkpoint: v3 lambda 6.2e-3, v3 lambda 1.85e-3, dense original.
# Fixes the v1 runner bug (one `local` line expanded $T before assigning it, so every tag wrote <scene>-.json and the
# later tags were skipped as "exists"): the 11 files <scene>-.json / .log are renamed to <scene>-v3-lam0.0062.*.
# Restartable.  Face rows -> $AN/planarity-v1.txt (PLANARITY ...), point rows -> $AN/planarity-v1b.txt (PLANES ...).
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
AN=/scratch/zr2245/analysis; OUT=$AN/planarity-v1; TABF=$AN/planarity-v1.txt; TABP=$AN/planarity-v1b.txt
PYF=$AN/diag/planarity_v1.py; PYP=$AN/diag/planarity_points_v1.py; mkdir -p $OUT
SCENES=${SCENES:-"room kitchen counter bonsai garden stump bicycle flowers treehill truck train"}
for S in $SCENES; do
  for ext in json log; do [ -f $OUT/$S-.$ext ] && [ ! -f $OUT/$S-v3-lam0.0062.$ext ] && mv $OUT/$S-.$ext $OUT/$S-v3-lam0.0062.$ext && echo "renamed $S-.$ext -> $S-v3-lam0.0062.$ext"; done
done
n_ok=0; n_fail=0; n_skip=0
run_one () {
  local S; local T; local CK; S=$1; T=$2; CK=$3
  local JF=$OUT/$S-$T.json; local LF=$OUT/$S-$T.log; local JP=$OUT/$S-$T-points.json; local LP=$OUT/$S-$T-points.log
  if [ ! -f "$CK" ]; then echo "missing: $S $T ($CK)"; n_skip=$((n_skip+1)); return; fi
  if [ -s $JF ]; then echo "exists (faces): $S $T"; else
    echo "=== $S $T faces ($(date +%H:%M:%S)) ==="
    if python $PYF --state "$CK" --label "$S $T" --out $JF > $LF 2>&1; then grep -h "^PLANARITY" $LF | tee -a $TABF; else echo "FAILED: $S $T faces"; tail -5 $LF; n_fail=$((n_fail+1)); fi
  fi
  if [ -s $JP ]; then echo "exists (points): $S $T"; n_skip=$((n_skip+1)); return; fi
  echo "=== $S $T points ($(date +%H:%M:%S)) ==="
  if python $PYP --state "$CK" --label "$S $T" --out $JP --save_labels $OUT/$S-$T-plane-labels.npy > $LP 2>&1; then
    grep -h "^PLANES" $LP | tee -a $TABP; n_ok=$((n_ok+1))
  else echo "FAILED: $S $T points"; tail -5 $LP; n_fail=$((n_fail+1)); fi
}
dense_ckpt () {  # first existing dense checkpoint of a scene
  local S=$1; local c
  for c in /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt \
           /scratch/zr2245/$S-fixed-sharing-0N4OSU/run/original/point_cloud/iteration_30000/point_cloud_state_dict.pt \
           /scratch/zr2245/$S-rd-error-lam0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; do
    [ -f "$c" ] && { echo "$c"; return; }
  done
  echo "/scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt"
}
for S in $SCENES; do
  run_one $S v3-lam0.0062  /scratch/zr2245/$S-rd3-lam0.0062-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  run_one $S v3-lam0.00185 /scratch/zr2245/$S-rd3-lam0.00185-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
done
for S in $SCENES; do run_one $S dense "$(dense_ckpt $S)"; done
echo "===== $TABP ====="; cat $TABP
echo "run_planarity_v1b done: ok $n_ok failed $n_fail skipped $n_skip  $(date)"
SHEOF_PLANRUN1B
bash $AN/run_planarity_v1b.sh 2>&1 | tee $AN/planarity-v1b-run.log
```

只跑部分场景：`SCENES="room kitchen garden" bash $AN/run_planarity_v1b.sh`。

## §3 看结果

```bash
AN=/scratch/zr2245/analysis; cat $AN/planarity-v1b.txt; echo; grep -h "PLANARITY" $AN/planarity-v1.txt | grep -v "lam0.0062"; tail -2 $AN/planarity-v1b-run.log
for S in room kitchen garden; do echo "== $S"; grep -h "radius\|eps" $AN/planarity-v1/$S-v3-lam0.0062-points.log; done
```
把 `planarity-v1b.txt` 整个、`planarity-v1.txt` 里新增的 λ1.85e-3 / dense 行、以及 room / kitchen / garden 三段 eps 明细贴回来。
