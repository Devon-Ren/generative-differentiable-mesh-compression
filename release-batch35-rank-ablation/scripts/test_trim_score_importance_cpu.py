"""test_trim_score_importance_cpu.py (batch35: the same checks with an importance-like cost: values in [0, 1], 40 % of the faces exactly 1.0)
test_select_feasible_cpu.py: select_feasible on synthetic meshes against the original select_collapses + joint fold-over filter.
  python3 test_select_feasible_cpu.py <select_feasible_v1.py or patched training script> <dir with rd_prune.py>"""
import sys, re, torch
sys.path.insert(0, sys.argv[2])
import rd_prune
from rd_prune import select_collapses, edge_table, _face_normals
src = open(sys.argv[1]).read(); a = src.index('# >>> select_feasible'); b = src.index('# <<< select_feasible')
ns = {'torch': torch, 'edge_table': edge_table, '_face_normals': _face_normals}; exec(src[a:b], ns); select_feasible = ns['select_feasible']
def mesh(n, rough, seed):
    torch.manual_seed(seed); xs, ys = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    V = torch.stack([xs.flatten().float(), ys.flatten().float(), rough * torch.randn(n * n)], 1); V[:, :2] += 0.35 * rough * torch.randn(n * n, 2) + 0.2 * torch.rand(n * n, 2)
    idx = lambda i, j: i * n + j; i, j = torch.meshgrid(torch.arange(n - 1), torch.arange(n - 1), indexing='ij'); i, j = i.flatten(), j.flatten()
    F = torch.cat([torch.stack([idx(i, j), idx(i + 1, j), idx(i, j + 1)], 1), torch.stack([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)], 1)])
    cx = V[F].mean(1); kept_f = torch.rand(F.shape[0]) > 0.45
    err = ((torch.sin(cx[:, 0] / 17.0) * torch.cos(cx[:, 1] / 23.0) + 1.2) ** 4) * torch.exp(0.8 * torch.randn(F.shape[0])); err[~kept_f] = 0.0
    kept_v = torch.zeros(V.shape[0], dtype=torch.bool); kept_v[F[kept_f].reshape(-1)] = True
    return V, F, err, kept_v, kept_f
def joint_flips(V, F, pairs):
    p2 = V.clone(); u, v = pairs[:, 0], pairs[:, 1]; mid = 0.5 * (V[u] + V[v]); p2[u] = mid; p2[v] = mid; nid = torch.arange(V.shape[0]); nid[v] = u; f2 = nid[F]
    deg = (f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2]); return int((((_face_normals(V, F) * _face_normals(p2, F)).sum(1) < 0) & ~deg).sum())
def freed(F, pairs, kept_v, kept_f, b_v, b_f):
    V = kept_v.numel(); both = int((kept_v[pairs[:, 0]] & kept_v[pairs[:, 1]]).sum()); key = lambda a, b: torch.minimum(a, b) * V + torch.maximum(a, b)
    pk = set(key(pairs[:, 0], pairs[:, 1]).tolist()); n = 0
    for a, b in ((0, 1), (1, 2), (0, 2)): n += int((torch.tensor([k in pk for k in key(F[:, a], F[:, b]).tolist()]) & kept_f).sum())
    return both * b_v + n * b_f
ok = True; b_v, b_f = 8.3, 1.8
for rough, label in ((0.15, 'smooth'), (1.2, 'crumpled')):
    V, F, err, kept_v, kept_f = mesh(160, rough, 2); V0 = V.shape[0]; total = float(kept_v.sum()) * b_v + float(kept_f.sum()) * b_f
    torch.manual_seed(7); err = torch.round(torch.rand(F.shape[0]).clamp(0.05, 1.0) * 1000) / 1000; err[torch.rand(F.shape[0]) < 0.4] = 1.0; err = err.float()
    for share in (0.01, 0.04, 0.10):
        want = share * total; maxp = int(0.15 * V0)
        # original route: rank threshold for the pairs a 9 B/pair estimate asks for, joint filter once
        fin = None
        sc = rd_prune.edge_table(F); pairs0, ps0 = select_collapses(V, F, err, 1e30, 1.0, b_v + 2 * b_f, b_f, min(0.15, (want / 9.0) / V0))
        f0 = freed(F, pairs0, kept_v, kept_f, b_v, b_f)
        edges, sel, st = select_feasible(V, F, err, kept_v, kept_f, b_v, b_f, want, maxp); pairs = edges[sel]
        f1 = freed(F, pairs, kept_v, kept_f, b_v, b_f); disjoint = torch.unique(pairs.reshape(-1)).numel() == 2 * pairs.shape[0]; nflip = joint_flips(V, F, pairs)
        hit = f1 / want; capped = pairs.shape[0] >= maxp
        good = disjoint and nflip == 0 and (0.97 <= hit <= 1.06 or capped) and abs(st['planned_bytes'] - f1) < 1e-6 * max(f1, 1)
        ok &= bool(good)
        print('%-9s want %5.1f%% of the file | original: %5d pairs free %3.0f%% of the wanted bytes | feasible: %5d pairs free %5.1f%% (%d batches, %d banned, joint flips %d, disjoint %s%s) | %s' % (label, 100 * share, pairs0.shape[0], 100 * f0 / want, pairs.shape[0], 100 * hit, st['iters'], st['n_banned'], nflip, disjoint, ', pair cap reached' if capped else '', 'ok' if good else 'NOT OK'))
print('TEST', 'PASSED' if ok else 'FAILED')
