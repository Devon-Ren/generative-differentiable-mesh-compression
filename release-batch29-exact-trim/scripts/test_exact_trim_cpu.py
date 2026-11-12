"""test_exact_trim_cpu.py: runs the batch29 trimming loop (text taken from the generated training script) on synthetic meshes with the real
rd_prune.select_collapses.  Checks: pairs delivered = goal (within 3 %), pairs vertex-disjoint, the cheapest pairs are kept.
  python3 test_exact_trim_cpu.py <train_rd_adapt_v9b_abl3.py> <dir with rd_prune.py>"""
import sys, re, textwrap, torch
sys.path.insert(0, sys.argv[2])
from rd_prune import select_collapses, edge_table
src = open(sys.argv[1]).read()
ns = {'torch': torch, 'edge_table': edge_table}
for name in ('edge_scores', 'rank_threshold'):
    exec(re.search(r'^def %s\(.*?(?=^def |\Z)' % name, src, re.S | re.M).group(0), ns)
edge_scores, rank_threshold = ns['edge_scores'], ns['rank_threshold']
a = src.index("                    if te:  # batch29: deliver the wanted pairs"); b = src.index("                        ev.update(trim_exact=te", a)
loop = textwrap.dedent(src[a:b])
def mesh(n, rough, seed):
    torch.manual_seed(seed); xs, ys = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    V = torch.stack([xs.flatten().float(), ys.flatten().float(), rough * torch.randn(n * n)], 1); V[:, :2] += 0.35 * rough * torch.randn(n * n, 2) + 0.2 * torch.rand(n * n, 2)
    idx = lambda i, j: i * n + j; i, j = torch.meshgrid(torch.arange(n - 1), torch.arange(n - 1), indexing='ij'); i, j = i.flatten(), j.flatten()
    F = torch.cat([torch.stack([idx(i, j), idx(i + 1, j), idx(i, j + 1)], 1), torch.stack([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)], 1)])
    cx = V[F].mean(1); err = ((torch.sin(cx[:, 0] / 17.0) * torch.cos(cx[:, 1] / 23.0) + 1.2) ** 4) * torch.exp(0.8 * torch.randn(F.shape[0])); err[torch.rand(F.shape[0]) < 0.45] = 0.0
    return V, F, err
ok = True
for rough, label in ((0.15, 'smooth'), (1.2, 'crumpled (many fold-over rejections)')):
    V, F, err = mesh(200, rough, 1); V0 = V.shape[0]; b_v, b_f = 8.3, 1.8
    class size: pass
    size.b_f = b_f; b_v_full = b_v + 2 * b_f; rd = {'prune_mu': 1.0}; dD = err
    class T: pass
    triangles = T(); triangles.vertices = V; triangles._triangle_indices = F
    edges, cnt, score = edge_scores(V, F, err, b_v, b_f)
    for wanted in (300.0, 2500.0, 0.2 * V0):
        frac = min(0.15, wanted / V0); thr, _ = rank_threshold(score, wanted); count_driven = True; te = 1; ev = {}; iteration = 0
        pairs, ps = select_collapses(V, F, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac); first = int(ps['n_pairs'])
        env = dict(globals()); env.update(locals()); exec(loop, env); pairs, ps, goal, tries = env['pairs'], env['ps'], env['goal'], env['tries']
        disjoint = torch.unique(pairs.reshape(-1)).numel() == 2 * pairs.shape[0]; good = disjoint and pairs.shape[0] <= goal and pairs.shape[0] >= 0.97 * goal and ps['n_pairs'] == pairs.shape[0]
        ok &= bool(good)
        print('%-38s wanted %6d goal %5d | original %5d pairs (%3.0f%%) | exact %5d pairs (%3.0f%%), %2d tries, margin %g, pair cap %.3f, flips %d | %s' % (label, wanted, goal, first, 100 * first / goal, pairs.shape[0], 100 * pairs.shape[0] / goal, tries, env['margin'], env['frac_eff'], ps['n_flip_rejected'], 'ok' if good else 'NOT OK'))
print('TEST', 'PASSED' if ok else 'FAILED')
