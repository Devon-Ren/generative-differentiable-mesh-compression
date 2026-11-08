"""CPU test of v9d (needs torch, CPU is enough).  (1) rd_prune_v2.select_collapses WITHOUT masks == rd_prune.select_collapses (same pairs, same stats
keys + n_priced / priced_frac); (2) WITH masks on a grid mesh whose right half is 'unkept': no selected pair has dR = 0, every selected pair touches a
kept face, n_priced < E, unkept-only region never collapses; (3) edge_scores of v9d with masks returns +inf exactly on the unpriced edges and agrees
with rd_prune_v2's dR; (4) the trainer diff vs v9c below the header is the expected set of lines."""
import ast, difflib, importlib.util, os, sys
import torch
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(R)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
p1 = load('rd_prune', os.path.join(ROOT, 'release-adapt-v7/code/rdgrowth/rd_prune.py')); p2 = load('rd_prune_v2', os.path.join(R, 'code/rdgrowth/rd_prune_v2.py'))
assert p2.PRUNE_VERSION == 'rd_prune_v2'
# grid mesh n x n
n = 24; torch.manual_seed(0)
ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij'); pos = torch.stack([xs.flatten().double(), ys.flatten().double(), torch.zeros(n * n, dtype=torch.float64)], 1) + 0.05 * torch.randn(n * n, 3, dtype=torch.float64)
idx = lambda i, j: i * n + j
faces = []
for i in range(n - 1):
    for j in range(n - 1):
        faces.append([idx(i, j), idx(i, j + 1), idx(i + 1, j)]); faces.append([idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)])
faces = torch.tensor(faces, dtype=torch.int32); T = faces.shape[0]; V = n * n
err = torch.rand(T, dtype=torch.float64) * 1e-3
# unkept: faces whose centroid x > n/2 (right half); give them ZERO residual (nobody renders them), the trap v9b/v9c fell into
cx = pos[faces.long()][:, :, 0].mean(1); kept_f = cx < n / 2; err[~kept_f] = 0.0
kept_v = torch.zeros(V, dtype=torch.bool); kept_v[faces.long()[kept_f].reshape(-1)] = True
lam, mu, b_v_full, b_f, frac = 1e9, 1.0, 10.0, 2.0, 0.15  # huge lambda: every priced edge is a candidate -> the selection is count/geometry-limited
pairs1, st1 = p1.select_collapses(pos, faces, err, lam, mu, b_v_full, b_f, frac)
pairs2, st2 = p2.select_collapses(pos, faces, err, lam, mu, b_v_full, b_f, frac)
assert torch.equal(pairs1, pairs2) and st2['n_priced'] == st2['edges'] and st2['priced_frac'] == 1.0 and all(st1[k] == st2[k] for k in st1), 'v2 without masks must equal rd_prune'
print(f'(1) without masks: identical to rd_prune.select_collapses ({pairs1.shape[0]} pairs of {st1["edges"]} edges; cheapest-first picks the zero-residual right half: '
      f'{int((~kept_v[pairs1.long()].all(1)).sum())} of them have an unkept endpoint)')
pairs3, st3 = p2.select_collapses(pos, faces, err, lam, mu, b_v_full, b_f, frac, kept_v=kept_v, kept_f=kept_f)
edges, inv, cnt = p2.edge_table(faces.long()); E = edges.shape[0]
cnt_kept = torch.zeros(E, dtype=torch.float64); cnt_kept.index_add_(0, inv, kept_f.double().repeat(3))
both = (kept_v[edges[:, 0]] & kept_v[edges[:, 1]]).double(); dR = (b_v_full - 2 * b_f) * both + cnt_kept * b_f
key = edges[:, 0] * V + edges[:, 1]; pk = torch.minimum(pairs3[:, 0], pairs3[:, 1]) * V + torch.maximum(pairs3[:, 0], pairs3[:, 1])
sel = torch.searchsorted(key, pk); assert torch.equal(key[sel], pk)
assert st3['n_priced'] == int((dR > 0).sum()) < E and bool((dR[sel] > 0).all()) and bool((cnt_kept[sel] > 0).all()), 'a selected pair has no delivered bytes'
assert pairs3.shape[0] > 0 and bool(kept_v[pairs3.long()].any(1).all())
print(f'(2) with masks: {st3["n_priced"]} of {E} edges priced ({st3["priced_frac"]:.2f}); {pairs3.shape[0]} pairs selected, all with dR > 0 and >= 1 kept face; unkept-only region untouched')
# (3) trainer's edge_scores with masks
src = open(os.path.join(R, 'code/rdgrowth/train_rd_adapt_v9d.py')).read(); ns = {'torch': torch, 'edge_table': p2.edge_table}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name == 'edge_scores': exec(ast.get_source_segment(src, node), ns)
e2, c2, sc = ns['edge_scores'](pos, faces, err, b_v_full - 2 * b_f, b_f, kept_v, kept_f)
assert torch.equal(e2, edges) and bool(torch.isinf(sc[dR == 0]).all()) and bool(torch.isfinite(sc[dR > 0]).all())
e1, c1, sc0 = ns['edge_scores'](pos, faces, err, b_v_full - 2 * b_f, b_f)
assert bool(torch.isfinite(sc0).all())
print(f'(3) edge_scores: +inf on the {int((dR == 0).sum())} unpriced edges, finite elsewhere; without masks finite everywhere (v9c behaviour)')
# (4) diff audit
v9c = open(os.path.join(ROOT, 'release-adapt-v9c/code/rdgrowth/train_rd_adapt_v9c.py')).read()
d = [l for l in difflib.unified_diff(v9c[v9c.index('import '):].splitlines(), src[src.index('import '):].splitlines(), lineterm='', n=0) if l[:1] == '-' and l[:3] != '---']
assert sum('rd_prune import' in l for l in d) == 1 and sum('def edge_scores' in l for l in d) == 1 and sum('b_coll = size.b_v *' in l for l in d) == 1 and sum('select_collapses(triangles' in l for l in d) == 1 and sum("adapt-v9c" in l for l in d) == 4, d
print(f'(4) trainer diff v9c -> v9d below the header: {len(d)} removed lines (import, edge_scores body, b_coll prior, edge_scores call, select_collapses call, prune print, 4 version strings)')
print('ALL OK')
