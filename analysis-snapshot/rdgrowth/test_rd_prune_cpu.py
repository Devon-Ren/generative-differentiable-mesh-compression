import sys, math, torch
sys.path.insert(0, '/mnt/project-files/analysis/rdgrowth')
from torch import nn
from rd_prune import select_collapses, collapse_edges, edge_table
torch.manual_seed(0)

def inverse_sigmoid(x): return torch.log(x / (1 - x))

class Mock:
    """upstream members used by rd_prune, with _prune_vertex_optimizer/_prune_vertices copied verbatim from triangle_model.py"""
    def __init__(s, pos, faces):
        s.eps = 1e-6; s.opacity_floor = 0.0
        s.opacity_activation = lambda x: s.opacity_floor + (1.0 - s.opacity_floor) * torch.sigmoid(x)
        s.inverse_opacity_activation = lambda y: inverse_sigmoid(((y.clamp(s.opacity_floor + s.eps, 1.0 - s.eps) - s.opacity_floor) / (1.0 - s.opacity_floor + s.eps)))
        V = pos.shape[0]
        s.vertices = nn.Parameter(pos.clone()); s.vertex_weight = nn.Parameter(torch.randn(V, 1))
        s._features_dc = nn.Parameter(torch.randn(V, 1, 3)); s._features_rest = nn.Parameter(torch.randn(V, 15, 3))
        s._triangle_indices = faces.to(torch.int32)
        T = faces.shape[0]
        s.image_size = torch.rand(T); s.importance_score = torch.rand(T); s.pixel_count = torch.randint(0, 9, (T,))
        s.optimizer = torch.optim.Adam([{'params': [s._features_dc], 'lr': 1e-3, 'name': 'f_dc'}, {'params': [s._features_rest], 'lr': 1e-4, 'name': 'f_rest'},
                                        {'params': [s.vertices], 'lr': 1e-3, 'name': 'vertices'}, {'params': [s.vertex_weight], 'lr': 1e-2, 'name': 'vertex_weight'}], lr=0.0, eps=1e-15)
    def step(s):
        loss = (s.vertices[s._triangle_indices.long()] ** 2).sum() + s.vertex_weight.sum() + s._features_dc.sum() + s._features_rest.sum()
        loss.backward(); s.optimizer.step(); s.optimizer.zero_grad(set_to_none=True)
    def _prune_vertex_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["vertices", "vertex_weight", "f_dc", "f_rest"]:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    del self.optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group['params'][0]
        for name, tensor in optimizable_tensors.items():
            if name == "vertices": self.vertices = tensor
            elif name == "vertex_weight": self.vertex_weight = tensor
            elif name == "f_dc": self._features_dc = tensor
            elif name == "f_rest": self._features_rest = tensor
    def _prune_vertices(self, vertex_mask):
        device = vertex_mask.device; oldV = vertex_mask.numel()
        new_id = torch.full((oldV,), -1, dtype=torch.long, device=device)
        kept = torch.nonzero(vertex_mask, as_tuple=True)[0]
        new_id[kept] = torch.arange(kept.numel(), device=device, dtype=torch.long)
        if self._triangle_indices.numel() > 0:
            remapped = new_id[self._triangle_indices.long()]
            valid_tris = (remapped >= 0).all(dim=1)
            remapped = remapped[valid_tris]
            self._triangle_indices = remapped.to(torch.int32).contiguous()
            if isinstance(self.image_size, torch.Tensor) and self.image_size.numel() > 0: self.image_size = self.image_size[valid_tris]
            if isinstance(self.importance_score, torch.Tensor) and self.importance_score.numel() > 0: self.importance_score = self.importance_score[valid_tris]
            if isinstance(self.pixel_count, torch.Tensor) and self.pixel_count.numel() > 0: self.pixel_count = self.pixel_count[valid_tris]
        self._prune_vertex_optimizer(vertex_mask)
        current_vertex_count = self.vertices.shape[0]
        if current_vertex_count > 0:
            if self._triangle_indices.numel() > 0:
                referenced_vertices = torch.unique(self._triangle_indices)
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device); mask_referenced[referenced_vertices] = True
            else:
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)
            if not mask_referenced.all():
                self._prune_vertex_optimizer(mask_referenced)
                if self._triangle_indices.numel() > 0:
                    new_id2 = torch.full((current_vertex_count,), -1, dtype=torch.long, device=device)
                    kept2 = torch.nonzero(mask_referenced, as_tuple=True)[0]
                    new_id2[kept2] = torch.arange(kept2.numel(), device=device, dtype=torch.long)
                    self._triangle_indices = new_id2[self._triangle_indices.long()].to(torch.int32).contiguous()

def grid(n):
    ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    pos = torch.stack([xs.flatten().float(), ys.flatten().float(), torch.zeros(n * n)], 1)
    pos[:, 2] = 0.1 * torch.rand(n * n)
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j; b = a + 1; c = a + n; d = c + 1
            f.append([a, b, c]); f.append([b, d, c])
    return pos, torch.tensor(f)

# ---- 1. selection on a grid with a zero-error region ----
n = 40; pos, faces = grid(n); V, T = pos.shape[0], faces.shape[0]
err = torch.rand(T) * 1e-2
cx = pos[faces].mean(1)
err[(cx[:, 0] < 10) & (cx[:, 1] < 10)] = 0.0           # bottom-left 10x10 block: nothing rendered there
m = Mock(pos, faces); [m.step() for _ in range(3)]
pairs, st = select_collapses(m.vertices, m._triangle_indices, err, lam=1e-3, mu=1.0, bytes_per_vertex=10.0, bytes_per_face=2.0, max_frac=0.1)
print('grid select:', {k: st[k] for k in ('vertices', 'faces', 'edges', 'n_cand', 'n_pass_pairs', 'n_pairs', 'max_pairs', 'capped', 'n_flip_rejected', 'passes', 'dR_mean_sel')})
print('  shares sel/cand/all', st['faces_removed_share_sel'], st['faces_removed_share_cand'], st['faces_removed_share_all'])
assert pairs.shape[0] > 0 and pairs.shape[0] <= st['max_pairs']
assert torch.unique(pairs.reshape(-1)).numel() == 2 * pairs.shape[0], 'pairs not vertex-disjoint'
# all selected pairs come from the zero-error block (score 0) because the block alone has more edges than the cap? check that scores are <= threshold
edges, inv, cnt = edge_table(m._triangle_indices)
sel_in_block = ((pos[pairs[:, 0], 0] < 10) & (pos[pairs[:, 0], 1] < 10)).float().mean()
print('  fraction of selected pairs inside the zero-error block', float(sel_in_block))
assert sel_in_block > 0.95
# ---- 2. collapse: counts, references, averages, Adam ----
u, v = pairs[:, 0], pairs[:, 1]
exp_pos = 0.5 * (m.vertices.data[u] + m.vertices.data[v]).clone()
exp_dc = 0.5 * (m._features_dc.data[u] + m._features_dc.data[v]).clone()
st_v = m.optimizer.state[m.vertices]; exp_m = 0.5 * (st_v['exp_avg'][u] + st_v['exp_avg'][v]).clone()
cnt_sel = cnt[torch.tensor([int(((edges == p).all(1)).nonzero()[0]) for p in pairs])]
exp_faces_removed = int(cnt_sel.sum())
cs = collapse_edges(m, pairs)
print('grid collapse:', cs)
assert cs['vertices_after'] == V - pairs.shape[0], cs
assert cs['faces_degenerate'] == exp_faces_removed, (cs, exp_faces_removed)
assert cs['faces_after'] == T - cs['faces_degenerate'] - cs['faces_duplicate']
f2 = m._triangle_indices.long(); assert f2.max() < m.vertices.shape[0] and f2.min() >= 0
assert not ((f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2])).any()
assert m.image_size.numel() == f2.shape[0] and m.pixel_count.numel() == f2.shape[0]
for name in ('vertices', 'vertex_weight', '_features_dc', '_features_rest'):
    p = getattr(m, name); stt = m.optimizer.state[p]
    assert stt['exp_avg'].shape == p.shape and stt['exp_avg_sq'].shape == p.shape, name
# merged vertex now sits at the compacted index of u
keep = torch.ones(V, dtype=torch.bool); keep[v] = False; comp = torch.cumsum(keep.long(), 0) - 1
assert torch.allclose(m.vertices.data[comp[u]], exp_pos) and torch.allclose(m._features_dc.data[comp[u]], exp_dc)
assert torch.allclose(m.optimizer.state[m.vertices]['exp_avg'][comp[u]], exp_m)
m.step(); print('  optimizer step after collapse OK; V', m.vertices.shape[0], 'T', m._triangle_indices.shape[0])
# ---- 3. all-zero error, no cap: ties resolve, matching is large ----
m3 = Mock(pos, faces)
pairs3, st3 = select_collapses(m3.vertices, m3._triangle_indices, torch.zeros(T), 1e-3, 1.0, 10.0, 2.0, max_frac=1.0)
print('all-zero: pairs', pairs3.shape[0], 'of max', V // 2, 'passes', st3['passes'], 'flip rejected', st3['n_flip_rejected'])
assert pairs3.shape[0] > 0.6 * (V // 2)
assert torch.unique(pairs3.reshape(-1)).numel() == 2 * pairs3.shape[0]
cs3 = collapse_edges(m3, pairs3); print("all-zero collapse:", cs3); assert cs3["vertices_after"] <= V - pairs3.shape[0] and cs3["vertices_isolated_removed"] >= 0
# ---- 4. foldover guard ----
from rd_prune import foldover_filter
P, Q, R, S = [0., 0., 0.], [2., 0., 0.], [1., 0.1, 0.], [1., -3., 0.]
pos4 = torch.tensor([P, Q, R, S]); faces4 = torch.tensor([[0, 1, 2], [2, 1, 3]])
edges4, _, _ = edge_table(faces4)
i_rs = int(((edges4 == torch.tensor([2, 3])).all(1)).nonzero()[0]); i_pq = int(((edges4 == torch.tensor([0, 1])).all(1)).nonzero()[0])
kept, nrej = foldover_filter(pos4, faces4, edges4, torch.tensor([i_rs]))
print('foldover: collapsing RS alone ->', kept.tolist(), 'rejected', nrej); assert nrej == 1 and kept.numel() == 0
kept, nrej = foldover_filter(pos4, faces4, edges4, torch.tensor([i_pq, i_rs]))
print('foldover: PQ and RS together (face PQR degenerates, no survivor flips) ->', kept.tolist(), 'rejected', nrej); assert nrej == 0 and kept.numel() == 2
m4 = Mock(pos4, faces4)
pairs4, st4 = select_collapses(m4.vertices, m4._triangle_indices, torch.zeros(2), 1e-3, 1.0, 10.0, 2.0, max_frac=1.0)
print('foldover via select:', pairs4.tolist(), 'rejected', st4['n_flip_rejected'])
# ---- 5. threshold: nothing below -> nothing selected; everything below with mu huge -> capped ----
_, st5 = select_collapses(m3.vertices, m3._triangle_indices, torch.ones(m3._triangle_indices.shape[0]), 1e-9, 1.0, 10.0, 2.0, 0.1)
assert st5['n_pairs'] == 0 and st5['n_cand'] == 0
_, st6 = select_collapses(m3.vertices, m3._triangle_indices, torch.ones(m3._triangle_indices.shape[0]), 1e-3, 1e9, 10.0, 2.0, 0.05)
assert st6['capped'] and st6['n_pairs'] == st6['max_pairs'], st6
print('all rd_prune unit tests passed')
