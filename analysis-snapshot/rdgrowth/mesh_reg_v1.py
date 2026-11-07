"""mesh_reg_v1.py (VERSION mesh_reg_v1 2026-11-05): mesh-quality regularisers for MeshSplatting training (pure torch, CPU-testable).

Three scale-free terms (all dimensionless, so one weight works across scenes):
  sliver_loss(pos, faces, a0)   per-face aspect = longest_edge^2 / (2 * area)  (= 1/height ratio); loss = mean(relu(log(aspect / a0)))
                                  -> penalises long thin "jump" triangles only; large flat well-shaped triangles are untouched.
  laplacian_loss(pos, faces)    uniform Laplacian delta_v = mean(neighbours) - v, normalised by the vertex's mean incident edge length
                                  -> mean(|delta_v| / s_v)^2 over interior vertices (open-boundary vertices excluded, they would only shrink inward);
                                  only meaningful once vertices are shared (after the restricted Delaunay).
  normal_loss(pos, faces, adj)  1 - n_i . n_j over interior edges (faces i, j share the edge); adj from face_adjacency(faces) (cached by caller).
  jump_face_mask(pos, faces, a1, k)  hard filter: faces with aspect > a1 AND longest edge > k * median edge length -> drop (post-Delaunay).
                                  aspect ~ length/base for a spike, so a1=20 = spikes at least 20x longer than their base; big flat well-shaped faces survive.
"""
import torch

VERSION = 'mesh_reg_v1 2026-11-05b'


def _edges_of(pos, faces):
    t = pos[faces.long()]                           # [T,3,3]
    e = torch.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 1], t[:, 0] - t[:, 2]], dim=1)   # [T,3,3]
    return t, e


def face_aspect(pos, faces, eps=1e-12):
    """aspect = longest_edge^2 / (2*area) >= ~1.15 (equilateral); long thin slivers -> large. Returns (aspect [T], longest [T], area [T])."""
    t, e = _edges_of(pos, faces)
    l2 = (e * e).sum(-1)                            # [T,3] squared lengths
    longest2 = l2.max(dim=1).values
    area = 0.5 * torch.linalg.norm(torch.cross(e[:, 0], -e[:, 2], dim=1), dim=1)
    aspect = longest2 / (2.0 * area + eps)
    return aspect, torch.sqrt(longest2), area


def sliver_loss(pos, faces, a0=20.0, max_log=8.0):
    """mean over faces of min(relu(log(aspect/a0)), max_log); the cap gives numerically degenerate faces (aspect > a0*e^8) no gradient
    (they render nothing and are removed by the upstream importance prune) instead of a 1/area gradient spike into Adam's moments."""
    aspect, _, _ = face_aspect(pos, faces)
    return torch.relu(torch.log(aspect.clamp_min(1e-6) / a0)).clamp_max(max_log).mean()


@torch.no_grad()
def jump_face_mask(pos, faces, a1=20.0, k=6.0):
    """keep-mask [T] (True = keep). Drops faces that are both very thin (aspect > a1) and long (longest edge > k * median edge)."""
    aspect, longest, _ = face_aspect(pos, faces)
    med = torch.median(longest)
    drop = (aspect > a1) & (longest > k * med)
    return ~drop, dict(median_edge=float(med), dropped=int(drop.sum()), aspect_q50=float(aspect.median()), aspect_q99=float(torch.quantile(aspect, 0.99)) if aspect.numel() < 16_000_000 else -1.0)


@torch.no_grad()
def boundary_vertices(faces, V):
    """bool [V]: vertex lies on an edge used by exactly one face (open boundary)."""
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], dim=0)
    e, _ = torch.sort(e, dim=1)
    key = e[:, 0] * V + e[:, 1]
    uk, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    bnd_edge = cnt[inv] == 1
    out = torch.zeros(V, dtype=torch.bool, device=faces.device)
    out[e[bnd_edge].flatten()] = True
    return out


def laplacian_loss(pos, faces, exclude_boundary=True, boundary=None):
    """mean over vertices of (|mean(neighbours) - v| / mean incident edge length)^2 (dimensionless). Vertices with no edges contribute 0;
    with exclude_boundary, vertices on an open boundary edge are excluded too (the uniform Laplacian would only pull them inward);
    boundary = precomputed boundary_vertices(faces, V) mask (cached by the caller per topology), else computed here."""
    f = faces.long()
    i = torch.cat([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = torch.cat([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    V = pos.shape[0]
    nbr_sum = torch.zeros_like(pos).index_add_(0, i, pos[j])
    cnt = torch.zeros(V, device=pos.device, dtype=pos.dtype).index_add_(0, i, torch.ones_like(i, dtype=pos.dtype))
    elen = torch.linalg.norm(pos[j] - pos[i], dim=1)
    esum = torch.zeros(V, device=pos.device, dtype=pos.dtype).index_add_(0, i, elen)
    has = cnt > 0
    if exclude_boundary:
        has = has & ~(boundary if boundary is not None else boundary_vertices(faces, V))
    c = cnt.clamp_min(1.0)
    delta = nbr_sum / c[:, None] - pos
    s = (esum / c).clamp_min(1e-12)
    r = torch.linalg.norm(delta, dim=1) / s
    r = torch.where(has, r, torch.zeros_like(r))
    return (r * r).sum() / has.sum().clamp_min(1)


@torch.no_grad()
def face_adjacency(faces):
    """Pairs of faces sharing an edge (interior edges only; edges with >2 faces contribute their first two). Returns long [P,2]."""
    f = faces.long()
    T = f.shape[0]
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], dim=0)
    e, _ = torch.sort(e, dim=1)
    fid = torch.arange(T, device=f.device).repeat(3)
    key = e[:, 0] * (int(f.max()) + 1) + e[:, 1]
    order = torch.argsort(key)
    key, fid = key[order], fid[order]
    same = key[1:] == key[:-1]
    first = torch.ones_like(same)
    first[1:] = ~same[:-1]                      # pair only the first two faces of each edge
    sel = same & first
    return torch.stack([fid[:-1][sel], fid[1:][sel]], dim=1)


def face_normals(pos, faces):
    t, e = _edges_of(pos, faces)
    n = torch.cross(e[:, 0], -e[:, 2], dim=1)
    return n / torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(1e-12)


def normal_loss(pos, faces, adj):
    if adj.numel() == 0:
        return pos.sum() * 0.0
    n = face_normals(pos, faces)
    return (1.0 - (n[adj[:, 0]] * n[adj[:, 1]]).sum(-1)).mean()


@torch.no_grad()
def mesh_report(pos, faces, adj=None, a0=20.0):
    aspect, longest, area = face_aspect(pos, faces)
    rep = dict(faces=int(faces.shape[0]), vertices=int(pos.shape[0]), aspect_q50=float(aspect.median()), sliver_frac=float((aspect > a0).float().mean()),
               lap=float(laplacian_loss(pos, faces)))
    if adj is not None and adj.numel():
        n = face_normals(pos, faces); c = (n[adj[:, 0]] * n[adj[:, 1]]).sum(-1)
        rep.update(normal=float((1 - c).mean()), dihedral_gt60_frac=float((c < 0.5).float().mean()))
    return rep


# ---- training-loop helpers (duck-typed: triangles has .vertices/._triangle_indices/.prune_triangles(keep); acc has .reset(n); log has .write(**kw)) ----
class AdjCache:
    """face_adjacency recomputed only when the topology signature (V, T) changes; callers call .invalidate() after any topology event."""
    def __init__(self):
        self.sig = None; self.adj = None; self.boundary = None

    def invalidate(self):
        self.sig = None

    def _refresh(self, triangles):
        sig = (int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0]))
        if self.sig != sig:
            self.adj = face_adjacency(triangles._triangle_indices)
            self.boundary = boundary_vertices(triangles._triangle_indices, sig[0])
            self.sig = sig

    def get(self, triangles):
        self._refresh(triangles); return self.adj

    def get_boundary(self, triangles):
        self._refresh(triangles); return self.boundary


def apply_jump_filter(triangles, acc, log, a1, k, tag, iteration, adj_cache=None):
    """Hard jump-triangle filter on the live model: drop faces with aspect > a1 AND longest edge > k * median edge; asserts the face count after
    prune_triangles equals keep.sum() (fails loudly if the upstream prune semantics ever change); resets the RD accumulator. Returns dropped count."""
    if a1 <= 0:
        return 0
    T0 = int(triangles._triangle_indices.shape[0])
    keep, st = jump_face_mask(triangles.vertices.detach(), triangles._triangle_indices, a1, k)
    n_keep = int(keep.sum()); n_drop = T0 - n_keep
    if n_drop > 0:
        triangles.prune_triangles(keep)
        T1 = int(triangles._triangle_indices.shape[0])
        if T1 != n_keep:
            raise RuntimeError(f'jump filter: prune_triangles(keep) left {T1} faces, expected keep.sum() = {n_keep} (keep-semantics assumption broken)')
        if acc is not None:
            acc.reset(T1)
        if adj_cache is not None:
            adj_cache.invalidate()
    ev = dict(event='jumpfilter', tag=tag, iteration=int(iteration), faces_before=T0, faces_after=T0 - n_drop, dropped=n_drop, a1=a1, k=k)
    ev.update({kk: vv for kk, vv in st.items() if kk != 'dropped'})
    log.write(**ev)
    print(f"\n[REG jumpfilter {tag} {iteration}] faces {T0}->{T0 - n_drop} dropped {n_drop} ({100.0 * n_drop / max(1, T0):.3f} %) | median edge {st['median_edge']:.4g} aspect q50/q99 {st['aspect_q50']:.2f}/{st['aspect_q99']:.1f}")
    return n_drop


def write_report(triangles, log, tag, iteration, a0=20.0, adj_cache=None):
    adj = adj_cache.get(triangles) if adj_cache is not None else None
    rep = mesh_report(triangles.vertices.detach(), triangles._triangle_indices, adj, a0)
    log.write(event='mesh', tag=tag, iteration=int(iteration), **rep)
    print(f"\n[REG mesh {tag} {iteration}] V {rep['vertices']} F {rep['faces']} | aspect q50 {rep['aspect_q50']:.2f} sliver(>{a0:g}) {100 * rep['sliver_frac']:.2f} % | lap {rep['lap']:.4f}"
          + (f" | normal {rep['normal']:.4f} dihedral>60deg {100 * rep['dihedral_gt60_frac']:.2f} %" if 'normal' in rep else ''))
    return rep
