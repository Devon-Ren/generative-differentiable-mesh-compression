"""mesh_stats_v2.py (VERSION mesh_stats_v2 2026-11-04; v1 + detach: checkpoint tensors carry requires_grad): CPU geometry statistics of a MeshSplatting checkpoint (triangle soup after Delaunay/growth/prune).

Answers "how ill-formed is the mesh": per-edge face incidence (manifold = 2, boundary = 1, non-manifold >= 3), sliver share
(min angle < 5 / 10 deg), degenerate/zero-area faces, duplicate faces, connected components (by shared vertices), dihedral angle
distribution across manifold edges, face-area distribution, unused vertices. Pure numpy, no CUDA; a 3 M-face mesh takes ~1 min.

Usage:
  python mesh_stats_v2.py --ckpt .../point_cloud/iteration_30000/point_cloud_state_dict.pt [--label lego-30k] [--out stats.json]
"""
import argparse, json, sys
import numpy as np
import torch

VERSION = 'mesh_stats_v2 2026-11-04'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--label', default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    V = sd['triangles_points'].detach().cpu().double().numpy(); F = sd['_triangle_indices'].detach().cpu().long().numpy()
    nv, nf = len(V), len(F)
    label = a.label or a.ckpt
    # --- faces: area, angles
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(B - A, C - A); area2 = np.linalg.norm(n, axis=1); area = 0.5 * area2
    def ang(P, Q, R):
        u, w = Q - P, R - P
        c = (u * w).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1), 1e-30)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    amin = np.minimum.reduce([ang(A, B, C), ang(B, C, A), ang(C, A, B)])
    degenerate = (area2 <= 1e-12 * np.median(area2)) | (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    # --- edges
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); E = np.sort(E, axis=1)
    key = E[:, 0] * (nv + 1) + E[:, 1]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inc = cnt[inv]  # per half-edge: how many faces share this edge
    ne = len(uk); n_bd = int((cnt == 1).sum()); n_man = int((cnt == 2).sum()); n_nm = int((cnt >= 3).sum())
    # faces touching a non-manifold / boundary edge
    inc_f = inc.reshape(3, nf)
    f_nm = (inc_f >= 3).any(0); f_bd = (inc_f == 1).any(0)
    # duplicate faces
    Fs = np.sort(F, axis=1); fkey = (Fs[:, 0] * (nv + 1) + Fs[:, 1]) * (nv + 1) + Fs[:, 2]
    n_dup = int(nf - len(np.unique(fkey)))
    # --- connected components via union-find on vertices (faces connect their vertices)
    parent = np.arange(nv)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    # vectorised-ish union: iterate edges in chunks with python loop over unique edges (ne up to ~5M -> fine in numpy loop? use scipy if available)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        e0, e1 = uk // (nv + 1), uk % (nv + 1)
        g = coo_matrix((np.ones(ne), (e0, e1)), shape=(nv, nv))
        ncomp, lab = connected_components(g, directed=False)
        used = np.zeros(nv, bool); used[F.reshape(-1)] = True
        comp_sizes = np.bincount(lab[used]); comp_sizes = np.sort(comp_sizes[comp_sizes > 0])[::-1]
        ncomp_used = int(len(comp_sizes)); largest_frac = float(comp_sizes[0] / used.sum()); n_tiny = int((comp_sizes <= 10).sum())
    except ImportError:
        ncomp_used, largest_frac, n_tiny = -1, -1.0, -1
        used = np.zeros(nv, bool); used[F.reshape(-1)] = True
    # --- dihedral angles across manifold edges
    fn = n / np.maximum(area2, 1e-30)[:, None]
    order = np.argsort(inv, kind='stable'); inv_s = inv[order]; fid = (np.arange(3 * nf) % nf)[order]
    man_edges = np.flatnonzero(cnt == 2)
    starts = np.searchsorted(inv_s, man_edges)
    fa, fb = fid[starts], fid[starts + 1]
    cosd = np.clip((fn[fa] * fn[fb]).sum(1), -1, 1); dih = np.degrees(np.arccos(np.abs(cosd)))  # 0 = coplanar (orientation-agnostic)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float('nan')
    stats = dict(version=VERSION, label=label, ckpt=a.ckpt, vertices=nv, faces=nf, unused_vertices=int((~used).sum()),
                 edges=ne, boundary_edges=n_bd, manifold_edges=n_man, nonmanifold_edges=n_nm,
                 boundary_edge_frac=n_bd / ne, nonmanifold_edge_frac=n_nm / ne,
                 faces_touching_nonmanifold_edge_frac=float(f_nm.mean()), faces_touching_boundary_edge_frac=float(f_bd.mean()),
                 duplicate_faces=n_dup, degenerate_faces=int(degenerate.sum()),
                 sliver_minangle_lt5_frac=float((amin < 5).mean()), sliver_minangle_lt10_frac=float((amin < 10).mean()), min_angle_median_deg=float(np.median(amin)),
                 dihedral_median_deg=q(dih, 50), dihedral_p90_deg=q(dih, 90), dihedral_gt30_frac=float((dih > 30).mean()) if len(dih) else float('nan'),
                 area_median=float(np.median(area)), area_p10=q(area, 10), area_p90=q(area, 90), area_max_over_median=float(area.max() / max(np.median(area), 1e-30)),
                 components=ncomp_used, largest_component_vertex_frac=largest_frac, components_le10_vertices=n_tiny)
    print(f"MESH_STATS {label} | V {nv} F {nf} unused_v {stats['unused_vertices']} | edges {ne}: boundary {100*stats['boundary_edge_frac']:.1f} % manifold {100*n_man/ne:.1f} % non-manifold {100*stats['nonmanifold_edge_frac']:.2f} % "
          f"| faces on non-manifold edge {100*stats['faces_touching_nonmanifold_edge_frac']:.1f} %, on boundary edge {100*stats['faces_touching_boundary_edge_frac']:.1f} % | dup {n_dup} degenerate {stats['degenerate_faces']} "
          f"| sliver <5deg {100*stats['sliver_minangle_lt5_frac']:.1f} % <10deg {100*stats['sliver_minangle_lt10_frac']:.1f} % (median min angle {stats['min_angle_median_deg']:.1f}) "
          f"| dihedral median {stats['dihedral_median_deg']:.1f} p90 {stats['dihedral_p90_deg']:.1f} >30deg {100*stats['dihedral_gt30_frac']:.1f} % "
          f"| components {ncomp_used} (largest {100*largest_frac:.1f} % of used vertices, {n_tiny} with <=10 v) | area p10/med/p90 {stats['area_p10']:.2e}/{stats['area_median']:.2e}/{stats['area_p90']:.2e}")
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(stats, f, indent=1)


if __name__ == '__main__':
    main()
