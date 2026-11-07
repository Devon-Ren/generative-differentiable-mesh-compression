"""delaunay_check_v1.py (2026-10-26, decoder-side check for direction A):
does every delivered face belong to Delaunay(delivered vertices), and what would the face-selection labels cost?

  python delaunay_check_v1.py --state <point_cloud_state_dict.pt | shared-final.pt | decoded.pt> --xyz q14 [--label room-lam0.0062]
        [--asset deliver/delivery/asset.mslb --stage2 /scratch/zr2245/shared-collapse-stage2-v2]   (current topology bytes)
        [--jitter 0.02] [--out check.json]

Steps: quantise positions with the delivery rule (idempotent on v4 outputs; on v3 outputs this is what the codec would do),
merge duplicate grid points, canonical jittered points (delaunay_canon_v1), scipy Delaunay, interior faces = candidates,
coverage = share of delivered faces that are candidates, H0 / H1 = order-0 / sibling-context entropy of the selection bit
in bits per vertex.  Prints one TOPO-CHECK line and writes JSON.  CPU only; ~1 min and ~3 GB per million vertices."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from delaunay_canon_v1 import quantize_xyz, dequantize_xyz, canonical_points, delaunay_tets, interior_faces, face_rows_to_ids, label_entropy, VERSION as CANON_VERSION  # noqa: E402

VERSION = 'delaunay_check_v1 2026-10-26'


def arr(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True); p.add_argument('--xyz', required=True, help='q13..q18 (the delivery grid of this scene)')
    p.add_argument('--jitter', type=float, default=0.02); p.add_argument('--label', default='')
    p.add_argument('--asset', default=None); p.add_argument('--stage2', default='/scratch/zr2245/shared-collapse-stage2-v2')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    t0 = time.time(); bits = int(a.xyz[1:])
    w = torch.load(a.state, map_location='cpu', weights_only=False)
    x = arr(w['triangles_points']).astype(np.float32).reshape(-1, 3); f = arr(w['_triangle_indices']).astype(np.int64).reshape(-1, 3)
    V, F = len(x), len(f)
    q, low, scale = quantize_xyz(x, bits); xq = dequantize_xyz(q, low, scale)
    drift = float(np.abs(x - xq).max()); q2, _, _ = quantize_xyz(xq, bits); idem = bool(np.array_equal(q2, q))
    uq, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True); n_dup = int(V - len(uq))
    rep = first[inv.reshape(-1)]; fr = rep[f]
    degenerate = (fr[:, 0] == fr[:, 1]) | (fr[:, 1] == fr[:, 2]) | (fr[:, 0] == fr[:, 2]); fr = fr[~degenerate]
    keep = np.zeros(V, bool); keep[first] = True; new_id = np.full(V, -1, np.int64); new_id[first] = np.arange(len(first)); fr = new_id[fr]
    qu = q[first]; xu = xq[first]
    perm, P32 = canonical_points(quantize_xyz(xu, bits)[0], *quantize_xyz(xu, bits)[1:], a.jitter)   # exactly the decoder's input
    inv_perm = np.empty_like(perm); inv_perm[perm] = np.arange(len(perm))
    t1 = time.time(); tets = delaunay_tets(P32); t_del = time.time() - t1
    cand, tet_pair = interior_faces(tets)
    ids = face_rows_to_ids(inv_perm[fr], cand)
    covered = int((ids >= 0).sum()); coverage = covered / max(len(fr), 1)
    ent = label_entropy(cand, tet_pair, ids, len(qu))
    res = dict(version=VERSION, canon=CANON_VERSION, state=str(a.state), label=a.label, xyz=a.xyz, jitter=a.jitter, vertices=V, faces=F,
               duplicates_merged=n_dup, degenerate_faces_dropped=int(degenerate.sum()), on_grid=(drift == 0.0), max_drift=drift, quantiser_idempotent=idem,
               tets=int(len(tets)), delaunay_s=round(t_del, 2), coverage=round(coverage, 6), faces_not_delaunay=int(len(fr) - covered), **ent)
    if a.asset:
        try:
            sys.path.insert(0, str(Path(a.stage2) / 'codec')); sys.path.insert(0, str(Path(a.stage2)))
            from codec_benchmark import unpack
            wire = unpack(a.asset); top = int(np.asarray(wire['top_stream']).size); fb = Path(a.asset).stat().st_size
            res.update(asset_bytes=fb, topology_bytes=top, topology_bits_per_vertex=round(top * 8 / V, 2),
                       est_bytes_with_labels_H1=int(fb - top + ent['H1_bits_per_vertex'] * V / 8),
                       est_MB_with_labels_H1=round((fb - top + ent['H1_bits_per_vertex'] * V / 8) / 1e6, 3), asset_MB=round(fb / 1e6, 3))
        except Exception as e:  # noqa: BLE001
            res['asset_error'] = repr(e)
    res['elapsed_s'] = round(time.time() - t0, 1)
    print(f"TOPO-CHECK {a.label} | V {V} F {F} | grid {a.xyz} on_grid {res['on_grid']} drift {drift:.2e} dup {n_dup} | tets {res['tets']} cand/V {ent['candidates_per_vertex']} "
          f"| coverage {100 * coverage:.2f} % ({res['faces_not_delaunay']} faces outside) | sel/cand {ent['p_selected']} | H0 {ent['H0_bits_per_vertex']} H1 {ent['H1_bits_per_vertex']} bits/v"
          + (f" | topology now {res['topology_bits_per_vertex']} bits/v, asset {res['asset_MB']} MB -> est {res['est_MB_with_labels_H1']} MB with labels" if 'topology_bytes' in res else '')
          + f" | {res['elapsed_s']} s")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
