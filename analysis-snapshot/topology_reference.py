"""Connectivity-coding reference for a MeshSplatting mesh: how many bits/vertex does Google Draco need for the
SAME face list, versus naive index streams and versus our top_stream (strip-fetch + edge-arithmetic)?

Answers the question "the vertex indices are close after reordering, is there a smarter encoding?" with numbers.

    python topology_reference.py --mesh <state_dict.pt | coarse-initial.pt | .npz with faces> \
        [--summary deliver/summary.json | --top-stream-bytes N] [--out report.json]

Rows reported (bits per vertex and per face):
  raw            3 x ceil(log2 V) bits per face, no coding
  int32+lzma     the "write bytes, entropy-code them" baseline (lzma -9 over the int32 index stream, original order)
  delta+lzma     same but each index delta-coded against the previous one
  draco_qp1      Draco mesh encoder, compression level 10, positions quantised to 1 bit (= connectivity + ~0 geometry)
  draco_qp14     Draco with 14-bit positions (full mesh, for scale)
  ours           top_stream bytes from --summary / --top-stream-bytes (if given)

Draco reorders vertices itself (Edgebreaker traversal), so it is an apples-to-apples "best known general connectivity
coder" number. Non-manifold edges are handled by Draco by splitting vertices; the decoded vertex count is reported so
the split overhead is visible.  Requires: pip install DracoPy (pure wheel, no CUDA).
"""
import argparse, json, lzma, math, time
from pathlib import Path
import numpy as np


def load_mesh(path):
    path = Path(path)
    if path.suffix == '.npz':
        z = np.load(path); faces = np.asarray(z['faces'], np.int64)
        xyz = np.asarray(z['xyz'] if 'xyz' in z else z['vertices'], np.float32)
        return xyz, faces
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'triangles_points' in sd:
        xyz = sd['triangles_points'].detach().float().cpu().numpy()
        faces = sd['_triangle_indices'].detach().cpu().numpy().astype(np.int64)
    else:
        raise SystemExit(f'unrecognised checkpoint keys: {list(sd)[:10]}')
    return xyz, faces


def bits_row(name, nbytes, V, F, extra=None):
    row = dict(name=name, bytes=int(nbytes), MB=round(nbytes / 1e6, 4), bits_per_vertex=round(8 * nbytes / V, 3), bits_per_face=round(8 * nbytes / F, 3))
    if extra: row.update(extra)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mesh', required=True)
    p.add_argument('--summary', help='deliver_coarse summary.json (reads codec.payload_bytes.top_stream)')
    p.add_argument('--codec-report', help='codec-report.json with payload_bytes.top_stream')
    p.add_argument('--top-stream-bytes', type=int, help='our topology stream size in bytes, if known')
    p.add_argument('--compression-level', type=int, default=10, help='Draco 0-10 (10 = Edgebreaker + valence prediction)')
    p.add_argument('--skip-draco-qp14', action='store_true')
    p.add_argument('--out')
    a = p.parse_args()

    xyz, faces = load_mesh(a.mesh)
    V, F = len(xyz), len(faces)
    used = np.unique(faces); print(f'{V} vertices ({len(used)} referenced by faces), {F} faces')
    rows = []
    rows.append(bits_row('raw', 3 * math.ceil(math.log2(V)) * F / 8, V, F, dict(bits_per_index=math.ceil(math.log2(V)))))
    stream = faces.astype(np.int32).reshape(-1)
    rows.append(bits_row('int32+lzma', len(lzma.compress(stream.tobytes(), preset=9 | lzma.PRESET_EXTREME)), V, F))
    delta = np.diff(stream, prepend=np.int32(0)).astype(np.int32)
    rows.append(bits_row('delta+lzma', len(lzma.compress(delta.tobytes(), preset=9 | lzma.PRESET_EXTREME)), V, F))

    try:
        import DracoPy
    except ImportError:
        raise SystemExit('pip install DracoPy')
    for qp, skip in ((1, False), (14, a.skip_draco_qp14)):
        if skip: continue
        t = time.time()
        buf = DracoPy.encode(xyz.astype(np.float32), faces.astype(np.uint32), quantization_bits=qp, compression_level=a.compression_level, preserve_order=False)
        enc_s = time.time() - t
        dec = DracoPy.decode(buf)
        dv, df = len(dec.points), len(dec.faces)
        rows.append(bits_row(f'draco_qp{qp}', len(buf), V, F, dict(decoded_vertices=int(dv), decoded_faces=int(df), vertex_split_overhead=round(dv / V - 1, 4), encode_seconds=round(enc_s, 1))))
        print(f'draco qp{qp}: {len(buf)} bytes, decoded {dv} vertices / {df} faces, {enc_s:.1f}s')
    ours = a.top_stream_bytes
    for src in (a.summary, a.codec_report):
        if src and ours is None:
            try:
                r = json.load(open(src)); r = r.get('codec', r); ours = r['payload_bytes']['top_stream']
            except (OSError, KeyError) as e:
                print(f'WARNING could not read top_stream from {src}: {e!r}')
    if ours is not None: rows.append(bits_row('ours_top_stream', ours, V, F))

    print(f'\n{"row":18s} {"MB":>9s} {"bits/vertex":>12s} {"bits/face":>10s}')
    for r in rows: print(f'{r["name"]:18s} {r["MB"]:9.3f} {r["bits_per_vertex"]:12.2f} {r["bits_per_face"]:10.2f}')
    report = dict(mesh=str(a.mesh), vertices=V, faces=F, referenced_vertices=int(len(used)), rows=rows)
    if a.out: Path(a.out).write_text(json.dumps(report, indent=1)); print('wrote', a.out)


if __name__ == '__main__':
    main()
