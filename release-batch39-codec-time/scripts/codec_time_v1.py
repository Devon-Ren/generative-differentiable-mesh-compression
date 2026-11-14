"""codec_time_v1.py (2026-11-14): encode and decode time of our codec on ONE delivered mesh.  CPU only.  Run inside the stage2 env:
    bash $PKG/stage2_run.sh $AN/codec_time_v1.py --shared <deliver>/shared-final.pt --xyz q15 --dc q7 --out <json> --label "garden cap1073000"
Input is the shared representation the delivery wrote (mesh + base colour + codebooks + indices), i.e. what the encoder receives.
Timed, with the same calls as stage2 runtime.export_asset / codec topology.read_asset:
  compile  : building the native coders (once per machine; not part of encode or decode)
  encode   : quantise + reorder vertices + predict + entropy-code every stream + write the one file
  decode   : read the file + entropy-decode every stream (repeated, the fastest run is reported) ; restore = de-quantise to float arrays
The decoded mesh is compared with the encoder's input order exactly as export_asset does."""
import argparse, json, os, platform, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch

def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage2', required=True); p.add_argument('--shared', required=True); p.add_argument('--out', required=True)
    p.add_argument('--xyz', default='q14'); p.add_argument('--dc', default='q7'); p.add_argument('--label', default=''); p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args(); stage2 = Path(a.stage2).resolve()
    for d in (stage2 / 'renderer', stage2, stage2 / 'codec'): sys.path.insert(0, str(d))
    import delivery, topology as tc
    from codec_benchmark import pack, unpack, equal
    delivery.XYZ_MODES = tuple(dict.fromkeys(delivery.XYZ_MODES + ('q15', 'q13', 'q12'))); delivery.DC_MODES = tuple(dict.fromkeys(delivery.DC_MODES + ('q7', 'q6', 'q5', 'q4')))
    wire = torch.load(a.shared, map_location='cpu', weights_only=False); wire.setdefault('opacity_floor', .999)
    V, F = len(wire['triangles_points']), len(wire['_triangle_indices'])
    work = Path(tempfile.mkdtemp(prefix='codec-time-', dir=str(Path(a.out).resolve().parent))); native = work / 'native'; T = {}
    try:
        t = time.time(); exe = tc.lc.native(native); edge = tc.edge_native(native); ctx = tc.ic.native(native); T['compile'] = time.time() - t
        enc = []
        for i in range(2):   # the second run has warm files; the faster one is reported
            tmp = work / f'enc{i}'; tmp.mkdir(); t = time.time()
            quant = delivery.prepare(wire, a.xyz, a.dc, True, 'const'); ordered, perm = tc.reorder(quant, 'strip-fetch')
            pred = tc.lc.graph(exe, ordered, tmp); encoded = tc.encode(ordered, 'edge-arithmetic', 1, exe, edge, tmp, pred, ctx)
            path = work / f'asset{i}.mslb'; pack(encoded, path, 'auto'); enc.append(time.time() - t); del encoded
        size = path.stat().st_size; dec, res = [], []
        for i in range(max(1, a.repeats)):
            tmp = work / f'dec{i}'; tmp.mkdir(); t = time.time(); decoded = tc.read_asset(path, native, tmp); dec.append(time.time() - t)
            t = time.time(); state = delivery.restore(decoded); res.append(time.time() - t)
        equal({k: v for k, v in ordered.items() if k != '_triangle_indices'}, {k: decoded[k] for k in ordered if k != '_triangle_indices'})
        if not np.array_equal(tc.cyclic(ordered['_triangle_indices'].numpy()), tc.cyclic(decoded['_triangle_indices'].numpy())): raise AssertionError('topology roundtrip')
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out = dict(label=a.label, vertices=V, faces=F, file_MB=round(size / 1e6, 3), xyz=a.xyz, dc=a.dc, compile_s=round(T['compile'], 2), encode_s=round(min(enc), 2), encode_runs=[round(x, 2) for x in enc],
               decode_s=round(min(dec), 2), decode_runs=[round(x, 2) for x in dec], restore_s=round(min(res), 3), threads=os.environ.get('OMP_NUM_THREADS', ''), cpu=platform.processor() or platform.machine(), host=platform.node(), roundtrip='exact')
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"TIME {a.label} | V {V} F {F} | {out['file_MB']} MB | ours encode {out['encode_s']} s | decode {out['decode_s']} s (+ restore {out['restore_s']} s) | compile once {out['compile_s']} s | roundtrip exact", flush=True)

if __name__ == '__main__':
    main()
