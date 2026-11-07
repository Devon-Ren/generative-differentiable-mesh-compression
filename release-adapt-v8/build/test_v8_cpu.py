"""CPU test of the v8 pieces that need no torch/upstream: SizeModel (constants -> update_attributes -> update) and the phase-A probe gating."""
import ast, sys
src = open(sys.argv[1]).read(); tree = ast.parse(src)
ns = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in ('TOPO_KEYS', 'BANK_KEYS') for t in node.targets): exec(ast.get_source_segment(src, node), ns)
    if isinstance(node, ast.ClassDef) and node.name == 'SizeModel': exec(ast.get_source_segment(src, node), ns)
SM = ns['SizeModel']
rd = dict(bytes_per_vertex=10.0, bytes_per_face=2.0, overhead_mb=0.6, faces_per_vertex=2.5)
size = SM(rd); assert size.b_v == 6.0 and size.b_f == 2.0 and size.faces_per_vertex == 2.5 and size.overhead == 6e5 and size.source == 'constants' and size.probes_a == []
V0 = size.vertices_for(2.5); assert abs(V0 - (2.5e6 - 6e5) / 11.0) < 1e-6, V0   # 172727 as in every v4-v7 log
# a phase-A soup probe shaped like probe_bytes_v1's json (+ the fields probe_bytes adds): 60000 kept vertices (= 3 x 20000 soup faces)
Vk, Fk = 60000, 20000
pb = dict(app_bank=600000, app_residual_bank=60000, top_stream=30000, _triangle_indices=5000, local_faces=1000, local_rotations=500, xyz=240000, dc=120000, app_index=130000)
pa = dict(file_bytes=sum(pb.values()) + 8000, payload_bytes=pb, vertices=Vk, faces=Fk, iteration=3500)
size.update_attributes(pa)
topo = 30000 + 5000 + 1000 + 500; over = 600000 + 60000 + 8000
assert size.overhead == over and abs(size.b_v - (pa['file_bytes'] - topo - over) / Vk) < 1e-9, (size.b_v, size.overhead)
assert size.b_f == 2.0 and size.faces_per_vertex == 2.5, 'topology must keep the prior after a phase-A probe'
assert size.source == 'probe_a@3500' and size.probes_a == [pa] and size.probes == []
assert pa['b_v'] == round(size.b_v, 3) and pa['soup_b_f'] == round(topo / Fk, 3) and pa['soup_faces_per_vertex'] == round(Fk / Vk, 4) and set(pa['topo_keys_found']) == {'top_stream', '_triangle_indices', 'local_faces', 'local_rotations'}
V1 = size.vertices_for(2.5); print(f"b_v {size.b_v:.3f} overhead {size.overhead/1e6:.3f} MB -> V* 2.5 MB {V0:.0f} -> {V1:.0f}; mb(V1) = {size.mb(V1):.4f}")
assert abs(size.mb(V1) - 2.5) < 1e-9
# then the 12k mesh probe replaces everything (v7b update, unchanged)
pm = dict(file_bytes=2160000, payload_bytes=dict(app_bank=600000, app_residual_bank=60000, top_stream=500000, _triangle_indices=30000, local_faces=1000, local_rotations=500, xyz=500000, dc=300000, app_index=160000), vertices=117680, faces=302038, iteration=12000)
size.update(pm); assert size.source == 'probe@12000' and abs(size.b_f - 531500 / 302038) < 1e-9 and abs(size.faces_per_vertex - 302038 / 117680) < 1e-9 and size.probes == [pm] and size.probes_a == [pa]
# gating rule as written in the densify block
def schedule(start_pruning, every, smoke, intervals):
    last = None; out = []
    for it in intervals:
        if every > 0 and (it > start_pruning or smoke) and (last is None or it - last >= every): out.append(it); last = it
    return out
iv = list(range(1500, 10000, 500)); assert len(iv) == 17
assert schedule(3000, 1000, False, iv) == [3500, 4500, 5500, 6500, 7500, 8500, 9500]
assert schedule(3000, 0, False, iv) == []
assert schedule(3000, 1000, True, [1500]) == [1500]  # the 1600-it smoke: one phase-A probe at the first interval
assert schedule(3000, 500, False, iv) == list(range(3500, 10000, 500))
print('v8 CPU test OK: SizeModel attribute-only update, mesh update, probe-a schedule (7 probes on 17 intervals, start_pruning 3000)')
