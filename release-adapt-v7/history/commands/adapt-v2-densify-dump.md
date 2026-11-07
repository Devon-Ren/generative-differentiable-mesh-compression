# adapt-v2 densify dump（2026-11-06）：看 phase A 为什么没长到 aim

```bash
AN=/scratch/zr2245/analysis
cat > $AN/dump_densify_v1.py <<'PY'
import json, sys
for d in sys.argv[1:]:
    ev = [json.loads(l) for l in open(d + '/original/rd_log.jsonl')]; dn = [e for e in ev if e['event'] == 'densify']
    print('===', d, len(dn), 'densify events; keys', sorted(k for k in dn[0] if k not in ('event',)))
    print('it     n_tri    n_pass   n_split  max_split  desired    cap   V_before    V_after  dV/3split  rhat   lam_used  lam_next  pass/des  valid')
    for e in dn:
        g = (e['vertices_after'] - e['vertices_before']) / max(3 * e['n_split'], 1)
        print(f"{e['iteration']:5d} {e['n_tri']:8d} {e['n_pass']:8d} {e['n_split']:8d} {e['max_split']:9d} {e['desired']:8.0f} {e['cap']:7d} {e['vertices_before']:10d} {e['vertices_after']:10d} {g:9.2f} {e['rhat_mb']:6.2f} {e['lam_used']:9.2e} {e['lam_next']:9.2e} {e['pass_ratio']:8.2f} {e.get('valid_frac')}")
    fin = [e for e in ev if e['event'] == 'final'][0]; pr = [e for e in ev if e['event'] == 'prune']
    print('prune probes (it, probe_mb, probe_vertices, probe_faces, faces_kept/faces_model, desired_pairs):', [(e['iteration'], e['probe_mb'], e['probe_vertices'], e['probe_faces'], round(e.get('faces_kept', 0) / max(e.get('faces_model', 1), 1), 3) if 'faces_kept' in e else None, e['desired_pairs']) for e in pr])
    print('final:', {k: fin[k] for k in ('vertices', 'triangles', 'rhat_mb', 'probe_final_mb')})
PY
python $AN/dump_densify_v1.py /scratch/zr2245/room-adapt2-t20mb /scratch/zr2245/kitchen-adapt2-t20mb /scratch/zr2245/garden-adapt2-t20mb /scratch/zr2245/room-adapt2-t5mb
for d in /scratch/zr2245/room-adapt2-t20mb /scratch/zr2245/room-adapt2-t5mb; do echo "--- $d"; grep -h "^\[RD adapt\|^\[RD prune\|^Training is done" $d/train.log | tail -8; done
```
