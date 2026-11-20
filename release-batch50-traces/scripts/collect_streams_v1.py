"""collect_streams_v1.py (2026-11-20): bytes of every stream of the ByteMesh files, and the wall time of the stages after training, read from the
files that the runs wrote (no computation, no GPU).  One line per run:
  STREAMS <scene> <target MB> | file <bytes> | V <vertices> F <faces> | <stream>:<bytes> ... | stages s: convert <a> finetune <b> encode <c>
The stage times are differences of file times: trained checkpoint -> converted mesh -> fine-tuned mesh -> delivery summary.
Usage: python collect_streams_v1.py > /scratch/zr2245/analysis/streams-b50.txt"""
import glob, json, os, re
runs = sorted(glob.glob('/scratch/zr2245/*-abl24-hold-t*mb') + glob.glob('/scratch/zr2245/*-abl24b-hold-t*mb-best'))
mt = lambda p: os.path.getmtime(p) if os.path.exists(p) else None
for d in runs:
    m = re.search(r'/([a-z]+)-abl24b?-hold-t([\d.]+)mb(-best)?$', d)
    if not m: continue
    a = d + '/axisB/v1.0'; rep = glob.glob(a + '/deliver/**/codec-report.json', recursive=True)
    if not rep: print('note: no codec report under', a); continue
    r = json.load(open(rep[0])); pb = r.get('payload_bytes', {}); V = F = None
    try: b = json.load(open(a + '/report.json')); V, F = b.get('vertices'), b.get('faces')
    except OSError: pass
    t0 = mt(d + '/original/point_cloud/iteration_30000/point_cloud_state_dict.pt'); t1 = mt(a + '/report.json'); t2 = mt(a + '/finetune/summary.json'); t3 = mt(a + '/deliver/summary.json')
    dur = lambda x, y: ('%.0f' % (y - x)) if x and y else 'n/a'
    print('STREAMS %s %s | file %d | V %s F %s | %s | stages s: convert %s finetune %s encode %s' % (m[1], m[2], r['file_bytes'], V, F, ' '.join('%s:%d' % (k, v) for k, v in sorted(pb.items(), key=lambda kv: -kv[1]) if v), dur(t0, t1), dur(t1, t2), dur(t2, t3)))
