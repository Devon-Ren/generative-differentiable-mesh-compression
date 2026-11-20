"""collect_traces_v1.py (2026-11-20): the size measured at every collapse round of the ByteMesh runs, from the training logs (rd_log.jsonl).
One line per run:  TRACE <scene> <target MB> | <iteration>:<probe MB>:<MB removed in this round>:<pairs> ... | final <probe MB at 30k>
Reads small JSON files only (no GPU).   Usage: python collect_traces_v1.py > /scratch/zr2245/analysis/traces-b50.txt"""
import glob, json, re
runs = sorted(glob.glob('/scratch/zr2245/*-abl24-hold-t*mb') + glob.glob('/scratch/zr2245/*-abl24b-hold-t*mb'))
for d in runs:
    m = re.search(r'/([a-z]+)-abl24b?-hold-t([\d.]+)mb$', d)
    if not m: continue
    try: ev = [json.loads(l) for l in open(d + '/original/rd_log.jsonl')]
    except OSError: print('note: no log in', d); continue
    pr = [e for e in ev if e.get('event') == 'prune']; fin = [e for e in ev if e.get('event') == 'final']
    pa = [e for e in ev if e.get('event') == 'probe_a']
    rows = ' '.join('%d:%.3f:%.3f:%d' % (e['iteration'], e['probe_mb'], e.get('removed_mb') or 0.0, e.get('n_pairs') or 0) for e in pr)
    grow = ' '.join('%d:%.3f' % (e['iteration'], e['probe_mb']) for e in pa)
    print('TRACE %s %s | %s | final %s | growth %s' % (m[1], m[2], rows, ('%.3f' % fin[-1]['probe_final_mb']) if fin and fin[-1].get('probe_final_mb') is not None else 'n/a', grow))
