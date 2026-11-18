"""probe_time_v1.py (2026-11-18): share of the training time spent in the size probes, from the training logs (rd_log.jsonl) of the ByteMesh runs.
Probe time = seconds of the probe subprocess (k-means on the copy + the real encoder) over all probes, plus the seconds of the hidden-face
cleaning (rendering the training views) where the log records them: the growth probes record it, the collapse rounds do not, so the printed share
is a LOWER bound.  Training time = elapsed seconds at the final event (30k iterations, without the fine-tuning and the final encoding).
    python probe_time_v1.py <run dir> [<run dir> ...]"""
import json, sys, statistics as st, re
rows = []
for d in sys.argv[1:]:
    try: ev = [json.loads(l) for l in open(d + '/original/rd_log.jsonl')]
    except OSError: print('note: no log in', d); continue
    fin = [e for e in ev if e.get('event') == 'final']
    if not fin: print('note: unfinished log in', d); continue
    pr = [e for e in ev if e.get('event') in ('probe_a', 'prune', 'probe_extra')]
    ps = sum((e.get('probe_seconds') or 0) + (e.get('seconds_cleaning') or 0) for e in pr); tot = fin[-1]['elapsed_s']
    pa = [e for e in pr if e.get('event') == 'probe_a']; cl = sum(e.get('seconds_cleaning') or 0 for e in pa); pp = sum(e.get('probe_seconds') or 0 for e in pa)
    m = re.search(r'/([a-z]+)-abl24-hold-t(\d+)mb', d.rstrip('/')); name = '%s %s MB' % (m[1], m[2]) if m else d
    rows.append((name, len(pr), ps, tot)); print('PROBETIME %-16s | probes %2d | probe time %6.0f s | training %6.0f s | share %4.1f %% (lower bound) | growth probes: encoder %4.0f s, cleaning %4.0f s' % (name, len(pr), ps, tot, 100 * ps / tot, pp, cl))
if rows: print('PROBETIME mean over %d runs | probes %.1f | probe time %.0f s | training %.0f s | share %.1f %% (range %.1f..%.1f)' % (len(rows), st.mean(r[1] for r in rows), st.mean(r[2] for r in rows), st.mean(r[3] for r in rows),
              st.mean(100 * r[2] / r[3] for r in rows), min(100 * r[2] / r[3] for r in rows), max(100 * r[2] / r[3] for r in rows)))
