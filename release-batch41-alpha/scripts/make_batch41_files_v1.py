#!/usr/bin/env python3
"""make_batch41_files_v1.py (2026-11-15): growth-aim ablation of the hold controller.
Adds three ABL names to the hold sbatch; each is the hold run with a different growth aim (--phase_a_overshoot), nothing else changed:
    holda100 : aim 1.00 x target (no growth past the target)     holda120 : 1.20     holda150 : 1.50      (hold itself: 1.35, unchanged)
The training script is the unchanged abl7.  Output: ablation-b24-r10.sbatch.
  python3 make_batch41_files_v1.py --sbatch <ablation-b24-r8.sbatch> --out <dir>"""
import argparse, hashlib, pathlib, sys
IN = '82259b73557bfcdb760648f42fad08cc'
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
def rep(s, old, new, what):
    n = s.count(old)
    if n != 1: sys.exit(f'FAILED: edit "{what}": anchor found {n} times (expected 1)')
    return s.replace(old, new)
ap = argparse.ArgumentParser(); ap.add_argument('--sbatch', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
s = pathlib.Path(a.sbatch).read_text()
if md5(s) != IN: sys.exit(f'FAILED: {a.sbatch} md5 {md5(s)} != {IN}')
H = '  hold) XARGS="--trim_exact 5 --trim_hold --allow_any_prune_iter"; OVER=1.35; PRUNE="$(seq -s " " 12500 500 29500)"; EXTRA="" ;;\n'
new = H + ''.join('  holda%s) XARGS="--trim_exact 5 --trim_hold --allow_any_prune_iter"; OVER=%s; PRUNE="$(seq -s " " 12500 500 29500)"; EXTRA="" ;;   # batch41: hold with growth aim %s\n' % (n, v, v) for n, v in (('100', '1.0'), ('120', '1.2'), ('150', '1.5')))
s = rep(s, H, new, 'hold variants'); s = rep(s, '|final|hold via --export}', '|final|hold|holda100|holda120|holda150 via --export}', 'ABL list')
s = rep(s, 'VERSION abl24-r8', 'VERSION abl24-r10 (batch41: holda100 / holda120 / holda150)', 'version')
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / 'ablation-b24-r10.sbatch').write_text(s)
print('written ablation-b24-r10.sbatch  lines %d  md5 %s' % (s.count('\n'), md5(s)))
