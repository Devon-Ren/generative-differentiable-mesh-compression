#!/usr/bin/env python3
"""gate_batch15_v1.py (2026-11-08): pass bar for the six v9c test rows, fixed before the jobs ran.
  Mip (adapt-v9c.txt: room 5, counter 5, kitchen 10, counter 10, kitchen 20 = the v9b over-target rows): >= 4 of 5 present; >= 4 of the present rows
      within 0.95..1.05 x target; none outside 0.90..1.08; every row's test PSNR >= the fixed-lambda hull of its scene at the delivered bytes - 0.15 dB
      (hull = rd-hull6-v2, lam 1.85e-3 / 6.2e-3 / 1.85e-2 / 6.2e-2 points, log-linear).
  Blender (adapt-v9c-blender.txt: lego 2.5): within 0.95..1.05, test PSNR >= 32.383 - 0.15 (v9b lego 2.5 at 2.496 MB).
Prints GATE MIP PASS|FAIL, GATE BLENDER PASS|FAIL, GATE SUBMIT all|mip|blender|none.  Usage: gate_batch15_v1.py <adapt-v9c.txt> <adapt-v9c-blender.txt>"""
import math, os, re, sys

HULL = {'room': [(22.1, 28.25), (12.4, 28.02), (5.4, 27.30), (2.9, 26.38)], 'counter': [(20.7, 25.95), (13.1, 25.84), (6.4, 25.32), (3.3, 24.54)],
        'kitchen': [(24.5, 27.09), (16.5, 26.89), (8.0, 25.97), (3.8, 24.62)]}
MIP = [('room', 5.0), ('counter', 5.0), ('kitchen', 10.0), ('counter', 10.0), ('kitchen', 20.0)]
BLD = {('lego', 2.5): 32.383}
LO, HI, LO2, HI2, DPS = 0.95, 1.05, 0.90, 1.08, 0.15


def hull_psnr(scene, mb):
    pts = sorted(HULL[scene]); xs = [math.log(m) for m, _ in pts]; x = math.log(mb)
    i = 0 if x <= xs[0] else len(pts) - 2 if x >= xs[-1] else max(j for j in range(len(pts) - 1) if xs[j] <= x)
    (x0, y0), (x1, y1) = (xs[i], pts[i][1]), (xs[i + 1], pts[i + 1][1])
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


def rows(path):
    out = {}
    if not os.path.exists(path): return out
    for line in open(path):
        m = re.match(r'^(\w+) \| target ([0-9.]+) MB', line)
        if not m or 'delivered' not in line: continue
        hit = re.search(r'hit ([0-9.]+) x target', line); ps = re.search(r'\| test ([0-9.]+) ', line); cap = re.search(r'growth-capped (True|False)', line)
        dl = re.search(r'delivered ([0-9.]+) MB', line); act = re.search(r'prune rounds (\d+) active (\d+) collapsed (\d+)', line)
        if not (hit and ps and dl): continue
        out[(m.group(1), float(m.group(2)))] = dict(hit=float(hit.group(1)), psnr=float(ps.group(1)), capped=(cap.group(1) == 'True') if cap else False, mb=float(dl.group(1)),
                                                    rounds=int(act.group(1)) if act else None, active=int(act.group(2)) if act else None, collapsed=int(act.group(3)) if act else None)
    return out


def judge(name, keys, refof, got, need_present, dps):
    present = {k: got[k] for k in keys if k in got}; missing = [k for k in keys if k not in present]
    inside = [k for k, v in present.items() if LO <= v['hit'] <= HI or v['capped']]
    wild = [k for k, v in present.items() if not (LO2 <= v['hit'] <= HI2) and not v['capped']]
    low = [k for k, v in present.items() if v['psnr'] < refof(k, v) - dps]
    need_in = min(need_present, len(present))
    ok = len(present) >= need_present and len(inside) >= need_in and not wild and not low
    for k, v in present.items():
        ref = refof(k, v)
        print(f"  {name} {k[0]} {k[1]:g} MB: hit {v['hit']:.3f} ({v['mb']} MB) test {v['psnr']:.3f} vs ref {ref:.3f} ({v['psnr'] - ref:+.3f}) phase-B {v['active']}/{v['rounds']} rounds, {v['collapsed']} pairs, capped {v['capped']}")
    for k in missing: print(f"  {name} {k[0]} {k[1]:g} MB: NO ROW (job failed or still running)")
    why = []
    if len(present) < need_present: why.append(f'only {len(present)} rows')
    if len(inside) < need_in: why.append(f'{len(inside)} within {LO}..{HI} (need {need_in})')
    if wild: why.append('outside %s..%s: %s' % (LO2, HI2, ' '.join(f'{k[0]}{k[1]:g}' for k in wild)))
    if low: why.append('PSNR below ref-%.2f: %s' % (dps, ' '.join(f'{k[0]}{k[1]:g}' for k in low)))
    print(f"GATE {name} {'PASS' if ok else 'FAIL'} | rows {len(present)}/{len(keys)} | within {LO}..{HI}: {len(inside)}" + (' | ' + '; '.join(why) if why else ''))
    return ok


mip_ok = judge('MIP', MIP, lambda k, v: hull_psnr(k[0], v['mb']), rows(sys.argv[1]), 4, DPS)
bld_ok = judge('BLENDER', list(BLD), lambda k, v: BLD[k], rows(sys.argv[2]), 1, DPS)
mode = 'all' if (mip_ok and bld_ok) else 'mip' if mip_ok else 'blender' if bld_ok else 'none'
print(f'GATE SUBMIT {mode}')
