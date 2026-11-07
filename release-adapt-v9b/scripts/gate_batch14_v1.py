#!/usr/bin/env python3
"""gate_batch14_v1.py (2026-11-07): reads the six v9b test rows (adapt-v9b.txt: room 10, bicycle 10, room 20, garden 5, truck 10;
adapt-v9b-blender.txt: lego 2.5), applies the pass bar written down BEFORE the jobs ran, and prints one verdict line per part:
  GATE MIP PASS|FAIL ...   GATE BLENDER PASS|FAIL ...   GATE SUBMIT all|mip|blender|none
The sbatch wrapper submits the sweep part(s) that passed.  Pass bar (per part):
  Mip/T&T: >= 4 of the 5 rows present; of the present rows >= 4 land within 0.93..1.07 x target and none outside 0.88..1.12;
           every present row's test PSNR >= its reference - 0.08 dB (references: the v9 batch12 / v7b sweep rows of the same scene and
           target, which were delivered at FEWER bytes, so more bytes must not lose PSNR; 0.08 = noise + the lambda-prune's +-0.05).
  Blender: lego 2.5 present, hit within 0.93..1.07 (or growth-capped True), test PSNR >= 32.173 - 0.10 (v9 batch12 lego 2.5 at 2.479 MB).
Usage: gate_batch14_v1.py <adapt-v9b.txt> <adapt-v9b-blender.txt>   (missing files = no rows)"""
import os, re, sys

MIP = {('room', 10.0): 27.686, ('bicycle', 10.0): 22.276, ('room', 20.0): 27.987, ('garden', 5.0): 22.633, ('truck', 10.0): 22.119}
BLD = {('lego', 2.5): 32.173}
LO, HI, LO2, HI2, DPS = 0.93, 1.07, 0.88, 1.12, 0.08


def rows(path):
    out = {}
    if not os.path.exists(path): return out
    for line in open(path):
        m = re.match(r'^(\w+) \| target ([0-9.]+) MB', line)
        if not m or 'delivered' not in line: continue
        hit = re.search(r'hit ([0-9.]+) x target', line); ps = re.search(r'\| test ([0-9.]+) ', line); cap = re.search(r'growth-capped (True|False)', line)
        dl = re.search(r'delivered ([0-9.]+) MB', line); act = re.search(r'prune rounds (\d+) active (\d+) collapsed (\d+)', line)
        if not (hit and ps): continue
        out[(m.group(1), float(m.group(2)))] = dict(hit=float(hit.group(1)), psnr=float(ps.group(1)), capped=(cap.group(1) == 'True') if cap else False,
                                                    mb=float(dl.group(1)) if dl else None, active=int(act.group(2)) if act else None, collapsed=int(act.group(3)) if act else None)
    return out


def judge(name, refs, got, need_present, need_in, dps):
    present = {k: v for k, v in got.items() if k in refs}
    missing = [k for k in refs if k not in present]
    inside = [k for k, v in present.items() if LO <= v['hit'] <= HI or v['capped']]
    wild = [k for k, v in present.items() if not (LO2 <= v['hit'] <= HI2) and not v['capped']]
    low = [k for k, v in present.items() if v['psnr'] < refs[k] - dps]
    ok = len(present) >= need_present and len(inside) >= min(need_in, len(present)) and not wild and not low
    for k, v in sorted(present.items()):
        print(f"  {name} {k[0]} {k[1]:g} MB: hit {v['hit']:.3f} ({v['mb']} MB) test {v['psnr']:.3f} vs ref {refs[k]:.3f} ({v['psnr'] - refs[k]:+.3f}) phase-B active {v['active']} collapsed {v['collapsed']} capped {v['capped']}")
    for k in missing: print(f"  {name} {k[0]} {k[1]:g} MB: NO ROW (job failed or still running)")
    why = []
    if len(present) < need_present: why.append(f'only {len(present)} rows')
    if len(inside) < min(need_in, len(present)): why.append(f'{len(inside)} within {LO}..{HI} (need {min(need_in, len(present))})')
    if wild: why.append('outside %s..%s: %s' % (LO2, HI2, ' '.join(f'{k[0]}{k[1]:g}' for k in wild)))
    if low: why.append('PSNR below ref-%.2f: %s' % (dps, ' '.join(f'{k[0]}{k[1]:g}' for k in low)))
    print(f"GATE {name} {'PASS' if ok else 'FAIL'} | rows {len(present)}/{len(refs)} | within {LO}..{HI}: {len(inside)}" + (' | ' + '; '.join(why) if why else ''))
    return ok


mip_ok = judge('MIP', MIP, rows(sys.argv[1]), 4, 4, DPS)
bld_ok = judge('BLENDER', BLD, rows(sys.argv[2]), 1, 1, 0.10)
mode = 'all' if (mip_ok and bld_ok) else 'mip' if mip_ok else 'blender' if bld_ok else 'none'
print(f'GATE SUBMIT {mode}')
