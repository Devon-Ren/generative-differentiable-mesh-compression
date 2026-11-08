"""CPU test of v9c: (1) diff vs v9b below the header = 1 fix + 4 version strings; (2) extrapolate_probe semantics on the kitchen-10 v9b probes:
the v9b estimate at 13000 falls below the probe (negative two-point slope x 11k), v9c's does not; a positive slope is unchanged."""
import ast, difflib, sys
v9b, src = open(sys.argv[1]).read(), open(sys.argv[2]).read()
assert src.count("version='adapt-v9c'") == 2 and "version='adapt-v9b'" not in src
d = [l for l in difflib.unified_diff(v9b[v9b.index('import '):].splitlines(), src[src.index('import '):].splitlines(), lineterm='', n=0) if l[:1] == '-' and l[:3] != '---']
assert len(d) == 5 and sum('return r_meas + slope' in l for l in d) == 1 and sum('adapt-v9b' in l for l in d) == 4, d
print('(1) code diff v9b -> v9c below the header: 5 lines (1 fix + 4 version strings)')
ns9b, ns9c = {}, {}
for text, ns in ((v9b, ns9b), (src, ns9c)):
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == 'extrapolate_probe': exec(ast.get_source_segment(text, node), ns)
old, new = ns9b['extrapolate_probe'], ns9c['extrapolate_probe']
# kitchen 10 (v9b run): probe 9.431 at 12500; at 13000 the probe read a little lower -> the two-point slope is negative
trend = [(12500, 9.431, 0.0)]
e_old, s_old = old(trend, 13000, 9.39, 24000); e_new, s_new = new(trend, 13000, 9.39, 24000)
print(f'(2) kitchen 10 at 13000: probe 9.39 MB, raw slope {s_old:+.3f} MB/1k -> v9b estimate {e_old:.2f} MB (gate closed: < 10), v9c estimate {e_new:.2f} MB (= the probe; final was 11.34)')
assert e_old < 9.0 and abs(e_new - 9.39) < 1e-9 and s_old == s_new < 0
trend2 = [(12500, 9.431, 0.0), (13000, 9.39, 0.0)]
e_old2, _ = old(trend2, 13500, 9.70, 24000); e_new2, _ = new(trend2, 13500, 9.70, 24000)
assert e_old2 == e_new2 > 9.70, 'a positive slope must be unchanged'
print(f'    at 13500 with the probe rising to 9.70: both estimates {e_new2:.2f} MB (positive slope unchanged) -> gate opens here instead of 14500')
e3, _ = new([(12500, 5.236, 0.0)], 13000, 5.20, 24000); assert e3 == 5.20
print('    counter 5 at 13000: probe 5.20 > target 5.0 -> v9c gate opens at round 2 (v9b: round 6)')
print('ALL OK')
