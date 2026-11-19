"""make_code_release_v1.py (2026-11-19): assembles the anonymous code supplement from the sources in this repository.
Python and our C++ files are stripped of comments and docstrings; every stripped file is checked to be equivalent to its source
(Python: identical AST after removing the string statements; C++: identical g++ comment-free token stream).  The renderer (derived from
third-party code) is copied unchanged with its licence files.   Usage: python make_code_release_v1.py <stage2 dir> <vis_collapse dir> <out dir>"""
import ast, io, os, re, shutil, subprocess, sys, tokenize
from pathlib import Path
R = Path(__file__).resolve().parents[1]; ST, VIS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
RG = R / 'analysis-snapshot/rdgrowth'
PY = {  # destination -> (source, [(old, new, count)])
 'training/train_bytemesh.py': (R / 'release-batch31-hold/build/train_rd_adapt_v9b_abl7.py', [('from rd_growth_v2 import', 'from rd_growth import', 1)]),
 'training/rd_growth.py': (RG / 'rd_growth_v2.py', []), 'training/rd_prune.py': (RG / 'rd_prune.py', []), 'training/size_probe.py': (RG / 'probe_bytes_v1.py', []),
 'postprocess/vertex_footprint.py': (VIS / 'vertex_footprint.py', []), 'postprocess/coarse_vis.py': (VIS / 'coarse_vis.py', []), 'postprocess/build_vis.py': (VIS / 'build_vis.py', []),
 'postprocess/finetune.py': (R / 'analysis-snapshot/coarse_finetune.py', []),
 'encode.py': (VIS / 'deliver_coarse.py', [("default='/scratch/zr2245/shared-collapse-stage2-v2'", "default=str(Path(__file__).resolve().parent / 'codec')", 1),
                                           ("default='/scratch/zr2245/room-shared-collapse-QGQgyr/run/config.json'", "default='none'", 1)]),
}
for n in ('algorithm', 'model', 'optimize', 'runtime', 'sh_utils', 'sharing_after', 'geometry_trust', 'preflight', 'train_collapse_first', 'train_stage2'): PY['codec/%s.py' % n] = (ST / (n + '.py'), [])
for n in ('appearance', 'codec_benchmark', 'delivery', 'index_context', 'local_codec', 'topology'): PY['codec/codec/%s.py' % n] = (ST / 'codec' / (n + '.py'), [])
CPP = {'codec/codec/edge.cpp': ST / 'codec/edge.cpp', 'codec/codec/index_ctx.cpp': ST / 'codec/index_ctx.cpp', 'codec/codec/native.cpp': ST / 'codec/native.cpp',
       'postprocess/native/native_qem_vis.cpp': VIS / 'native/native_qem_vis.cpp', 'postprocess/native/collapse_policy.h': VIS / 'native/collapse_policy.h', 'postprocess/native/topology_diagnostics.h': VIS / 'native/topology_diagnostics.h'}

def string_stmts(tree):
    out = []
    for node in ast.walk(tree):
        for field in ('body', 'orelse', 'finalbody'):
            body = getattr(node, field, None)
            if isinstance(body, list):
                for s in body:
                    if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str): out.append((s, body))
    return out

def reference_dump(src):
    tree = ast.parse(src)
    for s, body in string_stmts(tree):
        if len(body) == 1: body[0] = ast.Pass()
        else: body.remove(s)
    return ast.dump(tree)

def strip_python(src, name):
    tree = ast.parse(src); lines = src.split('\n'); cuts = []
    for s, body in string_stmts(tree):
        a, b = (s.lineno - 1, s.col_offset), (s.end_lineno - 1, s.end_col_offset)
        before = lines[a[0]].encode()[:a[1]].decode(); after = lines[b[0]].encode()[b[1]:].decode()
        if before.strip() or (after.strip() and not after.strip().startswith('#')): raise SystemExit('%s: string statement shares line %d with code' % (name, s.lineno))
        cuts.append((a[0], b[0], before + 'pass' if len(body) == 1 else None))
    for a, b, rep in sorted(cuts, reverse=True): lines[a:b + 1] = [rep] if rep is not None else []
    src = '\n'.join(lines); lines = src.split('\n'); dead = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT: dead.append((t.start[0] - 1, t.start[1]))
    touched = set()
    for row, col in dead: lines[row] = lines[row][:col].rstrip(); touched.add(row)
    lines = [l.rstrip() for i, l in enumerate(lines) if not (i in touched and not l.strip())]
    out = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip('\n') + '\n'
    return out

def drop_help(src, name):
    """Remove the help= keyword of every call (argparse help texts are explanations, not code)."""
    tree = ast.parse(src); b = src.encode(); starts = [0]
    for line in b.split(b'\n'): starts.append(starts[-1] + len(line) + 1)
    off = lambda ln, col: starts[ln - 1] + col; cuts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            items = sorted(list(node.args) + [k for k in node.keywords], key=lambda x: (x.lineno, x.col_offset) if not isinstance(x, ast.keyword) else (x.value.lineno if x.arg is None else x.lineno, x.col_offset))
            for i, k in enumerate(items):
                if isinstance(k, ast.keyword) and k.arg == 'help':
                    if i == 0: raise SystemExit('%s: help= is the first argument at line %d' % (name, k.lineno))
                    prev = items[i - 1]; cuts.append((off(prev.end_lineno, prev.end_col_offset), off(k.end_lineno, k.end_col_offset)))
    for a, e in sorted(cuts, reverse=True): b = b[:a] + b[e:]
    out = b.decode(); ref = ast.parse(src)
    for node in ast.walk(ref):
        if isinstance(node, ast.Call): node.keywords = [k for k in node.keywords if k.arg != 'help']
    if ast.dump(ast.parse(out)) != ast.dump(ref): raise SystemExit('%s: removing help= changed the code' % name)
    return out

def strip_cpp(src):
    out = []; i = 0; n = len(src)
    while i < n:
        c = src[i]; d = src[i:i + 2]
        if d == '//':
            while i < n and src[i] != '\n': i += 1
        elif d == '/*':
            j = src.find('*/', i + 2); i = n if j < 0 else j + 2; out.append(' ')
        elif c in '"\'':
            j = i + 1
            while j < n and src[j] != c: j += 2 if src[j] == '\\' else 1
            out.append(src[i:j + 1]); i = j + 1
        else: out.append(c); i += 1
    lines = [l.rstrip() for l in ''.join(out).split('\n')]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip('\n') + '\n'

def tokens_cpp(path):
    return subprocess.run(['g++', '-fpreprocessed', '-dD', '-E', '-P', '-x', 'c++', str(path)], capture_output=True, text=True, check=True).stdout.split()

if OUT.exists(): shutil.rmtree(OUT)
for dst, (srcp, edits) in PY.items():
    src = srcp.read_text()
    for old, new, cnt in edits:
        if src.count(old) != cnt: raise SystemExit('%s: expected %d x %r, found %d' % (dst, cnt, old, src.count(old)))
        src = src.replace(old, new)
    src = drop_help(src, dst); new = strip_python(src, dst)
    if re.search(r'(?<![A-Za-z_])help=', new): raise SystemExit('%s: help= left' % dst)
    if ast.dump(ast.parse(new)) != reference_dump(src): raise SystemExit('%s: stripped file is not equivalent' % dst)
    if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(new).readline)) or string_stmts(ast.parse(new)): raise SystemExit('%s: comment left' % dst)
    p = OUT / dst; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(new)
for dst, srcp in CPP.items():
    p = OUT / dst; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(strip_cpp(srcp.read_text()))
    if tokens_cpp(p) != tokens_cpp(srcp): raise SystemExit('%s: stripped file is not equivalent' % dst)
    if '//' in re.sub(r'"(\\.|[^"\\])*"', '""', p.read_text()) or '/*' in p.read_text(): raise SystemExit('%s: comment left' % dst)
shutil.copytree(ST / 'renderer', OUT / 'codec/renderer', ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__', '*.so', '*.pyc'))
for f in sorted((Path(__file__).resolve().parent / 'static').iterdir()): shutil.copy(f, OUT / f.name)
import zipfile
Z = OUT.parent / (OUT.name + '.zip'); Z.unlink(missing_ok=True)
with zipfile.ZipFile(Z, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob('*')):
        if f.is_file():
            info = zipfile.ZipInfo(str(Path(OUT.name) / f.relative_to(OUT)), date_time=(2026, 11, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if f.suffix == '.sh' else 0o644) << 16; z.writestr(info, f.read_bytes())
print('zip %s: %d files, %.2f MB' % (Z.name, len(zipfile.ZipFile(Z).namelist()), Z.stat().st_size / 1e6))
print('python files %d, c++ files %d, renderer files %d' % (len(PY), len(CPP), sum(1 for _ in (OUT / 'codec/renderer').rglob('*') if _.is_file())))
