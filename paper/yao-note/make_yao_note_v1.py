"""make_yao_note_v1.py: writes paper/yao-note/yao-note-2026-11-13.tex (self-contained: the figure is drawn with pgfplots from the cap runs
in release-*/results/*.txt; no image file)."""
import re, collections, glob
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)')
cap = collections.defaultdict(dict)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = C.match(l)
        if m: cap[m[1]].setdefault(int(m[2]), float(m[4]))
CAPT = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train','drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
HI = {'garden': 'sA', 'bonsai': 'sB', 'playroom': 'sC', 'lego': 'sA', 'mic': 'sB', 'ficus': 'sC'}
def plots(scenes):
    out = []
    for s in sorted(scenes, key=lambda s: s in HI):   # highlighted scenes drawn last, on top
        co = ' '.join('(%.3f,%.3f)' % (c / 1000, mb) for c, mb in sorted(cap[s].items()))
        out.append('\\addplot[%s] coordinates {%s}; %% %s' % (('hi,' + HI[s]) if s in HI else 'other', co, s))
    return '\n'.join(out)
def band(scenes, lo, hi): v = [mb for s in scenes for c, mb in cap[s].items() if lo <= c <= hi]; n = len({s for s in scenes if any(lo <= c <= hi for c in cap[s])}); return n, min(v), max(v)
print('captured band', band(CAPT, 1069000, 1086000), 'objects band', band(OBJ, 170000, 183000))
import subprocess, sys; subprocess.run([sys.executable, 'paper/yao-note/make_cap_onefig_v2.py', 'paper/yao-note/cap-onefig-block.tex'], check=True)
tex = open('paper/yao-note/yao-note-template.tex').read().replace('%%CAPFIG%%', open('paper/yao-note/cap-onefig-block.tex').read())
open('paper/yao-note/yao-note-2026-11-13.tex', 'w').write(tex); print('written')
