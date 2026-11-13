"""make_cap_onefig_v3.py <out.tex>: one figure, all 21 scenes, file size against vertex cap on LINEAR axes, from the smallest caps to the
released default (4 million).  The default-cap files are the level-1.0 deliveries of the original models (same fine-tuning and codec as the
cap runs): release-batch22-v1/results/generic-codecs-all-scenes-2026-11-10.md ('ours MB') and the ORIGINAL-INDOOR rows for Deep Blending."""
import re, collections, glob, sys
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)')
cap = collections.defaultdict(dict)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = C.match(l)
        if m: cap[m[1]].setdefault(int(m[2]), float(m[4]))
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
D4 = {}
for l in open('release-batch22-v1/results/generic-codecs-all-scenes-2026-11-10.md'):
    c = [x.strip() for x in l.split('|')]
    if len(c) > 4 and c[1] in MIP: D4[c[1]] = float(re.match(r'[\d.]+', c[3]).group())
for f in glob.glob('release-*/results/*.txt') + glob.glob('analysis-snapshot/rd/*.txt'):
    for l in open(f, errors='ignore'):
        m = re.match(r'(drjohnson|playroom) ORIGINAL-INDOOR \| vertices \d+ faces \d+ \|.*?delivered ([\d.]+) MB', l)
        if m: D4[m[1]] = float(m[2])
assert len(D4) == 13, D4
print('default cap:', sorted(D4.items(), key=lambda kv: kv[1])); print('range %.1f .. %.1f MB, ratio %.2f' % (min(D4.values()), max(D4.values()), max(D4.values()) / min(D4.values())))
def plots(scenes, col):
    out = []
    for s in scenes:
        p = sorted(cap[s].items()); out.append('\\addplot[ds, %s, mark options={fill=%s, draw=white, line width=0.2pt}, forget plot] coordinates {%s}; %% %s' % (col, col, ' '.join('(%.4f,%.3f)' % (c / 1e6, mb) for c, mb in p), s))
        if s in D4: out.append('\\addplot[ds, %s, dash pattern=on 2.5pt off 2pt, mark options={solid, fill=%s, draw=white, line width=0.2pt}, forget plot] coordinates {(%.4f,%.3f) (4.0000,%.3f)}; %% %s, 3 million to the default' % (col, col, p[-1][0] / 1e6, p[-1][1], D4[s], s))
    return '\n'.join(out)
fig = r'''\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[width=\textwidth, height=66mm, xmin=0, xmax=4.62, ymin=0, ymax=43, xtick={0,1,2,3,4}, ytick={0,10,20,30,40},
  axis lines=left, axis line style={gridc}, tick style={gridc}, tick align=outside, tick label style={font=\scriptsize, text=muted},
  label style={font=\scriptsize, text=muted}, ymajorgrids, grid style={gridc, line width=0.3pt}, clip=true, clip marker paths=true,
  xlabel={vertex cap (millions)}, ylabel={file size (MB)},
  legend style={font=\scriptsize, text=ink, draw=none, fill=none, at={(0.27,0.985)}, anchor=north west, row sep=-1pt}, legend cell align=left]
\path[fill=otherc, fill opacity=0.28] (axis cs:0.585,0) rectangle (axis cs:0.615,43);
\path[fill=otherc, fill opacity=0.28] (axis cs:1.062,0) rectangle (axis cs:1.093,43);
\path[fill=otherc, fill opacity=0.28] (axis cs:2.985,0) rectangle (axis cs:3.015,43);
\path[fill=otherc, fill opacity=0.28] (axis cs:3.985,0) rectangle (axis cs:4.015,43);
\addlegendimage{ds, cA, mark options={fill=cA, draw=white, line width=0.2pt}}\addlegendentry{Mip-NeRF 360 and Tanks and Temples, 11 scenes}
\addlegendimage{ds, cB, mark options={fill=cB, draw=white, line width=0.2pt}}\addlegendentry{Deep Blending, 2 scenes}
\addlegendimage{ds, cC, mark options={fill=cC, draw=white, line width=0.2pt}}\addlegendentry{NeRF-Synthetic objects, 8 scenes}
''' + plots(MIP, 'cA') + '\n' + plots(DB, 'cB') + '\n' + plots(OBJ, 'cC') + r'''
\node[font=\scriptsize, text=ink, anchor=south east, align=right] at (axis cs:0.575,8.6) {cap 0.6 million:\\5.8 to 7.5 MB};
\node[font=\scriptsize, text=ink, anchor=south east, align=right] at (axis cs:1.05,15.5) {cap 1.08 million:\\8.3 to 11.6 MB};
\node[font=\scriptsize, text=ink, anchor=north east, align=right] at (axis cs:2.97,15.5) {cap 3 million:\\19.9 to 31.6 MB};
\node[font=\scriptsize, text=ink, anchor=north east, align=right] at (axis cs:3.97,15.5) {released default,\\cap 4 million:\\20.0 to 39.8 MB};
\node[font=\scriptsize, text=ink, anchor=west] at (axis cs:4.04,39.8) {drjohnson};
\node[font=\scriptsize, text=ink, anchor=west] at (axis cs:4.04,37.3) {playroom};
\node[font=\scriptsize, text=ink, anchor=west] at (axis cs:4.04,32.5) {garden};
\node[font=\scriptsize, text=ink, anchor=west] at (axis cs:4.04,20.0) {truck};
\node[font=\scriptsize, text=ink, anchor=west] at (axis cs:0.60,2.4) {8 objects: they flatten between 1.5 and 5.9 MB};
\end{axis}
\end{tikzpicture}
\caption{CAPTION}\label{fig:cap}
\end{figure}'''
fig = fig.replace('CAPTION', r"File size (with our codec) against vertex cap, all 21 scenes on the same axes, one line per scene, every cap run we have. Dashed: from the cap of 3 million to the released default, with no runs in between. Dark vertical strips: caps at which all 13 captured scenes were trained.")
open(sys.argv[1], 'w').write(fig); print('written')
