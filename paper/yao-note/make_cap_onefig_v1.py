"""make_cap_onefig_v1.py <out.tex>: one figure, all 21 scenes, file size against vertex cap on log axes (pgfplots block, no image file)."""
import re, collections, glob, sys
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)')
cap = collections.defaultdict(dict)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = C.match(l)
        if m: cap[m[1]].setdefault(int(m[2]), float(m[4]))
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']
def plots(scenes, sty): return '\n'.join('\\addplot[%s] coordinates {%s}; %% %s' % (sty, ' '.join('(%.3f,%.3f)' % (c / 1000, mb) for c, mb in sorted(cap[s].items())), s) for s in scenes)
def rng(scenes, lo, hi): v = [mb for s in scenes for c, mb in cap[s].items() if lo <= c <= hi]; return len({s for s in scenes if any(lo <= c <= hi for c in cap[s])}), min(v), max(v)
print('176k', rng(OBJ, 170000, 183000), '600k', rng(MIP + DB, 600000, 600000), '1.08M', rng(MIP + DB, 1069000, 1086000), 'runs', sum(len(v) for v in cap.values()))
fig = r'''\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[width=\textwidth, height=88mm, xmode=log, ymode=log, log ticks with fixed point,
  xmin=30, xmax=3300, ymin=0.9, ymax=26, xtick={50,100,200,500,1000,2000}, ytick={1,2,5,10,20},
  xticklabels={50,100,200,500,{1,000},{2,000}}, yticklabels={1,2,5,10,20}, minor tick style={draw=none},
  axis lines=left, axis line style={gridc}, tick style={gridc}, tick align=outside, tick label style={font=\scriptsize, text=muted},
  label style={font=\scriptsize, text=muted}, ymajorgrids, grid style={gridc, line width=0.3pt}, clip=true, clip marker paths=true,
  xlabel={vertex cap (thousands, log scale)}, ylabel={file size (MB, log scale)},
  legend style={font=\scriptsize, text=ink, draw=none, fill=none, at={(0.015,0.985)}, anchor=north west, row sep=-1pt}, legend cell align=left]
\path[fill=otherc, fill opacity=0.28] (axis cs:170,0.9) rectangle (axis cs:183,26);
\path[fill=otherc, fill opacity=0.28] (axis cs:590,0.9) rectangle (axis cs:610,26);
\path[fill=otherc, fill opacity=0.28] (axis cs:1069,0.9) rectangle (axis cs:1086,26);
\addlegendimage{ds, cA, mark options={fill=cA, draw=white, line width=0.2pt}}\addlegendentry{Mip-NeRF 360 and Tanks and Temples, 11 scenes}
\addlegendimage{ds, cB, mark options={fill=cB, draw=white, line width=0.2pt}}\addlegendentry{Deep Blending, 2 scenes}
\addlegendimage{ds, cC, mark options={fill=cC, draw=white, line width=0.2pt}}\addlegendentry{NeRF-Synthetic objects, 8 scenes}
''' + plots(MIP, 'ds, cA, mark options={fill=cA, draw=white, line width=0.2pt}, forget plot') + '\n' + plots(DB, 'ds, cB, mark options={fill=cB, draw=white, line width=0.2pt}, forget plot') + '\n' + plots(OBJ, 'ds, cC, mark options={fill=cC, draw=white, line width=0.2pt}, forget plot') + r'''
\node[font=\scriptsize, text=ink, anchor=south west, align=left] at (axis cs:187,5.6) {cap 176 thousand,\\8 objects:\\1.5 to 2.8 MB};
\node[font=\scriptsize, text=ink, anchor=north west, align=left] at (axis cs:618,3.4) {cap 600,000,\\13 scenes:\\5.8 to 7.5 MB};
\node[font=\scriptsize, text=ink, anchor=north west, align=left] at (axis cs:1100,5.2) {cap 1.08 million,\\13 scenes:\\8.3 to 11.6 MB};
\node[font=\scriptsize, text=ink, anchor=west] at (axis cs:530,1.50) {ficus};
\node[font=\scriptsize, text=ink, anchor=west] at (axis cs:255,2.05) {mic};
\node[font=\scriptsize, text=ink, anchor=south east] at (axis cs:540,6.0) {lego};
\node[font=\scriptsize, text=ink, anchor=south east] at (axis cs:1100,13.4) {garden};
\end{axis}
\end{tikzpicture}
\caption{File size (with our codec) against vertex cap for all 21 scenes, one line per scene, every cap run we have. Both axes are logarithmic. Dark vertical strips: caps at which a whole group of scenes was trained.}\label{fig:cap}
\end{figure}'''
open(sys.argv[1], 'w').write(fig); print('written', sys.argv[1])
