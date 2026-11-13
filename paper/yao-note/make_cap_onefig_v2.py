"""make_cap_onefig_v2.py <out.tex>: one figure, all 21 scenes: how far each cap run is from the best single cap-to-size rule
(least squares in log space over all runs: MB = k * cap^b).  pgfplots block, no image file."""
import re, collections, glob, sys, math, numpy as np
C = re.compile(r'^(\w+) budget (\d+) TRAIN-AT-BUDGET(?:-BLENDER)? \| vertices (\d+) .*?\| delivered ([\d.]+) MB test ([\d.]+)')
cap = collections.defaultdict(dict)
for f in sorted(glob.glob('release-*/results/*.txt')):
    for l in open(f, errors='ignore'):
        m = C.match(l)
        if m: cap[m[1]].setdefault(int(m[2]), float(m[4]))
MIP = ['room','kitchen','counter','bonsai','garden','bicycle','stump','flowers','treehill','truck','train']; DB = ['drjohnson','playroom']; OBJ = ['lego','chair','drums','ficus','hotdog','materials','mic','ship']; ALL = MIP + DB + OBJ
x = np.array([math.log(c) for s in ALL for c in cap[s]]); y = np.array([math.log(cap[s][c]) for s in ALL for c in cap[s]]); b, a = np.polyfit(x, y, 1)
dev = lambda c, mb: 100 * (mb / math.exp(a + b * math.log(c)) - 1)
E = [dev(c, mb) for s in ALL for c, mb in cap[s].items()]
print('rule: MB = %.4f * (cap in thousands)^%.3f | runs %d | within 10%%: %d | range %+.0f .. %+.0f' % (math.exp(a + b * math.log(1000)), b, len(E), sum(abs(v) <= 10 for v in E), min(E), max(E)))
for s in ('ficus', 'mic', 'garden', 'bonsai', 'lego', 'drjohnson', 'playroom'): print(s, [(c // 1000, round(dev(c, mb))) for c, mb in sorted(cap[s].items())])
def plots(scenes, col): return '\n'.join('\\addplot[ds, %s, mark options={fill=%s, draw=white, line width=0.2pt}, forget plot] coordinates {%s}; %% %s' % (col, col, ' '.join('(%.3f,%.1f)' % (c / 1000, dev(c, mb)) for c, mb in sorted(cap[s].items())), s) for s in scenes)
fig = r'''\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[width=\textwidth, height=68mm, xmode=log, log ticks with fixed point,
  xmin=30, xmax=3300, ymin=-82, ymax=48, xtick={50,100,200,500,1000,2000}, ytick={-80,-60,-40,-20,0,20,40},
  xticklabels={50,100,200,500,{1,000},{2,000}}, yticklabels={$-$80\%,$-$60\%,$-$40\%,$-$20\%,0,+20\%,+40\%}, minor tick style={draw=none},
  axis lines=left, axis line style={gridc}, tick style={gridc}, tick align=outside, tick label style={font=\scriptsize, text=muted},
  label style={font=\scriptsize, text=muted}, ymajorgrids, grid style={gridc, line width=0.3pt}, clip=true, clip marker paths=true,
  xlabel={vertex cap (thousands, log scale)}, ylabel={file size, relative to the common rule},
  legend style={font=\scriptsize, text=ink, draw=none, fill=none, at={(0.015,0.03)}, anchor=south west, row sep=-1pt}, legend cell align=left]
\path[fill=bandc] (axis cs:30,-10) rectangle (axis cs:3300,10);
\draw[muted, line width=0.5pt, dash pattern=on 3pt off 2pt] (axis cs:30,0) -- (axis cs:3300,0);
\addlegendimage{ds, cA, mark options={fill=cA, draw=white, line width=0.2pt}}\addlegendentry{Mip-NeRF 360 and Tanks and Temples, 11 scenes}
\addlegendimage{ds, cB, mark options={fill=cB, draw=white, line width=0.2pt}}\addlegendentry{Deep Blending, 2 scenes}
\addlegendimage{ds, cC, mark options={fill=cC, draw=white, line width=0.2pt}}\addlegendentry{NeRF-Synthetic objects, 8 scenes}
''' + plots(MIP, 'cA') + '\n' + plots(DB, 'cB') + '\n' + plots(OBJ, 'cC') + r'''
\node[font=\scriptsize, text=muted, anchor=south east] at (axis cs:3200,10) {$\pm$10\%};
\node[font=\scriptsize, text=ink, anchor=west] at (axis cs:535,-75) {ficus};
\node[font=\scriptsize, text=ink, anchor=west] at (axis cs:256,-40) {mic};
\end{axis}
\end{tikzpicture}
\caption{Every cap run of all 21 scenes on one pair of axes. The common rule is the best single curve that turns a cap into a file size for all scenes together (file in MB = 0.0588 $\times$ cap in thousands to the power 0.738, fitted on these same runs). If the cap decided the file size, every point would sit on the dashed line. 114 of the 178 runs are within $\pm$10\% of it (light band); the others are between $-$75\% and +36\%.}\label{fig:cap}
\end{figure}'''
open(sys.argv[1], 'w').write(fig); print('written')
