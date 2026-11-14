"""make_pipeline_drawio_v1.py: the pipeline figure as an editable draw.io file, with the decision rules written as formulas.
Every formula is the one in train_rd_adapt_v9b_abl7.py / rd_growth_v2.py (checked 2026-11-14).  Illustrations are left as labelled placeholders."""
import xml.etree.ElementTree as ET
root = ET.Element('mxfile', host='app.diagrams.net'); dia = ET.SubElement(root, 'diagram', name='pipeline', id='p1')
model = ET.SubElement(dia, 'mxGraphModel', dx='1600', dy='900', grid='1', gridSize='10', guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1', pageWidth='1920', pageHeight='800', math='0', shadow='0')
R = ET.SubElement(model, 'root'); ET.SubElement(R, 'mxCell', id='0'); ET.SubElement(R, 'mxCell', id='1', parent='0'); N = [1]
def cell(value, x, y, w, h, style):
    N[0] += 1; c = ET.SubElement(R, 'mxCell', id=str(N[0]), value=value, style=style, vertex='1', parent='1'); ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h)).set('as', 'geometry'); return str(N[0])
def edge(a, b, style='endArrow=block;html=1;strokeColor=#2a78d6;strokeWidth=1.5;', label=''):
    N[0] += 1; c = ET.SubElement(R, 'mxCell', id=str(N[0]), value=label, style=style, edge='1', parent='1', source=a, target=b); ET.SubElement(c, 'mxGeometry', relative='1').set('as', 'geometry')
PANEL = 'rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor=#f7f7f4;strokeColor=#c9c8c2;verticalAlign=top;align=left;spacingLeft=10;spacingTop=6;fontSize=14;fontStyle=1;'
OURS = 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#e6f0fc;strokeColor=#2a78d6;align=left;verticalAlign=top;spacing=8;fontSize=11;'
UP = 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#e9e8e3;strokeColor=#8a8983;align=left;verticalAlign=top;spacing=8;fontSize=11;'
PLACE = 'rounded=0;whiteSpace=wrap;html=1;dashed=1;fillColor=none;strokeColor=#8a8983;fontColor=#52514e;fontSize=10;'
TXT = 'text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;fontSize=11;fontColor=#52514e;'
B = lambda s: '<b>' + s + '</b>'
# ---------------- A grow
cell('A&nbsp;&nbsp;Budgeted growth &nbsp;<span style="font-weight:normal;font-size:11px;color:#52514e">iterations 0 to 10k, every 500</span>', 20, 40, 600, 520, PANEL)
cell('draw: semi-transparent, unconnected triangles; one being split into 4, one left alone, one faint triangle removed', 34, 76, 170, 110, PLACE)
a1 = cell(B('1&nbsp; Error of triangle t') + '<br>e<sub>t</sub> = mean over training views of &Sigma;<sub>p &isin; P(t)</sub> | I(p) &minus; I<sub>gt</sub>(p) |<br><i>P(t)</i>: pixels where t is the front triangle; <i>I, I<sub>gt</sub></i>: rendered and real image', 216, 76, 390, 96, OURS)
a2 = cell(B('2&nbsp; Size model, from block M (from 4k, every 1k)') + '<br>R(V) = b<sub>v</sub>&middot;V + b<sub>f</sub>&middot;&phi;&middot;V + R<sub>0</sub><br><i>V</i>: vertices; <i>b<sub>v</sub></i>: bytes per vertex and <i>R<sub>0</sub></i>: fixed bytes, both measured on the soup; <i>b<sub>f</sub></i>: bytes per face and <i>&phi;</i>: faces per vertex, fixed priors (the soup has no final connectivity)', 34, 200, 572, 96, OURS)
a3 = cell(B('3&nbsp; How many to split') + '<br>V* = (1.35&middot;T &minus; R<sub>0</sub>) / (b<sub>v</sub> + &phi;&middot;b<sub>f</sub>)<br>n = V&middot;[ (V*/V)<sup>1/k</sup> &minus; 1 ] / (3&sigma;)<br><i>T</i>: target file size; <i>k</i>: growth rounds left; <i>&sigma;</i>: share of new vertices that survive MeshSplatting&rsquo;s next clean-up (measured); 3: a split adds up to 3 vertices', 34, 310, 572, 112, OURS)
a4 = cell(B('4&nbsp; Which to split') + '<br>the n triangles with the largest e<sub>t</sub> (not smaller than MeshSplatting&rsquo;s size limits); all others stay as they are', 34, 436, 330, 76, OURS)
cell(B('MeshSplatting, unchanged') + '<br>removes faint or oversized triangles every 500 iterations; picks triangles by sampling &prop; importance<sub>t</sub> = largest blending weight (replaced by step 4)', 376, 436, 230, 110, UP)
cell('no collapse in this phase: the soup has no shared edges', 34, 522, 330, 30, TXT)
edge(a1, a2); edge(a2, a3); edge(a3, a4)
# ---------------- B connect
cell('B&nbsp;&nbsp;Connect', 640, 40, 170, 520, PANEL.replace('#f7f7f4', '#e9e8e3'))
cell('draw: the same triangles as one connected mesh', 654, 76, 142, 110, PLACE)
cell('restricted Delaunay triangulation at 11k (MeshSplatting, unchanged): from now on triangles share vertices and edges', 654, 200, 142, 150, TXT)
# ---------------- C hold
cell('C&nbsp;&nbsp;Rate-locked refinement &nbsp;<span style="font-weight:normal;font-size:11px;color:#52514e">12.5k to 29.5k, every 500 iterations, 35 rounds; nothing is grown</span>', 830, 40, 640, 520, PANEL)
c1 = cell(B('1&nbsp; Excess') + '<br>E = R &minus; T<br><i>R</i>: bytes of the whole file, measured by block M; if E &le; 0 nothing is removed this round', 844, 76, 300, 96, OURS)
cell('draw: an edge (u,v) between two triangles collapsing to one vertex, before and after', 1156, 76, 300, 96, PLACE)
c2 = cell(B('2&nbsp; Cost of collapsing edge (u,v)') + '<br>c<sub>uv</sub> = &Sigma;<sub>f touching u or v</sub> e<sub>f</sub> &nbsp;/&nbsp; &Delta;R<sub>uv</sub><br>&Delta;R<sub>uv</sub> = b<sub>v</sub>&middot;[u and v are both in the file] + b<sub>f</sub>&middot;(faces on the edge that are in the file)<br><i>e<sub>f</sub></i>: error of face f as in A1; &ldquo;in the file&rdquo;: kept by block M', 844, 186, 612, 112, OURS)
c3 = cell(B('3&nbsp; Which edges') + '<br>take edges in order of increasing c<sub>uv</sub>, skipping an edge if it shares a vertex with one already taken or if collapsing it flips a face normal, until &Sigma; &Delta;R<sub>uv</sub> &ge; E', 844, 312, 612, 80, OURS)
c4 = cell(B('4&nbsp; Collapse and continue') + '<br>u and v merge at their midpoint, colours averaged; then 500 training iterations', 844, 406, 400, 66, OURS)
cell('the mesh stays at the target size while it trains', 844, 486, 400, 30, TXT)
edge(c1, c2); edge(c2, c3); edge(c3, c4); edge(c4, c1, 'endArrow=block;html=1;strokeColor=#2a78d6;strokeWidth=1.5;edgeStyle=orthogonalEdgeStyle;exitX=1;exitY=0.5;entryX=1;entryY=0.5;', 'next round')
# ---------------- M
m = cell('M&nbsp;&nbsp;Measure: how large would the file be if training stopped now? &nbsp;<span style="font-weight:normal;font-size:11px;color:#52514e">used by A2 and C1</span>', 20, 580, 1450, 150, PANEL.replace('strokeColor=#c9c8c2', 'strokeColor=#2a78d6;dashed=1'))
m1 = cell(B('a') + '&nbsp; copy the mesh (training is not touched)', 34, 616, 200, 60, OURS)
m2 = cell(B('b') + '&nbsp; render the copy at its final opacity from every training view; w<sub>f</sub> = largest blending weight of face f', 264, 616, 320, 60, OURS)
m3 = cell(B('c') + '&nbsp; keep faces with w<sub>f</sub> &gt; 0.5, drop unused vertices (MeshSplatting&rsquo;s final clean-up)', 614, 616, 300, 60, OURS)
m4 = cell(B('d') + '&nbsp; run the real encoder (the one in D) on what is kept', 944, 616, 260, 60, OURS)
m5 = cell(B('R') + ' = file size in bytes;<br>also b<sub>v</sub>, b<sub>f</sub>, R<sub>0</sub>', 1234, 616, 220, 60, OURS)
edge(m1, m2); edge(m2, m3); edge(m3, m4); edge(m4, m5)
cell('in A the faces are not yet connected, so only b<sub>v</sub> and R<sub>0</sub> are taken from the measurement', 34, 686, 700, 30, TXT)
# ---------------- D encode, E decode
cell('D&nbsp;&nbsp;Encode &nbsp;<span style="font-weight:normal;font-size:11px;color:#52514e">after training</span>', 1490, 40, 270, 690, PANEL)
d1 = cell(B('Shared colour codebook') + '<br>main: 4,096 entries<br>residual: 256 entries<br>each vertex stores a base colour and two indices', 1504, 76, 242, 100, OURS)
d2 = cell(B('Encoder') + ' (one lossless coder per stream; share of the file, garden 10 MB)<br>connectivity 39%<br>positions 21%<br>colour indices 19%<br>base colour 15%<br>codebooks 7%', 1504, 200, 242, 150, OURS)
d3 = cell('one file', 1574, 380, 100, 40, 'rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#0b0b0b;fontSize=11;')
edge(d1, d2, 'endArrow=block;html=1;strokeWidth=1.5;'); edge(d2, d3, 'endArrow=block;html=1;strokeWidth=1.5;')
cell('draw the encoder as a trapezoid with four bands whose widths follow the shares', 1504, 440, 242, 60, PLACE)
cell('E&nbsp;&nbsp;Decode', 1780, 40, 130, 690, PANEL)
e1 = cell(B('Decoder') + '<br>needs only the file; CPU', 1790, 200, 110, 70, OURS)
cell('draw: opaque mesh', 1790, 300, 110, 80, PLACE)
edge(d3, e1, 'endArrow=block;html=1;strokeWidth=1.5;')
# ---------------- timeline
cell('training iterations: &nbsp; 0 &mdash; A &mdash; 10k &nbsp;|&nbsp; 11k connect &nbsp;|&nbsp; 12.5k &mdash; C &mdash; 29.5k &nbsp;|&nbsp; 30k end. &nbsp; Opacity of the triangles rises from semi-transparent (5k) to fully opaque (24k).', 20, 745, 1450, 30, TXT)
cell('<span style="background-color:#e6f0fc">&nbsp;ours&nbsp;</span> &nbsp; <span style="background-color:#e9e8e3">&nbsp;as in MeshSplatting&nbsp;</span> &nbsp; dashed boxes: illustrations to draw', 20, 10, 700, 24, TXT)
ET.ElementTree(root).write('paper/pipeline-sketch/pipeline-v3.drawio', encoding='utf-8', xml_declaration=True)
ET.parse('paper/pipeline-sketch/pipeline-v3.drawio'); print('ok cells', N[0])
