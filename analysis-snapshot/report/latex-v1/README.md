# report-latex-v1 (2026-11-02)

Technical report "Generative Compression for Differentiable Opaque Meshes" as an Overleaf-ready LaTeX project.

- `main.tex` — the whole report (methodology + experiments). Compile with pdflatex; no bibliography file needed.
- `figures/fig-codecs-room-v2.pdf` — Fig. 2 (same Room asset, generic codecs vs ours; 10/24 Room asset, no --indoor)
- `figures/fig-mesh-class-v2.pdf` — Fig. 3 (engine-ready mesh class, Mip-360 means, indoor = --indoor rerun)
- `figures/fig-gauss-class-v2.pdf` — Fig. 4 (3DGS-class codecs, absolute + retained quality)
- `figures/fig-in-the-loop-v1.pdf` — Fig. 5 (in-the-loop ablation, 7 scenes, older parameter set)
- Fig. 1 (pipeline schematic) is TikZ inside main.tex; edit the node texts directly.

Numbers come from analysis/report/tables-v2.md, analysis/report/report-v4.md and analysis/rd/rd-11scenes-v3.csv.
Sources for the figures: analysis/report/fig_*_v2.py / fig_loop_v1.py with figstyle_v2.py.
