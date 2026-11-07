# report-latex-v2 (2026-11-03)

Changes vs v1:
- Byte prices: two independent constants only. B_a = 6 B = attribute bytes of one vertex (position residual, DC, SH indices); B_f = 2 B = connectivity bytes of one face. B_v is gone. All rate formulas are sums: split dR_t = 3B_a + 6B_f = 30 B; collapse dR_uv = B_a + n_uv B_f = 10 B (interior edge). Same numbers the code uses (BPV=10, BPF=2).
- Collapse distortion written as a sum over the union of the two vertex rings (no D_u + D_v - D_uv subtraction).
- §1.2 rewritten: Distortion / Rate / The two rules / Why fixed prices; new Fig. 2 (figures/fig-split-prune-v2.pdf) works through one split and one collapse.
- TikZ Fig. 1: split, prune and byte-price boxes use B_a / B_f.
- Cross-method table split into two upright tables (tab:cross-mesh, tab:cross-3dgs); rotating package dropped.
- Author/date lines removed (as in your Overleaf copy).

Files: main.tex, main.pdf (compiled, 7 pages, no errors or overfull boxes), figures/*.pdf.
