# report-latex-v5 (2026-11-06 morning)

v4 + the overnight batch (node-batch2-v1, 16 jobs on gl030):
- Section 2.7 (sec:fan) now covers all 11 scenes: Table 6 = per-scene summary (joint-hull gain over the lambda-only hull, hull membership of K1024 / R1024-K16384 / DC 8 bit, mean lever prices); the full 44-row table (Table 10) and the 11-panel figure (Fig. 11) moved to Appendix B. Readings rewritten with the 11-scene numbers (within 0.16 dB everywhere; K1024 on the hull at the two low lambdas on 11/11; larger banks at the top on 11/11; DC / dictionary matter indoors, not outdoors).
- New Section 2.9 (sec:adapt): lambda rate controller, first run (Table 8, Fig. 8): quality on the fixed-lambda curve (mean +0.005 dB), size 52-77 % of target, decomposed into 1.12 x 0.87 x 0.85 x 0.77 (phase A, phase B, post-control shrink, double-counted connectivity bytes in the projection).
- New Section 2.10 (sec:basens): price-constant sensitivity (Table 9, Fig. 9): B_v 8 / 13 B points lie on the B_v 10 curve (-0.01..+0.07 dB), lambda_eff/lambda 0.78 / 1.31 vs predicted 0.80 / 1.30.
- "Choosing lambda" paragraph points to sec:adapt; "What is still open" updated (fan item closed, controller rerun item added); sec:headline label added.
Source data: analysis/rd/fan-analysis-v2.txt, adapt-v1-analysis.txt, ba-sens-v1-analysis.txt. Compile: pdflatex main.tex (twice). 16 pages, 0 errors.
