# batch40 v2: codec design choices (run 2026-11-15 02:30-02:52 EST, host cs604, CPU, 11 scenes, 0 failures, round trips exact)

Meshes: MeshSplatting with a vertex cap near 1.08 million, fine-tuned (same as batch39).  bits per vertex (b/v).
"shipped" = the stream in the file.  The four position predictors and the "no context" index coder use the test coder of codec_choices_v2.py.

## Positions (b/v)

| scene | shipped (first neighbour, shipped coder) | prev | first | mean | para | mean vs first | file change with mean |
|---|---|---|---|---|---|---|---|
| garden | 20.58 | 21.53 | 21.03 | 19.91 | 21.97 | -5.3 % | -1.21 % |
| room | 17.98 | 18.90 | 18.39 | 17.37 | 19.38 | -5.6 % | -1.20 % |
| truck | 26.59 | 28.08 | 27.47 | 26.57 | 28.23 | -3.3 % | -1.00 % |
| kitchen | 22.94 | 24.06 | 23.53 | 22.46 | 24.52 | -4.5 % | -1.09 % |
| counter | 21.84 | 22.91 | 22.38 | 21.39 | 23.33 | -4.5 % | -1.10 % |
| bonsai | 21.84 | 22.89 | 22.26 | 21.27 | 23.14 | -4.5 % | -1.13 % |
| bicycle | 22.87 | 24.04 | 23.43 | 22.48 | 24.15 | -4.0 % | -1.12 % |
| stump | 21.47 | 22.43 | 21.84 | 20.87 | 22.45 | -4.4 % | -1.16 % |
| flowers | 21.32 | 22.56 | 21.94 | 21.04 | 22.64 | -4.1 % | -1.12 % |
| treehill | 21.35 | 22.52 | 21.94 | 20.97 | 22.59 | -4.4 % | -1.17 % |
| train | 23.77 | 25.19 | 24.64 | 23.67 | 25.50 | -4.0 % | -1.08 % |
| **mean** | 22.05 | 23.19 | 22.62 | 21.64 | 23.45 | **-4.4 %** | **-1.13 %** |

Order on all 11 scenes: mean < first < prev < para.  Parallelogram is available for 81-91 % of the vertices and is the worst.
The shipped coder is 2.5 % smaller than the test coder with the same predictor (22.05 vs 22.62), so the gain of 'mean' is quoted relative to 'first' in the same coder.

## SH indices (b/v)

| scene | main: shipped (neighbour context) | main: no context | residual: shipped | residual: no context | context saves (index streams) | (file) |
|---|---|---|---|---|---|---|
| garden | 11.25 | 10.57 | 7.61 | 7.13 | -6.5 % | -1.25 % |
| room | 9.72 | 9.90 | 7.27 | 7.08 | -0.1 % | -0.02 % |
| truck | 10.29 | 10.75 | 7.31 | 7.21 | +2.0 % | +0.40 % |
| kitchen | 9.64 | 9.94 | 7.24 | 7.02 | +0.5 % | +0.08 % |
| counter | 9.71 | 10.16 | 7.30 | 7.09 | +1.4 % | +0.26 % |
| bonsai | 9.99 | 9.96 | 7.23 | 6.87 | -2.3 % | -0.44 % |
| bicycle | 10.79 | 10.55 | 7.26 | 6.82 | -3.9 % | -0.80 % |
| stump | 11.04 | 10.83 | 7.47 | 7.05 | -3.5 % | -0.74 % |
| flowers | 10.73 | 10.51 | 7.19 | 6.73 | -4.0 % | -0.85 % |
| treehill | 10.47 | 10.46 | 7.40 | 7.05 | -2.0 % | -0.43 % |
| train | 9.49 | 10.28 | 7.44 | 7.34 | +3.9 % | +0.76 % |
| **mean** | 10.28 | 10.36 | 7.34 | 7.04 | **-1.3 %** | **-0.28 %** |

Fixed-length codes: 12 and 8 bits.  Positive = the neighbour context makes the streams smaller.
Reading: the neighbour context does not help on average (better on 4 scenes, worse on 7); on the residual index it is worse on all 11.
The saving against fixed-length codes (12 -> 10.3, 8 -> 7.0-7.3) comes from the uneven use of the codewords, which a model without context already captures.

## DC colour: see dc-predict-2026-11-15.md (identical numbers in this run): DC -10.4 %, file -1.46 %.

## Sum: with mean prediction for positions and DC and no neighbour context for the indices, the file would be about 2.9 % smaller
(1.46 + 1.13 + 0.28).  None of this is in the reported file sizes.
