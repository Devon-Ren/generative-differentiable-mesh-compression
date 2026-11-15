# batch40: neighbour prediction on the DC colour stream (run 2026-11-15, host cs604, CPU, 11 scenes, 0 failures, all round trips exact)

Same meshes as batch39 (MeshSplatting with a vertex cap near 1.08 million, fine-tuned), DC colour at 7 bits.
Shipped: zstd, no prediction.  Tested: prediction from earlier-decoded neighbours + context-adaptive binary arithmetic coding (real bitstream).

| scene | file (MB) | DC zstd (MB, bits/vertex) | first neighbour (bits/vertex) | mean of neighbours (bits/vertex) | DC change | file change |
|---|---|---|---|---|---|---|
| garden | 11.641 | 1.833, 14.58 | 14.07 | 13.38 | -8.2 % | -1.29 % |
| room | 10.671 | 1.264, 10.13 | 9.41 | 8.87 | -12.5 % | -1.48 % |
| truck | 9.951 | 1.484, 13.41 | 12.30 | 11.72 | -12.6 % | -1.87 % |
| kitchen | 11.634 | 1.408, 11.77 | 10.69 | 10.03 | -14.8 % | -1.79 % |
| counter | 9.311 | 1.192, 11.67 | 10.44 | 9.85 | -15.6 % | -2.00 % |
| bonsai | 8.315 | 1.087, 11.53 | 11.04 | 10.45 | -9.4 % | -1.23 % |
| bicycle | 10.042 | 1.474, 12.45 | 12.16 | 11.63 | -6.6 % | -0.97 % |
| stump | 10.212 | 1.587, 13.02 | 12.41 | 11.84 | -9.0 % | -1.40 % |
| flowers | 9.416 | 1.534, 13.18 | 12.68 | 12.14 | -7.9 % | -1.29 % |
| treehill | 10.596 | 1.647, 12.87 | 12.73 | 12.18 | -5.4 % | -0.84 % |
| train | 10.357 | 1.631, 14.22 | 13.12 | 12.46 | -12.4 % | -1.95 % |
| **mean** | | 12.62 bits/vertex | | 11.32 bits/vertex | **-10.4 %** | **-1.46 %** (-0.84 to -2.00) |

"DC change" and "file change" use the better predictor (mean of earlier neighbours, on all 11 scenes).
The shipped codec is unchanged; no reported file size includes this.
