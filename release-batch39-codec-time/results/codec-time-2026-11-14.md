# batch39: encode / decode time (measured 2026-11-14, host cs611, Intel Xeon Platinum 8592+, 8 threads requested, CPU only)

Same mesh for every codec: MeshSplatting trained with a vertex cap near 1.08 million, fine-tuned.  11 Mip-NeRF 360 / T&T scenes, 0 failures.
Generic codecs at 8-bit SH.  Our round trip was exact on all 11 scenes.

| scene | vertices | ours encode | ours decode | zstd enc | zstd dec | Draco enc | Draco dec | TFAN enc+dec |
|---|---|---|---|---|---|---|---|---|
| garden | 1,005,902 | 5.31 | 4.15 | 11.68 | 0.23 | 20.00 | 4.50 | 9.74 |
| room | 997,819 | 5.45 | 4.23 | 17.50 | 0.22 | 19.64 | 4.57 | 9.75 |
| truck | 885,368 | 4.19 | 3.27 | 15.95 | 0.19 | 14.36 | 3.35 | 7.77 |
| kitchen | 956,939 | 5.63 | 4.38 | 12.38 | 0.23 | 20.98 | 4.85 | 9.58 |
| counter | 817,438 | 4.38 | 3.49 | 17.18 | 0.18 | 15.90 | 3.64 | 7.51 |
| bonsai | 754,307 | 3.78 | 2.99 | 16.02 | 0.16 | 13.70 | 3.15 | 6.57 |
| bicycle | 947,525 | 4.40 | 3.57 | 16.63 | 0.20 | 15.43 | 3.59 | 7.79 |
| stump | 975,273 | 4.51 | 3.57 | 14.45 | 0.20 | 16.11 | 3.58 | 7.66 |
| flowers | 930,919 | 4.03 | 3.23 | 13.61 | 0.18 | 13.71 | 3.12 | 6.96 |
| treehill | 1,023,422 | 4.69 | 3.72 | 15.96 | 0.22 | 16.34 | 3.81 | 8.29 |
| train | 917,466 | 4.57 | 3.57 | 17.69 | 0.22 | 16.19 | 3.78 | 8.31 |
| **mean** | 928,398 | **4.63** | **3.65** | 15.37 | 0.20 | 16.58 | 3.81 | 8.18 |

Seconds.  Ours: encode + decode = 8.28 s; de-quantisation adds 0.04 s; compiling the native coders once per machine took 6.4 to 7.5 s and is not included.
Our encode starts from the shared representation (codebooks already built) and ends with the file.  TFAN: one driver call that encodes and decodes.
zstd -19 used all cores of the node; Draco is single-threaded; threading of our native coders was not checked.

## After training: fine-tuning, codebook, encoding (file time stamps, 13 captured scenes, target 10 MB / cap near 1.08 million)

| | fine-tuning (no effect, see below) | codebook (k-means + 400-step fit) | encode + decode check + test renders |
|---|---|---|---|
| our controller, 13 scenes | 4.9 min (3.8 to 6.6) | 1.5 min (1.2 to 1.9) | 0.4 min |
| vertex cap, 13 scenes | 4.6 min (3.3 to 5.9) | 1.6 min (1.3 to 2.0) | 0.4 min |

Fine-tuning kept the unchanged mesh in 59 of 65 runs (36 of 39 controller runs, 23 of 26 cap runs); the other 6 changed by less than 0.01 dB.

## One training, with the fine-tuning subtracted (minutes)

| | whole run | without the fine-tuning |
|---|---|---|
| our controller, 13 scenes | 53.8 (41.2 to 70.8) | 48.8 (37.4 to 64.3) |
| our controller, the 9 scenes below | 54.5 | 49.5 |
| vertex cap, first training, 9 scenes with stage stamps | 32.6 (24.8 to 42.4) | 28.0 (21.0 to 36.8) |

9 scenes: bicycle bonsai counter drjohnson flowers room stump train treehill.  The subtraction is arithmetic on measured parts, not a run without fine-tuning.
