# batch33: rate-distortion curves of the generic codecs (11 Mip-NeRF 360 / T&T scenes, measured 2026-11-12/13)

12 operating points per scene and codec: 4 meshes (FP32 model; simplified after training + fine-tuned at three vertex counts) x SH precision 8 / 6 / 4 bits. 132 rows, no failures. Means over the 11 scenes; the curve of a codec is the upper convex hull of its 12 mean points in (log size, PSNR), the reading most favourable to the codec.

| codec | size it needs for BudgetMesh's PSNR at 5 / 7.6 / 10.2 / 15.1 / 19.9 MB | ratio | BudgetMesh minus codec at the same size |
|---|---|---|---|
| MPEG TFAN | 41 / 48 / 56 / 64 / 82 MB | 8.2 / 6.4 / 5.5 / 4.2 / 4.1 x | no point below 6.9 MB; +5.1 / +4.6 / +3.7 / +3.1 dB |
| quantized PLY + zstd | 57 / 68 / 80 / 92 / 115 MB | 11.4 / 8.9 / 7.8 / 6.1 / 5.8 x | no point below 9.8 MB; +5.6 / +4.7 / +4.0 dB |
| Draco | 150 / 184 / 219 / 255 / 331 MB | 30 / 24 / 22 / 17 / 17 x | no point below 17.6 MB; +5.8 dB at 19.9 MB |

SH precision on the FP32 model (loss against the mesh before quantisation): 8 bit -0.13 dB, 6 bit -0.31 dB, 4 bit -2.63 dB; sizes TFAN 95.7 / 65.1 / 38.9 MB.

Codec alone (same meshes, FP32 model and its simplified versions): our codec against TFAN at equal PSNR is about 2.4-2.5 x smaller (23.65 dB: 26 MB vs 65 MB; 23.1 dB: 19 MB vs about 47 MB; 21.4 dB: 10 MB vs about 25 MB).

Limits: the BudgetMesh curve is the controller used in the paper draft (not the final design of batch30); the "no vertices removed" point of our codec uses the earlier FP32 models for the four indoor scenes, the generic rows use the --indoor models; the hull connects operating points that a single file cannot mix.
