# Compression ratio decomposed: fewer vertices x fewer bytes per vertex (Mip-NeRF 360 / T&T, 11 scenes, 2026-11-12)

FP32 bytes per vertex = FP32 MB / V_FP32 (52 floats + faces); ours = file MB / V_ours.

| target | FP32 V (mean) | ours V (mean) | vertex ratio | FP32 B/vertex | ours B/vertex | bytes-per-vertex ratio | total ratio |
|---|---|---|---|---|---|---|---|
| 20 MB | 2.63 M | 1.82 M | 1.44x (1.13..1.75) | 231 | 10.9 | 21.3x (18.4..22.6) | 31x |
| 10 MB | 2.63 M | 0.88 M | 3.00x (2.24..3.76) | 231 | 11.6 | 20.0x (17.5..21.6) | 60x |
| 5 MB | 2.63 M | 0.40 M | 6.65x (4.95..8.71) | 231 | 12.7 | 18.3x (16.5..19.6) | 122x |

FP32 per-scene: room V 2.55M F 5.49M, kitchen V 2.14M F 5.24M, counter V 2.22M F 4.79M, bonsai V 2.94M F 5.47M, garden V 3.17M F 6.67M, bicycle V 2.97M F 5.19M, stump V 3.04M F 5.35M, flowers V 2.94M F 4.69M, treehill V 3.03M F 5.11M, truck V 1.88M F 3.48M, train V 2.09M F 4.18M