# Vertex cap from a prior vs BudgetMesh as size control (2026-11-12, no new runs)

Bytes per vertex of the encoded file, same codec: BudgetMesh 11.8 (10.1..14.4), MeshSplatting with a vertex cap 11.2 (9.3..14.0). At the same file size the two have the same vertex count within a few percent; the PSNR gain (+0.27 dB same V, +0.21 dB same size) comes from which triangles exist, not from how many.

Size control on the 11 Mip-NeRF 360 / T&T scenes. Cap baselines are read on each scene's measured cap -> file size curve (4-6 real cap runs per scene, log-log interpolation); the prior MB = a*cap + c is fitted on the other 10 scenes (leave-one-scene-out).

| method | trainings | T=5 MB: mean abs err, range, within 10% | T=10 MB | T=20 MB |
|---|---|---|---|---|
| vertex cap from a prior | 1 | 5.6%, -17.9..+5.4%, 9/11 | 5.6%, -5.6..+25.1%, 10/11 | 6.3%, -18.8..+10.5%, 8/11 |
| vertex cap, rescaled after measuring the first file | 2 | 1.6%, -5.9..+0.7%, 11/11 | 1.1%, -4.4..+2.1%, 11/11 | 3.9%, -21.6..+6.6%, 10/11 |
| BudgetMesh | 1 | 3.9%, -4.8..+9.2%, 11/11 | 4.1%, -4.7..+13.5%, 10/11 | 5.1%, -6.6..+10.0%, 10/11 |

Wall-clock per scene on one L40S, training + fine-tune + encode (sacct): BudgetMesh 35 / 42 / 57 min (truck / garden / room); the same without collapse-round probes 26 / 34 / 44 min, which is about what a MeshSplatting cap run costs. BudgetMesh is about 1.3x one cap run; the two-training cap baseline is 2x.
