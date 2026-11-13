# batch35: selection rules of the hold controller against MeshSplatting's importance score (target 10 MB, one seed)

| scene | variant | file (MB) | PSNR | SSIM | LPIPS | PSNR minus hold |
|---|---|---|---|---|---|---|
| garden | hold | 9.997 | 23.672 | 0.6691 | 0.3843 | reference |
| garden | holdis (importance split) | 9.917 | 23.507 | 0.6655 | 0.3927 | -0.17 |
| garden | holdic (importance as collapse cost) | 9.957 | 22.889 | 0.6531 | 0.3948 | -0.78 |
| room | hold | 9.992 | 27.991 | 0.8418 | 0.4271 | reference |
| room | holdis | 9.966 | 27.563 | 0.8396 | 0.4340 | -0.43 |
| room | holdic | 9.999 | 25.609 | 0.8269 | 0.4364 | -2.38 |

All files within 1 % of the target, so the comparison is at the same size.  Both differences exceed the 0.1 dB threshold fixed before the runs.
Other readings from the rows: with importance split the survival of new vertices at the next upstream prune drops (sigma median 0.55 -> 0.38
garden, 0.48 -> 0.31 room); with importance as collapse cost more rounds are active (17 -> 26 garden, 20 -> 21 room).
Limits: two scenes, one seed; MeshSplatting has no edge collapse, holdic puts its importance score into our collapse step.
