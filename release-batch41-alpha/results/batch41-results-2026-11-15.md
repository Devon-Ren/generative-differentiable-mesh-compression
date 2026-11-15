# batch41: growth aim of the hold controller (target 10 MB, one run each) and frame rate (2026-11-15)

## Growth aim (alpha = --phase_a_overshoot)

| scene | alpha | file (MB) | size error | PSNR | SSIM | LPIPS | V before collapse -> final | active collapse rounds |
|---|---|---|---|---|---|---|---|---|
| garden | 1.00 | 8.462 | -15.4 % | 23.332 | 0.6427 | 0.4131 | 730,955 -> 730,955 (trained V 693,195) | 0 of 35 |
| garden | 1.20 | 9.937 | -0.6 % | 23.568 | 0.6623 | 0.3906 | 895,974 -> 838,971 | 11 |
| garden | 1.35 (ByteMesh, batch31) | 9.997 | -0.0 % | 23.672 | 0.6691 | 0.3843 | 1,018,671 -> 822,913 | 17 |
| garden | 1.50 | 9.958 | -0.4 % | 23.758 | 0.6738 | 0.3799 | 1,141,929 -> 811,712 | 28 |
| room | 1.00 | 8.724 | -12.8 % | 27.649 | 0.8328 | 0.4441 | 822,840 -> 822,840 | 0 |
| room | 1.20 | 10.025 | +0.2 % | 27.901 | 0.8431 | 0.4287 | 988,764 -> 908,245 | 11 |
| room | 1.35 (ByteMesh, batch31) | 9.992 | -0.1 % | 27.991 | 0.8418 | 0.4271 | 1,142,744 -> 889,576 | 20 |
| room | 1.50 | 9.703 | -3.0 % | 27.836 | 0.8425 | 0.4266 | 1,275,459 -> 872,843 | 21 |

Reading: with alpha 1.0 the file ends 13-15 % below the target and no collapse round is active: vertices are lost between the end of
densification and the final mesh, and nothing can add them back.  From 1.2 to 1.5 the file is on target (room at 1.5: 3 % below, after two
late rounds removed more than the next probes showed was needed) and PSNR stays within 0.19 dB: garden rises with alpha (+0.09 dB at 1.5),
room is best at 1.35 (-0.09 at 1.2, -0.16 at 1.5).  Two scenes, one run each.

## Frame rate (viewer-v3.html, WebGL, Chrome, MacBook with Apple M2 Max, 1920x958, 300 frames with forced completion, SH degree 3)

| scene | target | faces | ms per frame (mean) | fps |
|---|---|---|---|---|
| bicycle | 5 MB | 879,574 | 0.99 | 1011 |
| bicycle | 10 MB | 1,972,925 | 1.31 | 765 |
| bicycle | 20 MB | 4,020,609 | 1.94 | 515 |
| garden | 5 MB | 970,370 | 1.10 | 906 |
| garden | 10 MB | 2,140,445 | 1.32 | 757 |
| garden | 20 MB | 4,395,647 | 2.09 | 477 |
| room | 5 MB | 1,055,118 | 1.05 | 955 |
| room | 10 MB | 2,240,586 | 1.33 | 752 |
| room | 20 MB | 4,430,520 | 2.00 | 500 |

Mean: 5 MB 957 fps (1.05 ms), 10 MB 758 fps (1.32 ms), 20 MB 497 fps (2.01 ms).  The FP32 models were not measured (not downloaded).
The face counts are those of the stored mesh.  Resolution is the browser window, not the capture resolution.
