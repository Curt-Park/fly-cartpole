# Results

Mean length of the final 100 episodes, per seed. 500 is the CartPole-v1 ceiling.

| condition | mean ± std | episodes reaching 500 | per seed |
|---|---|---|---|
| fly | 22.2 ± 13.6 | 0% | 9, 31, 30, 44, 10, 10, 10, 10, 29, 9, 9, 31, 9, 51, 9, 35, 36, 30, 30, 10 |
| fly-best | 9.8 ± 0.3 | 0% | 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 9, 10, 9, 10, 10, 10, 11, 10 |
| fly-shuffled | 27.0 ± 12.8 | 0% | 10, 29, 35, 32, 29, 49, 26, 10, 10, 9, 33, 25, 35, 9, 32, 39, 50, 36, 10, 31 |
| fly-frozen | 18.7 ± 17.5 | 0% | 27, 10, 32, 11, 9, 11, 9, 9, 10, 12, 9, 9, 23, 33, 18, 24, 9, 9, 86, 10 |
| fly-rpe | 22.7 ± 10.1 | 0% | 9, 22, 27, 36, 26, 10, 33, 30, 30, 9, 27, 32, 9, 35, 9, 10, 33, 30, 28, 10 |
| fly-rpe-wired | 21.9 ± 11.9 | 0% | 9, 27, 30, 46, 26, 10, 10, 34, 31, 9, 9, 35, 9, 38, 9, 10, 28, 27, 30, 10 |
| fly-bilateral | 233.8 ± 43.1 | 6% | 199, 198, 209, 238, 189, 226, 234, 249, 261, 194, 304, 270, 169, 280, 186, 161, 298, 275, 292, 241 |
| fly-bilateral-shuffled | 218.3 ± 66.3 | 7% | 226, 209, 147, 91, 274, 179, 283, 125, 152, 188, 224, 236, 155, 226, 298, 343, 156, 285, 273, 296 |
| fly-bilateral-frozen | 21.3 ± 13.9 | 0% | 25, 12, 31, 66, 9, 32, 19, 19, 29, 11, 12, 12, 14, 13, 46, 12, 16, 10, 24, 13 |
| td | 302.3 ± 168.0 | 38% | 10, 500, 164, 216, 270, 473, 227, 113, 494, 448, 500, 500, 498, 298, 203, 9, 168, 189, 266, 500 |
| random | 22.2 ± 1.0 | 0% | 22, 24, 21, 23, 21, 21, 23, 22, 23, 23, 21, 23, 22, 23, 24, 20, 23, 21, 21, 22 |

| claim | comparison | permutation p |
|---|---|---|
| learning | fly vs fly-frozen (greater) | 0.2571 |
| wiring contributes | fly vs fly-shuffled (greater) | 0.8646 |
| baseline mode | fly vs fly-best (two-sided) | 0.0010 |
| prediction error helps | fly-rpe vs fly (greater) | 0.4553 |
| measured feedback wiring works | fly-rpe-wired vs fly (greater) | 0.5297 |
| bilateral fly learns | fly-bilateral vs fly-bilateral-frozen (greater) | 0.0001 |
| bilateral fly beats chance | fly-bilateral vs random (greater) | 0.0001 |
| bilateral wiring contributes | fly-bilateral vs fly-bilateral-shuffled (greater) | 0.1953 |
