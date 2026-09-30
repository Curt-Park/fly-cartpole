# Results

Mean length of the final 100 episodes, per seed. 500 is the CartPole-v1 ceiling.

| condition | mean ± std | episodes reaching 500 | per seed |
|---|---|---|---|
| fly | 20.7 ± 3.6 | 0% | 20, 18, 23, 20, 19, 30, 17, 20, 18, 22, 21, 22, 26, 25, 17, 16, 22, 20, 23, 15 |
| fly-best | 20.7 ± 3.6 | 0% | 20, 18, 20, 22, 20, 31, 18, 19, 20, 22, 19, 22, 23, 27, 17, 16, 22, 21, 23, 15 |
| fly-shuffled | 21.4 ± 2.7 | 0% | 23, 23, 22, 22, 15, 19, 20, 21, 21, 20, 21, 22, 21, 27, 20, 28, 22, 21, 18, 23 |
| fly-frozen | 20.5 ± 3.7 | 0% | 22, 18, 24, 20, 13, 29, 21, 21, 22, 19, 28, 18, 22, 19, 20, 17, 20, 21, 24, 15 |
| fly-rpe | 23.1 ± 1.1 | 0% | 25, 23, 23, 24, 23, 22, 23, 22, 24, 24, 26, 22, 25, 22, 24, 21, 22, 22, 23, 22 |
| fly-rpe-wired | 22.5 ± 1.3 | 0% | 23, 22, 22, 21, 22, 25, 21, 22, 22, 22, 24, 23, 24, 22, 22, 23, 25, 22, 25, 20 |
| fly-bilateral | 392.5 ± 103.1 | 46% | 492, 377, 473, 497, 492, 205, 441, 464, 376, 498, 310, 338, 469, 162, 500, 258, 316, 410, 476, 294 |
| fly-bilateral-shuffled | 390.3 ± 96.4 | 44% | 357, 150, 334, 500, 428, 253, 456, 395, 499, 441, 412, 222, 487, 456, 491, 428, 450, 317, 431, 298 |
| fly-bilateral-frozen | 18.7 ± 9.9 | 0% | 18, 28, 10, 43, 11, 16, 11, 21, 40, 10, 10, 13, 19, 25, 11, 11, 12, 11, 30, 23 |
| td | 500.0 ± 0.0 | 100% | 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500 |
| random | 22.8 ± 0.9 | 0% | 21, 22, 22, 24, 25, 23, 23, 21, 23, 22, 22, 23, 23, 23, 23, 23, 24, 23, 23, 24 |

| claim | comparison | permutation p |
|---|---|---|
| learning | fly vs fly-frozen (greater) | 0.4238 |
| wiring contributes | fly vs fly-shuffled (greater) | 0.7511 |
| baseline mode | fly vs fly-best (two-sided) | 0.9982 |
| prediction error helps | fly-rpe vs fly (greater) | 0.0053 |
| measured feedback wiring works | fly-rpe-wired vs fly (greater) | 0.0212 |
| bilateral fly learns | fly-bilateral vs fly-bilateral-frozen (greater) | 0.0001 |
| bilateral fly beats chance | fly-bilateral vs random (greater) | 0.0001 |
| bilateral wiring contributes | fly-bilateral vs fly-bilateral-shuffled (greater) | 0.4771 |
