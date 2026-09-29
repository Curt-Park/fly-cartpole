# Results

Mean length of the final 100 episodes, per seed. 500 is the CartPole-v1 ceiling.

| condition | mean ± std | episodes reaching 500 | per seed |
|---|---|---|---|
| fly | 23.1 ± 11.5 | 0% | 9, 31, 29, 24, 10, 51, 24, 10, 31, 28, 23, 31, 10, 9, 39, 10, 26, 32, 9, 28 |
| fly-best | 10.7 ± 1.9 | 0% | 10, 10, 15, 12, 10, 10, 10, 10, 10, 15, 10, 9, 10, 10, 10, 10, 10, 15, 9, 10 |
| fly-shuffled | 28.2 ± 10.4 | 0% | 32, 45, 10, 29, 30, 33, 40, 32, 10, 29, 33, 28, 10, 27, 34, 29, 41, 35, 9, 28 |
| fly-frozen | 16.7 ± 10.2 | 0% | 9, 9, 14, 17, 9, 34, 10, 10, 24, 9, 49, 29, 9, 17, 16, 12, 14, 10, 9, 24 |
| fly-rpe | 22.7 ± 8.4 | 0% | 9, 31, 25, 24, 18, 29, 27, 10, 27, 26, 24, 32, 28, 9, 30, 10, 25, 36, 9, 26 |
| fly-rpe-wired | 24.1 ± 10.8 | 0% | 9, 28, 25, 25, 10, 45, 24, 10, 29, 27, 10, 36, 28, 35, 37, 10, 25, 35, 9, 27 |
| fly-bilateral | 191.7 ± 49.5 | 3% | 311, 188, 168, 195, 182, 124, 233, 182, 258, 232, 152, 154, 158, 141, 206, 148, 224, 149, 281, 148 |
| fly-bilateral-shuffled | 221.7 ± 55.3 | 8% | 256, 166, 199, 257, 217, 206, 180, 219, 256, 209, 334, 138, 260, 169, 196, 304, 337, 202, 158, 170 |
| fly-bilateral-frozen | 24.5 ± 28.1 | 0% | 45, 20, 10, 11, 140, 17, 32, 33, 20, 13, 13, 11, 18, 9, 15, 10, 12, 15, 12, 34 |
| td | 205.4 ± 137.4 | 13% | 249, 479, 161, 310, 9, 138, 302, 14, 500, 149, 317, 274, 170, 361, 131, 122, 9, 123, 174, 116 |
| random | 22.0 ± 0.9 | 0% | 22, 21, 21, 21, 22, 23, 21, 21, 22, 22, 23, 24, 21, 24, 22, 21, 23, 23, 23, 21 |

| claim | comparison | permutation p |
|---|---|---|
| learning | fly vs fly-frozen (greater) | 0.0410 |
| wiring contributes | fly vs fly-shuffled (greater) | 0.9212 |
| baseline mode | fly vs fly-best (two-sided) | 0.0001 |
| prediction error helps | fly-rpe vs fly (greater) | 0.5383 |
| measured feedback wiring works | fly-rpe-wired vs fly (greater) | 0.3878 |
| bilateral fly learns | fly-bilateral vs fly-bilateral-frozen (greater) | 0.0001 |
| bilateral fly beats chance | fly-bilateral vs random (greater) | 0.0001 |
| bilateral wiring contributes | fly-bilateral vs fly-bilateral-shuffled (greater) | 0.9600 |
