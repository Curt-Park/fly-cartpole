# Results

Mean length of the final 100 episodes, per seed. 500 is the CartPole-v1 ceiling.

| condition | mean ± std | episodes reaching 500 | per seed |
|---|---|---|---|
| fly | 21.4 ± 7.7 | 0% | 27, 21, 10, 25, 26, 11, 10, 28, 31, 25 |
| fly-best | 15.3 ± 5.9 | 0% | 12, 15, 11, 23, 14, 11, 11, 11, 29, 16 |
| fly-shuffled | 26.0 ± 3.7 | 0% | 28, 20, 23, 31, 27, 29, 21, 30, 25, 27 |
| fly-frozen | 12.6 ± 3.0 | 0% | 9, 9, 10, 15, 13, 18, 10, 17, 14, 11 |
| td | 272.4 ± 147.7 | 29% | 212, 122, 495, 500, 228, 283, 236, 443, 105, 100 |

| claim | comparison | permutation p |
|---|---|---|
| learning | fly vs fly-frozen (greater) | 0.0030 |
| wiring contributes | fly vs fly-shuffled (greater) | 0.9413 |
| baseline mode | fly vs fly-best (two-sided) | 0.0708 |
