# Results

Mean length of the final 100 episodes, per seed. 500 is the CartPole-v1 ceiling.

| condition | mean ± std | episodes reaching 500 | per seed |
|---|---|---|---|
| fly-reflex | 275.6 ± 9.4 | 17% | 273, 278, 285, 287, 265, 288, 287, 276, 258, 279, 268, 282, 278, 276, 282, 278, 257, 259, 281, 275 |
| fly-reflex-adaptive | 499.9 ± 0.3 | 100% | 500, 500, 499, 500, 500, 500, 500, 500, 500, 500, 500, 499, 500, 500, 500, 500, 500, 500, 500, 500 |
| fly-reflex-adaptive-shuffled | 165.9 ± 182.8 | 20% | 9, 9, 9, 196, 182, 9, 500, 499, 41, 176, 183, 9, 500, 214, 500, 9, 181, 41, 19, 33 |
| random | 22.1 ± 0.8 | 0% | 24, 21, 22, 22, 22, 22, 21, 23, 22, 21, 23, 23, 22, 22, 21, 22, 23, 22, 22, 24 |
| linear | 310.2 ± 11.1 | 11% | 323, 289, 305, 304, 290, 330, 310, 319, 313, 299, 318, 313, 317, 307, 297, 320, 315, 316, 298, 322 |
| linear-adaptive | 499.7 ± 0.7 | 100% | 500, 500, 498, 499, 500, 500, 500, 500, 500, 500, 497, 500, 500, 500, 500, 500, 500, 500, 500, 500 |

| claim | comparison | permutation p |
|---|---|---|
| reflex beats chance | fly-reflex vs random (greater) | 0.0001 |
| self-tuning helps | fly-reflex-adaptive vs fly-reflex (greater) | 0.0001 |
| wiring contributes | fly-reflex-adaptive vs fly-reflex-adaptive-shuffled (greater) | 0.0001 |
| the untuned circuit differs from its linear map | fly-reflex vs linear (two-sided) | 0.0001 |
| the self-tuned circuit differs from its linear map | fly-reflex-adaptive vs linear-adaptive (two-sided) | 0.3388 |
