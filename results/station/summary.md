# Results

Mean length of the final 100 episodes, per seed. 500 is the CartPole-v1 ceiling.

| condition | mean ± std | episodes reaching 500 | per seed |
|---|---|---|---|
| fly-reflex-station-fixed | 353.7 ± 6.4 | 11% | 357, 350, 356, 354, 361, 348, 349, 355, 360, 350, 361, 358, 354, 359, 356, 332, 356, 349, 352, 357 |
| fly-reflex-adaptive | 499.9 ± 0.2 | 100% | 500, 500, 500, 500, 500, 500, 500, 499, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 499 |
| fly-reflex-station | 499.6 ± 1.4 | 100% | 499, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 493, 500, 500, 500, 500, 500, 500, 499, 500 |

| claim | comparison | permutation p |
|---|---|---|
| position helps within the 500-step cap | fly-reflex-station vs fly-reflex-adaptive (greater) | 0.7578 |
| self-tuning helps with a landmark | fly-reflex-station vs fly-reflex-station-fixed (greater) | 0.0001 |

## Past the 500-step cap: 10 episodes of up to 2000 steps per seed

Tuned gains, exploration off. Gains are shown relative to the ocelli; only their ratios steer the fly.

| condition | mean length ± std | track exits | mean distance from centre (m) | median gains relative to the ocelli (angle, rate, drift, position) |
|---|---|---|---|---|
| fly-reflex-station-fixed | 356 ± 34 | 100% | 0.81 ± 0.06 | 1.00, 1.00, 1.00, 1.00 |
| fly-reflex-adaptive | 1580 ± 151 | 59% | 0.94 ± 0.13 | 1.00, 0.07, 0.25, 0.00 |
| fly-reflex-station | 1982 ± 49 | 5% | 0.51 ± 0.17 | 1.00, 0.10, 0.28, 0.20 |

| claim | comparison | permutation p |
|---|---|---|
| position holds the pole longer | fly-reflex-station vs fly-reflex-adaptive, long-episode length (greater) | 0.0001 |
| position keeps the cart near the centre | fly-reflex-station vs fly-reflex-adaptive, distance from centre (smaller) | 0.0001 |
| self-tuning holds the pole longer with a landmark | fly-reflex-station vs fly-reflex-station-fixed, long-episode length (greater) | 0.0001 |
