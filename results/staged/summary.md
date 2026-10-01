# Results

Mean length of the final 100 episodes, per seed. 500 is the CartPole-v1 ceiling.

| condition | mean ± std | episodes reaching 500 | per seed |
|---|---|---|---|
| fly-reflex-station | 499.8 ± 0.4 | 100% | 500, 499, 500, 500, 500, 500, 500, 500, 500, 500, 499, 500, 500, 499, 500, 500, 500, 500, 500, 499 |
| fly-reflex-station-staged-motion | 500.0 ± 0.0 | 100% | 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500 |

| claim | comparison | permutation p |
|---|---|---|
| curriculum learning changes the 500-step score | fly-reflex-station-staged-motion vs fly-reflex-station (two-sided) | 0.0090 |

## Past the 500-step cap: 10 episodes of up to 2000 steps per seed

Tuned gains, exploration off. Gains are shown relative to the ocelli; only their ratios steer the fly.

| condition | mean length ± std | track exits | mean distance from centre (m) | median gains relative to the ocelli (angle, rate, drift, position, motion) |
|---|---|---|---|---|
| fly-reflex-station | 1990 ± 20 | 5% | 0.48 ± 0.09 | 1.00, 0.10, 0.29, 0.22, 0.00 |
| fly-reflex-station-staged-motion | 2000 ± 0 | 0% | 0.12 ± 0.06 | 1.00, 0.21, 0.45, 0.92, 2.48 |

| claim | comparison | permutation p |
|---|---|---|
| curriculum learning keeps the cart nearer the centre | fly-reflex-station-staged-motion vs fly-reflex-station, distance from centre (smaller) | 0.0001 |
| curriculum learning holds the pole longer | fly-reflex-station-staged-motion vs fly-reflex-station, long-episode length (greater) | 0.0237 |
