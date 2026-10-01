# Results

Mean length of the final 100 episodes, per seed. 500 is the CartPole-v1 ceiling.

| condition | mean ± std | episodes reaching 500 | per seed |
|---|---|---|---|
| fly-reflex-station | 499.8 ± 0.4 | 100% | 500, 500, 499, 500, 500, 499, 500, 500, 500, 500, 500, 499, 500, 500, 500, 500, 500, 500, 500, 499 |
| fly-reflex-station-balanced | 497.5 ± 2.8 | 99% | 496, 500, 500, 500, 496, 496, 500, 499, 498, 499, 500, 490, 499, 500, 496, 496, 499, 492, 496, 498 |

| claim | comparison | permutation p |
|---|---|---|
| a balanced record changes the 500-step score | fly-reflex-station-balanced vs fly-reflex-station (two-sided) | 0.0002 |

## Past the 500-step cap: 10 episodes of up to 2000 steps per seed

Tuned gains, exploration off. Gains are shown relative to the ocelli; only their ratios steer the fly.

| condition | mean length ± std | track exits | mean distance from centre (m) | median gains relative to the ocelli (angle, rate, drift, position) |
|---|---|---|---|---|
| fly-reflex-station | 1984 ± 45 | 4% | 0.50 ± 0.22 | 1.00, 0.10, 0.28, 0.20 |
| fly-reflex-station-balanced | 1965 ± 102 | 4% | 0.14 ± 0.07 | 1.00, 0.04, 0.34, 0.57 |

| claim | comparison | permutation p |
|---|---|---|
| a balanced record keeps the cart nearer the centre | fly-reflex-station-balanced vs fly-reflex-station, distance from centre (smaller) | 0.0001 |
| a balanced record holds the pole longer | fly-reflex-station-balanced vs fly-reflex-station, long-episode length (greater) | 0.7165 |
