# Results

Mean length of the final 100 episodes, per seed. 500 is the CartPole-v1 ceiling.

| condition | mean ± std | episodes reaching 500 | per seed |
|---|---|---|---|
| fly-reflex-station | 499.9 ± 0.4 | 100% | 500, 500, 500, 500, 500, 498, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500 |
| fly-reflex-station-centred | 496.7 ± 2.8 | 99% | 500, 491, 492, 500, 496, 493, 500, 497, 500, 500, 495, 498, 495, 497, 494, 497, 496, 498, 494, 500 |

| claim | comparison | permutation p |
|---|---|---|
| a centred record changes the 500-step score | fly-reflex-station-centred vs fly-reflex-station (two-sided) | 0.0001 |

## Past the 500-step cap: 10 episodes of up to 2000 steps per seed

Tuned gains, exploration off. Gains are shown relative to the ocelli; only their ratios steer the fly.

| condition | mean length ± std | track exits | mean distance from centre (m) | median gains relative to the ocelli (angle, rate, drift, position) |
|---|---|---|---|---|
| fly-reflex-station | 1979 ± 44 | 8% | 0.52 ± 0.22 | 1.00, 0.10, 0.26, 0.21 |
| fly-reflex-station-centred | 1956 ± 140 | 3% | 0.15 ± 0.20 | 1.00, 0.04, 0.50, 0.71 |

| claim | comparison | permutation p |
|---|---|---|
| a centred record keeps the cart nearer the centre | fly-reflex-station-centred vs fly-reflex-station, distance from centre (smaller) | 0.0001 |
| a centred record holds the pole longer | fly-reflex-station-centred vs fly-reflex-station, long-episode length (greater) | 0.6642 |
