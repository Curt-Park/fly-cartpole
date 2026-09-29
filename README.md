# fly-cartpole

A fruit fly mushroom body, wired from the [MaleCNS v1.0](https://male-cns.janelia.org/)
connectome, tries to learn Gymnasium's CartPole with the fly's own dopamine-gated
plasticity rule. A web viewer replays which mushroom body neurons light up while it
balances the pole.

**Result: the fly's rule does not learn to balance the pole.** Learning moves the fly
from a one-sided innate bias up to chance level (about 22 steps, the same as pressing
random buttons) and stops there. A TD actor-critic trained on the *same* Kenyon cell
code reaches 272 steps on average, so the Kenyon cell code carries enough information;
the bottleneck is the learning rule.

![Learning curves](results/learning_curves.png)

| condition | final-100 mean ± std (10 seeds) | episodes reaching 500 |
|---|---|---|
| `fly` (real wiring, fly rule) | 21.4 ± 7.7 | 0% |
| `fly-best` (baseline = best episode) | 15.3 ± 5.9 | 0% |
| `fly-shuffled` (degree-preserving random wiring) | 26.0 ± 3.7 | 0% |
| `fly-frozen` (no plasticity) | 12.6 ± 3.0 | 0% |
| `td` (TD actor-critic on the same KC code) | 272.4 ± 147.7 | 29% |
| random buttons (reference, 1,000 episodes) | 22.6 | 0% |

Permutation tests on per-seed final-100 means (full table in [results/summary.md](results/summary.md)):

- `fly` > `fly-frozen`: p = 0.003. The frozen fly's innate preferences push the cart one
  way and it falls in about 13 steps; plasticity removes that bias within the first few dozen
  episodes. That is the whole effect.
- `fly` > `fly-shuffled`: p = 0.94. The measured wiring does not help; shuffled wiring
  does slightly better.
- `fly` vs `fly-best`: p = 0.07 (two-sided). Rewarding only record-breaking runs is,
  if anything, worse than rewarding runs longer than the recent mean.

Hyperparameters were searched on separate tuning seeds (100-104; 40 fly settings, 6 TD
settings) and applied unchanged to evaluation seeds 0-9. The best fly setting scored 27
steps on the tuning seeds. See [results/tuning.json](results/tuning.json).

## How it works

**Connectome slice.** `fly-cartpole extract` downloads the official MaleCNS v1.0 flat
connectome and keeps the right mushroom body: 343 antennal lobe projection neurons (PN),
1,912 Kenyon cells (KC) with both PN input and MBON output, 49 mushroom body output
neurons (MBON), and the 332 PAM/PPL1 dopaminergic neurons (DAN) that synapse onto them.
Every edge is kept with its synapse count; each cell's inputs are normalised to sum to
one. A KC receives on average 5.8 PNs, matching the literature. The result is committed
as `data/mb_right.npz` (110 KB).

**Encoder.** The 83 glomeruli (PN groups) are split at random per seed: 25% encode the
candidate action (push left or push right), the rest tile the four CartPole state
variables with Gaussian tuning curves. Sister PNs of a glomerulus fire together.

**Kenyon cells.** PN activity times the measured PN→KC matrix; the top 5% of KCs fire
(k-winners-take-all standing in for APL inhibition).

**Choice.** The mushroom body outputs valence, not direction, so the fly chooses T-maze
style: it "smells" state + push-left and state + push-right, scores each as approach
MBONs minus avoidance MBONs, and picks stochastically (`sigmoid(beta * difference)`).

**Valence.** An MBON is labelled approach if PPL1 (punishment) dopamine dominates its
dopaminergic input and avoidance if PAM (reward) dopamine does (Aso et al. 2014): 28
approach, 21 avoidance.

**Dopamine.** PPL1 fires when the pole falls. PAM fires on every step the current
episode is longer than the baseline (mean of the last 20 episodes). Where dopamine lands
is read from the measured DAN→MBON wiring.

**Plasticity.** The measured rule: KC activity shortly before dopamine depresses that
KC→MBON synapse (Hige et al. 2015; Handler et al. 2019).
`gain -= learning_rate * outer(trace, dopamine_at_mbon)`, `gain += gain_decay * (1 - gain)`,
clipped to [0, 1]; the trace is a decaying sum of the chosen option's KC codes.

## Why the fly fails and TD does not

Dopamine arrives only when the pole falls and while an episode beats the baseline. With
an eligibility trace long enough to reach the mistake, the whole short episode is
punished almost equally; with a short trace, only the last hopeless steps are. The TD
learner has a critic that predicts trouble before the fall and turns every step into a
signed teaching signal. The fly rule has no such prediction, and no setting in the
search found a way around that. This matches earlier attempts to put mushroom body
plasticity on control tasks (DOOMFLY, FlyPong), while associative choice tasks
(fly-blackjack) do work.

A natural next experiment is to let the MBON→DAN feedback wiring, which the connectome
contains, compute a prediction error, as proposed by Bennett et al. (2021).

## Measured versus invented

- **Measured:** every PN→KC, KC→MBON and DAN→MBON connection and its synapse count;
  which dopamine population dominates each MBON; cell-body positions.
- **Invented:** the encoder, k-winners-take-all instead of the APL neuron, rate units,
  the baseline reward, and the hyperparameters in `results/hyperparameters.json`.

## Run it

```bash
uv sync
uv run fly-cartpole extract    # 1.1 GB download once; data/mb_right.npz is already committed
uv run fly-cartpole tune       # searches on tuning seeds 100-104 (add --configs 40 to match these results)
uv run fly-cartpole compare    # all conditions on seeds 0-9, writes results/
uv run fly-cartpole record     # writes web/data/ for the viewer
python3 -m http.server -d web  # then open http://localhost:8000
uv run pytest
```

`compare` takes about 5 minutes on 12 CPU cores; the TD runs dominate.

## Viewer

`web/index.html` replays one episode before learning and one after 1,000 training
episodes. Left: the cart, the two option scores and the current dopamine signal. Right:
the right mushroom body at its cell-body positions; PNs (blue) follow the glomerulus
code, active KCs light up white, MBONs (magenta) brighten with their response, PAM
(green, ▲ reward) and PPL1 (amber, ✕ punishment) flash with dopamine. Cell bodies sit
at the brain surface, so a dot identifies a neuron, not where its synapses compute.

## Limitations

- Only the right mushroom body is simulated; nothing else in the brain is computed or
  drawn.
- Rate units, no spikes, no APL, KC→KC or MBON→MBON connections.
- MBON valence labels follow the compartmental account and were not checked against
  individually characterised MBONs.
- One dataset, one hemisphere, ten evaluation seeds.

## Credits

- MaleCNS v1.0 connectome, CC BY 4.0; see [NOTICE.md](NOTICE.md).
- Method from [fly-blackjack](https://github.com/WilliamJones/fly-blackjack) (MIT).
- Aso et al. 2014, *eLife* (MBON valence); Hige et al. 2015, *Neuron* (KC→MBON
  depression); Handler et al. 2019, *Cell* (timing of dopamine plasticity); Barto,
  Sutton and Anderson 1983 (failure-only reinforcement on cart-pole); Bennett et al.
  2021, *Nature Communications* (prediction errors in the mushroom body).
