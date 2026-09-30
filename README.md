<p align="center">
  <img src="assets/fly-cartpole.svg" alt="A worried fruit fly on a cart, holding a wobbling pole upright" width="640">
</p>

# fly-cartpole

**English** · [한국어](README.ko.md)

Two fruit fly mushroom bodies, wired synapse by synapse from the
[MaleCNS v1.0](https://male-cns.janelia.org/) connectome, learn to balance Gymnasium's
CartPole with dopamine-gated plasticity. A web viewer runs the trained fly live in the
browser and shows which mushroom body neurons light up while it balances the pole.

![The web viewer: the cart and pole, both sides' values and the dopamine signal on the left; both mushroom bodies at their MaleCNS cell-body positions on the right](assets/viewer.png)

## Result

**The fly learns to balance the pole, and runs the full 500 steps in almost half of its final
episodes.** With its innate wiring it drops the pole after about 19 steps
(`fly-bilateral-frozen`). After 3,000 episodes it holds the pole for **392.5 steps** on average
over the last 100 episodes, across 20 evaluation seeds that no earlier experiment touched, and
46% of those episodes reach the 500-step cap. 19 of the 20 flies clear CartPole-v0's "solved"
mark of 195 on their own, and 6 clear CartPole-v1's 475. A textbook TD actor-critic reading the
same Kenyon cells, under the same reward, balances perfectly on every seed (500). The Kenyon
cell code and the reward are enough; what holds the fly back is its learning rule.

![Learning curves](results/learning_curves.png)

| condition | what it is | final-100 mean ± std (20 seeds) | episodes reaching 500 |
|---|---|---|---|
| `fly-bilateral` | two mushroom bodies, prediction error dopamine, push-pull plasticity | **392.5 ± 103.1** | 46% |
| `fly-bilateral-shuffled` | same, with PN→KC and KC→MBON wiring shuffled (degree-preserving) | 390.3 ± 96.4 | 44% |
| `fly-bilateral-frozen` | same, without plasticity | 18.7 ± 9.9 | 0% |
| `fly` | the first design: one mushroom body, actions smelled as odours | 20.7 ± 3.6 | 0% |
| `td` | TD actor-critic on the same Kenyon cell code (not a fly) | 500.0 ± 0.0 | 100% |
| `random` | uniform random pushes (chance) | 22.8 ± 0.9 | 0% |

Permutation tests on the per-seed final-100 means (every condition and test is in
[results/summary.md](results/summary.md)):

- the fly learns: `fly-bilateral` > `fly-bilateral-frozen`, p = 0.0001;
- it beats chance: `fly-bilateral` > `random`, p = 0.0001;
- there is no evidence that the measured wiring helps: `fly-bilateral` >
  `fly-bilateral-shuffled`, p = 0.48. The shuffle keeps the cell counts, how many partners
  each cell has, and where each dopamine neuron lands; it only changes which Kenyon cell talks
  to which. The learning comes from that layout, not from the exact synapse map. Earlier
  evaluations pointed both ways (real wiring ahead on seeds 0-9, p = 0.037; shuffled ahead on
  seeds 30-49).

Hyperparameters were searched on tuning seeds 100-139 only and applied unchanged to the
evaluation seeds 70-89 ([results/tuning.json](results/tuning.json)).

## How we got there

The first design followed [fly-blackjack](https://github.com/WilliamJones/fly-blackjack)
closely and stayed at chance. Each row below is one hypothesis, tested on tuning seeds
(100 and up) before anything touched evaluation seeds. The score is the mean length of the
final 100 episodes, over five tuning seeds unless the row says otherwise.

| # | hypothesis | what changed | tuning seeds | verdict |
|---|---|---|---|---|
| 0 | The fly-blackjack recipe transfers | One mushroom body; the fly smells "state + push left" and "state + push right" and picks the better one (T-maze); dopamine only depresses synapses; punishment when the pole falls, reward while beating the recent mean | 27 | chance |
| 1 | Dopamine should be a prediction error | Dopamine = TD error computed from the fly's own MBON valence (Bennett et al. 2021) | 29 | rejected |
| – | *Diagnostic: is the encoding the problem?* | Unconstrained linear SARSA on the same state + action Kenyon cell code | 34-55 | yes: the action-as-odour code hides the state |
| 2 | One mushroom body per push direction | The left mushroom body values pushing left, the right one pushing right; both see only the state | up to 67 | first signal, unstable |
| 3 | An innate left/right bias blocks learning | Centre each side on its innate average value | 61 | rejected |
| 4 | Naive flies are indifferent | Value = learned change only | 63 | rejected |
| 5 | The unchosen side should learn the opposite lesson | Opponent teaching between the two sides | 39 | rejected |
| 6 | The targets exceed what bounded synapses can express | Fall-only punishment, scaled down | 49 | rejected |
| – | *Diagnostic: is the synapse rule the problem?* | Unconstrained linear SARSA on the two state-only codes | 275 | yes: bounded, depression-only synapses |
| 7 | The MBON readout is too weak | Amplify the MBON-to-choice readout | 62 | rejected |
| 8a | Dopamine cross-talk cancels learning | Punishment only on approach MBONs, reward only on avoidance MBONs | 49 | rejected |
| **8b** | **Depression-only plasticity exhausts synapses** | **Push-pull: traced synapses weaken where the released dopamine lands and recover where the opposing dopamine lands** | **237** | **learns** (evaluation seeds 0-9: 194) |
| 9 | Better generalisation across states | Wider glomerulus tuning, more active Kenyon cells | 234 | no gain |
| 10 | The reward schedule is off | Smaller record reward, shorter discount, or punishment at falls only | 235 at best; fall-only 91-187 | no gain |
| 11 | An innate bias still hurts under push-pull | Centre each side again | 174 | rejected |
| 12 | It just needs more practice | 2,000 episodes | 147, 237, 167, 182 at 500, 1,000, 1,500, 2,000 | no: performance drifts |
| 13 | Sleep consolidates | Replay each episode once more after it ends | 176 | rejected |
| 14 | Plasticity should settle with experience | Learning rate falls as lr / (1 + episode / τ) | 178 | rejected |
| – | *Diagnostic: do the synapses saturate?* | Track gains and values during training | 2-7% of gains at a bound; one side can express values from about −0.6 to +0.7, while the targets run from −1 (a fall) to +10 (a long record run) | the targets, not the synapses, are out of range |
| 15 | Outcomes must fit the value range | Scale punishment and record reward by 0.1, with a larger learning rate | 275 | looked better, but scored 115 on evaluation seeds 0-9: three of ten flies never learned |
| – | *Diagnostic: why do some flies never learn?* | Follow both sides' values in a stuck fly | One side's value sank below the scaled punishment (−0.1). The other side's value cannot fall below that, so it wins every choice, and the sunken side is never tried again | a lock-in; 8 of 40 fresh tuning seeds lock in at scale 0.1, 1 at 0.3, 2 at 1.0 |
| 16 | The record reward's moving target causes the collapses | Slower baseline (50 or 100 episodes), or a fixed reward for every step survived | 215 at best | rejected: the record reward over the last 20 episodes works best |
| 17 | A softer choice lets the sunken side back in | β = 30 instead of 100 at scale 0.1 | 44 (40 seeds, 300 episodes), 6 locked in | rejected |
| **18** | **Five tuning seeds are too few to see a one-in-five failure** | **Tune on 40 seeds (100-139) over outcome scales 0.1, 0.3 and 1.0, with the record reward fixed at 0.1** | **221 (40 seeds), no seed locked in** | **adopted: outcome scale 0.3; 191.7 after 1,000 episodes on fresh evaluation seeds 30-49** |
| **19** | **With outcomes in range, more practice keeps helping** | **2,000 episodes instead of 1,000** | **194, 221, 231, 227, 232 at 500, 1,000, 1,250, 1,500, 2,000 (40 seeds)** | **adopted: gains flatten after 1,250 episodes but never collapse; 233.8 on fresh evaluation seeds 50-69** |

Three findings carried the result. First, the fly-blackjack choice scheme, where the fly
smells each action as an odour and compares the two, buries the pole's state under the action
code; giving each push direction its own mushroom body fixed that. Second, synapses that
dopamine can only weaken run out: every mistake is punished by depression, the useful synapses
sink to zero and never come back. Letting the opposing dopamine population restore traced
synapses (push-pull) turned a stuck fly into one that balances the pole for hundreds of steps.
Third, the values one mushroom body can express through bounded synapses span only about −0.6
to +0.7, while a fall is worth −1 and a long record run up to +10. Shrinking the outcomes toward
that range (to 0.3 of their size) made learning faster and more even across seeds; shrinking
them too far (0.1) let one side sink below the punishment and lock the fly into pushing one way.

The reward schedule came from a human analogy: punishment when the pole falls, and a reward
on every step once the fly beats its recent record, like a person whose best is 5 seconds
feeling a rush at second 6. Without that record reward the fly still learned with a short
discount (187 at γ = 0.95), but at the tuned discount (γ = 0.99) it collapsed on three of
five tuning seeds (91). A record that moves more slowly, or a fixed reward for every step
survived, also did worse (hypothesis 16).

### Aiming for 500

Once the fly passed 195, the goal was raised to the 500-step cap. A diagnostic showed where the
trained fly failed: 58% of its episodes ended with the cart leaving the track rather than the
pole falling, and a fly that always picked its better side did no better than one choosing
stochastically. The rows below ran on ten tuning seeds for 3,000 episodes unless noted.
Hypotheses 24, 30-31 and 32-33 were the project author's ideas, as was the record reward.

| # | hypothesis | what changed | tuning seeds | verdict |
|---|---|---|---|---|
| 20-21 | The fly cannot see the track edge coming | Discount 0.998 instead of 0.99, outcomes scaled to 0.1, learning rate 0.05 | 332 | adopted |
| – | *Diagnostic: does the record reward push the cart off centre?* | Punishment at falls only | 40 | no: the record reward is essential |
| 22 | Velocity glomeruli are too coarse | Tile only the velocities the fly actually meets | 298-315 | rejected |
| **24** | **Reward posture, not only survival** | **Potential-based shaping: reward for moving toward upright and centred, punishment for moving away** | **388** | **adopted** |
| 25 | A stronger posture signal | Larger shaping, or three times the weight on position | track exits fall to 11-12%, but flies lock in again (174-284) | rejected |
| 26 | Active forgetting prevents lock-in | Gains drift back toward 0.5 between episodes | 91-199 | rejected |
| 27 | Teach against the state's expected value, or rebalance the posture signal | Actor-critic style prediction; more weight on position within the same budget | 329-376 | no gain |
| 28 | Noisy choices random-walk the cart off the track | β = 300 or 1,000 instead of 100 | 366-386 | no gain alone |
| **29** | **A fly that balances well should stop unlearning** | **Learning rate 0.05 / (1 + episode / 1,000), β = 300** | **416** | **adopted** |
| 30-31 | Credit should reach further back | Eligibility traces lasting 0.27 to 1.4 s instead of 0.06 s | 170-339; track exits fall to 15% but learning slows | rejected |
| – | *Diagnostic: does the fly explore?* | Chance of picking the lower-valued side | picked on 4-5% of steps, but on half of all steps its chance is below 0.1% | it explores only where it is unsure |
| **32-33** | **Explore everywhere, not only where unsure** | **1-3% of choices ignore the values (spontaneous behaviour)** | **446 at 1%, 440 at 2%, 399 at 3%; 4,000 episodes slipped to 418** | **adopted, with 3,000 episodes** |
| **34** | **Confirm on forty seeds** | Spontaneous rate 1% or 2% on tuning seeds 100-139 | 407 at 2%, no fly locked in | 392.5 on fresh evaluation seeds 70-89 |

Three more findings took the fly from 234 to 393. The posture reward tells the fly at once
when a push tilts the pole or carries the cart away from the centre, instead of only when the
episode ends hundreds of steps later. Settling plasticity keeps a fly that has learned to
balance from unlearning it. And exploration: the fly explored only where its two values were
close, so in half of all states it never tried the other side, and a confident but wrong value
was never corrected. Making 2% of choices spontaneous let those values be corrected, cut track
exits from 44% to 28%, and lifted the tuning score from 416 to 440.

**Evaluation seeds were spent five times, and every spend is reported here.** Seeds 0-9
evaluated the push-pull fly at outcome scale 1.0 (194) and then at 0.1 (115, the lock-in).
Seeds 10-29 evaluated a run in which the original fly's search had also chosen the size of the
record reward for every other condition (0.5 instead of 0.1; bilateral fly 162.5); the reward
is part of the task, so it is now fixed and never searched. Seeds 30-49 evaluated the
forty-seed tuned fly after 1,000 episodes (191.7), and seeds 50-69 the fly after 2,000 episodes
(233.8), which met the first goal of 195. The numbers at the top come from seeds 70-89, which no
earlier experiment touched.

## How it works

**Connectome slice.** `fly-cartpole extract` downloads the official MaleCNS v1.0 flat
connectome and keeps both mushroom bodies. Right: 343 antennal lobe projection neurons (PN),
1,912 Kenyon cells (KC) with both PN input and MBON output, 49 mushroom body output neurons
(MBON), and the 332 PAM and PPL1 dopaminergic neurons (DAN) that synapse onto those MBONs.
Left: 343 PNs, 1,893 KCs and 48 MBONs. Every edge keeps its synapse count, and each cell's
inputs are normalised to sum to one. A KC receives 5.8 PNs on average, matching the
literature. The slices are committed as `data/mb_right.npz` and `data/mb_left.npz`.

**Two sides, two actions.** The left mushroom body values pushing left, the right one
pushing right. Both hear the same state, and neither is told which action it stands for.

**Encoder.** The glomeruli (PN groups) tile the four CartPole state variables with Gaussian
tuning curves; sister PNs of a glomerulus fire together.

**Kenyon cells.** PN activity times the measured PN→KC matrix; the top 5% of KCs fire
(k-winners-take-all standing in for the APL neuron).

**Value and choice.** An MBON counts as approach if PPL1 (punishment) dopamine dominates its
dopaminergic input and as avoidance if PAM (reward) dopamine does (Aso et al. 2014). Each
side's value is its approach MBON output minus its avoidance MBON output, and the fly picks
a side with `sigmoid(beta * (left - right))`. 2% of choices ignore the values altogether
(spontaneous behaviour, as flies show; Maye et al. 2007), so a side the fly has written off
still gets retried.

**Reward.** PPL1 punishment when the pole falls; a small PAM reward on every step once the
episode is longer than the mean of the last 20 episodes; and a posture signal on every step,
`3 * (0.998 * phi(next) - phi(now))` with `phi = -(angle / 12°)² - (position / 2.4 m)²`.
Moving toward upright and centred is rewarded and moving away is punished (potential-based
shaping; Ng, Harada and Russell 1999). The reward is part of the task: every condition,
including the TD reference, gets the same one.

**Dopamine as a prediction error.** Dopamine carries the surprise rather than the raw reward:
`delta = scale * (reward - punishment) + gamma * expected next value - current value`. PAM
fires for `max(delta, 0)` and PPL1 for `max(-delta, 0)`, with `gamma = 0.998`. The outcome
scale (0.1) keeps the values the fly must learn inside the range its bounded synapses can
express. Where each dopamine population lands is read from the measured DAN→MBON wiring.

**Push-pull plasticity.** KC→MBON synapses have gains between 0 and 1, starting at 0.5. A
decaying trace remembers which KCs of the chosen side were active. Released dopamine
depresses traced synapses in the compartments it reaches (Hige et al. 2015; Handler et al.
2019), and the opposing dopamine population restores traced synapses in its own
compartments. The learning rate settles with experience, as `0.05 / (1 + episode / 1000)`.

## Measured versus invented

- **Measured:** every PN→KC, KC→MBON, DAN→MBON and MBON→DAN connection and its synapse
  count; which dopamine population dominates each MBON; cell-body positions.
- **Invented:** the glomerulus encoder; k-winners-take-all instead of the APL neuron; rate
  units; one mushroom body per push direction; dopamine as an idealised prediction error;
  recovery of synapses under the opposing dopamine; bounded gains starting at 0.5; settling
  plasticity; spontaneous choices; the reward schedule, its posture signal and its scale; and
  the hyperparameters in
  [results/hyperparameters.json](results/hyperparameters.json).

## Viewer

```bash
python3 -m http.server -d web   # then open http://localhost:8000
```

The viewer runs one trained fly live: seed 0, the default rather than a seed picked for looks,
after 3,000 episodes. It averaged 185 steps over its last 100 training episodes, well below
the evaluation median of 425, and its longest run reached the 500-step cap. Every episode starts from a random state, and the fly's
choices are computed in the browser from the exported synapse gains (a test checks them
against the Python model to 1e-9). Left: the cart, both sides' values and the dopamine
signal. Right: both mushroom bodies at their MaleCNS cell-body positions; PNs (blue) follow
the glomerulus code, active KCs light up (the chosen side brightest), MBONs (magenta)
brighten with their output, and PAM (green, ▲ better than expected) and PPL1 (amber,
✕ worse than expected) flash with the prediction error. Switch the brain to "before
learning" to watch the naive fly fail. Cell bodies sit at the brain surface, so a dot
identifies a neuron, not where its synapses compute.

## Run it

```bash
uv sync
uv run fly-cartpole extract    # 1.1 GB download once; data/*.npz are already committed
uv run fly-cartpole tune       # searches on tuning seeds 100-139
uv run fly-cartpole compare    # every condition on seeds 70-89, writes results/
uv run fly-cartpole export     # trains the fly on seed 0 and writes web/data/ for the viewer
uv run pytest
```

On 12 CPU cores `tune` takes about 2 hours 40 minutes and `compare` about 1 hour 35 minutes.

## Limitations

- Only the mushroom bodies are simulated; nothing else in the brain is computed or drawn.
- Rate units, no spikes, no APL, KC→KC or MBON→MBON connections.
- The fly balances well but not perfectly: it averages 393 steps, individual flies range from
  162 to 500, and a TD learner on the same code and reward is perfect.
- The posture signal tells the fly more than whether the pole fell. The fly without it
  (233.8 on seeds 50-69, after 2,000 episodes) is the cleaner test of learning from sparse
  outcomes.
- The outcome scale, the posture signal, exploration and the training length were chosen after
  earlier evaluations had fallen short; each choice was tested on tuning seeds first and then
  evaluated on unused seeds, but the sequence of choices was guided by what those shortfalls
  revealed.
- MBON valence labels follow the compartmental account and were not checked against
  individually characterised MBONs.
- One dataset and twenty evaluation seeds. Each seed also draws its own glomerulus-to-variable
  assignment, shared by every condition of that seed, so the spread across seeds mixes that
  assignment with learning noise.

## Credits

- MaleCNS v1.0 connectome, CC BY 4.0; see [NOTICE.md](NOTICE.md).
- Method adapted from [fly-blackjack](https://github.com/WilliamJones/fly-blackjack) (MIT).
- Aso et al. 2014, *eLife* (MBON valence); Hige et al. 2015, *Neuron* (KC→MBON
  depression); Handler et al. 2019, *Cell* (timing of dopamine plasticity); Bennett et al.
  2021, *Nature Communications* (prediction errors in the mushroom body); Barto, Sutton and
  Anderson 1983 (cart-pole and the actor-critic); Ng, Harada and Russell 1999, *ICML*
  (potential-based reward shaping); Maye et al. 2007, *PLoS ONE* (spontaneous behaviour in
  flies).
