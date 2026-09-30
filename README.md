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

**The fly learns to balance the pole, and passes the "solved" mark.** With its innate wiring
it drops the pole after about 21 steps (`fly-bilateral-frozen`). After 2,000 episodes it
holds the pole for **233.8 steps** on average over the last 100 episodes, across 20
evaluation seeds that no earlier experiment touched. CartPole-v0 counts an average of 195
over 100 consecutive episodes as solved: 15 of the 20 flies clear that on their own, every
fly learned (161 to 304 steps), and 6% of their final episodes reached the 500-step cap. A
textbook TD actor-critic reading the same Kenyon cells scores higher on average (302), with
a far wider spread: it failed outright on two seeds.

![Learning curves](results/learning_curves.png)

| condition | what it is | final-100 mean ± std (20 seeds) | episodes reaching 500 |
|---|---|---|---|
| `fly-bilateral` | two mushroom bodies, prediction error dopamine, push-pull plasticity | **233.8 ± 43.1** | 6% |
| `fly-bilateral-shuffled` | same, with PN→KC and KC→MBON wiring shuffled (degree-preserving) | 218.3 ± 66.3 | 7% |
| `fly-bilateral-frozen` | same, without plasticity | 21.3 ± 13.9 | 0% |
| `fly` | the first design: one mushroom body, actions smelled as odours | 22.2 ± 13.6 | 0% |
| `td` | TD actor-critic on the same Kenyon cell code (not a fly) | 302.3 ± 168.0 | 38% |
| `random` | uniform random pushes (chance) | 22.2 ± 1.0 | 0% |

Permutation tests on the per-seed final-100 means (every condition and test is in
[results/summary.md](results/summary.md)):

- the fly learns: `fly-bilateral` > `fly-bilateral-frozen`, p = 0.0001;
- it beats chance: `fly-bilateral` > `random`, p = 0.0001;
- there is no clear evidence that the measured wiring helps: `fly-bilateral` >
  `fly-bilateral-shuffled`, p = 0.20. The shuffle keeps the cell counts, how many partners
  each cell has, and where each dopamine neuron lands; it only changes which Kenyon cell
  talks to which. Most of the learning comes from that layout rather than the exact synapse
  map. Earlier evaluations pointed both ways (real wiring ahead on seeds 0-9, p = 0.037;
  shuffled ahead on seeds 30-49).

Hyperparameters were searched on tuning seeds 100-139 only and applied unchanged to the
evaluation seeds 50-69 ([results/tuning.json](results/tuning.json)).

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

**Evaluation seeds were spent four times, and every spend is reported here.** Seeds 0-9
evaluated the push-pull fly at outcome scale 1.0 (194) and then at 0.1 (115, the lock-in).
Seeds 10-29 evaluated a run in which the original fly's search had also chosen the size of
the record reward for every other condition (0.5 instead of 0.1; bilateral fly 162.5); the
reward is part of the task, so it is now fixed and never searched. Seeds 30-49 evaluated the
40-seed tuned fly after 1,000 episodes (191.7, three steps short). Hypothesis 19 was then
tested on tuning seeds only, and the numbers at the top come from seeds 50-69, which no
earlier experiment touched.

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
a side with `sigmoid(beta * (left - right))`.

**Reward.** PPL1 punishment when the pole falls, and a small PAM reward on every step once
the episode is longer than the mean of the last 20 episodes.

**Dopamine as a prediction error.** Dopamine carries the surprise rather than the raw reward:
`delta = scale * (reward - punishment) + gamma * expected next value - current value`. PAM
fires for `max(delta, 0)` and PPL1 for `max(-delta, 0)`. The outcome scale (0.3 after tuning)
keeps the values the fly must learn inside the range its bounded synapses can express. Where
each dopamine population lands is read from the measured DAN→MBON wiring.

**Push-pull plasticity.** KC→MBON synapses have gains between 0 and 1, starting at 0.5. A
decaying trace remembers which KCs of the chosen side were active. Released dopamine
depresses traced synapses in the compartments it reaches (Hige et al. 2015; Handler et al.
2019), and the opposing dopamine population restores traced synapses in its own
compartments.

## Measured versus invented

- **Measured:** every PN→KC, KC→MBON, DAN→MBON and MBON→DAN connection and its synapse
  count; which dopamine population dominates each MBON; cell-body positions.
- **Invented:** the glomerulus encoder; k-winners-take-all instead of the APL neuron; rate
  units; one mushroom body per push direction; dopamine as an idealised prediction error;
  recovery of synapses under the opposing dopamine; bounded gains starting at 0.5; the
  reward schedule and its scale; and the hyperparameters in
  [results/hyperparameters.json](results/hyperparameters.json).

## Viewer

```bash
python3 -m http.server -d web   # then open http://localhost:8000
```

The viewer runs one trained fly live: seed 0, the default rather than a seed picked for looks,
after 2,000 episodes. It averaged 166 steps over its last 100 training episodes, and its
longest run reached the 500-step cap. Every episode starts from a random state, and the fly's
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
uv run fly-cartpole compare    # every condition on seeds 50-69, writes results/
uv run fly-cartpole export     # trains the fly on seed 0 and writes web/data/ for the viewer
uv run pytest
```

On 12 CPU cores `tune` takes about 45 minutes and `compare` about two and a half hours.

## Limitations

- Only the mushroom bodies are simulated; nothing else in the brain is computed or drawn.
- Rate units, no spikes, no APL, KC→KC or MBON→MBON connections.
- The fly balances, but not perfectly: it averages 234 steps, individual flies range from 161
  to 304, only 6% of final episodes reach the 500-step cap, and the TD reference averages more.
- The outcome scale and the training length were chosen after earlier evaluations had failed;
  each choice was tested on tuning seeds first and then evaluated on unused seeds, but the
  sequence of choices was guided by what those failures revealed.
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
  Anderson 1983 (cart-pole and the actor-critic).
