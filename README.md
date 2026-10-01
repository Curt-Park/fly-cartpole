# fly-cartpole

<p align="center">
  <img src="assets/fly-cartpole.svg" alt="A worried fruit fly on a cart, holding a wobbling pole upright" width="640">
</p>

**English** · [한국어](README.ko.md)

## Background and goal

If you took a fruit fly's brain as it is, could it solve CartPole? To find out, I cut neural circuits
out of [MaleCNS v1.0](https://male-cns.janelia.org/), the connectome of the adult male fruit fly, and
wired them to Gymnasium's CartPole-v1.

The rule I cared about most was **to learn without backpropagation or any other machine learning
machinery, using nothing but the fly's own neurons**. There is no trained artificial network: every
computation that keeps the pole up is done by fly neurons connected exactly as the connectome records
them. Learning is limited to what a fly could plausibly do. Every episode the fly nudges its sensor gains
(how strongly it listens to each sense) at random, and a dopamine signal reports whether that episode
beat its recent record. The gains then move toward the nudge if it did and away from it if it did not,
in proportion to the dopamine. The
wiring comes straight from the connectome, but this learning rule does not: it is an assumption
inspired by fly biology, not something shown in real flies.

The circuit I ended up with is the **flight-stabilisation circuit**, which a fly uses to keep its
balance in the air. It maps the fly's flight onto CartPole: the circuit balances the pole as a reflex
and holds its place by watching a landmark. The web viewer simulates all 5,459 of its neurons live in
the browser, so you can watch the fly at work.

![The web viewer: the cart and pole, the steering signal and the sensor gains, the fly seen from behind, and the flight circuit at its MaleCNS cell-body positions](assets/viewer.gif)

## Results

Each condition was evaluated once, on 20 seeds that had never been used before. A CartPole-v1 episode
lasts at most 500 steps.

| fly | what it tunes | seeds | mean of the last 100 episodes |
|---|---|---|---|
| wiring only | nothing | 160-179 | 275.6 ± 9.4 |
| self-tuning | three sensor gains | 160-179 | 499.9 ± 0.3 |
| **landmark + curriculum learning** | five sensor gains | 240-259 | **500.0 ± 0.0** |
| random pushes (baseline) | – | 160-179 | 22.1 ± 0.8 |

Once a fly tunes itself, though, it scores close to 500 almost every time, so differences barely show
within the cap. I therefore ran the landmark flies for up to 2,000 steps per episode, ten episodes per
seed, with their tuned gains and no exploration. The comparison is a landmark fly tuned only to keep
the pole up for as long as possible.

| fly | track exits | pole falls | mean distance from the centre |
|---|---|---|---|
| tuned only to keep the pole up | 10 of 200 | 0 | 0.48 m |
| **curriculum learning** | **0 of 200** | **0** | **0.12 m** |

The full numbers and permutation tests are in [reflex fly](results/reflex/summary.md) and
[landmark + curriculum learning](results/staged/summary.md).

## What finally worked

- **Using the flight circuit.** I built on something the fly brain is already good at. The ocelli read
  the pole's tilt as if the body were rolling, the halteres read how fast it is falling, and the HS
  cells read the cart's velocity as optic flow. Which way to push the cart is decided by the difference
  between the left and right wing steering motor neurons (b1, b2); in effect, the cart is pushed toward
  the wing that beats harder.
- **Self-tuning.** The fly fine-tunes its own sensor gains. Every episode it tries slightly different
  gains, and if it lasts longer than its recent record, it moves toward them. What really matters is
  the ratio between gains: in the original wiring the haltere signal is about 48 times stronger than
  the ocellar one, and tuning brings that down to about 5.
- **Landmark: position.** This came from how flies hover in place. A fly that wants to move sideways
  first banks toward where it is going, so when the cart is right of centre, the fly reads it as if the
  pole were leaning right.
- **Landmark: motion.** How fast the landmark slides across the fly's view is fed in as well. It damps
  the wobble in position as the cart returns to the centre.
- **Curriculum learning.** At first the fly is rewarded for keeping the pole up for long; once it is
  good at that, it is rewarded for holding its position. In the second stage the balance gains are
  frozen, so that holding position cannot erode balance.

## Experiment log

Every hypothesis, including the ones that failed, the seed protocol and the analysis behind each step
are in [EXPERIMENTS.md](EXPERIMENTS.md). It also covers my first attempt before the flight circuit,
with the fly's learning centre, the mushroom body (MB), which learned by changing its synapses with
dopamine and reached an average of 392.5 steps.

## Run it

```bash
uv sync
python3 -m http.server -d web        # the web viewer at http://localhost:8000
uv run fly-cartpole reflex-tune      # pick the self-tuning rates on tuning seeds 100-109
uv run fly-cartpole station-compare --seeds 240-259 \
  --conditions fly-reflex-station-staged-motion,fly-reflex-station --results results/staged
uv run fly-cartpole export-flight    # tune the viewer's fly on seed 0 and write web/data/flight.json
uv run pytest
```

The extracted circuit is already in the repository. To extract it again from MaleCNS, run
`uv run fly-cartpole extract-flight` (it downloads about 1.1 GB the first time). On 12 CPU cores,
`reflex-tune` takes about 1.5 hours and `station-compare` about an hour.

## Working with an AI assistant

In broad strokes, this is how Claude Code (running Claude Opus 5.5 at high effort) and I worked together.

- **I set the direction.** Most of the big turns started from my questions or ideas: switching to the
  flight circuit, letting the fly tune itself, taking a hint from hovering, and curriculum learning.
- **Claude Code built and verified.** It turned each idea into a working mechanism, tried it cheaply
  on tuning seeds first, and reported the numbers, failures included. I decided the next step from
  those numbers.

Because Claude Code quickly built and ran whatever question or idea I brought to it, I was able to
test many hypotheses in a short time.

In the early experiments I also left Claude Code to improve the fly on its own overnight, but it could
not push the performance past a certain level. Rather than leaving everything to the AI, I think it
matters that the person actively brings ideas and proposes directions too.

## Credits

- MaleCNS v1.0 connectome (CC BY 4.0); see [NOTICE.md](NOTICE.md).
- Flight: Fayyazuddin and Dickinson 1996, *Journal of Neuroscience* (haltere input to the b1 steering
  motor neuron); Dickinson 1999, *Philosophical Transactions of the Royal Society B* (haltere-mediated
  equilibrium reflexes); Chan, Prete and Dickinson 1998, *Science* (visual input to the haltere's
  steering muscles); Reichardt and Poggio 1976, *Quarterly Reviews of Biophysics* (position and motion
  in fly orientation).
- Barto, Sutton and Anderson 1983 (the cart-pole task).
