// Checks the browser reflex and CartPole against values Python exported: node reflex_parity.mjs flight.json
import { readFileSync } from "node:fs";

import { step } from "../../web/cartpole.js";
import { linearSteer, lqrForce } from "../../web/controllers.js";
import { createReflex } from "../../web/reflex.js";

const model = JSON.parse(readFileSync(process.argv[2], "utf8"));
const TOLERANCE = 1e-9;
const failures = [];

const reflex = createReflex(model);
reflex.reset();
const { states, actions, steers, gains } = model.trajectory;
states.forEach((state, index) => {
  const { action, steer } = reflex.act(state, gains);
  if (action !== actions[index]) failures.push(`step ${index}: action ${action} != ${actions[index]}`);
  if (Math.abs(steer - steers[index]) > TOLERANCE) failures.push(`step ${index}: steer ${steer} != ${steers[index]}`);
});

// The controllers the viewer runs without the circuit must act as Python did on the same states.
const others = [
  ["linear", (state) => linearSteer(state, model.state_limits, model.linear.weights, model.linear.trajectory.gains)],
  ["lqr", (state) => lqrForce(state, model.lqr.gains)],
];
for (const [name, steerFor] of others) {
  const { states: seen, actions: chosen, steers: steered } = model[name].trajectory;
  seen.forEach((state, index) => {
    const steer = steerFor(state);
    if ((steer > 0 ? 1 : 0) !== chosen[index]) failures.push(`${name} step ${index}: action differs`);
    if (Math.abs(steer - steered[index]) > TOLERANCE * Math.max(1, Math.abs(steered[index]))) failures.push(`${name} step ${index}: steer ${steer} != ${steered[index]}`);
  });
}

let state = model.physics.start;
model.physics.actions.forEach((action, index) => {
  state = step(state, action);
  state.forEach((value, dimension) => {
    const expected = model.physics.states[index][dimension];
    if (Math.abs(value - expected) > TOLERANCE) failures.push(`physics step ${index} dim ${dimension}: ${value} != ${expected}`);
  });
});

if (failures.length) {
  console.error(failures.slice(0, 10).join("\n"));
  process.exit(1);
}
