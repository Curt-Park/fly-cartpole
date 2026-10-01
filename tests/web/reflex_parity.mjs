// Checks the browser reflex and CartPole against values Python exported: node reflex_parity.mjs flight.json
import { readFileSync } from "node:fs";

import { step } from "../../web/cartpole.js";
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
