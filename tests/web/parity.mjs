// Checks the browser fly and CartPole against values Python exported: node parity.mjs model.json
import { readFileSync } from "node:fs";

import { step } from "../../web/cartpole.js";
import { evaluate } from "../../web/fly.js";

const model = JSON.parse(readFileSync(process.argv[2], "utf8"));
const TOLERANCE = 1e-9;
const failures = [];

for (const sample of model.samples) {
  for (const [trained, key] of [[true, "values_trained"], [false, "values_naive"]]) {
    evaluate(model, sample.state, trained).values.forEach((value, side) => {
      if (Math.abs(value - sample[key][side]) > TOLERANCE) failures.push(`${key}[${side}] ${value} != ${sample[key][side]}`);
    });
  }
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
