import assert from "node:assert/strict";
import { test } from "node:test";

import { flySignals } from "../../web/flyview.js";

// Haltere, ocellus, HS cell and b1 on each side, plus a right steering motor neuron that is not b1 or b2.
const MODEL = {
  roles: ["interneuron", "haltere", "ocellar", "hs", "wing_motor"],
  role: [1, 1, 2, 2, 3, 3, 4, 4, 4],
  side: [-1, 1, -1, 1, -1, 1, -1, 1, 1],
  amplitude: [false, false, false, false, false, false, true, true, false],
};

function activityAt(index, value = 1) {
  const activity = new Float64Array(MODEL.role.length);
  activity[index] = value;
  return activity;
}

test("the right b1 drives the right wing only", () => {
  const signals = flySignals(MODEL)(activityAt(7));
  assert.deepEqual(signals.wings, [0, 1]);
});

test("steering motor neurons other than b1 and b2 leave the wingbeat alone", () => {
  assert.deepEqual(flySignals(MODEL)(activityAt(8)).wings, [0, 0]);
});

test("each sense lights its own side", () => {
  const read = flySignals(MODEL);
  assert.deepEqual(read(activityAt(1)).halteres, [0, 1]);
  assert.deepEqual(read(activityAt(2)).ocelli, [1, 0]);
  assert.deepEqual(read(activityAt(4, -0.5)).hs, [-0.5, 0]);
});
