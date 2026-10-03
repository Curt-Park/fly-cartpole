import assert from "node:assert/strict";
import { test } from "node:test";

import { linearSteer, lqrForce } from "../../web/controllers.js";

const LIMITS = [2.4, 3.0, 0.21, 3.5];

test("the linear controller weighs each sense by its gain and the circuit's settled weight", () => {
  // Senses read angle, angular velocity, cart velocity, cart position and cart velocity again, clipped to ±1.
  const state = [1.2, 1.5, 0.105, 7.0];
  const steer = linearSteer(state, LIMITS, [1, 2, 3, 4, 5], [1, 1, 1, 0.5, 2]);
  assert.equal(steer, 1 * 0.5 + 2 * 1 + 3 * 0.5 + 4 * 0.5 * 0.5 + 5 * 2 * 0.5);
});

test("the LQR pushes with minus its gains times the state", () => {
  assert.equal(lqrForce([1, 2, 3, 4], [-1, 0.5, -2, 0]), -(-1 + 1 - 6 + 0));
});
