import assert from "node:assert/strict";
import { test } from "node:test";

import { framesDue } from "../../web/playback.js";

test("advances one frame per elapsed step interval", () => {
  assert.equal(framesDue(40, 20), 2);
});

test("waits when less than one step interval has passed", () => {
  assert.equal(framesDue(16.7, 20), 0);
});

test("keeps fast playback faster than the display refresh", () => {
  // 4x speed at 50 steps/s is a 5 ms step; a 60 Hz frame is ~16.7 ms.
  assert.equal(framesDue(16.7, 5), 3);
});
