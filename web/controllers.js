// Controllers the viewer runs without the circuit; they mirror the memoryless fly and the LQR in fly_cartpole.
// The state each sense reads, in sense order: drift and the landmark's motion both see the cart's velocity.
const SENSE_STATE = [2, 3, 1, 0, 1];

export function linearSteer(state, stateLimits, weights, gains) {
  let steer = 0;
  SENSE_STATE.forEach((index, sense) => {
    const normalised = Math.min(1, Math.max(-1, state[index] / stateLimits[index]));
    steer += gains[sense] * weights[sense] * normalised;
  });
  return steer;
}

export function lqrForce(state, gains) {
  let product = 0;
  state.forEach((value, index) => { product += gains[index] * value; });
  return -product;
}
