// CartPole-v1 dynamics exactly as Gymnasium integrates them (explicit Euler).
const GRAVITY = 9.8;
const MASS_CART = 1.0;
const MASS_POLE = 0.1;
const TOTAL_MASS = MASS_CART + MASS_POLE;
const HALF_LENGTH = 0.5;
const POLE_MASS_LENGTH = MASS_POLE * HALF_LENGTH;
const FORCE = 10.0;
const TAU = 0.02;

export const TRACK_LIMIT = 2.4;
export const ANGLE_LIMIT = (12 * 2 * Math.PI) / 360;
export const MAX_STEPS = 500;

export function randomState(random = Math.random) {
  return [0, 0, 0, 0].map(() => random() * 0.1 - 0.05);
}

export function step([x, xDot, theta, thetaDot], action) {
  const force = action === 1 ? FORCE : -FORCE;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const temp = (force + POLE_MASS_LENGTH * (thetaDot * thetaDot) * sin) / TOTAL_MASS;
  const thetaAcc = (GRAVITY * sin - cos * temp) / (HALF_LENGTH * (4.0 / 3.0 - (MASS_POLE * (cos * cos)) / TOTAL_MASS));
  const xAcc = temp - (POLE_MASS_LENGTH * thetaAcc * cos) / TOTAL_MASS;
  return [x + TAU * xDot, xDot + TAU * xAcc, theta + TAU * thetaDot, thetaDot + TAU * thetaAcc];
}

export function fell([x, , theta]) {
  return x < -TRACK_LIMIT || x > TRACK_LIMIT || theta < -ANGLE_LIMIT || theta > ANGLE_LIMIT;
}
