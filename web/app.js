import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { ANGLE_LIMIT, MAX_STEPS, TRACK_LIMIT, fell, randomState, step } from "./cartpole.js";
import { evaluate, probabilityLeft } from "./fly.js";
import { framesDue } from "./playback.js";

const STEPS_PER_SECOND = 50; // CartPole advances 0.02 s per step
const HOLD_LAST_FRAME_MS = 1200;
const COLOURS = {
  pn: new THREE.Color("#3987e5"),
  kc: new THREE.Color("#c3c2b7"),
  mbon: new THREE.Color("#d55181"),
  pam: new THREE.Color("#0ca30c"),
  ppl1: new THREE.Color("#fab219"),
};
const SIZES = { pn: 0.03, kc: 0.02, mbon: 0.06, pam: 0.045, ppl1: 0.06 };
// Large populations rest dimmer: overlapping additive points would otherwise hide which cells are active.
const RESTING = { pn: 0.15, kc: 0.04, mbon: 0.15, pam: 0.05, ppl1: 0.15 };
const SIDE_NAMES = ["left", "right"];

const [cells, model] = await Promise.all([
  fetch("data/cells.json").then((response) => response.json()),
  fetch("data/model.json").then((response) => response.json()),
]);

// Scene: one point cloud per population and side, so sizes and colours can differ.
const container = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor("#1a1a19");
container.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;

function makeCloud(population, points) {
  const kept = [];
  points.forEach((point, index) => { if (point) kept.push([index, point]); });
  const positions = new Float32Array(kept.length * 3);
  kept.forEach(([, point], row) => positions.set(point, row * 3));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(kept.length * 3), 3));
  const material = new THREE.PointsMaterial({
    size: SIZES[population], vertexColors: true, sizeAttenuation: true,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  scene.add(new THREE.Points(geometry, material));
  // rowOf maps a circuit index to its drawn row; cells without a soma have none.
  return { population, geometry, rowOf: new Map(kept.map(([index], row) => [index, row])), intensity: new Float32Array(points.length) };
}

const clouds = {
  pam: makeCloud("pam", cells.dan.map((point, index) => (cells.dan_is_punishment[index] ? null : point))),
  ppl1: makeCloud("ppl1", cells.dan.map((point, index) => (cells.dan_is_punishment[index] ? point : null))),
};
for (const side of SIDE_NAMES) {
  for (const population of ["pn", "kc", "mbon"]) clouds[`${side}_${population}`] = makeCloud(population, cells[side][population]);
}

function paint(cloud) {
  const colours = cloud.geometry.attributes.color;
  const base = COLOURS[cloud.population];
  const resting = RESTING[cloud.population];
  for (const [index, row] of cloud.rowOf) {
    const level = resting + (1 - resting) * Math.min(1, cloud.intensity[index]);
    colours.setXYZ(row, base.r * level, base.g * level, base.b * level);
  }
  colours.needsUpdate = true;
}

// Frame the bulk of the cells; a few distant dopamine somas may fall outside.
const radii = [cells.left, cells.right].flatMap((side) => [...side.pn, ...side.kc, ...side.mbon]).concat(cells.dan)
  .filter(Boolean).map((point) => Math.hypot(...point)).sort((first, second) => first - second);
const radius = radii[Math.floor(radii.length * 0.95)];
camera.position.set(0, radius * 0.3, (radius / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.1);

function resize() {
  const width = container.clientWidth;
  renderer.setSize(width, width);
  camera.aspect = 1;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// CartPole panel.
const canvas = document.getElementById("cartpole");
const context = canvas.getContext("2d");
const POLE_LENGTH = 1.0;

function drawCartPole([x, , theta], action, done) {
  const { width, height } = canvas;
  const scale = (width - 80) / (2 * TRACK_LIMIT);
  const groundY = height * 0.78;
  const cartX = width / 2 + x * scale;
  context.fillStyle = "#1a1a19";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#383835";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(40, groundY);
  context.lineTo(width - 40, groundY);
  context.stroke();
  context.fillStyle = "#c3c2b7";
  context.fillRect(cartX - 36, groundY - 22, 72, 22);
  context.strokeStyle = done && Math.abs(theta) > ANGLE_LIMIT ? "#fab219" : "#ffffff";
  context.lineWidth = 6;
  context.lineCap = "round";
  context.beginPath();
  context.moveTo(cartX, groundY - 22);
  context.lineTo(cartX + Math.sin(theta) * POLE_LENGTH * scale, groundY - 22 - Math.cos(theta) * POLE_LENGTH * scale);
  context.stroke();
  context.fillStyle = "#898781";
  context.font = "13px system-ui, sans-serif";
  context.fillText(action === 0 ? "◀ push left" : "push right ▶", action === 0 ? cartX - 110 : cartX + 44, groundY - 6);
}

// Live simulation: the exported brain is frozen; dopamine shows the prediction error it would signal.
const elements = Object.fromEntries(
  ["play", "restart", "speed", "brain", "episode", "step", "dopamine", "best", "mean", "value-left", "value-right", "bar-left", "bar-right"]
    .map((id) => [id, document.getElementById(id)]),
);
const lengths = [];
let trained = true;
let state;
let steps;
let evaluation;
let latest;
let done;
let heldSince = null;
let playing = true;
let lastTick = performance.now();
let rewardGlow = 0;
let punishGlow = 0;
let valueScale = 1e-6;
const mbonScale = [1e-6, 1e-6];

function baseline() {
  if (!lengths.length) return model.initial_baseline;
  const recent = lengths.slice(-model.baseline_window);
  return recent.reduce((total, length) => total + length, 0) / recent.length;
}

function newEpisode() {
  state = randomState();
  steps = 0;
  done = false;
  heldSince = null;
  evaluation = evaluate(model, state, trained);
  latest = { action: evaluation.values[0] >= evaluation.values[1] ? 0 : 1, error: 0, evaluation };
  rewardGlow = punishGlow = 0;
  render();
}

// Closer to upright and centred is better; mirrors fly_cartpole.dopamine.posture_potential.
function posturePotential([x, , theta]) {
  return -((theta / ANGLE_LIMIT) ** 2 + (x / TRACK_LIMIT) ** 2);
}

function advance() {
  const current = evaluation;
  const action = Math.random() < probabilityLeft(model, current.values) ? 0 : 1;
  const before = state;
  state = step(state, action);
  steps += 1;
  const fellOver = fell(state);
  done = fellOver || steps >= MAX_STEPS;
  const limit = baseline();
  const posture = model.posture_weight * (model.gamma * posturePotential(state) - posturePotential(before));
  const reward = (!fellOver && limit !== null && steps > limit ? model.reward_per_step : 0) + posture;
  const punish = fellOver ? 1 : 0;
  evaluation = evaluate(model, state, trained);
  let upcoming = 0;
  if (!fellOver) {
    const pLeft = probabilityLeft(model, evaluation.values);
    upcoming = pLeft * evaluation.values[0] + (1 - pLeft) * evaluation.values[1];
  }
  const error = model.outcome_scale * (reward - punish) + model.gamma * upcoming - current.values[action];
  latest = { action, error, evaluation: current };
  rewardGlow = Math.max(rewardGlow * 0.85, Math.min(1, Math.max(error, 0) * 10));
  punishGlow = Math.max(punishGlow * 0.92, Math.min(1, Math.max(-error, 0) * 3));
  if (done) lengths.push(steps);
}

function render() {
  const { action, error, evaluation: shown } = latest;
  drawCartPole(state, action, done);
  shown.sides.forEach((side, index) => {
    const name = SIDE_NAMES[index];
    model.sides[index].pn_group.forEach((group, pn) => { clouds[`${name}_pn`].intensity[pn] = shown.activity[group]; });
    const kcCloud = clouds[`${name}_kc`];
    kcCloud.intensity.fill(0);
    for (const kc of side.active) kcCloud.intensity[kc] = index === action ? 1 : 0.35;
    mbonScale[index] = Math.max(mbonScale[index], ...side.mbon);
    side.mbon.forEach((value, mbon) => { clouds[`${name}_mbon`].intensity[mbon] = value / mbonScale[index]; });
  });
  clouds.pam.intensity.fill(rewardGlow);
  clouds.ppl1.intensity.fill(punishGlow);
  Object.values(clouds).forEach(paint);

  valueScale = Math.max(valueScale, ...shown.values.map(Math.abs));
  ["left", "right"].forEach((name, index) => {
    const value = shown.values[index];
    elements[`value-${name}`].textContent = value.toFixed(4);
    elements[`bar-${name}`].firstElementChild.style.width = `${(50 + (50 * value) / valueScale).toFixed(1)}%`;
    elements[`bar-${name}`].classList.toggle("chosen", action === index);
  });
  elements.episode.textContent = lengths.length + (done ? 0 : 1);
  elements.step.textContent = steps;
  if (error < -1e-3) {
    elements.dopamine.textContent = `✕ PPL1: worse than expected (${error.toFixed(3)})`;
    elements.dopamine.style.color = "var(--punish)";
  } else if (error > 1e-3) {
    elements.dopamine.textContent = `▲ PAM: better than expected (+${error.toFixed(3)})`;
    elements.dopamine.style.color = "var(--reward)";
  } else {
    elements.dopamine.textContent = "no surprise";
    elements.dopamine.style.color = "var(--ink-muted)";
  }
  const recent = lengths.slice(-10);
  elements.best.textContent = lengths.length ? Math.max(...lengths) : "–";
  elements.mean.textContent = recent.length ? (recent.reduce((total, length) => total + length, 0) / recent.length).toFixed(0) : "–";
}

function tick(now) {
  const stepMs = 1000 / (STEPS_PER_SECOND * Number(elements.speed.value));
  const due = playing ? framesDue(now - lastTick, stepMs) : 0;
  if (!playing) lastTick = now;
  if (due > 0) {
    lastTick += due * stepMs;
    if (!done) {
      for (let frame = 0; frame < due && !done; frame += 1) advance();
      render();
    } else if (heldSince === null) {
      heldSince = now;
    } else if (now - heldSince > HOLD_LAST_FRAME_MS) {
      newEpisode();
    }
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

elements.play.addEventListener("click", () => {
  playing = !playing;
  elements.play.textContent = playing ? "Pause" : "Play";
});
elements.restart.addEventListener("click", newEpisode);
elements.brain.addEventListener("change", () => {
  trained = elements.brain.value === "trained";
  lengths.length = 0;
  newEpisode();
});

newEpisode();
requestAnimationFrame(tick);
