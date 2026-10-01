import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { ANGLE_LIMIT, MAX_STEPS, TRACK_LIMIT, fell, randomState, step } from "./cartpole.js";
import { framesDue } from "./playback.js";
import { createReflex } from "./reflex.js";

const STEPS_PER_SECOND = 50; // CartPole advances 0.02 s per step
const HOLD_LAST_FRAME_MS = 1200;
// Large populations rest dimmer: overlapping additive points would otherwise hide which cells are active.
const STYLES = {
  haltere: { colour: "#fab219", size: 0.03, resting: 0.2 },
  ocellar: { colour: "#0ca30c", size: 0.06, resting: 0.25 },
  hs: { colour: "#3987e5", size: 0.06, resting: 0.25 },
  interneuron: { colour: "#c3c2b7", size: 0.014, resting: 0.04 },
  wing_motor: { colour: "#d55181", size: 0.05, resting: 0.2 },
};

const model = await fetch("data/flight.json").then((response) => response.json());
const reflex = createReflex(model);

// Scene: one point cloud per role.
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

function makeCloud(role) {
  const roleIndex = model.roles.indexOf(role);
  const kept = model.role.flatMap((value, index) => (value === roleIndex && model.positions[index] ? [index] : []));
  const positions = new Float32Array(kept.length * 3);
  kept.forEach((index, row) => positions.set(model.positions[index], row * 3));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(kept.length * 3), 3));
  const material = new THREE.PointsMaterial({
    size: STYLES[role].size, vertexColors: true, sizeAttenuation: true,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  scene.add(new THREE.Points(geometry, material));
  return { role, geometry, neurons: kept, base: new THREE.Color(STYLES[role].colour), scale: 1e-6 };
}

const clouds = Object.keys(STYLES).map(makeCloud);

function paint(cloud, activity) {
  const colours = cloud.geometry.attributes.color;
  const { resting } = STYLES[cloud.role];
  let peak = 0;
  for (const index of cloud.neurons) peak = Math.max(peak, Math.abs(activity[index]));
  // Each population is scaled to its own recent peak: sensors and interneurons differ by orders of magnitude.
  cloud.scale = Math.max(cloud.scale * 0.995, peak, 1e-6);
  cloud.neurons.forEach((index, row) => {
    const level = resting + (1 - resting) * Math.min(1, Math.abs(activity[index]) / cloud.scale);
    colours.setXYZ(row, cloud.base.r * level, cloud.base.g * level, cloud.base.b * level);
  });
  colours.needsUpdate = true;
}

// The view turns about the vertical axis, so the horizontal radius bounds what must stay in frame.
const radii = model.positions.filter(Boolean).map(([x, , z]) => Math.hypot(x, z)).sort((first, second) => first - second);
const radius = radii[radii.length - 1];
// Start from the fly's side, where the brain and nerve cord lie end to end.
camera.position.set((radius * 1.1) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)), radius * 0.3, 0);

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
  if (gains[3] > 0) {
    // The landmark the fly holds station on: a vertical stripe above the track's centre.
    context.fillStyle = "#383835";
    context.fillRect(width / 2 - 4, 24, 8, groundY - 140);
    context.fillStyle = "#898781";
    context.font = "12px system-ui, sans-serif";
    context.fillText("landmark", width / 2 + 10, 36);
  }
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

// Live simulation: the reflex runs on the exported circuit; nothing learns in the browser.
const elements = Object.fromEntries(
  ["play", "restart", "speed", "gains", "episode", "step", "steer", "best", "mean", "steer-bar", "gain-angle", "gain-rate", "gain-drift", "gain-position"]
    .map((id) => [id, document.getElementById(id)]),
);
const lengths = [];
const GAIN_SETS = { fixed: model.gains_fixed, adapted: model.gains_adapted, station: model.gains_station };
let gains = GAIN_SETS[elements.gains.value];
let state;
let steps;
let latest;
let done;
let heldSince = null;
let playing = true;
let lastTick = performance.now();
let steerScale = 1e-6;
const idle = new Float64Array(reflex.size);

function newEpisode() {
  state = randomState();
  reflex.reset();
  steps = 0;
  done = false;
  heldSince = null;
  latest = { action: 0, steer: 0, activity: idle };
  render();
}

function advance() {
  latest = reflex.act(state, gains);
  state = step(state, latest.action);
  steps += 1;
  done = fell(state) || steps >= MAX_STEPS;
  if (done) lengths.push(steps);
}

function render() {
  const { action, steer, activity } = latest;
  drawCartPole(state, action, done);
  clouds.forEach((cloud) => paint(cloud, activity));
  steerScale = Math.max(steerScale * 0.995, Math.abs(steer), 1e-6);
  const bar = elements["steer-bar"].firstElementChild;
  const share = Math.min(1, Math.abs(steer) / steerScale) * 50;
  bar.style.left = `${steer >= 0 ? 50 : 50 - share}%`;
  bar.style.width = `${share}%`;
  elements.steer.textContent = `${steer >= 0 ? "+" : ""}${steer.toExponential(2)}`;
  elements.episode.textContent = lengths.length + (done ? 0 : 1);
  elements.step.textContent = steps;
  const recent = lengths.slice(-10);
  elements.best.textContent = lengths.length ? Math.max(...lengths) : "–";
  elements.mean.textContent = recent.length ? (recent.reduce((total, length) => total + length, 0) / recent.length).toFixed(0) : "–";
  ["angle", "rate", "drift", "position"].forEach((name, index) => { elements[`gain-${name}`].textContent = gains[index].toFixed(2); });
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
elements.gains.addEventListener("change", () => {
  gains = GAIN_SETS[elements.gains.value];
  lengths.length = 0;
  newEpisode();
});

newEpisode();
requestAnimationFrame(tick);
