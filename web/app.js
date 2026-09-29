import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

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
const SIZES = { pn: 0.035, kc: 0.022, mbon: 0.07, pam: 0.05, ppl1: 0.07 };
// Large populations rest dimmer: overlapping additive points would otherwise hide which cells are active.
const RESTING = { pn: 0.15, kc: 0.04, mbon: 0.15, pam: 0.05, ppl1: 0.15 };

const [cells, recording] = await Promise.all([
  fetch("data/cells.json").then((response) => response.json()),
  fetch("data/episodes.json").then((response) => response.json()),
]);

// Scene: one point cloud per population so sizes can differ.
const container = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor("#1a1a19");
container.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
camera.position.set(0, 0.4, 3.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.6;

function makeCloud(name, points) {
  const kept = [];
  points.forEach((point, index) => { if (point) kept.push([index, point]); });
  const positions = new Float32Array(kept.length * 3);
  kept.forEach(([, point], row) => positions.set(point, row * 3));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(kept.length * 3), 3));
  const material = new THREE.PointsMaterial({
    size: SIZES[name], vertexColors: true, sizeAttenuation: true,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  scene.add(new THREE.Points(geometry, material));
  // rowOf maps a circuit index to its drawn row; cells without a soma have none.
  const rowOf = new Map(kept.map(([index], row) => [index, row]));
  return { name, geometry, rowOf, intensity: new Float32Array(points.length) };
}

const clouds = {
  pn: makeCloud("pn", cells.pn),
  kc: makeCloud("kc", cells.kc),
  mbon: makeCloud("mbon", cells.mbon),
  pam: makeCloud("pam", cells.dan.map((point, index) => (cells.dan_is_punishment[index] ? null : point))),
  ppl1: makeCloud("ppl1", cells.dan.map((point, index) => (cells.dan_is_punishment[index] ? point : null))),
};

function paint(cloud) {
  const colours = cloud.geometry.attributes.color;
  const base = COLOURS[cloud.name];
  const resting = RESTING[cloud.name];
  for (const [index, row] of cloud.rowOf) {
    const level = resting + (1 - resting) * Math.min(1, cloud.intensity[index]);
    colours.setXYZ(row, base.r * level, base.g * level, base.b * level);
  }
  colours.needsUpdate = true;
}

// Frame the bulk of the cells; a few distant dopamine somas may fall outside.
function fitCamera() {
  const radii = [cells.pn, cells.kc, cells.mbon, cells.dan].flat().filter(Boolean).map((point) => Math.hypot(...point)).sort((a, b) => a - b);
  const radius = radii[Math.floor(radii.length * 0.95)];
  camera.position.set(0, radius * 0.3, radius / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.15);
}
fitCamera();

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
const TRACK_LIMIT = 2.4;
const POLE_LENGTH = 1.0;

function drawCartPole(x, theta, action) {
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
  context.strokeStyle = "#ffffff";
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

// Playback.
const elements = Object.fromEntries(
  ["play", "speed", "episode", "scrub", "step", "baseline", "dopamine", "score-left", "score-right", "bar-left", "bar-right"]
    .map((id) => [id, document.getElementById(id)]),
);
let frames = recording[elements.episode.value];
let frameIndex = 0;
let playing = true;
let lastTick = performance.now();
let heldSince = null;
let rewardGlow = 0;
let punishGlow = 0;
let scoreScale = 1;
let mbonScale = 1;

function selectEpisode(name) {
  frames = recording[name];
  frameIndex = 0;
  elements.scrub.max = frames.step.length - 1;
  scoreScale = Math.max(1e-9, ...frames.score_left.map(Math.abs), ...frames.score_right.map(Math.abs));
  mbonScale = Math.max(1e-9, ...frames.mbon.flat());
  rewardGlow = punishGlow = 0;
  render();
}

function render() {
  const index = frameIndex;
  drawCartPole(frames.x[index], frames.theta[index], frames.action[index]);

  const glomeruli = frames.glomeruli[index];
  cells.pn_group.forEach((group, pn) => { clouds.pn.intensity[pn] = glomeruli[group]; });
  clouds.kc.intensity.fill(0);
  for (const kc of frames.kc[index]) clouds.kc.intensity[kc] = 1;
  frames.mbon[index].forEach((value, mbon) => { clouds.mbon.intensity[mbon] = value / mbonScale; });
  if (frames.reward[index] > 0) rewardGlow = 1;
  if (frames.punish[index] > 0) punishGlow = 1;
  clouds.pam.intensity.fill(rewardGlow);
  clouds.ppl1.intensity.fill(punishGlow);
  Object.values(clouds).forEach(paint);

  const [left, right] = [frames.score_left[index], frames.score_right[index]];
  elements["score-left"].textContent = left.toFixed(4);
  elements["score-right"].textContent = right.toFixed(4);
  elements["bar-left"].firstElementChild.style.width = `${(50 + 50 * left / scoreScale).toFixed(1)}%`;
  elements["bar-right"].firstElementChild.style.width = `${(50 + 50 * right / scoreScale).toFixed(1)}%`;
  elements["bar-left"].classList.toggle("chosen", frames.action[index] === 0);
  elements["bar-right"].classList.toggle("chosen", frames.action[index] === 1);
  elements.step.textContent = `${frames.step[index]} / ${frames.step.length}`;
  const baseline = frames.baseline[index];
  elements.baseline.textContent = baseline === null ? "none yet" : baseline.toFixed(1);
  if (frames.punish[index] > 0) {
    elements.dopamine.textContent = "✕ punishment (PPL1): the pole fell";
    elements.dopamine.style.color = "var(--punish)";
  } else if (frames.reward[index] > 0) {
    elements.dopamine.textContent = "▲ reward (PAM): beating the baseline";
    elements.dopamine.style.color = "var(--reward)";
  } else {
    elements.dopamine.textContent = "no dopamine";
    elements.dopamine.style.color = "var(--ink-muted)";
  }
  elements.scrub.value = index;
}

function tick(now) {
  const stepMs = 1000 / (STEPS_PER_SECOND * Number(elements.speed.value));
  const due = playing ? framesDue(now - lastTick, stepMs) : 0;
  if (!playing) lastTick = now;
  if (due > 0) {
    lastTick += due * stepMs;
    const advance = Math.min(due, frames.step.length - 1 - frameIndex);
    if (advance > 0) {
      for (let frame = 0; frame < advance; frame += 1) {
        frameIndex += 1;
        rewardGlow *= 0.85;
        punishGlow *= 0.92;
        if (frames.reward[frameIndex] > 0) rewardGlow = 1;
        if (frames.punish[frameIndex] > 0) punishGlow = 1;
      }
      render();
    } else if (heldSince === null) {
      heldSince = now;
    } else if (now - heldSince > HOLD_LAST_FRAME_MS) {
      heldSince = null;
      frameIndex = 0;
      rewardGlow = punishGlow = 0;
      render();
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
elements.episode.addEventListener("change", () => selectEpisode(elements.episode.value));
elements.scrub.addEventListener("input", () => {
  frameIndex = Number(elements.scrub.value);
  render();
});

selectEpisode(elements.episode.value);
requestAnimationFrame(tick);
