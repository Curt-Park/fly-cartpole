// The same moment seen as flight, from behind the fly: the pole's lean is its roll, the cart's motion the ground flowing past.
import { TRACK_LIMIT } from "./cartpole.js";

const SIDES = [-1, 1]; // left, right: seen from behind, the fly's sides match the screen's
const GROUND_SPACING = 0.25; // metres between ground marks
const WING_LENGTH = 92;
const WING_RAISE = -0.25; // stroke midline, radians above horizontal
const RESTING_AMPLITUDE = 0.6;
const AMPLITUDE_SWING = 0.4;

export function flySignals(model) {
  const roleIndex = (name) => model.roles.indexOf(name);
  const perSide = (belongs) => SIDES.map((side) => model.role.flatMap((role, index) => (belongs(role, index) && model.side[index] === side ? [index] : [])));
  const groups = {
    // Only b1 and b2 count, the motor neurons the steering signal reads.
    wings: perSide((role, index) => model.amplitude[index]),
    halteres: perSide((role) => role === roleIndex("haltere")),
    ocelli: perSide((role) => role === roleIndex("ocellar")),
    hs: perSide((role) => role === roleIndex("hs")),
  };
  return (activity) => Object.fromEntries(Object.entries(groups).map(([name, sides]) => [
    name, sides.map((members) => members.reduce((total, index) => total + activity[index], 0)),
  ]));
}

export function createFlyView(canvas, model) {
  const read = flySignals(model);
  const context = canvas.getContext("2d");
  const scales = { wings: 1e-9, halteres: 1e-9, ocelli: 1e-9, hs: 1e-9 };
  let levels = { wings: [0, 0], halteres: [0, 0], ocelli: [0, 0], hs: [0, 0] };

  function update(activity) {
    const signals = read(activity);
    levels = {};
    for (const [name, values] of Object.entries(signals)) {
      // Each population is scaled to its own recent peak, as in the circuit panel.
      scales[name] = Math.max(scales[name] * 0.995, ...values.map(Math.abs), 1e-9);
      levels[name] = values.map((value) => Math.max(-1, Math.min(1, value / scales[name])));
    }
  }

  function draw([x, , theta], gains, phase) {
    const { width, height } = canvas;
    const horizon = height * 0.72;
    const metres = (width - 80) / (2 * TRACK_LIMIT);
    // The view follows the fly, so the world slides past at the cart's speed.
    const screenX = (worldX) => width / 2 + (worldX - x) * metres;
    context.fillStyle = "#1a1a19";
    context.fillRect(0, 0, width, height);
    context.fillStyle = "#222220";
    context.fillRect(0, horizon, width, height - horizon);
    context.strokeStyle = "#383835";
    context.lineWidth = 2;
    const first = Math.floor((x - width / 2 / metres) / GROUND_SPACING);
    for (let mark = first; mark * GROUND_SPACING < x + width / 2 / metres; mark += 1) {
      const markX = screenX(mark * GROUND_SPACING);
      context.beginPath();
      context.moveTo(markX, horizon + 12);
      context.lineTo(markX, horizon + 30);
      context.stroke();
    }
    context.fillStyle = "#898781";
    context.font = "12px system-ui, sans-serif";
    for (const edge of [-TRACK_LIMIT, TRACK_LIMIT]) {
      const edgeX = screenX(edge);
      if (edgeX < -10 || edgeX > width + 10) continue;
      context.fillRect(edgeX - 1, horizon - 40, 2, height - horizon + 40);
      context.fillText("track edge", edgeX + (edge < 0 ? 6 : -68), horizon - 46);
    }
    if (gains[3] > 0) {
      context.fillStyle = "#383835";
      context.fillRect(screenX(0) - 5, 16, 10, horizon - 16);
      context.fillStyle = "#898781";
      context.fillText("landmark", screenX(0) + 10, 28);
    }
    context.fillText("the fly, seen from behind", 12, 20);

    context.save();
    context.translate(width / 2, height * 0.42);
    context.rotate(theta);
    context.scale(1.3, 1.3);
    for (const [index, side] of SIDES.entries()) drawWing(side, levels.wings[index], phase);
    drawBody();
    for (const [index, side] of SIDES.entries()) drawHaltere(side, levels.halteres[index], phase);
    drawHead();
    context.restore();
  }

  function drawWing(side, level, phase) {
    const amplitude = RESTING_AMPLITUDE + AMPLITUDE_SWING * level;
    context.save();
    context.translate(side * 16, -10);
    context.scale(side, 1);
    // The swept fan shows the stroke amplitude even when paused; it widens as b1 and b2 fire.
    context.fillStyle = `rgba(213, 81, 129, ${0.1 + 0.18 * Math.max(level, 0)})`;
    context.beginPath();
    context.moveTo(0, 0);
    context.arc(0, 0, WING_LENGTH, WING_RAISE - amplitude, WING_RAISE + amplitude);
    context.closePath();
    context.fill();
    context.rotate(WING_RAISE + amplitude * Math.sin(phase));
    context.fillStyle = "rgba(230, 228, 218, 0.35)";
    context.strokeStyle = "rgba(255, 255, 255, 0.5)";
    context.lineWidth = 1;
    context.beginPath();
    context.ellipse(WING_LENGTH / 2, 0, WING_LENGTH / 2, 10, 0, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.restore();
  }

  function drawBody() {
    context.strokeStyle = "#6b6457";
    context.lineWidth = 2;
    for (const side of SIDES) {
      for (let leg = 0; leg < 3; leg += 1) {
        context.beginPath();
        context.moveTo(side * 8, 6 + leg * 5);
        context.lineTo(side * (22 + leg * 5), 38 + leg * 6);
        context.stroke();
      }
    }
    context.fillStyle = "#a39a87";
    context.beginPath();
    context.ellipse(0, 24, 17, 22, 0, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "#6b6457";
    for (const stripe of [16, 26, 36]) {
      context.beginPath();
      context.ellipse(0, stripe - 6, 15, 6, 0, 0.15 * Math.PI, 0.85 * Math.PI);
      context.stroke();
    }
    context.fillStyle = "#8c7f6a";
    context.beginPath();
    context.ellipse(0, -4, 21, 18, 0, 0, Math.PI * 2);
    context.fill();
  }

  function drawHaltere(side, level, phase) {
    // Halteres beat in antiphase to the wings; the one on the side moving down is excited.
    const knobY = 12 + 5 * Math.sin(phase + Math.PI);
    context.strokeStyle = "#8c7f6a";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(side * 12, 10);
    context.lineTo(side * 24, knobY);
    context.stroke();
    glow("#fab219", Math.max(level, 0), () => {
      context.beginPath();
      context.arc(side * 25, knobY, 4.5, 0, Math.PI * 2);
      context.fill();
    });
  }

  function drawHead() {
    context.fillStyle = "#8c7f6a";
    context.beginPath();
    context.arc(0, -30, 14, 0, Math.PI * 2);
    context.fill();
    for (const [index, side] of SIDES.entries()) {
      context.fillStyle = "#a33a2c";
      context.beginPath();
      context.ellipse(side * 11, -31, 7, 10, 0, 0, Math.PI * 2);
      context.fill();
      // HS cells sit behind the eyes and read the ground flowing past.
      const flow = Math.max(levels.hs[index], 0);
      context.strokeStyle = `rgba(57, 135, 229, ${0.15 + 0.85 * flow})`;
      context.lineWidth = 2;
      context.stroke();
      glow("#0ca30c", Math.max(levels.ocelli[index], 0), () => {
        context.beginPath();
        context.arc(side * 5, -41, 2.6, 0, Math.PI * 2);
        context.fill();
      });
    }
    context.fillStyle = "rgba(12, 163, 12, 0.3)";
    context.beginPath();
    context.arc(0, -44, 2.6, 0, Math.PI * 2);
    context.fill();
  }

  function glow(colour, level, shape) {
    context.save();
    context.globalAlpha = 0.3 + 0.7 * level;
    context.fillStyle = colour;
    context.shadowColor = colour;
    context.shadowBlur = 14 * level;
    shape();
    context.restore();
  }

  return { update, draw };
}
