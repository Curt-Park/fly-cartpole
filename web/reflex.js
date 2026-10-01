// The reflex fly in the browser; mirrors fly_cartpole.reflex.ReflexFly operation by operation.
const ANGLE = 2;
const ANGLE_DOT = 3;
const X = 0;
const X_DOT = 1;

export function createReflex(model) {
  const size = model.role.length;
  const pre = Int32Array.from(model.pre);
  const post = Int32Array.from(model.post);
  const coupling = Float64Array.from(model.coupling);
  const side = model.side;
  const roleIndex = (name) => model.roles.indexOf(name);
  const members = (name) => model.role.flatMap((role, index) => (role === roleIndex(name) ? [index] : []));
  const ocellar = members("ocellar");
  const haltere = members("haltere");
  const hs = members("hs");
  const rightAmplitude = model.amplitude.flatMap((amplitude, index) => (amplitude && side[index] > 0 ? [index] : []));
  const leftAmplitude = model.amplitude.flatMap((amplitude, index) => (amplitude && side[index] < 0 ? [index] : []));
  const activity = new Float64Array(size);
  const synaptic = new Float64Array(size);
  const drive = new Float64Array(size);
  const leak = model.leak;

  function sensorDrive(state, gains) {
    const normalised = state.map((value, index) => Math.min(1, Math.max(-1, value / model.state_limits[index])));
    drive.fill(0);
    for (const index of ocellar) drive[index] = (gains[0] * normalised[ANGLE] + gains[3] * normalised[X] + gains[4] * normalised[X_DOT]) * side[index];
    for (const index of haltere) drive[index] = gains[1] * normalised[ANGLE_DOT] * side[index];
    for (const index of hs) drive[index] = -gains[2] * normalised[X_DOT] * side[index];
  }

  function act(state, gains) {
    sensorDrive(state, gains);
    for (let substep = 0; substep < model.substeps; substep += 1) {
      synaptic.fill(0);
      for (let edge = 0; edge < pre.length; edge += 1) synaptic[post[edge]] += activity[pre[edge]] * coupling[edge];
      for (let index = 0; index < size; index += 1) activity[index] = (1 - leak) * activity[index] + leak * (synaptic[index] + drive[index]);
    }
    let right = 0;
    let left = 0;
    for (const index of rightAmplitude) right += activity[index];
    for (const index of leftAmplitude) left += activity[index];
    const steer = right - left;
    return { action: steer > 0 ? 1 : 0, steer, activity };
  }

  return { act, reset: () => activity.fill(0), size };
}
