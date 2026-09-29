// The exported bilateral fly: glomeruli -> PN -> KC (top k) -> MBON -> value, one mushroom body per push direction.
const ROLE_PUSH_LEFT = 4;

export function glomeruli(model, state) {
  const { group_role: roles, preferred, sigma } = model.encoder;
  const limits = model.state_limits;
  return roles.map((role, group) => {
    if (role >= ROLE_PUSH_LEFT) return 0;
    const value = Math.min(Math.max(state[role], -limits[role]), limits[role]);
    return Math.exp(-((value - preferred[group]) ** 2) / (2 * sigma[group] ** 2));
  });
}

export function hemisphere(side, activity, trained) {
  const drive = new Float64Array(side.n_kc);
  const { rows, cols, values } = side.pn_kc;
  for (let edge = 0; edge < rows.length; edge += 1) drive[cols[edge]] += activity[side.pn_group[rows[edge]]] * values[edge];

  const kc = new Float64Array(side.n_kc);
  const mbon = new Float64Array(side.n_mbon);
  // Ties go to the lowest index, matching the Python kenyon_code.
  const order = Array.from(drive.keys()).sort((first, second) => drive[second] - drive[first] || first - second);
  if (drive[order[0]] <= 0) return { kc, active: [], mbon, value: 0 };
  const winners = order.slice(0, side.k);
  const top = drive[winners[0]];
  for (const index of winners) kc[index] = drive[index] / top;

  const weights = trained ? side.kc_mbon_trained : side.kc_mbon_naive;
  const { rows: kcRows, cols: mbonCols } = side.kc_mbon;
  for (let edge = 0; edge < kcRows.length; edge += 1) {
    const level = kc[kcRows[edge]];
    if (level) mbon[mbonCols[edge]] += level * weights[edge];
  }
  let value = 0;
  side.mbon_valence.forEach((sign, index) => { value += sign * mbon[index]; });
  return { kc, active: winners.filter((index) => kc[index] > 0), mbon, value };
}

export function evaluate(model, state, trained) {
  const activity = glomeruli(model, state);
  const sides = model.sides.map((side) => hemisphere(side, activity, trained));
  return { activity, sides, values: sides.map((side) => side.value) };
}

export function probabilityLeft(model, values) {
  return 0.5 * (1 + Math.tanh(0.5 * model.beta * (values[0] - values[1])));
}
