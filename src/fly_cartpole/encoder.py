"""CartPole state plus a candidate action, as antennal lobe projection neuron activity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Cart position, cart velocity, pole angle (rad), pole angular velocity.
STATE_LIMITS = np.array([2.4, 3.0, 0.21, 3.5])
LEFT, RIGHT = 0, 1
ROLE_PUSH_LEFT, ROLE_PUSH_RIGHT = 4, 5


@dataclass(frozen=True)
class Encoder:
    pn_group: np.ndarray
    group_role: np.ndarray
    preferred: np.ndarray
    sigma: np.ndarray

    @property
    def n_groups(self) -> int:
        return len(self.group_role)

    def glomeruli(self, state: np.ndarray, action: int | None) -> np.ndarray:
        activity = np.zeros(self.n_groups)
        is_state = self.group_role < ROLE_PUSH_LEFT
        values = np.clip(state, -STATE_LIMITS, STATE_LIMITS)[self.group_role[is_state]]
        activity[is_state] = np.exp(-((values - self.preferred[is_state]) ** 2) / (2 * self.sigma[is_state] ** 2))
        if action is not None:
            activity[self.group_role == ROLE_PUSH_LEFT + action] = 1.0
        return activity

    def pn(self, state: np.ndarray, action: int | None) -> np.ndarray:
        # Sister PNs of one glomerulus carry the same signal.
        return self.glomeruli(state, action)[self.pn_group]


def build_encoder(pn_glomerulus: np.ndarray, action_fraction: float, tuning_width: float, seed: int) -> Encoder:
    names, pn_group = np.unique(pn_glomerulus, return_inverse=True)
    n_groups = len(names)
    per_action = max(1, round(action_fraction * n_groups / 2))
    if n_groups - 2 * per_action < len(STATE_LIMITS):
        raise ValueError(f"{n_groups} glomeruli cannot cover two actions and {len(STATE_LIMITS)} state variables")
    order = np.random.default_rng(seed).permutation(n_groups)
    role = np.empty(n_groups, dtype=np.int64)
    preferred = np.zeros(n_groups)
    sigma = np.ones(n_groups)
    role[order[:per_action]] = ROLE_PUSH_LEFT
    role[order[per_action : 2 * per_action]] = ROLE_PUSH_RIGHT
    for variable, groups in enumerate(np.array_split(order[2 * per_action :], len(STATE_LIMITS))):
        limit = STATE_LIMITS[variable]
        role[groups] = variable
        preferred[groups] = np.linspace(-limit, limit, len(groups)) if len(groups) > 1 else 0.0
        sigma[groups] = 2 * limit / max(len(groups) - 1, 1) * tuning_width
    return Encoder(pn_group=pn_group, group_role=role, preferred=preferred, sigma=sigma)
