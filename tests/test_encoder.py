import numpy as np
import pytest

from fly_cartpole.encoder import LEFT, RIGHT, ROLE_PUSH_LEFT, ROLE_PUSH_RIGHT, build_encoder

GLOMERULI = np.repeat(np.array([f"G{index}" for index in range(12)]), 2)
STATE = np.array([0.3, -0.5, 0.05, 1.0])


def test_every_role_gets_glomeruli():
    roles = build_encoder(GLOMERULI, action_fraction=0.25, tuning_width=1.0, seed=0).group_role.tolist()
    assert roles.count(ROLE_PUSH_LEFT) == roles.count(ROLE_PUSH_RIGHT) == 2
    assert all(roles.count(variable) >= 1 for variable in range(4))


def test_left_and_right_differ_only_on_action_glomeruli():
    encoder = build_encoder(GLOMERULI, 0.25, 1.0, seed=0)
    left, right = encoder.glomeruli(STATE, LEFT), encoder.glomeruli(STATE, RIGHT)
    assert set(encoder.group_role[left != right].tolist()) == {ROLE_PUSH_LEFT, ROLE_PUSH_RIGHT}
    assert np.all(left[encoder.group_role == ROLE_PUSH_LEFT] == 1.0)
    assert not encoder.glomeruli(STATE, None)[encoder.group_role >= ROLE_PUSH_LEFT].any()


def test_state_glomerulus_peaks_at_its_preferred_value():
    encoder = build_encoder(GLOMERULI, 0.25, 1.0, seed=0)
    group = int(np.flatnonzero(encoder.group_role == 2)[0])
    state = np.zeros(4)
    state[2] = encoder.preferred[group]
    assert encoder.glomeruli(state, None)[group] == pytest.approx(1.0)


def test_out_of_range_state_saturates_at_the_edge_glomerulus():
    encoder = build_encoder(GLOMERULI, 0.25, 1.0, seed=0)
    activity = encoder.glomeruli(np.array([10.0, -50.0, 1.0, 99.0]), None)
    assert np.all(np.isfinite(activity))
    for variable in range(4):
        assert activity[encoder.group_role == variable].max() == pytest.approx(1.0)


def test_projection_neurons_copy_their_glomerulus():
    encoder = build_encoder(GLOMERULI, 0.25, 1.0, seed=0)
    assert np.array_equal(encoder.pn(STATE, LEFT), encoder.glomeruli(STATE, LEFT)[encoder.pn_group])


def test_too_few_glomeruli_is_rejected():
    with pytest.raises(ValueError):
        build_encoder(np.array(["A", "B", "C", "D"]), 0.25, 1.0, seed=0)
