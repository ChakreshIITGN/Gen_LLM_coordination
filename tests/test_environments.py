import numpy as np
import pytest

from taxisim.environments.linear_1d import Linear1DEnvironment


def test_reset_returns_valid_step_result():
    env = Linear1DEnvironment(seed=0, start_position=10.0)
    r = env.reset()
    assert r.observation == r.info["noisy_concentration"]
    assert "position" in r.info
    assert r.info["action_taken"] == "reset"
    assert r.reward == 0.0
    assert not r.done


def test_step_left_moves_position():
    env = Linear1DEnvironment(seed=0, start_position=50.0)
    env.reset()
    p0 = env._position
    r = env.step("left")
    assert r.info["position"] == p0 - 1.0


def test_step_right_moves_position():
    env = Linear1DEnvironment(seed=0, start_position=50.0)
    env.reset()
    p0 = env._position
    r = env.step("right")
    assert r.info["position"] == p0 + 1.0


def test_concentration_peaks_at_source():
    env = Linear1DEnvironment(source_position=50.0, decay_length=10.0)
    c50 = env.concentration(50.0)
    assert c50 > env.concentration(0.0)
    assert c50 > env.concentration(100.0)


def test_episode_terminates_at_max_steps():
    env = Linear1DEnvironment(max_steps=3, seed=0, start_position=0.0)
    env.reset()
    done = False
    while not done:
        r = env.step("stay")
        done = r.done
    assert env.step_count >= env.max_steps


def test_noise_added_when_sigma_positive():
    env = Linear1DEnvironment(noise_sigma=5.0, seed=123, start_position=0.0)
    env.reset()
    obs = [env.step("stay").observation for _ in range(10)]
    assert np.std(obs) > 0.1


def test_boundary_clipping():
    env = Linear1DEnvironment(seed=0, start_position=0.0)
    env.reset()
    r = env.step("left")
    assert r.info["position"] == 0.0


def test_goal_reward():
    env = Linear1DEnvironment(seed=0, start_position=2.0, source_position=1.0, length=100)
    env.reset()
    r = env.step("left")
    assert r.info["distance_to_source"] <= 1.0
    assert r.reward >= 100.0
    assert r.done
