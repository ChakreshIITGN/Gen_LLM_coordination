from taxisim.metrics.comparison import compare_agents
from taxisim.metrics.navigation import path_efficiency, steps_to_goal, success_rate


def test_steps_to_goal_none_when_not_reached():
    traj = [
        {"action_taken": "reset", "distance_to_source": 50.0},
        {"action_taken": "stay", "distance_to_source": 50.0},
        {"action_taken": "stay", "distance_to_source": 50.0},
    ]
    assert steps_to_goal(traj, goal_threshold=1.0) is None


def test_path_efficiency_straight_line():
    # Start distance 10, move toward goal in 10 steps, reach at step 10
    traj = [{"action_taken": "reset", "distance_to_source": 10.0}]
    for i in range(10):
        d = 10.0 - (i + 1)
        traj.append({"action_taken": "right", "distance_to_source": max(d, 0.0)})
    eff = path_efficiency(traj, goal_threshold=1.0)
    assert eff == 1.0


def test_success_rate():
    t_ok = [
        {"action_taken": "reset", "distance_to_source": 5.0},
        {"action_taken": "right", "distance_to_source": 0.0},
    ]
    t_fail = [
        {"action_taken": "reset", "distance_to_source": 50.0},
        {"action_taken": "stay", "distance_to_source": 50.0},
    ]
    assert success_rate([t_ok, t_fail], goal_threshold=1.0) == 0.5


def test_compare_agents_columns():
    df = compare_agents(
        {"random": [1.0, 2.0, 3.0], "smart": [0.5, 0.6, 0.7]},
        baseline_name="random",
    )
    assert set(df.columns) >= {
        "agent",
        "mean",
        "std",
        "median",
        "mannwhitney_p_vs_baseline",
        "cohens_d_vs_baseline",
    }
