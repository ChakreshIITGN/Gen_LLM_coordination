from taxisim.agents.base import AgentConfig
from taxisim.agents.bayesian_agent import BayesianAgent
from taxisim.agents.hillclimb_agent import HillClimbAgent
from taxisim.agents.llm_agent import LLMAgent
from taxisim.agents.random_agent import RandomAgent
from taxisim.environments.gradient_2d import Gradient2DEnvironment
from taxisim.environments.linear_1d import Linear1DEnvironment
from taxisim.prompts.builder import PromptBuilder


def test_random_agent_actions_in_space():
    env = Linear1DEnvironment(seed=0)
    cfg = AgentConfig(name="r", seed=0)
    ag = RandomAgent(cfg, env.action_space)
    for _ in range(20):
        s = ag.act(1.0, 0.0)
        assert s.action in env.action_space


def test_hillclimb_moves_toward_higher_concentration():
    env = Linear1DEnvironment(seed=0, start_position=40.0, source_position=50.0)
    cfg = AgentConfig(name="h", seed=0)
    ag = HillClimbAgent(cfg, env.action_space)
    r0 = env.reset()
    o0 = r0.observation
    s0 = ag.act(o0, r0.info["position"])
    r1 = env.step(s0.action)
    o1 = r1.observation
    s1 = ag.act(o1, r1.info["position"])
    assert s1.action == "right"


def test_bayesian_trends_toward_source_noiseless():
    env = Linear1DEnvironment(
        seed=0,
        start_position=20.0,
        source_position=50.0,
        noise_sigma=0.0,
    )
    cfg = AgentConfig(name="b", seed=0)
    ag = BayesianAgent(
        cfg,
        env.action_space,
        env_length=int(env.length),
        decay_length=env.decay_length,
        noise_sigma=1e-3,
    )
    r = env.reset()
    obs = r.observation
    pos = r.info["position"]
    for _ in range(5):
        step = ag.act(obs, pos)
        out = env.step(step.action)
        obs = out.observation
        pos = out.info["position"]
    assert out.info["position"] > 20.0


def test_bayesian_2d_act_vectorized_concentration():
    env = Gradient2DEnvironment(seed=0, start_position=(2.0, 2.0))
    cfg = AgentConfig(name="b2", seed=0)
    ag = BayesianAgent(
        cfg,
        env.action_space,
        env_length=int(env.width),
        decay_length=env.decay_length,
        noise_sigma=1.0,
        width=float(env.width),
        height=float(env.height),
    )
    r = env.reset()
    step = ag.act(r.observation, r.info["position"])
    assert step.action in env.action_space


def test_llm_agent_lazy_load():
    env = Linear1DEnvironment(seed=0)
    cfg = AgentConfig(name="llm", seed=0)
    ag = LLMAgent(cfg, env.action_space, backend="huggingface")
    assert ag._hf_model is None


def test_prompt_builder_padding():
    pb = PromptBuilder("abstract", "1d", history_length=5)
    d = pb.build([], 1.0, 2.0, {"length": 10})
    assert "N/A" in d["user"]


def test_prompt_builder_rounding():
    pb = PromptBuilder("abstract", "1d", history_length=2, round_digits=1)
    d = pb.build([1.23456], 2.0, 3.0, {"length": 10})
    assert "1.2" in d["user"] or "2.0" in d["user"]
