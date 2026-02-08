# %%
# loading necessary modules
import json
import math
from ollama import chat, ChatResponse
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ValidationError, model_validator, ConfigDict, Field

# %%
# setting the configuration parameters
# ---- World parameters ----
L = 50  # ring positions: 0..19
T = 500  # max steps

# ---- LLM (Ollama OpenAI-compatible) ----
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "mistral"
TEMPERATURE = 1  # study about temperature effects on coordination
MAX_TOKENS = 150

# ---- Output ----
# RUN_DIR = Path("runs/exp_e-coli") / time.strftime("%Y%m%d-%H%M%S")
# RUN_DIR.mkdir(parents=True, exist_ok=True)


# %%
# defining necessary classes and functions
ActionType = Literal["L", "R", "W", "F"]
X_0 = 20
agent_x0 = 5
RATE = 1 / 10
INITIAL = 10.0


def exp_decay(x: float, initial: float = INITIAL, rate: float = RATE, x_0: float = X_0) -> float:
    """Exponential decay function."""
    if x < x_0:
        return 0.0
    else:
        return initial * math.exp(-rate * (x - x_0))


def reward_value(x, min_val: float=-0.1, max_val: float=100) -> float:
    """Reward value by capping function value between min_val and max_val."""
    return max(min(exp_decay(x), max_val), min_val)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")  # forbid extra fields
    type: ActionType
    reason: str = Field(min_length=1, max_length=100)  

    @model_validator(mode="after")
    def _validate_shape(self):
        if self.type in ["L", "R", "W"]:
            return self
        else:
            raise ValueError("requires L, R, or W")
    



class LLMAgent:
    """
    Docstring for LLMAgent
    LLM Agent class representing an agent in the simulation.
    """

    def __init__(self, name: str, x_0: float=agent_x0, reward: list = [0], memory: list = []):
        self.name = name
        self.x = [x_0]
        self.reward = reward
        self.memory = memory

        return None

    def apply(self, action: Action, L: int = 50) -> None:
        # boundary blocking: if move would go out of bounds, don't move
        if action.type == "L":
            if self.x[-1] - 1 >= 0:
                self.x.append(self.x[-1] - 1)
        elif action.type == "R":
            if self.x[-1] + 1 <= L:
                self.x.append(self.x[-1] + 1)
        # W: no move

        self.reward.append(reward_value(self.x[-1]))
        self.memory.append(action.type)

    def state_dict(self, memory_last_k: Optional[int] = None) -> dict:
        mem = self.memory if memory_last_k is None else self.memory[-memory_last_k:]
        return {
            "name": self.name,
            "x": self.x,
            "reward": self.reward,
            "memory": mem,
        }


# --- Helpers: store dicts, send JSON strings to Ollama ---
def _serialize_messages_for_ollama(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        else:
            content = str(content)
        out.append({"role": msg["role"], "content": content})
    return out


# %%
# -- Continuous chat history (content can be dicts here) ---
_messages = [
    {
        "role": "system",
        "content": """
        You are an agent.
        You must output ONLY valid JSON.
        Schema:
        { "type": "L|R|W", "reason": "string" }
        Rules:
            - type must be exactly one of: L, R, W
            - reason must be one short sentence
            - Do not add extra fields
            - Do not add explanations""",
    },
]


def llm_call(user_payload, agent_payload) -> str:
    """Call Ollama chat endpoint with messages, return response content string."""
    global _messages
    if agent_payload is not None:

        _messages.extend(agent_payload)
        _messages.append(user_payload)
    
    else:
        _messages.append(user_payload)

    resp: ChatResponse = chat(
        model=OLLAMA_MODEL,
        messages=_serialize_messages_for_ollama(_messages),
        options={
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },

    )
    return resp["message"]["content"].strip()  # return the 'type' field from the response



def llm_policy_step(agent: LLMAgent, *, provide_memory: bool, memory_k: int = 5) -> Tuple[Action, str, dict]:
    """
    One step:
    - user message includes agent state (with/without last-k memory)
    - LLM returns single char
    - validate Action
    - update agent state
    - assistant message content is ALWAYS a dict of agent state
    Returns: (action, raw_choice, assistant_state_dict)
    """
    # Build what we SHOW the LLM in the user turn
    if provide_memory:
        if len(agent.memory) < memory_k:
            memory = agent.memory
        else:
            memory = agent.memory[-memory_k:]
        agent_payload = []
        user_payload = {
            "role": "user",
            "content": {
                "last_k_positions": memory,
                "last_k_rewards"  : agent.reward[-memory_k:],
            },
            "instruction": "Choose next action: L, R, or W.",
        }
        if len(agent.memory) < memory_k:
            for m in agent.memory:
                agent_payload.append({"role": "assistant", "content": m})
        else:
            for m in agent.memory[-memory_k:]:
                agent_payload.append({"role": "assistant", "content": m})

    else:
        # memory not given: omit it (or set to None; here we omit)
        user_payload = {
            "role": "user",
            "content": {
                "current_position": agent.x[-1],
                "current_reward"  : agent.reward[-1] if agent.reward else None,
            },
            "instruction": "Choose next action: L, R, or W.",
        }
        agent_payload = None  # no memory messages

    raw = llm_call(user_payload, agent_payload)

    # Validate with Action model; fallback is W (safe, deterministic)
    try:
        action = Action(type=raw)
        # Update agent
        agent.apply(action, L=L)

        # Assistant always writes agent state as a dict (including full memory in the log)
        assistant_state = agent.state_dict(memory_last_k=None)

        return action, raw, assistant_state
    
    except ValidationError:
        print("Validation error for action:", raw)
        pass


def run(agent: LLMAgent, N: int, *, provide_memory: bool) -> None:
    
    for _ in range(N):
        print("memory :>>>>>>>", agent.memory, "reward >>>>>", agent.reward)
        action, raw, state = llm_policy_step(agent, provide_memory=provide_memory, memory_k=5)
        print(f"LLM choice: {raw}, validated action: {action.type}, reason: {action.reason}")
        print(f"Agent state: position {agent.x[-1]}, reward {agent.reward[-1]}")
        print("-" * 30)
    
            
    
# %%
# ---- Example usage ----
agent = LLMAgent(name="a1", x_0=20)

print("\n--- Case 1: memory NOT given to LLM ---")
run(agent, N=25, provide_memory=True)

# print("\n--- Case 2: last 5 memory entries given to LLM ---")
# run(agent, N=10, provide_memory=True)


# %%
# plotting the movement trajectory
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(agent.x, agent.reward, marker='o')
plt.xlabel("Position")
plt.ylabel("Reward")
plt.title("Agent Movement and Reward Trajectory")
plt.grid(True)
plt.show()