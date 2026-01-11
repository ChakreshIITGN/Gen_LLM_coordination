#%%
# loading necessary modules
import json
import time 
import math
import random
import ollama
from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ValidationError,model_validator 

#%% 
# setting the configuration parameters
# ---- World parameters ----
L = 50                 # ring positions: 0..19
T = 500                 # max steps
START_X = 0

 
# ---- LLM (Ollama OpenAI-compatible) ----
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "mistral"
TEMPERATURE = 1 # study about temperature effects on coordination
MAX_TOKENS = 512

# ---- Output ----
# RUN_DIR = Path("runs/exp_e-coli") / time.strftime("%Y%m%d-%H%M%S")
# RUN_DIR.mkdir(parents=True, exist_ok=True)


#%%
# defining necessary classes and functions
ActionType = Literal["L", "R", "W"]

X_0 = 20.0
RATE = 1/0.5


def exp_decay(initial: float, x: float, rate: float=RATE , x_0: float=X_0) -> float:
    """Exponential decay function."""
    if x < x_0:
        return 0.0
    else:
        return initial * math.exp(-rate * (x - x_0))

def reward_value(x, min_val: float, max_val: float) -> float:
    """Reward value by capping function value between min_val and max_val."""
    return max(min(exp_decay(x), max_val), min_val)



class Action(BaseModel):
    type: ActionType

    @model_validator(mode="after")
    def _validate_shape(self):
        if self.type in ["L", "R", "W"]:
            return self  
        else:
            raise ValueError("requires L, R, or W")
        
class continuousAction(BaseModel):
    x: int

    @model_validator(mode="after")
    def _validate_shape(self):
        # bounds will be checked within the context
        return self

class LLMAgent():
    """
    Docstring for LLMAgent
    LLM Agent class representing an agent in the simulation.
    """
    
    def __init__(self, name:str, x_0: float, reward: list=[], memory:list = []):
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

        self.reward = 0.0
        self.memory.append(action.type)

    def apply_action_with_reward(
                self,
                action: continuousAction,
                *,
                x_0: float,
                initial: float,
                rate: float,
                min_reward: float,
                max_reward: float,
                L: int = 50,
        )-> None:
        old_x = self.x[-1]

        if not (0 <= action.x <= L):
            raise ValueError("out of bounds")

        new_x = action.x

        # --- reward update ---
        if new_x == old_x:
            return  # no movement → no reward change

        if abs(new_x - x_0) < abs(old_x - x_0):
            # moved towards target
            raw_reward = exp_decay(
                initial=initial,
                x=new_x,
                rate=rate,
                x_0=x_0,
            )
            reward = reward_value(raw_reward, min_reward, max_reward)

        else:
            # moved away
            reward = -0.1
        self.x.append(new_x)
        self.reward.append(reward)


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
    for m in messages:
        content = m["content"]
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        else:
            content = str(content)
        out.append({"role": m["role"], "content": content})
    return out

#%% 
# -- Continuous chat history (content can be dicts here) ---
_messages = [
    {
        "role": "system",
        "content": "You reply with exactly one character: L, R, or W. No other text."
    }
]

def llm_call(user_payload, NUM_PREDICT) -> str:

    """Call Ollama chat endpoint with messages, return response content string."""
    global _messages

    _messages.append({"role": "user", "content": user_payload})

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=_serialize_messages_for_ollama(_messages),
        options={
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "stop": ["\n", " ", ".", ",", "!", "?"],
        },
    )
    return resp["message"]["content"].strip()

def llm_policy_continuous(agent: LLMAgent, *, L: int, provide_memory: bool, memory_k: int = 5) -> Tuple[Action, str]:
    global _messages

    # what the LLM sees
    agent_state = agent.state_dict(memory_last_k=memory_k) if provide_memory else {
        "name": agent.name,
        "x": agent.x,
        "reward": agent.reward,
    }

    system_hint = (
        f"Choose the next target position as a single integer between 0 and {L} inclusive. "
        "Output ONLY the integer. No other text."
    )

    # (optional) refresh system instruction each step
    _messages[0]["content"] = system_hint

    user_payload = {
            "agent": agent_state,
            "instruction": f"Output one integer in [0,{L}]",
        }
    
    raw = llm_call(user_payload=user_payload, NUM_PREDICT=2)

    # parse + validate bounds
    try:
        x_choice = int(raw)
        action = continuousAction(x=x_choice)
    except (ValueError, ValidationError):
        # safe fallback: "stay" by choosing current position clipped into bounds
        x_fallback = min(max(int(agent.x), 0), L)
        action = continuousAction(x=x_fallback)

    agent.apply_action_with_reward(
        agent,
        action,
        x_0=X_0,
        initial=10,
        rate=RATE,
        min_reward=-0.1,
        max_reward=10.0,
        L=L,
    )
    # assistant always writes agent state dict
    assistant_state = agent.state_dict(memory_last_k=None)
    _messages.append({"role": "assistant", "content": json.dumps(assistant_state)})

    return action, raw, assistant_state


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
        user_payload = {
            "agent": agent.state_dict(memory_last_k=memory_k),
            "instruction": "Choose next action: L, R, or W."
        }
    else:
        # memory not given: omit it (or set to None; here we omit)
        user_payload = {
            "agent": {
                "name": agent.name,
                "x": agent.x,
                "reward": agent.reward,
            },
            "instruction": "Choose next action: L, R, or W."
        }

    raw = llm_call(user_payload, NUM_PREDICT=1)

    # Validate with Action model; fallback is W (safe, deterministic)
    try:
        action = Action(type=raw)
    except ValidationError:
        action = Action(type="W")

    # Update agent
    agent.apply(action, L=L)

    # Assistant always writes agent state as a dict (including full memory in the log)
    assistant_state = agent.state_dict(memory_last_k=None)

    _messages.append({"role": "assistant", "content": assistant_state})
    print(_messages)
    return action, raw, assistant_state


def run(agent: LLMAgent, N: int, *, provide_memory: bool, policy_override: str = "step") -> None:
    if policy_override == "step":
        for _ in range(N):
            action, raw, state = llm_policy_step(agent, provide_memory=provide_memory, memory_k=5)
            print("raw:", raw, "-> action:", action.type, "| x:", agent.x)
    else:
        for _ in range(N):
            action, raw, state = llm_policy_continuous(agent, L=10, provide_memory=provide_memory)
            print("raw:", raw, "-> action:", action.type, "| x:", agent.x)

#%%
# ---- Example usage ----
agent = LLMAgent(name="a1", x_0=1)

print("\n--- Case 1: memory NOT given to LLM ---")
run(agent, N=20, provide_memory=False)

print("\n--- Case 2: last 5 memory entries given to LLM ---")
run(agent, N=10, provide_memory=True)


#%%
# plotting the movement trajectory