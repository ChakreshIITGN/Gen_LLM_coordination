from schemas import Action, Observation
from pydantic import ValidationError
import json

def llm_policy(obs: Observation, MAX_STEPS_PER_ACTION: int) -> Tuple[Action, str]:
    """
    LLM policy:
    - Build prompt
    - Call LLM
    - Parse JSON -> Action
    - If invalid, fallback W
    Returns (Action, raw_llm_output)
    """
    system = (
    "You are an agent on a ring line. Output ONLY JSON.\n"
    "Valid outputs:\n"
    "1) {\"type\":\"W\"}\n"
    "2) {\"type\":\"M\",\"dir\":\"L\"|\"R\",\"steps\":1..%d}\n"
    "No extra keys. No extra text."
    ) % MAX_STEPS_PER_ACTION


    user = json.dumps({
        "observation": obs.model_dump(),
        "allowed_actions": {
            "W": {},
            "M": {
                "dir": ["L", "R"],
                "steps": {"min": 1, "max": MAX_STEPS_PER_ACTION}
            }
        },
        "output_format_examples": [
            {"type": "W"},
            {"type": "M", "dir": "R", "steps": 3}
        ],
        "notes": [
            "Output JSON only.",
            "Do not add extra keys.",
            "If energy is low, choose smaller steps or W."
        ]
    })


    raw = llm_call(system, user).strip()

    try:
        action = Action.model_validate_json(raw)
        return action, raw
    except ValidationError:
        # If the LLM output isn't valid JSON matching schema, we apply a safe fallback.
        return Action(type="W"), raw


def random_llm_policy() -> Tuple[Action, str]:
    """
    Random LLM policy:
    - LLM outputs exactly one character: aiban, duven, or beauki
    - No JSON, no reasoning
    - Simulator maps it to an Action
    Returns (Action, raw_llm_output)
    """
    system = (
        "You are an agent on a ring line. When asked to 'make a choice'\n"
        "Choose only between aiban, duven, or beauki.\n"
        "Output ONLY a single word: aiban, duven, or beauki.\n"
        "NO extra reasoning, NO JSON, NO extra text. Just a single word."
        "Very important: Do not assume anything apriori about the options, just make a random choice."
    )

    user = "Make a choice"

    raw = llm_call(system, user).strip()
    print(f"LLM output: {raw}")
    if raw == "aiban":
        return Action(type="M", dir="L", steps=1), raw
    elif raw == "duven":
        return Action(type="M", dir="R", steps=1), raw
    elif raw == "beauki":
        # includes "W" or any invalid output
        return Action(type="W"), raw
    else:
        return Action(type="W"), raw
