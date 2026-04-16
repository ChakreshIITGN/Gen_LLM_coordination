"""Biological (chemotaxis) prompt strings for 1D and 2D."""

SYSTEM_1D = """You are simulating a bacterium performing chemotaxis.
You can sense the local concentration of a nutrient gradient.
Use your recent concentration history to determine whether you are moving
toward or away from the nutrient source and adjust your movement accordingly.
Do not restate or enumerate the prompt. Do not list the history again.
Reply with ONLY one word: left, right, or stay."""

USER_TEMPLATE_1D = """Concentration history (oldest to newest): {h0}, {h1}, {h2}, {h3}, {h4} μM
Current concentration: {current} μM
Position: {position} μm (domain: 0–{length} μm)
Your answer (one word only — left, right, or stay):
"""

SYSTEM_2D = """You are simulating a bacterium performing chemotaxis on a 2D surface.
You sense local nutrient concentration. Use your history to move toward the source.
Do not restate or enumerate the prompt. Do not list the history again.
Reply with ONLY one word: north, south, east, west, or stay."""

USER_TEMPLATE_2D = """Concentration history (oldest to newest): {h0}, {h1}, {h2}, {h3}, {h4} μM
Current concentration: {current} μM
Position: ({x}, {y}) μm on a {width}x{height} μm grid
Your answer (one word only — north, south, east, west, or stay):
"""


class BiologicalPromptTemplate:
    """Holds biological style templates for documentation / reuse."""

    system_1d = SYSTEM_1D
    user_1d = USER_TEMPLATE_1D
    system_2d = SYSTEM_2D
    user_2d = USER_TEMPLATE_2D
