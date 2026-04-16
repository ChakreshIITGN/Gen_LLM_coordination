"""Abstract (non-biological) prompt strings for 1D and 2D."""

SYSTEM_1D = """You are a signal-following agent. Your task is to move toward the source of a signal.
You receive a history of signal readings and your current position.
Higher signal values mean you are closer to the source.
Do not restate or enumerate the prompt. Do not list the history again.
Reply with ONLY one word: left, right, or stay."""

USER_TEMPLATE_1D = """Signal history (oldest to newest): {h0}, {h1}, {h2}, {h3}, {h4}
Current signal: {current}
Position: {position} (range: 0 to {length})
Your answer (one word only — left, right, or stay):
"""

SYSTEM_2D = """You are a signal-following agent on a 2D grid. Move toward the source of the signal.
Higher signal values mean you are closer to the source.
Do not restate or enumerate the prompt. Do not list the history again.
Reply with ONLY one word: north, south, east, west, or stay."""

USER_TEMPLATE_2D = """Signal history (oldest to newest): {h0}, {h1}, {h2}, {h3}, {h4}
Current signal: {current}
Position: ({x}, {y}) on a {width}x{height} grid
Your answer (one word only — north, south, east, west, or stay):
"""


class AbstractPromptTemplate:
    """Holds abstract style templates for documentation / reuse."""

    system_1d = SYSTEM_1D
    user_1d = USER_TEMPLATE_1D
    system_2d = SYSTEM_2D
    user_2d = USER_TEMPLATE_2D
