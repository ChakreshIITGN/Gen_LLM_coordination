from __future__ import annotations

from taxisim.prompts import abstract, biological


class PromptBuilder:
    def __init__(
        self,
        style: str,  # "abstract" or "biological"
        env_type: str,  # "1d" or "2d"
        history_length: int = 5,
        round_digits: int = 2,  # round float values in prompts
    ):
        self.style = style
        self.env_type = env_type
        self.history_length = history_length
        self.round_digits = round_digits

    def _fmt_val(self, v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{round(float(v), self.round_digits):.{self.round_digits}f}"

    def build(
        self,
        history: list[float],
        current_observation: float,
        position,  # float for 1d, tuple for 2d
        env_params: dict,  # {"length": 100} for 1d, {"width": 20, "height": 20} for 2d
    ) -> dict:
        """
        Returns {"system": str, "user": str}
        Pads history with "N/A" if shorter than history_length.
        Rounds all float values to round_digits decimal places.
        """
        if self.style == "abstract":
            mod = abstract
        elif self.style == "biological":
            mod = biological
        else:
            raise ValueError(f"Unknown prompt style: {self.style}")

        # Templates use five fixed slots {h0}..{h4}; trim input by history_length then pad to 5.
        hist = list(history)[-self.history_length :]
        hist = hist[-5:]
        missing = 5 - len(hist)
        padded_nums: list[float | None] = [None] * missing + [float(x) for x in hist]
        padded = [self._fmt_val(x) for x in padded_nums]
        h0, h1, h2, h3, h4 = padded

        cur = self._fmt_val(float(current_observation))

        if self.env_type == "1d":
            length = env_params.get("length", 100)
            pos = self._fmt_val(float(position))
            system = mod.SYSTEM_1D
            user = mod.USER_TEMPLATE_1D.format(
                h0=h0,
                h1=h1,
                h2=h2,
                h3=h3,
                h4=h4,
                current=cur,
                position=pos,
                length=length,
            )
        elif self.env_type == "2d":
            width = env_params.get("width", 20)
            height = env_params.get("height", 20)
            x, y = float(position[0]), float(position[1])
            system = mod.SYSTEM_2D
            user = mod.USER_TEMPLATE_2D.format(
                h0=h0,
                h1=h1,
                h2=h2,
                h3=h3,
                h4=h4,
                current=cur,
                x=self._fmt_val(x),
                y=self._fmt_val(y),
                width=width,
                height=height,
            )
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")

        return {"system": system, "user": user}
