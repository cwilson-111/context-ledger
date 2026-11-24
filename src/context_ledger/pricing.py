"""Per-model token prices, used only where a log has token counts but no recorded cost.

omp writes the cost of every request into its log, and those recorded costs are always
preferred. Claude Code records tokens only, and some omp entries carry tokens with a zero
cost. For those the table below is used. Every row was checked against costs the
harnesses themselves recorded (omp per-request ``usage.cost``; Claude Code ``cost-state``
per-model totals) on 2026-11-23. Values the logs could not confirm are derived from
Anthropic's published ratios (5-minute cache write = 1.25 x input, cache read = 0.1 x input)
and are marked below. Override or extend with ``--prices prices.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from context_ledger.model import Cost, Usage


@dataclass(frozen=True)
class Rates:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float

    def price(self, usage: Usage) -> Cost:
        write_5m = usage.cache_write - usage.cache_write_1h
        return Cost(
            input=usage.input * self.input / 1e6,
            output=usage.output * self.output / 1e6,
            cache_read=usage.cache_read * self.cache_read / 1e6,
            cache_write=(
                write_5m * self.cache_write_5m + usage.cache_write_1h * self.cache_write_1h
            )
            / 1e6,
        )


def _rates(inp: float, out: float, read: float, write_1h: float) -> Rates:
    # 5-minute cache writes never occur in the reference logs; 1.25 x input is Anthropic's ratio.
    return Rates(inp, out, read, inp * 1.25, write_1h)


DEFAULT_PRICES: dict[str, Rates] = {
    "claude-opus-5-5": _rates(4.0, 20.0, 0.2, 8.0),
    "claude-opus-5": _rates(5.0, 25.0, 0.5, 10.0),
    "claude-fable-5": _rates(10.0, 50.0, 1.0, 20.0),
    "claude-fable-5-2": _rates(4.0, 20.0, 0.2, 8.0),
    "claude-haiku-5-5": _rates(0.1, 0.5, 0.01, 0.2),
    # Only input, output and 1h writes observed; cache read is 0.1 x input (assumed).
    "claude-sonnet-5-5": _rates(2.0, 10.0, 0.2, 4.0),
}

_DATE_SUFFIX = re.compile(r"-\d{8}$")


class PriceTable:
    def __init__(self, prices: dict[str, Rates] | None = None) -> None:
        self.prices = dict(DEFAULT_PRICES if prices is None else prices)

    def lookup(self, model: str) -> Rates | None:
        name = model.rsplit("/", maxsplit=1)[-1]
        return self.prices.get(name) or self.prices.get(_DATE_SUFFIX.sub("", name))

    @classmethod
    def from_file(cls, path: Path) -> PriceTable:
        """Load ``{"model": {"input": .., "output": .., "cache_read": .., "cache_write_1h": ..}}``
        on top of the defaults. ``cache_write_5m`` defaults to 1.25 x input."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a JSON object of model -> rates")
        table = cls()
        for model, r in raw.items():
            if not isinstance(r, dict):
                raise ValueError(f"{path}: rates for {model!r} must be an object")
            try:
                inp = float(r["input"])
                table.prices[str(model)] = Rates(
                    input=inp,
                    output=float(r["output"]),
                    cache_read=float(r["cache_read"]),
                    cache_write_5m=float(r.get("cache_write_5m", inp * 1.25)),
                    cache_write_1h=float(r["cache_write_1h"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: bad rates for {model!r}: {exc}") from exc
        return table
