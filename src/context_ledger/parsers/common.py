"""Helpers shared by the harness parsers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from context_ledger.model import Cost, CostSource, Usage
from context_ledger.pricing import PriceTable

JsonObj = dict[str, Any]

_LABEL_MAX = 80
# omp's hashline edit input names files as ``[path#ABCD]``.
_HASHLINE_PATH = re.compile(r"\[([^\]\n#]+)#[0-9A-Fa-f]{4}\]")
# omp read selectors: ``file.py:50-80``, ``file.py:raw``, ``file.py:-5`` ...
_READ_SELECTOR = re.compile(r":(?:raw|-?\d+[-+]?\d*(?:,[-\d+:]+)*)(?::raw)?$")


def read_jsonl(path: Path) -> Iterator[tuple[int, JsonObj | None]]:
    """Yield ``(line_number, object)``; ``object`` is None for unparseable lines."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield n, None
                continue
            yield n, obj if isinstance(obj, dict) else None


def parse_time(value: object) -> float | None:
    """Epoch seconds from an ISO-8601 string or epoch milliseconds."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value) / 1000.0 if value > 1e11 else float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def as_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def short(text: str, limit: int = _LABEL_MAX) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def price_usage(
    model: str, usage: Usage, recorded: Cost | None, prices: PriceTable
) -> tuple[Cost, CostSource]:
    """Prefer the harness's recorded cost, even zero (omp records 0 for history it replays
    into a new session without billing it); fall back to the price table."""
    if recorded is not None:
        return recorded, CostSource.RECORDED
    if usage.prompt + usage.output == 0:
        return Cost(), CostSource.RECORDED
    rates = prices.lookup(model)
    if rates is None:
        return Cost(), CostSource.UNPRICED
    return rates.price(usage), CostSource.TABLE


def normalize_read_path(arg: str) -> str:
    return _READ_SELECTOR.sub("", arg)


def edited_paths(tool: str, args: JsonObj) -> list[str]:
    """Files a tool call writes to, for the repeated-read detector."""
    name = tool.lower()
    paths: list[str] = []
    for key in ("path", "file_path", "notebook_path"):
        value = args.get(key)
        if isinstance(value, str) and name in {
            "write",
            "edit",
            "multiedit",
            "notebookedit",
            "ast_edit",
        }:
            paths.append(value)
    raw = args.get("input")
    if name == "edit" and isinstance(raw, str):
        paths.extend(m.group(1).strip() for m in _HASHLINE_PATH.finditer(raw))
    return paths


def call_label(tool: str, args: object) -> str:
    """``tool <main argument>``, e.g. ``read src/app.py`` or ``bash pytest -q``."""
    if isinstance(args, dict):
        for key in (
            "path",
            "file_path",
            "command",
            "pattern",
            "url",
            "query",
            "description",
            "i",
        ):
            value = args.get(key)
            if isinstance(value, str) and value:
                return short(f"{tool} {value}")
    return tool
