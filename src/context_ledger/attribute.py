"""Lifetime cost attribution.

For each turn the prompt is laid out in send order (system block, then conversation items).
Anthropic bills a prompt as ``cache_read`` tokens for the cached prefix, then
``cache_write`` tokens for the newly cached span, then ``input`` tokens for the uncached
tail. Each item is charged for the tokens it occupies in each region at that turn's
recorded per-token rate. Output cost is charged to the assistant parts the turn produced.

Identity, per turn and therefore per session, up to float rounding:

    sum(item costs) + unattributed == recorded turn cost

``unattributed`` is the cost of prompt tokens the calibrated items do not account for (a
positive residual), plus output cost of turns with no logged content. A negative residual
(items larger than the prompt) leaves the overflow uncharged and is counted separately.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from context_ledger.calibrate import Calibration, calibrate
from context_ledger.model import Kind, Session


@dataclass
class ItemCost:
    item_id: str
    birth: float = 0.0  # output cost of generating the item
    lifetime: float = 0.0  # cost of resending it in later prompts
    turns: int = 0  # prompts it rode in
    token_turns: int = 0  # sum of its size over those prompts
    per_turn: list[tuple[int, float]] = field(default_factory=list)  # (turn, cost)

    @property
    def total(self) -> float:
        return self.birth + self.lifetime


@dataclass
class TurnCost:
    index: int
    prompt: float
    output: float
    unattributed: float = 0.0
    residual_tokens: int = 0  # prompt tokens not covered by items (>0) or overflow (<0)

    @property
    def total(self) -> float:
        return self.prompt + self.output


@dataclass
class Attribution:
    session: Session
    calibration: Calibration
    items: dict[str, ItemCost]
    turns: list[TurnCost]
    side_calls: float

    @property
    def total(self) -> float:
        return sum(t.total for t in self.turns) + self.side_calls

    @property
    def unattributed(self) -> float:
        return sum(t.unattributed for t in self.turns)

    @property
    def attributed(self) -> float:
        return sum(c.total for c in self.items.values()) + self.side_calls

    def by_kind(self) -> dict[str, tuple[float, float]]:
        """kind -> (birth/output cost, lifetime/resend cost)."""
        out: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for item_id, cost in self.items.items():
            kind = self.session.items[item_id].kind.value
            out[kind][0] += cost.birth
            out[kind][1] += cost.lifetime
        return {k: (v[0], v[1]) for k, v in out.items()}


def _charge(sizes: list[float], regions: list[tuple[int, float]]) -> tuple[list[float], float]:
    """Charge consecutive token spans against consecutive priced regions.

    Returns each span's cost and the cost of prompt tokens after the last span.
    """
    bounds: list[tuple[float, float, float]] = []  # (start, end, cost per token)
    pos = 0.0
    for tokens, cost in regions:
        bounds.append((pos, pos + tokens, cost / tokens))
        pos += tokens
    charged: list[float] = []
    start = 0.0
    for size in sizes:
        end = start + size
        charged.append(
            sum(max(0.0, min(end, hi) - max(start, lo)) * rate for lo, hi, rate in bounds)
        )
        start = end
    left = sum(max(0.0, hi - max(start, lo)) * rate for lo, hi, rate in bounds)
    return charged, left


def attribute(session: Session, calibration: Calibration | None = None) -> Attribution:
    cal = calibration if calibration is not None else calibrate(session)
    items: dict[str, ItemCost] = {i: ItemCost(i) for i in session.items}
    turns: list[TurnCost] = []
    for turn in session.turns:
        t = turn.index
        u, c = turn.usage, turn.cost
        regions = [
            (n, cost)
            for n, cost in (
                (u.cache_read, c.cache_read),
                (u.cache_write, c.cache_write),
                (u.input, c.input),
            )
            if n > 0
        ]
        sizes = [session.items[i].size_at(t) for i in turn.context]
        covered = sum(sizes)
        residual = u.prompt - covered
        # Items larger than the recorded prompt: shrink every span proportionally rather
        # than leave the newest items uncharged. The overflow stays visible in ``residual``.
        scale = u.prompt / covered if covered > u.prompt and covered > 0 else 1.0
        charged, left_cost = _charge([s * scale for s in sizes], regions)
        for item_id, cost, size in zip(turn.context, charged, sizes, strict=True):
            ic = items[item_id]
            ic.lifetime += cost
            ic.turns += 1
            ic.token_turns += size
            ic.per_turn.append((t, cost))
        tc = TurnCost(t, c.prompt, c.output, unattributed=left_cost, residual_tokens=residual)

        outputs = [session.items[i] for i in turn.output_items]
        out_tokens = sum(max(i.tokens, 0) for i in outputs)
        if outputs and out_tokens > 0:
            for item in outputs:
                items[item.id].birth += c.output * max(item.tokens, 0) / out_tokens
        elif outputs:
            for item in outputs:
                items[item.id].birth += c.output / len(outputs)
        else:
            tc.unattributed += c.output
        turns.append(tc)
    side = sum(s.cost.total for s in session.side_calls)
    return Attribution(session, cal, items, turns, side)


KIND_ORDER = [k.value for k in Kind]
