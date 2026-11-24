"""Counterfactual re-pricing: what the same sessions would have cost under a policy.

Every estimate keeps the agent's behaviour fixed (same turns, same calls) and re-prices
each affected item at the per-token cost it actually paid at each turn. That makes them
upper bounds: a capped or expired tool output can make the agent re-run a command or
re-read a file, which these numbers do not charge for.
"""

from __future__ import annotations

from dataclasses import dataclass

from context_ledger.attribute import Attribution
from context_ledger.detectors import excess_lifetime_cost
from context_ledger.model import Kind, Session

# Size of the placeholder an expired tool result would leave behind.
STUB_TOKENS = 20


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    saved: float


def cap_tool_outputs(pairs: list[tuple[Session, Attribution]], cap: int) -> float:
    return sum(
        excess_lifetime_cost(attr, item, cap)
        for session, attr in pairs
        for item in session.items.values()
        if item.kind == Kind.TOOL_RESULT and item.tokens > cap
    )


def expire_tool_results(pairs: list[tuple[Session, Attribution]], keep_turns: int) -> float:
    """Replace every tool result with a ``STUB_TOKENS`` placeholder ``keep_turns`` turns
    after it entered the prompt."""
    saved = 0.0
    for session, attr in pairs:
        for item in session.items.values():
            if item.kind != Kind.TOOL_RESULT or item.first_turn is None:
                continue
            for t, cost in attr.items[item.id].per_turn:
                size = item.size_at(t)
                if t - item.first_turn >= keep_turns and size > STUB_TOKENS:
                    saved += cost * (size - STUB_TOKENS) / size
    return saved


def strip_thinking(pairs: list[tuple[Session, Attribution]]) -> float:
    """Resend no thinking blocks from earlier turns (generation cost is unchanged)."""
    return sum(
        attr.items[item.id].lifetime
        for session, attr in pairs
        for item in session.items.values()
        if item.kind == Kind.THINKING
    )


def scenarios(
    pairs: list[tuple[Session, Attribution]],
    caps: tuple[int, ...] = (2_000, 5_000, 10_000),
    keeps: tuple[int, ...] = (10, 25, 50),
) -> list[Scenario]:
    out = [
        Scenario(
            f"cap_tool_output_{cap}",
            f"cap every tool result at {cap:,} tokens",
            cap_tool_outputs(pairs, cap),
        )
        for cap in caps
    ]
    out += [
        Scenario(
            f"expire_tool_results_{keep}",
            f"replace tool results with a {STUB_TOKENS}-token stub {keep} turns after they arrive",
            expire_tool_results(pairs, keep),
        )
        for keep in keeps
    ]
    out.append(
        Scenario(
            "strip_old_thinking",
            "do not resend earlier turns' thinking blocks",
            strip_thinking(pairs),
        )
    )
    return out
