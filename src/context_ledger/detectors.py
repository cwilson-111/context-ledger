"""Rule-based waste detectors. Each is a pure function of an attributed session.

``Finding.cost`` is the money the pattern is responsible for under the stated definition;
the definitions are repeated in the README. Labels inside ``Finding.summary`` come from the
logs (file paths, commands) and are untrusted text: renderers must escape them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from context_ledger.attribute import Attribution
from context_ledger.model import Item, Kind, Session, Turn

HUGE_OUTPUT_TOKENS = 5_000
BUST_MIN_TOKENS = 2_048
BUST_MIN_SHARE = 0.10


@dataclass(frozen=True)
class Finding:
    detector: str
    cost: float
    tokens: int
    turn: int | None
    summary: str
    item_ids: tuple[str, ...] = ()
    detail: dict[str, str | int | float] = field(default_factory=dict)


def _item_cost(attr: Attribution, item_id: str | None) -> float:
    if item_id is None or item_id not in attr.items:
        return 0.0
    return attr.items[item_id].total


def _calls_and_results(session: Session) -> list[tuple[Item, Item | None]]:
    """Tool calls in session order, each with its result item (if logged).

    Calls made in replayed history (copied from another session at zero cost) are left
    out so each real call is counted once, in the session that made it.
    """
    results: dict[str, Item] = {}
    for item in session.items.values():
        if item.kind == Kind.TOOL_RESULT and item.tool_call_id:
            results.setdefault(item.tool_call_id, item)
    calls = [
        i
        for i in session.items.values()
        if i.kind == Kind.TOOL_CALL
        and not (i.born_turn is not None and session.turns[i.born_turn].replayed)
    ]
    calls.sort(key=lambda i: i.born_turn if i.born_turn is not None else 1 << 30)
    return [(c, results.get(c.tool_call_id or "")) for c in calls]


def _same_file(a: str, b: str) -> bool:
    """Paths name the same file, allowing one to be relative to an unknown working dir."""
    a, b = a.removeprefix("./"), b.removeprefix("./")
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def repeated_reads(session: Session, attr: Attribution) -> list[Finding]:
    """A read is redundant when the identical read request was made earlier, that earlier
    result is still in the prompt (not compacted or pruned away) when the new result
    arrives, and the file was not written in between. Cost: the new call and result."""
    in_context: dict[str, set[int]] = {i: {t for t, _ in c.per_turn} for i, c in attr.items.items()}
    last_read: dict[str, tuple[Item, Item]] = {}
    last_write: dict[str, int] = {}
    findings: list[Finding] = []
    for call, result in _calls_and_results(session):
        turn = call.born_turn if call.born_turn is not None else -1
        for path in call.writes:
            last_write[path] = turn
        if call.read_key is None or result is None:
            continue
        prev = last_read.get(call.read_key)
        last_read[call.read_key] = (call, result)
        if prev is None:
            continue
        prev_call, prev_result = prev
        prev_turn = prev_call.born_turn if prev_call.born_turn is not None else -1
        written = any(
            t >= prev_turn and _same_file(p, call.file_path or "") for p, t in last_write.items()
        )
        arrival = result.first_turn
        alive = (
            arrival is not None
            and arrival in in_context.get(prev_result.id, set())
            and prev_result.size_at(arrival) == prev_result.tokens
        )
        if written or not alive or result.is_error:
            continue
        cost = _item_cost(attr, call.id) + _item_cost(attr, result.id)
        findings.append(
            Finding(
                "repeated_reads",
                cost,
                result.tokens,
                result.first_turn,
                f"re-read {call.read_key} (first read at turn {prev_turn})",
                (call.id, result.id),
            )
        )
    return findings


def huge_tool_outputs(
    session: Session, attr: Attribution, threshold: int = HUGE_OUTPUT_TOKENS
) -> list[Finding]:
    """Tool results of at least ``threshold`` tokens. Cost: the lifetime cost of the part
    above ``threshold`` (what capping output at that size would have saved, assuming the
    agent needed nothing past the cap)."""
    findings: list[Finding] = []
    for item in session.items.values():
        if item.kind != Kind.TOOL_RESULT or item.tokens < threshold:
            continue
        excess_cost = excess_lifetime_cost(attr, item, threshold)
        findings.append(
            Finding(
                "huge_tool_outputs",
                excess_cost,
                item.tokens,
                item.first_turn,
                f"{item.label}: {item.tokens:,} tokens, carried "
                f"{attr.items[item.id].turns} turns, lifetime ${attr.items[item.id].total:.2f}",
                (item.id,),
            )
        )
    findings.sort(key=lambda f: f.cost, reverse=True)
    return findings


def excess_lifetime_cost(attr: Attribution, item: Item, cap: int) -> float:
    """Lifetime cost of the tokens of ``item`` above ``cap`` (pro rata per turn)."""
    total = 0.0
    for t, cost in attr.items[item.id].per_turn:
        size = item.size_at(t)
        if size > cap:
            total += cost * (size - cap) / size
    return total


def failed_tool_calls(session: Session, attr: Attribution) -> list[Finding]:
    """Tool calls whose result is an error. Consecutive failures of the same tool form a
    retry loop and are reported as one finding. Cost: the failing calls and their error
    results (generation plus lifetime)."""
    by_tool: dict[str, list[tuple[Item, Item]]] = defaultdict(list)
    for call, result in _calls_and_results(session):
        if result is not None:
            by_tool[call.tool_name or "tool"].append((call, result))
    findings: list[Finding] = []

    def flush(tool: str, run: list[tuple[Item, Item]]) -> None:
        ids = tuple(i.id for pair in run for i in pair)
        first = run[0][0]
        summary = (
            f"failed: {first.label}"
            if len(run) == 1
            else f"retry loop: {len(run)} consecutive failed {tool} calls"
        )
        findings.append(
            Finding(
                "failed_tool_calls",
                sum(_item_cost(attr, i) for i in ids),
                sum(r.tokens + c.tokens for c, r in run),
                first.born_turn,
                summary,
                ids,
                {"attempts": len(run), "loop": int(len(run) > 1)},
            )
        )

    for tool, seq in by_tool.items():
        run: list[tuple[Item, Item]] = []
        for call, result in seq:
            if result.is_error:
                run.append((call, result))
            elif run:
                flush(tool, run)
                run = []
        if run:
            flush(tool, run)
    findings.sort(key=lambda f: f.cost, reverse=True)
    return findings


def _rate(tokens: int, cost: float) -> float | None:
    return cost / tokens if tokens > 0 and cost > 0 else None


def _bust_cause(prev: Turn, turn: Turn) -> tuple[str, float]:
    kinds = {e.kind for e in turn.events}
    for kind, cause in (
        ("model_change", "model change"),
        ("thinking_level_change", "thinking level change"),
        ("system_prompt_change", "system prompt changed"),
        ("tools_change", "tool set changed"),
        ("system_size_change", "tool set changed"),
        ("thinking_dropped", "thinking blocks stripped"),
        ("context_edit", "context edited (pruning)"),
    ):
        if kind in kinds:
            return cause, 0.0
    touched = prev.completed_at or prev.timestamp
    for e in turn.events:
        if e.kind == "cache_warm":
            touched = max(touched, e.timestamp)
    gap = turn.timestamp - touched
    ttl = prev.cache_ttl_s or turn.cache_ttl_s
    if ttl is not None and gap > ttl:
        return "cache TTL expired", gap
    return "unknown", gap


def cache_busts(session: Session, attr: Attribution) -> list[Finding]:
    """A turn whose cache read falls short of the prefix the previous turn left cached by at
    least ``BUST_MIN_TOKENS`` tokens and ``BUST_MIN_SHARE`` of that prefix (excluding the
    first turn after a compaction, where the prompt is rebuilt on purpose, and replayed
    history). Cost: the lost tokens priced at this turn's cache-write minus cache-read rate.
    The cause is read from harness events since the previous turn, else from the time gap
    against the recorded cache TTL."""
    findings: list[Finding] = []
    for prev, turn in zip(session.turns, session.turns[1:], strict=False):
        if turn.after_compaction or turn.replayed:
            continue
        cached = prev.usage.cache_read + prev.usage.cache_write
        expected = min(cached, turn.usage.prompt)
        lost = expected - turn.usage.cache_read
        if lost < BUST_MIN_TOKENS or lost < BUST_MIN_SHARE * expected:
            continue
        write = _rate(turn.usage.cache_write, turn.cost.cache_write)
        read = _rate(turn.usage.cache_read, turn.cost.cache_read)
        if write is None:
            continue
        if read is None:
            # Nothing read this turn (full bust). Price the counterfactual reads at the
            # previous turn's read rate when the model is the same; otherwise count the
            # whole write (overstates by the read rate, a few percent of the write rate).
            same_model = prev.model == turn.model
            read = _rate(prev.usage.cache_read, prev.cost.cache_read) if same_model else None
        extra = lost * (write - (read or 0.0))
        cause, gap = _bust_cause(prev, turn)
        findings.append(
            Finding(
                "cache_busts",
                extra,
                lost,
                turn.index,
                f"turn {turn.index}: {lost:,} cached tokens rewritten ({cause}"
                + (f", gap {gap / 60:.0f} min" if gap > 60 else "")
                + ")",
                detail={"cause": cause, "gap_s": round(gap, 1)},
            )
        )
    return findings


def compactions(session: Session, attr: Attribution) -> list[Finding]:
    """Each compaction with an estimate of what it saved: the prompt tokens it removed,
    priced at the cache-read rate of every later turn until the next compaction or the end,
    minus the extra cost of re-writing the cache on the first turn after it. An upper bound:
    without compaction the session would have hit the context limit."""
    findings: list[Finding] = []
    starts = sorted(c.first_turn_after for c in session.compactions if c.first_turn_after)
    for comp in session.compactions:
        after = comp.first_turn_after
        if after is None or after == 0:
            continue
        # The harness's own pre-compaction count includes what arrived after the last turn.
        before_p = comp.tokens_before or session.turns[after - 1].usage.prompt
        first = session.turns[after]
        removed = max(before_p - first.usage.prompt, 0)
        end = next((s for s in starts if s > after), len(session.turns))
        saved = 0.0
        for t in session.turns[after:end]:
            read = _rate(t.usage.cache_read, t.cost.cache_read)
            if read is not None:
                saved += removed * read
        write = _rate(first.usage.cache_write, first.cost.cache_write) or 0.0
        read0 = _rate(first.usage.cache_read, first.cost.cache_read) or 0.0
        rewrite = first.usage.cache_write * max(write - read0, 0.0)
        findings.append(
            Finding(
                "compactions",
                saved - rewrite,
                removed,
                after,
                f"{comp.method} compaction before turn {after}: prompt {before_p:,} -> "
                f"{first.usage.prompt:,} tokens over {end - after} later turns",
                detail={"rewrite_cost": rewrite, "turns_after": end - after},
            )
        )
    return findings


def side_calls(session: Session, attr: Attribution) -> list[Finding]:
    """Billed requests outside the conversation, grouped by purpose (titles, cache warming,
    completion probes)."""
    by_purpose: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for call in session.side_calls:
        by_purpose[call.purpose][0] += call.cost.total
        by_purpose[call.purpose][1] += 1
    return [
        Finding("side_calls", cost, 0, None, f"{purpose}: {int(n)} calls", detail={"calls": int(n)})
        for purpose, (cost, n) in sorted(by_purpose.items(), key=lambda kv: -kv[1][0])
    ]


WASTE_DETECTORS = {
    "repeated_reads": repeated_reads,
    "huge_tool_outputs": huge_tool_outputs,
    "failed_tool_calls": failed_tool_calls,
    "cache_busts": cache_busts,
}
INFO_DETECTORS = {
    "compactions": compactions,
    "side_calls": side_calls,
}


def run_all(session: Session, attr: Attribution) -> dict[str, list[Finding]]:
    out: dict[str, list[Finding]] = {}
    for name, fn in {**WASTE_DETECTORS, **INFO_DETECTORS}.items():
        out[name] = fn(session, attr)
    return out
