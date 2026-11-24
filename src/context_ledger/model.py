"""Harness-neutral session model that every parser produces.

A *turn* is one billed model request. Its prompt is an ordered list of *items* (the system
block first, then conversation pieces in the order they were sent). Anthropic prompt caching
is prefix based, so position decides whether a token was billed as a cache read, a cache
write or uncached input; the attribution engine relies on that order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Kind(StrEnum):
    SYSTEM = "system"  # system prompt + tool schemas
    USER = "user"
    ASSISTANT_TEXT = "assistant_text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INJECTED = "injected"  # harness reminders, async results, developer/meta messages
    SUMMARY = "compaction_summary"


class TokenSource(StrEnum):
    RECORDED = "recorded"  # read from a usage field (assistant output, harness-reported size)
    DELTA = "delta"  # the only new item between two turns: the prompt-size delta is its size
    SCALED = "scaled"  # a delta pinned a group of new items; split across them by estimate
    ESTIMATE = "estimate"  # no usable delta: estimator output


@dataclass
class Item:
    """One piece of context. ``tokens`` is the calibrated size; ``est_tokens`` the estimate."""

    id: str
    kind: Kind
    label: str
    entry_id: str
    est_tokens: int
    tokens: int = 0
    source: TokenSource = TokenSource.ESTIMATE
    born_turn: int | None = None  # turn whose output generated it (assistant parts)
    first_turn: int | None = None  # first turn whose prompt carried it (set by calibration)
    tool_name: str | None = None
    # The logged text cannot size the item (pruned content, opaque compaction blobs, a
    # system prompt that is not logged): calibration gives it whatever a delta leaves over.
    size_unknown: bool = False
    image_tokens: int = 0  # part of est_tokens that estimates attached images
    tool_call_id: str | None = None
    file_path: str | None = None  # file a read call targets (range selectors stripped)
    read_key: str | None = None  # the exact read request (path plus any range)
    writes: tuple[str, ...] = ()  # files a tool call edits or writes
    is_error: bool = False
    content_hash: str | None = None
    # The harness replaced the content later (omp pruning). From ``shrink_turn`` on the item
    # is resent at ``shrunk_tokens``; the logged text is the replacement, so ``est_tokens``
    # describes the shrunk form and the original size is only knowable from a delta.
    shrink_turn: int | None = None
    shrunk_tokens: int | None = None

    def size_at(self, turn: int) -> int:
        shrunk = self.shrink_turn is not None and self.shrunk_tokens is not None
        if shrunk and turn >= (self.shrink_turn or 0):
            return self.shrunk_tokens or 0
        return self.tokens


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int = 0  # part of cache_write written with the 1 hour TTL
    thinking: int | None = None  # part of output that was thinking, when recorded

    @property
    def prompt(self) -> int:
        return self.input + self.cache_read + self.cache_write


@dataclass
class Cost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0

    @property
    def total(self) -> float:
        return self.input + self.output + self.cache_read + self.cache_write

    @property
    def prompt(self) -> float:
        return self.input + self.cache_read + self.cache_write


class CostSource(StrEnum):
    RECORDED = "recorded"  # the harness wrote per-request cost into the log
    TABLE = "table"  # priced from context_ledger.pricing (log had tokens but no cost)
    UNPRICED = "unpriced"  # unknown model: tokens counted, cost left at zero


@dataclass
class Event:
    """Something between two turns that can explain a cache bust or a context change."""

    kind: str  # model_change, thinking_level_change, system_prompt_change, tools_change, ...
    timestamp: float
    detail: str = ""


@dataclass
class Turn:
    index: int
    entry_id: str
    timestamp: float  # request start, epoch seconds
    completed_at: float | None
    model: str
    usage: Usage
    cost: Cost
    cost_source: CostSource
    stop_reason: str
    context: list[str] = field(default_factory=list)  # item ids in prompt order
    output_items: list[str] = field(default_factory=list)
    cache_ttl_s: int | None = None
    events: list[Event] = field(default_factory=list)  # since the previous turn
    after_compaction: bool = False
    # Recorded as unbilled despite carrying tokens: history copied into this session from
    # another one (omp "seance" subagents replay their source session at zero cost).
    replayed: bool = False
    # Harness-reported size of the system block, when the log records it.
    system_tokens: int | None = None


@dataclass
class SideCall:
    """A billed request that is not part of the conversation (titles, cache warmers, ...)."""

    entry_id: str
    purpose: str
    model: str
    timestamp: float
    usage: Usage
    cost: Cost
    cost_source: CostSource


@dataclass
class Compaction:
    entry_id: str
    timestamp: float
    first_turn_after: int | None
    tokens_before: int | None
    tokens_after: int | None
    method: str


@dataclass
class Session:
    id: str
    harness: str  # "omp" | "claude-code"
    path: str
    agent: str | None = None  # subagent name; None for a main session
    parent_id: str | None = None
    items: dict[str, Item] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    side_calls: list[SideCall] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
    # Total cost the harness itself reports for the session, if it records one separately
    # from the per-request entries (Claude Code's ``cost-state``).
    recorded_total: float | None = None
    # Subagent id -> id of the tool-call item that spawned it.
    spawned_by: dict[str, str] = field(default_factory=dict)
    subagents: list[Session] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped_lines: int = 0

    @property
    def name(self) -> str:
        return self.agent or self.id

    def add_item(self, item: Item) -> Item:
        self.items[item.id] = item
        return item


def spread(items: list[Item], total: int, weights: list[int]) -> None:
    """Set ``tokens`` so the items sum to ``total``, proportionally to ``weights``."""
    wsum = sum(weights)
    assigned = 0
    for idx, item in enumerate(items):
        if idx == len(items) - 1:
            item.tokens = total - assigned
        elif wsum > 0:
            item.tokens = total * weights[idx] // wsum
            assigned += item.tokens
        else:
            item.tokens = total // len(items)
            assigned += item.tokens


def split_output(parts: list[Item], total: int, thinking_tokens: int | None = None) -> None:
    """Size an assistant message's parts so they sum to its recorded output tokens.

    Text and tool calls are logged verbatim, so their estimates stand; thinking is often
    logged summarised or not at all, so it takes the remainder unless the harness recorded
    the thinking count itself.
    """
    if not parts:
        return
    thinking = [p for p in parts if p.kind == Kind.THINKING]
    visible = [p for p in parts if p.kind != Kind.THINKING]
    if thinking and thinking_tokens is not None and 0 <= thinking_tokens <= total:
        spread(thinking, thinking_tokens, [max(p.est_tokens, 1) for p in thinking])
        if visible:
            spread(visible, total - thinking_tokens, [max(p.est_tokens, 1) for p in visible])
    else:
        visible_est = sum(p.est_tokens for p in visible)
        if thinking and total >= visible_est:
            for p in visible:
                p.tokens = p.est_tokens
            spread(thinking, total - visible_est, [max(p.est_tokens, 1) for p in thinking])
        else:
            for p in thinking:
                p.tokens = 0
            target = visible or parts
            spread(target, total, [max(p.est_tokens, 1) for p in target])
    for p in parts:
        p.source = TokenSource.RECORDED
