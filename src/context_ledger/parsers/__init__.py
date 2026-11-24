"""Find session logs under a path, detect their harness, parse and group them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from context_ledger.model import Session
from context_ledger.parsers.claude_code import looks_like_claude_code, parse_claude_code
from context_ledger.parsers.common import JsonObj, read_jsonl
from context_ledger.parsers.omp import looks_like_omp, parse_omp
from context_ledger.pricing import PriceTable

_PEEK_LINES = 30


@dataclass
class LoadResult:
    sessions: list[Session] = field(default_factory=list)  # top level; subagents nested
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)

    def all_sessions(self) -> list[Session]:
        out: list[Session] = []
        stack = list(reversed(self.sessions))
        while stack:
            s = stack.pop()
            out.append(s)
            stack.extend(reversed(s.subagents))
        return out


def detect(path: Path) -> str | None:
    head: list[JsonObj] = []
    for _, obj in read_jsonl(path):
        if obj is not None:
            head.append(obj)
        if len(head) >= _PEEK_LINES:
            break
    if looks_like_omp(head):
        return "omp"
    if looks_like_claude_code(head):
        return "claude-code"
    return None


def load(target: Path, prices: PriceTable | None = None) -> LoadResult:
    table = prices or PriceTable()
    result = LoadResult()
    if target.is_file():
        # Both harnesses keep a session's subagent transcripts in a directory named after it.
        sibling = target.with_suffix("")
        files = [target, *(sorted(sibling.rglob("*.jsonl")) if sibling.is_dir() else [])]
    else:
        files = sorted(target.rglob("*.jsonl"))
    parsed: list[Session] = []
    for f in files:
        harness = detect(f)
        if harness is None:
            result.skipped.append((str(f), "not an omp or Claude Code session log"))
            continue
        try:
            session = parse_omp(f, table) if harness == "omp" else parse_claude_code(f, table)
        except (ValueError, OSError) as exc:
            result.skipped.append((str(f), f"parse error: {exc}"))
            continue
        parsed.append(session)

    # omp names a subagent's parent by session id or by the parent's file path.
    by_key: dict[str, Session] = {}
    for s in parsed:
        by_key[s.id] = s
        by_key[str(Path(s.path).resolve())] = s
    for s in parsed:
        parent = None
        if s.parent_id:
            parent = by_key.get(s.parent_id) or by_key.get(
                str(Path(s.parent_id).expanduser().resolve())
            )
        if parent is not None and parent is not s:
            parent.subagents.append(s)
        else:
            result.sessions.append(s)
    return result
