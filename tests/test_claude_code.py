from __future__ import annotations

from pathlib import Path

import pytest

from context_ledger.attribute import attribute
from context_ledger.model import CostSource, Kind
from context_ledger.parsers import detect, load
from context_ledger.parsers.claude_code import parse_claude_code
from tests.synth import ClaudeCodeSession, tool_use


def _session(tmp_path: Path) -> tuple[ClaudeCodeSession, Path]:
    s = ClaudeCodeSession()
    s.user("Find where the config is loaded")
    a = s.assistant(
        [
            {"type": "thinking", "thinking": "two searches in parallel"},
            tool_use("tu1", "Grep", pattern="load_config"),
            tool_use("tu2", "Read", file_path="/work/demo/config.py", offset=1, limit=40),
        ],
        output=120,
        thinking_tokens=60,
    )
    # Parallel results: each hangs off its own tool_use line, out of order.
    s.tool_result("tu2", "c" * 2220, parent=a[2])
    s.tool_result("tu1", "config.py:12: def load_config():", parent=a[1])
    s.assistant([{"type": "text", "text": "It is loaded in config.py line 12."}], output=20)
    s.cost_state(extra=0.01)  # the harness total includes a side call not in the transcript
    return s, s.write(tmp_path / "cc.jsonl")


def test_detects_claude_code(tmp_path: Path) -> None:
    _, path = _session(tmp_path)
    assert detect(path) == "claude-code"


def test_streamed_lines_merge_into_one_turn(tmp_path: Path) -> None:
    _, path = _session(tmp_path)
    session = parse_claude_code(path)
    assert len(session.turns) == 2
    t0 = session.turns[0]
    kinds = [session.items[i].kind for i in t0.output_items]
    assert kinds == [Kind.THINKING, Kind.TOOL_CALL, Kind.TOOL_CALL]
    thinking = session.items[t0.output_items[0]]
    assert thinking.tokens == 60  # recorded thinking count, not a remainder
    assert sum(session.items[i].tokens for i in t0.output_items) == 120
    assert all(t.cost_source == CostSource.TABLE for t in session.turns)


def test_parallel_tool_results_are_all_in_context(tmp_path: Path) -> None:
    _, path = _session(tmp_path)
    session = parse_claude_code(path)
    ctx = session.turns[1].context
    results = [session.items[i] for i in ctx if session.items[i].kind == Kind.TOOL_RESULT]
    assert len(results) == 2
    attr = attribute(session)
    assert attr.calibration.residual_tokens == [0, 0]
    read = next(r for r in results if r.tool_name == "Read")
    assert read.file_path == "/work/demo/config.py"
    assert read.label.startswith("Read /work/demo/config.py")


def test_recorded_total_and_reconciliation(tmp_path: Path) -> None:
    synth, path = _session(tmp_path)
    session = parse_claude_code(path)
    attr = attribute(session)
    assert session.recorded_total == pytest.approx(synth.total_cost + 0.01)
    assert attr.total == pytest.approx(synth.total_cost, rel=1e-9)
    assert attr.attributed + attr.unattributed == pytest.approx(attr.total, rel=1e-12)


def test_system_block_absorbs_first_turn_remainder(tmp_path: Path) -> None:
    _, path = _session(tmp_path)
    session = parse_claude_code(path)
    attribute(session)
    assert session.items["system#1"].tokens == 12_000


def test_sidechain_file_becomes_subagent(tmp_path: Path) -> None:
    main = ClaudeCodeSession(sid="00000000-0000-4000-8000-0000000000aa")
    main.user("delegate")
    main.assistant([tool_use("task1", "Task", description="explore")], output=30)
    main_path = main.write(tmp_path / "proj" / "main.jsonl")
    sub = ClaudeCodeSession(sid=main.sid, sidechain=True, agent_id="agent-1", system_tokens=3000)
    sub.user("explore the repo")
    sub.assistant([{"type": "text", "text": "found it"}], output=10)
    sub.write(tmp_path / "proj" / "main" / "subagents" / "agent-1.jsonl")
    loaded = load(main_path)
    assert len(loaded.sessions) == 1
    assert [s.agent for s in loaded.sessions[0].subagents] == ["agent-1"]
