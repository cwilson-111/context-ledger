from __future__ import annotations

import json
from pathlib import Path

from context_ledger.attribute import attribute
from context_ledger.model import CostSource, Kind, Session, TokenSource
from context_ledger.parsers import load
from context_ledger.parsers.omp import parse_omp
from context_ledger.tokens import estimate_text
from tests.synth import OmpSession, call, text, thinking


def test_turns_items_and_order(basic: tuple[Session, object]) -> None:
    session, _ = basic
    assert len(session.turns) == 4
    t3 = session.turns[3]
    kinds = [session.items[i].kind for i in t3.context]
    assert kinds[0] == Kind.SYSTEM
    assert kinds[1] == Kind.USER
    # Every earlier assistant part and tool result is resent, in order.
    assert kinds.count(Kind.TOOL_RESULT) == 3
    assert kinds.count(Kind.TOOL_CALL) == 3
    assert all(t.cost_source == CostSource.RECORDED for t in session.turns)
    assert [e.kind for e in session.turns[0].events] == ["model_change", "thinking_level_change"]


def test_tool_result_links_to_call(basic: tuple[Session, object]) -> None:
    session, _ = basic
    results = [i for i in session.items.values() if i.kind == Kind.TOOL_RESULT]
    assert results[0].label == "read calc.py"
    assert results[0].file_path == "calc.py"
    assert results[2].is_error


def test_unparseable_lines_are_counted(tmp_path: Path) -> None:
    s = OmpSession("broken00-0000-4000-8000-000000000001")
    s.user("hi")
    s.assistant([text("hello")], 5)
    path = s.write(tmp_path / "s.jsonl")
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    session = parse_omp(path)
    assert session.skipped_lines == 1
    assert len(session.turns) == 1


def test_compaction_replaces_history(tmp_path: Path) -> None:
    s = OmpSession("compact0-0000-4000-8000-000000000001")
    s.user("start")
    s.assistant([call("c1", "bash", command="ls")], 40)
    s.tool_result("c1", "bash", "a" * 4440)  # 2000 tokens
    kept = s.user("keep this")
    s.assistant([call("c2", "bash", command="ls -la")], 40)
    s.tool_result("c2", "bash", "b" * 222)  # 100 tokens
    # kept history: user (5) + assistant (40) + result (100)
    s.compact(kept, kept_tokens=145, summary_tokens=700)
    s.assistant([text("done")], 10)
    session = parse_omp(s.write(tmp_path / "c.jsonl"))
    last = session.turns[-1]
    assert last.after_compaction
    labels = [session.items[i].kind for i in last.context]
    assert labels[:3] == [Kind.SYSTEM, Kind.SUMMARY, Kind.USER]
    attr = attribute(session)
    summary = next(session.items[i] for i in last.context if session.items[i].kind == Kind.SUMMARY)
    assert summary.tokens == 700  # recovered from the prompt-size delta
    assert attr.calibration.residual_tokens[-1] == 0
    assert session.compactions[0].first_turn_after == len(session.turns) - 1


def test_replayed_history_is_not_billed(tmp_path: Path) -> None:
    s = OmpSession("seance00-0000-4000-8000-000000000001")
    s.user("old task")
    s.assistant([text("old answer")], 30, cost_recorded=False)
    s.user("new question")
    s.assistant([text("new answer")], 20)
    session = parse_omp(s.write(tmp_path / "r.jsonl"))
    assert session.turns[0].replayed and session.turns[0].cost.total == 0
    assert not session.turns[1].replayed
    assert session.turns[0].cost_source == CostSource.RECORDED


def test_pruned_result_shrinks_from_its_prune_time(tmp_path: Path) -> None:
    s = OmpSession("prune000-0000-4000-8000-000000000001")
    s.user("go")
    s.assistant([call("c1", "read", path="big.py")], 30)
    placeholder = "[Superseded by a newer read of this file]"
    rid = s.tool_result("c1", "read", placeholder, tokens=3000)
    s.assistant([call("c2", "bash", command="ls")], 30)
    s.tool_result("c2", "bash", "ok")
    prune_time = s.now + 1
    s._prompt_known -= 3000 - 19  # from the next request the placeholder (19 tokens) is sent
    s.assistant([text("a")], 10)  # starts after prune_time
    s.user("more")
    s.assistant([text("b")], 10)
    lines = s.lines
    for line in lines:
        if line.get("id") == rid:
            line["message"]["prunedAt"] = int(prune_time * 1000)
    session = parse_omp(s.write(tmp_path / "p.jsonl"))
    attr = attribute(session)
    item = session.items[rid]
    assert item.tokens == 3000
    assert item.shrink_turn == 2
    assert item.size_at(1) == 3000 and item.size_at(2) == item.shrunk_tokens
    assert any(e.kind == "context_edit" for e in session.turns[2].events)
    assert attr.calibration.residual_tokens[1] == 0


def test_dropped_thinking_stays_dropped(tmp_path: Path) -> None:
    s = OmpSession("drop0000-0000-4000-8000-000000000001")
    s.user("q")
    a1 = s.assistant([thinking("t" * 400), call("c1", "bash", command="ls")], 300)
    s.tool_result("c1", "bash", "ok")
    # thinking = output - estimated tool call (10 tokens): the harness stops resending it
    s._prompt_known -= 300 - 10
    s.assistant(
        [call("c2", "bash", command="pwd")],
        20,
        inputTransformations=[
            {"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "x"}
        ],
    )
    s.tool_result("c2", "bash", "/")
    s.assistant([text("done")], 10)
    session = parse_omp(s.write(tmp_path / "d.jsonl"))
    thinking_id = f"{a1}:0"
    assert thinking_id not in session.turns[1].context
    assert thinking_id not in session.turns[2].context
    assert "thinking_dropped" in [e.kind for e in session.turns[1].events]
    assert attribute(session).calibration.residual_tokens == [0, 0, 0]


def test_subagents_attach_by_id_or_path(tmp_path: Path) -> None:
    main = OmpSession("main0000-0000-4000-8000-000000000001")
    main.user("delegate")
    main.assistant([call("t1", "task", agent="explore")], 50)
    main.tool_result(
        "t1", "task", "spawned", details={"progress": [{"id": "Scout", "status": "done"}]}
    )
    main.assistant([text("ok")], 10)
    main_path = main.write(tmp_path / "main.jsonl")
    by_id = OmpSession("sub00000-0000-4000-8000-000000000001", parent=main.sid)
    by_id.user("explore")
    by_id.assistant([text("found")], 10)
    by_id.write(tmp_path / "main" / "Scout.jsonl")
    by_path = OmpSession("sub00000-0000-4000-8000-000000000002", parent=str(main_path))
    by_path.user("explore 2")
    by_path.assistant([text("found 2")], 10)
    by_path.write(tmp_path / "main" / "Other.jsonl")

    loaded = load(main_path)
    assert len(loaded.sessions) == 1
    root = loaded.sessions[0]
    assert sorted(s.agent or "" for s in root.subagents) == ["Other", "Scout"]
    assert root.items[root.spawned_by["Scout"]].kind == Kind.TOOL_CALL


def test_side_calls_and_cache_warm(tmp_path: Path) -> None:
    s = OmpSession("side0000-0000-4000-8000-000000000001")
    s.user("x")
    s.side_call("title")
    s.assistant([text("y")], 5)
    session = parse_omp(s.write(tmp_path / "s.jsonl"))
    assert [c.purpose for c in session.side_calls] == ["title"]
    assert session.side_calls[0].cost.total > 0


def test_non_session_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "changes.jsonl").write_text(json.dumps({"type": "change"}) + "\n")
    loaded = load(tmp_path)
    assert loaded.sessions == []
    assert loaded.skipped and "not an omp" in loaded.skipped[0][1]


def test_calibration_recovers_true_size(basic: tuple[Session, object]) -> None:
    session, _ = basic
    read_result = next(
        i
        for i in session.items.values()
        if i.label == "read calc.py" and i.kind == Kind.TOOL_RESULT
    )
    assert read_result.est_tokens == estimate_text("x" * 2220)
    assert read_result.tokens == 1300
    assert read_result.source == TokenSource.DELTA
    system = session.items["system#1"]
    assert system.tokens == 9_000 + 1_200  # harness count plus what it misses
