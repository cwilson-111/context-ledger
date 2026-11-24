from __future__ import annotations

from pathlib import Path

import pytest

from context_ledger.attribute import Attribution, attribute
from context_ledger.detectors import (
    cache_busts,
    compactions,
    failed_tool_calls,
    huge_tool_outputs,
    repeated_reads,
    side_calls,
)
from context_ledger.model import Session
from context_ledger.parsers.omp import parse_omp
from tests.synth import OmpSession, call, text


def _run(s: OmpSession, tmp_path: Path) -> tuple[Session, Attribution]:
    session = parse_omp(s.write(tmp_path / f"{s.sid[:6]}.jsonl"))
    return session, attribute(session)


def test_repeated_read_flagged_only_when_redundant(tmp_path: Path) -> None:
    s = OmpSession("reads000-0000-4000-8000-000000000001")
    s.user("go")
    s.assistant([call("r1", "read", path="a.py")], 20)
    s.tool_result("r1", "read", "A" * 444)
    s.assistant([call("r2", "read", path="a.py")], 20)  # redundant
    s.tool_result("r2", "read", "A" * 444)
    s.assistant([call("w1", "write", path="a.py", content="new")], 20)
    s.tool_result("w1", "write", "ok")
    s.assistant([call("r3", "read", path="a.py")], 20)  # after a write: justified
    s.tool_result("r3", "read", "B" * 444)
    s.assistant([call("r4", "read", path="a.py:1-5")], 20)  # different range: not identical
    s.tool_result("r4", "read", "B" * 50)
    s.assistant([text("done")], 5)
    session, attr = _run(s, tmp_path)
    found = repeated_reads(session, attr)
    assert len(found) == 1
    f = found[0]
    assert "a.py" in f.summary and f.turn == 2
    result_id = f.item_ids[1]
    assert f.cost == pytest.approx(attr.items[f.item_ids[0]].total + attr.items[result_id].total)


def test_huge_outputs_cost_is_the_excess_above_threshold(tmp_path: Path) -> None:
    s = OmpSession("huge0000-0000-4000-8000-000000000001")
    s.user("go")
    s.assistant([call("c1", "bash", command="cat big.log")], 20)
    s.tool_result("c1", "bash", "L" * 22_200)  # ~10,000 tokens
    s.assistant([call("c2", "bash", command="echo hi")], 20)
    s.tool_result("c2", "bash", "hi")
    s.assistant([text("done")], 5)
    session, attr = _run(s, tmp_path)
    found = huge_tool_outputs(session, attr, threshold=5_000)
    assert len(found) == 1
    item_id = found[0].item_ids[0]
    item = session.items[item_id]
    share = (item.tokens - 5_000) / item.tokens
    assert found[0].cost == pytest.approx(attr.items[item_id].lifetime * share, rel=1e-9)
    assert huge_tool_outputs(session, attr, threshold=20_000) == []


def test_failed_calls_and_retry_loops(tmp_path: Path) -> None:
    s = OmpSession("fail0000-0000-4000-8000-000000000001")
    s.user("go")
    for i in range(3):
        s.assistant([call(f"e{i}", "edit", path="x.py")], 30)
        s.tool_result(f"e{i}", "edit", "Error: no match", is_error=True)
    s.assistant([call("e3", "edit", path="x.py")], 30)
    s.tool_result("e3", "edit", "applied")
    s.assistant([call("b1", "bash", command="false")], 30)
    s.tool_result("b1", "bash", "exit 1", is_error=True)
    s.assistant([text("done")], 5)
    session, attr = _run(s, tmp_path)
    found = failed_tool_calls(session, attr)
    loops = [f for f in found if f.detail["loop"]]
    singles = [f for f in found if not f.detail["loop"]]
    assert len(loops) == 1 and loops[0].detail["attempts"] == 3
    assert len(singles) == 1 and "bash false" in singles[0].summary
    assert all(f.cost > 0 for f in found)


def test_cache_bust_causes(tmp_path: Path) -> None:
    s = OmpSession("bust0000-0000-4000-8000-000000000001")
    s.user("go")
    s.assistant([call("c1", "bash", command="ls")], 20)
    s.tool_result("c1", "bash", "z" * 9000)
    s.assistant([call("c2", "bash", command="ls")], 20)
    s.tool_result("c2", "bash", "ok")
    s.assistant([text("pause")], 10, gap=4000, bust=True)  # > 1h TTL
    s.user("next")
    s.entry("model_change", model="anthropic/claude-opus-5-5")
    s.assistant([text("switched")], 10, bust=True)
    s.user("small")
    s.assistant([text("fine")], 10)
    session, attr = _run(s, tmp_path)
    found = cache_busts(session, attr)
    causes = [str(f.detail["cause"]) for f in found]
    assert causes == ["cache TTL expired", "model change"]
    ttl = found[0]
    turn = session.turns[ttl.turn or 0]
    expected = ttl.tokens * (8e-6 - 0.2e-6)
    assert ttl.cost == pytest.approx(expected, rel=1e-6)
    assert turn.usage.cache_read == 0


def test_cache_warm_resets_ttl_clock(tmp_path: Path) -> None:
    s = OmpSession("warm0000-0000-4000-8000-000000000001")
    s.user("go")
    s.assistant([call("c1", "bash", command="ls")], 20)
    s.tool_result("c1", "bash", "z" * 9000)
    s.tick(3000)
    s.entry(
        "model_usage",
        purpose="cache-warm",
        model="claude-opus-5-5",
        usage={},
        cacheRefresh={"hit": True},
    )
    s.assistant([text("later")], 10, gap=2000, bust=True)
    session, attr = _run(s, tmp_path)
    found = cache_busts(session, attr)
    assert [f.detail["cause"] for f in found] == ["unknown"]


def test_compaction_saving_is_positive_when_many_turns_follow(tmp_path: Path) -> None:
    s = OmpSession("comp0000-0000-4000-8000-000000000001")
    s.user("go")
    s.assistant([call("c1", "bash", command="cat")], 20)
    s.tool_result("c1", "bash", "q" * 88_800)  # 40k tokens
    kept = s.user("continue")
    s.compact(kept, kept_tokens=5, summary_tokens=500)
    for i in range(20):
        s.assistant([call(f"k{i}", "bash", command="ls")], 20)
        s.tool_result(f"k{i}", "bash", "ok")
    session, attr = _run(s, tmp_path)
    found = compactions(session, attr)
    assert len(found) == 1
    assert found[0].tokens > 39_000
    assert found[0].cost > 0


def test_side_calls_grouped_by_purpose(tmp_path: Path) -> None:
    s = OmpSession("sidec000-0000-4000-8000-000000000001")
    s.user("go")
    s.side_call("title")
    s.side_call("title")
    s.side_call("cache-warm")
    s.assistant([text("ok")], 5)
    session, attr = _run(s, tmp_path)
    found = side_calls(session, attr)
    assert [f.detail["calls"] for f in found if "title" in f.summary] == [2]
    assert sum(f.cost for f in found) == pytest.approx(attr.side_calls)
