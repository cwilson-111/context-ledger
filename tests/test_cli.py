from __future__ import annotations

import json
from pathlib import Path

import pytest

from context_ledger.cli import main
from context_ledger.render import clean
from tests.synth import OmpSession, call, text

FIXTURES = Path(__file__).parent / "fixtures"
DEMO = FIXTURES / "demo" / "demo-session.jsonl"


def test_report_text(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", str(DEMO), "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "Spend by item kind" in out
    assert "huge_tool_outputs" in out
    assert "cache TTL expired" in out


def test_report_json_reconciles(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", str(DEMO), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    rec = rep["reconciliation"]
    assert rec["unattributed_pct"] == 0
    kinds = rep["by_kind"]
    assert sum(k["total"] for k in kinds.values()) == pytest.approx(rec["total_cost"], abs=1e-3)
    assert rep["sessions"][1]["spawned_at_turn"] is not None  # the demo's subagent


def test_report_html_is_self_contained(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    assert main(["report", str(DEMO), "--html", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert "<svg" in page and "<script" not in page
    assert "http://" not in page and "https://" not in page
    assert "id='turn-0'" in page


def test_summary_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["summary", str(FIXTURES / "demo"), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["harnesses"]["omp"]["sessions"] == 1
    assert {"cache_busts", "repeated_reads", "huge_tool_outputs", "failed_tool_calls"} <= set(
        rep["detectors"]
    )
    assert rep["whatif"][0]["saved_pct"] > 0


def test_missing_path_and_empty_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["report", str(tmp_path / "nope")])
    assert main(["summary", str(tmp_path)]) == 2


def test_log_text_is_escaped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    hostile = "<img src=x onerror=alert(1)>\x1b[31mred"
    s = OmpSession("evil0000-0000-4000-8000-000000000001")
    s.user(hostile)
    s.assistant([call("c1", "bash", command=hostile)], 20)
    s.tool_result("c1", "bash", "z" * 30_000)
    s.assistant([text(hostile)], 10)
    path = s.write(tmp_path / "evil.jsonl")
    out = tmp_path / "evil.html"
    assert main(["report", str(path), "--html", str(out), "--text"]) == 0
    page = out.read_text(encoding="utf-8")
    assert "<img" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "\x1b" not in capsys.readouterr().out
    assert clean("a\x1b[2Jb\u202e") == "a[2Jb"
