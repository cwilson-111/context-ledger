from __future__ import annotations

import math
from pathlib import Path

import pytest

from context_ledger.attribute import Attribution, _charge, attribute
from context_ledger.calibrate import estimator_error
from context_ledger.model import Kind, Session
from context_ledger.parsers import load
from context_ledger.parsers.omp import parse_omp
from tests.synth import OmpSession, call, text

FIXTURES = Path(__file__).parent / "fixtures"


def _recorded(session: Session) -> float:
    return sum(t.cost.total for t in session.turns) + sum(c.cost.total for c in session.side_calls)


def test_reconciles_exactly(basic: tuple[Session, Attribution]) -> None:
    session, attr = basic
    assert attr.total == pytest.approx(_recorded(session), rel=1e-12)
    assert attr.attributed + attr.unattributed == pytest.approx(attr.total, rel=1e-12)
    assert attr.unattributed == pytest.approx(0.0, abs=1e-12)
    assert attr.calibration.residual_tokens == [0, 0, 0, 0]


def test_demo_fixture_reconciles() -> None:
    loaded = load(FIXTURES / "demo")
    assert len(loaded.sessions) == 1
    root = loaded.sessions[0]
    for s in [root, *root.subagents]:
        attr = attribute(s)
        assert attr.attributed + attr.unattributed == pytest.approx(attr.total, rel=1e-12)
        assert abs(attr.unattributed) < 1e-9
        assert all(r == 0 for r in attr.calibration.residual_tokens)


def test_charge_is_positional() -> None:
    # 100 tokens read at $1, then 50 written at $5, then 10 input at $1.
    regions = [(100, 1.0), (50, 5.0), (10, 1.0)]
    charged, left = _charge([60.0, 60.0, 40.0], regions)
    assert charged[0] == pytest.approx(0.6)
    assert charged[1] == pytest.approx(0.4 + 20 * 0.1)  # 40 read + 20 write tokens
    assert charged[2] == pytest.approx(30 * 0.1 + 1.0)  # 30 write + 10 input
    assert left == pytest.approx(0.0)
    charged, left = _charge([50.0], regions)
    assert left == pytest.approx(0.5 + 5.0 + 1.0)


def test_lifetime_cost_grows_with_turns_carried(basic: tuple[Session, Attribution]) -> None:
    session, attr = basic
    read = next(
        i
        for i in session.items.values()
        if i.kind == Kind.TOOL_RESULT and i.label == "read calc.py"
    )
    cost = attr.items[read.id]
    assert cost.turns == 3  # turns 1, 2 and 3
    assert cost.birth == 0
    # Turn 1: written to cache except its last 2 tokens, the uncached input tail. Turn 2: read
    # from cache except those 2 tokens, which were not cached yet. Turn 3: all read.
    w, r, i = 8e-6, 0.2e-6, 4e-6
    expected = (1298 * w + 2 * i) + (1298 * r + 2 * w) + 1300 * r
    assert cost.lifetime == pytest.approx(expected, rel=1e-9)


def test_output_cost_goes_to_generated_parts(basic: tuple[Session, Attribution]) -> None:
    session, attr = basic
    t0 = session.turns[0]
    births = sum(attr.items[i].birth for i in t0.output_items)
    assert births == pytest.approx(t0.cost.output)


def test_unexplained_prompt_tokens_are_unattributed(tmp_path: Path) -> None:
    s = OmpSession("gap00000-0000-4000-8000-000000000001")
    s.user("q")
    s.assistant([call("c1", "bash", command="ls")], 20)
    s.tool_result("c1", "bash", "ok")
    s.entry(
        "model_change", model="anthropic/claude-opus-5-5"
    )  # an event: delta must look plausible
    s._pending += 5000  # something the log does not show
    s.assistant([text("a")], 10)
    session = parse_omp(s.write(tmp_path / "g.jsonl"))
    attr = attribute(session)
    assert attr.calibration.residual_tokens[1] > 4900
    assert attr.unattributed > 0
    assert attr.attributed + attr.unattributed == pytest.approx(attr.total, rel=1e-12)


def test_overflow_is_spread_not_dropped(tmp_path: Path) -> None:
    s = OmpSession("over0000-0000-4000-8000-000000000001")
    s.user("q")
    s.assistant([call("c1", "bash", command="ls")], 20)
    s.tool_result("c1", "bash", "y" * 2220)
    s.assistant([text("a")], 10)
    s._prompt_known -= 600  # the harness silently trimmed something
    s.user("again")
    s.assistant([text("b")], 10)
    session = parse_omp(s.write(tmp_path / "o.jsonl"))
    attr = attribute(session)
    assert attr.calibration.residual_tokens[-1] < 0
    last = attr.turns[-1]
    charged = sum(c for ic in attr.items.values() for t, c in ic.per_turn if t == last.index)
    assert charged == pytest.approx(last.prompt, rel=1e-9)


def test_estimator_error_math() -> None:
    rows = [(222.0, 100), (444.0, 200)]
    e = estimator_error(rows, 2.22)
    assert e.samples == 2
    assert e.aggregate_pct == pytest.approx(0.0, abs=1e-9)
    e4 = estimator_error(rows, 4.0)
    assert e4.median_signed_pct == pytest.approx(100 * (222 / 4 - 100) / 100)
    assert math.isclose(e4.aggregate_pct, 100 * ((666 / 4) / 300 - 1))
