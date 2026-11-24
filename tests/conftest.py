from __future__ import annotations

from pathlib import Path

import pytest

from context_ledger.attribute import Attribution, attribute
from context_ledger.model import Session
from context_ledger.parsers.omp import parse_omp
from tests.synth import OmpSession, call, text, thinking

FIXTURES = Path(__file__).parent / "fixtures"


def build_basic(tmp_path: Path) -> Path:
    """Four turns: read a file, run a command, fail an edit, answer."""
    s = OmpSession("basic000-0000-4000-8000-000000000001")
    s.user("Fix the failing test in calc.py")
    s.assistant([thinking("look at the file"), call("c1", "read", path="calc.py")], 120)
    s.tool_result("c1", "read", "x" * 2220, tokens=1300)  # estimate ~1000, true 1300
    s.assistant([call("c2", "bash", command="pytest -q")], 50)
    s.tool_result("c2", "bash", "1 failed, 3 passed")
    s.assistant([call("c3", "edit", path="calc.py", old="a", new="b")], 60)
    s.tool_result("c3", "edit", "Error: anchor not found", is_error=True)
    s.assistant([text("The test fails because of rounding.")], 30)
    return s.write(tmp_path / "basic.jsonl")


@pytest.fixture
def basic(tmp_path: Path) -> tuple[Session, Attribution]:
    session = parse_omp(build_basic(tmp_path))
    return session, attribute(session)
