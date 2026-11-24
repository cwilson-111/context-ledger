from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import pytest

from context_ledger import tokens
from context_ledger.attribute import attribute
from context_ledger.model import Cost, Item, Kind, Session, Usage, split_output
from context_ledger.parsers.common import edited_paths, normalize_read_path, price_usage
from context_ledger.pricing import PriceTable, Rates
from context_ledger.whatif import cap_tool_outputs, expire_tool_results, strip_thinking


def test_estimate_text_uses_fitted_ratio() -> None:
    assert tokens.estimate_text("") == 0
    assert tokens.estimate_text("x" * 222) == 100


def test_image_dimensions_from_headers() -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 800, 600)
    assert tokens.image_dimensions(png) == (800, 600)
    gif = b"GIF89a" + struct.pack("<HH", 32, 16)
    assert tokens.image_dimensions(gif) == (32, 16)
    jpeg = b"\xff\xd8" + b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 300, 400) + b"\x00" * 12
    assert tokens.image_dimensions(jpeg) == (400, 300)
    assert tokens.image_dimensions(b"not an image") is None
    b64 = base64.b64encode(png).decode()
    assert tokens.estimate_image(b64) == 640  # 800*600/750
    assert tokens.estimate_image(None) == 1600


def test_split_output_prefers_recorded_thinking() -> None:
    parts = [
        Item("a:0", Kind.THINKING, "thinking", "a", est_tokens=5),
        Item("a:1", Kind.TOOL_CALL, "read x", "a", est_tokens=20),
    ]
    split_output(parts, 100)
    assert [p.tokens for p in parts] == [80, 20]
    split_output(parts, 100, thinking_tokens=70)
    assert [p.tokens for p in parts] == [70, 30]


def test_rates_and_lookup() -> None:
    rates = Rates(4.0, 20.0, 0.2, 5.0, 8.0)
    cost = rates.price(
        Usage(input=10, output=100, cache_read=1000, cache_write=300, cache_write_1h=200)
    )
    assert cost.total == pytest.approx((40 + 2000 + 200 + 100 * 5 + 200 * 8) / 1e6)
    table = PriceTable()
    assert table.lookup("anthropic/claude-opus-5-5") is not None
    assert table.lookup("claude-opus-5-5-20261101") is not None
    assert table.lookup("unknown-model") is None


def test_price_table_file(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps({"my-model": {"input": 1, "output": 2, "cache_read": 0.1, "cache_write_1h": 2}})
    )
    table = PriceTable.from_file(path)
    rates = table.lookup("my-model")
    assert rates is not None and rates.cache_write_5m == 1.25
    path.write_text(json.dumps({"bad": {"input": 1}}))
    with pytest.raises(ValueError, match="bad rates"):
        PriceTable.from_file(path)


def test_price_usage_prefers_recorded_even_zero() -> None:
    usage = Usage(input=10, output=10)
    cost, source = price_usage("claude-opus-5-5", usage, Cost(), PriceTable())
    assert cost.total == 0 and source.value == "recorded"
    cost, source = price_usage("claude-opus-5-5", usage, None, PriceTable())
    assert cost.total > 0 and source.value == "table"
    _, source = price_usage("mystery", usage, None, PriceTable())
    assert source.value == "unpriced"


def test_path_helpers() -> None:
    assert normalize_read_path("src/a.py:10-20") == "src/a.py"
    assert normalize_read_path("src/a.py:raw") == "src/a.py"
    assert normalize_read_path("src/a.py") == "src/a.py"
    assert edited_paths("edit", {"input": "[src/a.py#1A2B]\nPUT 1.=2:\n+x\n[b.py#FFFF]"}) == [
        "src/a.py",
        "b.py",
    ]
    assert edited_paths("Write", {"file_path": "/w/x.py"}) == ["/w/x.py"]
    assert edited_paths("read", {"path": "x"}) == []


def test_whatif_bounds(basic: tuple[Session, object]) -> None:
    session, _ = basic
    attr = attribute(session)
    pairs = [(session, attr)]
    tool_resend = sum(
        attr.items[i].lifetime for i, it in session.items.items() if it.kind == Kind.TOOL_RESULT
    )
    assert 0 < cap_tool_outputs(pairs, 100) < tool_resend
    assert cap_tool_outputs(pairs, 10_000) == 0
    assert expire_tool_results(pairs, 0) < tool_resend
    assert expire_tool_results(pairs, 10) == 0
    assert strip_thinking(pairs) == pytest.approx(
        sum(attr.items[i].lifetime for i, it in session.items.items() if it.kind == Kind.THINKING)
    )
