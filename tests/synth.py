"""Builders for synthetic omp and Claude Code logs. Everything here is invented.

The builders keep a "true" token count for every piece of context and emit usage numbers
that a real provider would have recorded for it, so tests can check that calibration
recovers the true sizes and that attribution reconciles.

``python -m tests.synth`` regenerates the demo session in ``tests/fixtures/demo``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from context_ledger.tokens import estimate_text

# claude-opus-5-5 rates (USD per token), matching the package price table.
RATES = {"input": 4e-6, "output": 20e-6, "cacheRead": 0.2e-6, "cacheWrite": 8e-6}
T0 = 1_790_000_000.0  # 2026-09-21, arbitrary


def true_tokens(text: str) -> int:
    """The builders' stand-in for a real tokenizer: the estimator itself, so estimates are
    exact unless a test passes a different true size explicitly."""
    return estimate_text(text)


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class OmpSession:
    sid: str
    cwd: str = "/work/demo"
    model: str = "claude-opus-5-5"
    system_tokens: int = 9_000  # what the harness reports as nonMessageTokens
    hidden_system: int = 1_200  # provider-side additions the harness does not count
    parent: str | None = None
    rates: dict[str, float] = field(default_factory=lambda: dict(RATES))
    lines: list[dict[str, Any]] = field(default_factory=list)
    now: float = T0
    _n: int = 0
    _last_id: str | None = None
    _prompt_known: int = 0  # true prompt size up to the last turn
    _pending: int = 0  # true tokens added since the last turn
    _cached: int = 0  # prefix the cache holds after the last turn
    _turns: int = 0

    def __post_init__(self) -> None:
        self.lines.append({"type": "title", "v": 1, "title": "demo", "updatedAt": iso(self.now)})
        header: dict[str, Any] = {
            "type": "session",
            "version": 3,
            "id": self.sid,
            "timestamp": iso(self.now),
            "cwd": self.cwd,
        }
        if self.parent:
            header["parentSession"] = self.parent
        self.lines.append(header)
        self.entry("model_change", model=f"anthropic/{self.model}")
        self.entry("thinking_level_change", thinkingLevel="high")
        self._prompt_known = self.system_tokens + self.hidden_system

    def _id(self) -> str:
        self._n += 1
        return f"{self.sid[:4]}{self._n:04x}"

    def entry(self, etype: str, **fields: Any) -> str:
        eid = self._id()
        line = {"type": etype, "id": eid, "parentId": self._last_id, "timestamp": iso(self.now)}
        line.update(fields)
        self.lines.append(line)
        if etype not in ("model_usage",):
            self._last_id = eid
        return eid

    def tick(self, seconds: float = 3.0) -> None:
        self.now += seconds

    def user(self, text: str) -> str:
        self.tick()
        self._pending += true_tokens(text)
        return self.entry(
            "message",
            message={
                "role": "user",
                "content": [{"type": "text", "text": text}],
                "timestamp": int(self.now * 1000),
            },
        )

    def tool_result(
        self,
        call_id: str,
        tool: str,
        text: str,
        is_error: bool = False,
        tokens: int | None = None,
        **extra: Any,
    ) -> str:
        self.tick()
        self._pending += true_tokens(text) if tokens is None else tokens
        msg = {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": tool,
            "content": [{"type": "text", "text": text}],
            "details": extra.pop("details", {}),
            "isError": is_error,
            "timestamp": int(self.now * 1000),
        }
        msg.update(extra)
        return self.entry("message", message=msg)

    def custom_message(self, text: str, kind: str = "reminder") -> str:
        self._pending += true_tokens(text)
        return self.entry("custom_message", customType=kind, content=text, display=False)

    def assistant(
        self,
        parts: list[dict[str, Any]],
        output: int,
        gap: float = 3.0,
        bust: bool = False,
        cost_recorded: bool = True,
        **extra: Any,
    ) -> str:
        """One billed request. ``parts`` use omp content shapes; ``output`` is the true output
        size, which the next prompt carries for this message."""
        self.tick(gap)
        prompt = self._prompt_known + self._pending
        inp = 2
        if self._turns == 0 or bust:
            read, write = 0, prompt - inp
        else:
            read = min(self._cached, prompt - inp)
            write = prompt - inp - read
        usage = {
            "input": inp,
            "output": output,
            "cacheRead": read,
            "cacheWrite": write,
            "totalTokens": prompt + output,
            "cost": self._cost(inp, output, read, write) if cost_recorded else _zero_cost(),
            "cttl": {"ephemeral1h": write},
        }
        start = self.now
        self.tick(2.0)
        msg = {
            "role": "assistant",
            "content": parts,
            "api": "anthropic-messages",
            "provider": "anthropic",
            "model": self.model,
            "usage": usage,
            "stopReason": "toolUse" if any(p["type"] == "toolCall" for p in parts) else "stop",
            "timestamp": int(start * 1000),
            "completedAt": int(self.now * 1000),
            "contextSnapshot": {
                "promptTokens": prompt,
                "nonMessageTokens": self.system_tokens,
                "compactionEpoch": 0,
            },
        }
        msg.update(extra)
        eid = self.entry("message", message=msg)
        self._turns += 1
        self._prompt_known = prompt
        self._cached = prompt - inp
        self._pending = output
        return eid

    def _cost(self, inp: int, out: int, read: int, write: int) -> dict[str, float]:
        c = {
            "input": inp * self.rates["input"],
            "output": out * self.rates["output"],
            "cacheRead": read * self.rates["cacheRead"],
            "cacheWrite": write * self.rates["cacheWrite"],
        }
        c["total"] = sum(c.values())
        return c

    def side_call(self, purpose: str, output: int = 20, prompt: int = 400) -> str:
        return self.entry(
            "model_usage",
            purpose=purpose,
            api="anthropic-messages",
            provider="anthropic",
            model=self.model,
            usage={
                "input": prompt,
                "output": output,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": prompt + output,
                "cost": self._cost(prompt, output, 0, 0),
            },
            stopReason="stop",
        )

    def compact(self, first_kept: str, kept_tokens: int, summary_tokens: int) -> str:
        """Replace history before ``first_kept`` with a summary; ``kept_tokens`` is the true
        size of what stays (messages from ``first_kept`` on, excluding the system block)."""
        self.tick()
        before = self._prompt_known + self._pending
        eid = self.entry(
            "compaction",
            summary="## Goal\nsynthetic summary",
            shortSummary="Remote compaction",
            firstKeptEntryId=first_kept,
            tokensBefore=before,
            tokensAfter=summary_tokens + kept_tokens,
            method="remote",
        )
        self._prompt_known = self.system_tokens + self.hidden_system + summary_tokens + kept_tokens
        self._pending = 0
        self._cached = 0
        return eid

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(line) + "\n" for line in self.lines), encoding="utf-8")
        return path


def _zero_cost() -> dict[str, float]:
    return {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}


def call(call_id: str, name: str, **args: Any) -> dict[str, Any]:
    return {"type": "toolCall", "id": call_id, "name": name, "arguments": args}


def thinking(text: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": text, "thinkingSignature": "sig"}


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


# -- Claude Code -----------------------------------------------------------------------------


@dataclass
class ClaudeCodeSession:
    sid: str = "00000000-0000-4000-8000-000000000001"
    model: str = "claude-opus-5-5"
    system_tokens: int = 12_000
    sidechain: bool = False
    agent_id: str | None = None
    lines: list[dict[str, Any]] = field(default_factory=list)
    now: float = T0
    _n: int = 0
    _last: str | None = None
    _prompt_known: int = 0
    _pending: int = 0
    _cached: int = 0
    _turns: int = 0
    total_cost: float = 0.0

    def __post_init__(self) -> None:
        self._prompt_known = self.system_tokens

    def _uuid(self) -> str:
        self._n += 1
        return f"00000000-0000-4000-9000-{self._n:012d}"

    def _base(self, etype: str, parent: str | None) -> dict[str, Any]:
        self.now += 2
        line: dict[str, Any] = {
            "parentUuid": parent,
            "isSidechain": self.sidechain,
            "type": etype,
            "uuid": self._uuid(),
            "timestamp": iso(self.now),
            "sessionId": self.sid,
            "cwd": "/work/demo",
        }
        if self.agent_id:
            line["agentId"] = self.agent_id
        return line

    def user(
        self, content: str | list[dict[str, Any]], parent: str | None = None, **kw: Any
    ) -> str:
        line = self._base("user", parent if parent is not None else self._last)
        line["message"] = {"role": "user", "content": content}
        line.update(kw)
        self.lines.append(line)
        if isinstance(content, str):
            self._pending += true_tokens(content)
        else:
            for part in content:
                if part.get("type") == "tool_result":
                    self._pending += true_tokens(str(part.get("content", "")))
                elif part.get("type") == "text":
                    self._pending += true_tokens(part["text"])
        self._last = line["uuid"]
        return str(line["uuid"])

    def tool_result(
        self, tool_use_id: str, content: str, parent: str, is_error: bool = False
    ) -> str:
        part: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            part["is_error"] = True
        return self.user([part], parent=parent)

    def assistant(
        self,
        blocks: list[dict[str, Any]],
        output: int,
        thinking_tokens: int = 0,
        bust: bool = False,
    ) -> list[str]:
        """One response streamed as one line per block; returns the line uuids."""
        prompt = self._prompt_known + self._pending
        inp = 2
        if self._turns == 0 or bust:
            read, write = 0, prompt - inp
        else:
            read = min(self._cached, prompt - inp)
            write = prompt - inp - read
        usage = {
            "input_tokens": inp,
            "cache_creation_input_tokens": write,
            "cache_read_input_tokens": read,
            "output_tokens": output,
            "output_tokens_details": {"thinking_tokens": thinking_tokens},
            "cache_creation": {"ephemeral_1h_input_tokens": write, "ephemeral_5m_input_tokens": 0},
        }
        self.total_cost += (
            inp * RATES["input"]
            + output * RATES["output"]
            + read * RATES["cacheRead"]
            + write * RATES["cacheWrite"]
        )
        msg_id = f"msg_{self._turns:04d}"
        uuids = []
        for block in blocks:
            line = self._base("assistant", self._last)
            line["message"] = {
                "model": self.model,
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [block],
                "stop_reason": "tool_use",
                "usage": usage,
            }
            self.lines.append(line)
            self._last = line["uuid"]
            uuids.append(str(line["uuid"]))
        self._turns += 1
        self._prompt_known = prompt
        self._cached = prompt - inp
        self._pending = output
        return uuids

    def cost_state(self, extra: float = 0.0) -> None:
        self.lines.append(
            {"type": "cost-state", "sessionId": self.sid, "totalCostUSD": self.total_cost + extra}
        )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(line) + "\n" for line in self.lines), encoding="utf-8")
        return path


def tool_use(use_id: str, name: str, **args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": use_id, "name": name, "input": args}


# -- demo session for the README ------------------------------------------------------------

_SOURCE = "\n".join(
    f"{i:>4}  def handler_{i}(event):  return route(event, {i})" for i in range(1, 900)
)
_LOG = "\n".join(
    f"2026-09-21T10:{i // 60 % 60:02d}:{i % 60:02d} INFO request id={i:06d} ok" for i in range(1500)
)
DEMO_ID = "01demo00-0000-4000-8000-000000000001"


def demo_session(root: Path) -> Path:
    """A plausible 40-turn session with every pattern the detectors look for."""
    s = OmpSession(DEMO_ID, cwd="/work/shop-api")
    s.user("The checkout endpoint returns 500 after the last deploy. Find and fix it.")
    s.assistant(
        [thinking("Start by reading the handler."), call("c1", "read", path="src/checkout.py")],
        180,
    )
    s.tool_result("c1", "read", _SOURCE[:9000])
    s.assistant(
        [thinking("Need the server log."), call("c2", "bash", command="cat logs/server.log")], 120
    )
    s.tool_result("c2", "bash", _LOG)  # a huge tool output that rides along for the session
    s.assistant([call("c3", "grep", pattern="Traceback", path="logs/")], 90)
    s.tool_result("c3", "grep", "logs/server.log:812: Traceback (most recent call last)")
    s.assistant([call("c4", "read", path="src/checkout.py")], 60)  # identical re-read
    s.tool_result("c4", "read", _SOURCE[:9000])
    for attempt in range(3):  # a retry loop
        s.assistant(
            [call(f"e{attempt}", "edit", path="src/checkout.py", old="tax=", new="tax =")], 140
        )
        s.tool_result(
            f"e{attempt}", "edit", "Error: anchor not found in src/checkout.py", is_error=True
        )
    s.assistant(
        [
            thinking("Read the exact lines first."),
            call("c5", "read", path="src/checkout.py:300-340"),
        ],
        200,
    )
    s.tool_result("c5", "read", _SOURCE[3000:4800])
    s.assistant(
        [call("c6", "edit", path="src/checkout.py", old="tax=rate*", new="tax = rate *")], 160
    )
    s.tool_result("c6", "edit", "Applied 1 edit to src/checkout.py")
    s.side_call("title")
    s.assistant([call("k1", "task", agent="reviewer", assignment="Review the tax fix")], 150)
    sub_start = s.now
    s.tool_result(
        "k1",
        "task",
        "Reviewer: the fix is correct; add a test for rounding.",
        details={"results": [{"id": "Reviewer", "status": "done"}]},
    )
    for i in range(8):
        s.assistant(
            [call(f"t{i}", "bash", command=f"pytest tests/test_checkout.py -k case_{i}")], 70
        )
        s.tool_result(f"t{i}", "bash", f"1 passed in 0.{i}1s")
    # The user steps away for two hours: the 1 hour cache expires.
    s.user("Back. Also add a regression test.")
    s.assistant(
        [thinking("Cache is cold now."), call("w1", "write", path="tests/test_tax.py")],
        400,
        gap=7200,
        bust=True,
    )
    s.tool_result("w1", "write", "Wrote tests/test_tax.py")
    for i in range(12):
        s.assistant([call(f"r{i}", "bash", command="pytest -q")], 60)
        s.tool_result(f"r{i}", "bash", "42 passed in 3.1s")
    s.assistant(
        [text("Fixed: the tax calculation now rounds before summing; added a regression test.")], 90
    )
    path = s.write(root / "demo" / "demo-session.jsonl")

    sub = OmpSession("01demo00-0000-4000-8000-000000000002", cwd="/work/shop-api", parent=DEMO_ID)
    sub.now = sub_start
    sub.system_tokens, sub.hidden_system = 6_000, 1_100
    sub._prompt_known = sub.system_tokens + sub.hidden_system
    sub.user("Review the tax fix in src/checkout.py")
    sub.assistant([call("s1", "read", path="src/checkout.py")], 80)
    sub.tool_result("s1", "read", _SOURCE[:9000])
    sub.assistant([call("s2", "bash", command="git diff")], 60)
    sub.tool_result("s2", "bash", "-    tax=rate*amount\n+    tax = round(rate * amount, 2)")
    sub.assistant([text("The fix is correct; add a test for rounding.")], 40)
    sub.write(root / "demo" / "demo-session" / "Reviewer.jsonl")
    return path


if __name__ == "__main__":
    print(demo_session(Path(__file__).parent / "fixtures"))
