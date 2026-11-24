"""Assemble per-session and multi-session reports as plain JSON-able data."""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from context_ledger import tokens
from context_ledger.attribute import Attribution, attribute
from context_ledger.calibrate import estimator_error, single_result_samples
from context_ledger.detectors import WASTE_DETECTORS, Finding, run_all
from context_ledger.model import CostSource, Kind, Session
from context_ledger.whatif import scenarios

Json = dict[str, Any]


@dataclass
class Group:
    """A top-level session with all of its subagents, attributed."""

    root: Session
    pairs: list[tuple[Session, Attribution]]
    findings: dict[str, list[Finding]]

    @property
    def total(self) -> float:
        return sum(a.total for _, a in self.pairs)


def _flatten(session: Session) -> list[Session]:
    out = [session]
    for sub in session.subagents:
        out.extend(_flatten(sub))
    return out


def build_group(root: Session) -> Group:
    pairs = [(s, attribute(s)) for s in _flatten(root)]
    findings: dict[str, list[Finding]] = defaultdict(list)
    for s, a in pairs:
        for name, found in run_all(s, a).items():
            findings[name].extend(
                dataclasses.replace(f, detail={**f.detail, "session": s.name}) for f in found
            )
    for found in findings.values():
        found.sort(key=lambda f: f.cost, reverse=True)
    return Group(root, pairs, dict(findings))


def _r(x: float, nd: int = 4) -> float:
    return round(x, nd)


def reconciliation(pairs: list[tuple[Session, Attribution]]) -> Json:
    total = sum(a.total for _, a in pairs)
    unattributed = sum(a.unattributed for _, a in pairs)
    prompt_tokens = sum(t.usage.prompt for s, _ in pairs for t in s.turns)
    abs_residual = sum(abs(t.residual_tokens) for _, a in pairs for t in a.turns)
    overflow = sum(-t.residual_tokens for _, a in pairs for t in a.turns if t.residual_tokens < 0)
    sources: Counter[str] = Counter(t.cost_source.value for s, _ in pairs for t in s.turns)
    return {
        "total_cost": _r(total),
        "attributed_cost": _r(total - unattributed),
        "unattributed_cost": _r(unattributed),
        "unattributed_pct": _r(100 * unattributed / total if total else 0.0, 3),
        "prompt_tokens": prompt_tokens,
        "abs_token_residual_pct": _r(100 * abs_residual / prompt_tokens if prompt_tokens else 0, 3),
        "overflow_token_pct": _r(100 * overflow / prompt_tokens if prompt_tokens else 0, 3),
        "turn_cost_sources": dict(sources),
        "replayed_turns": sum(t.replayed for s, _ in pairs for t in s.turns),
    }


def by_kind(pairs: list[tuple[Session, Attribution]]) -> Json:
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for _, a in pairs:
        for kind, (birth, life) in a.by_kind().items():
            acc[kind][0] += birth
            acc[kind][1] += life
    side = sum(a.side_calls for _, a in pairs)
    unattributed = sum(a.unattributed for _, a in pairs)
    total = sum(a.total for _, a in pairs) or 1.0
    rows: Json = {}
    for kind in [k.value for k in Kind]:
        if kind in acc:
            birth, life = acc[kind]
            rows[kind] = {
                "output_cost": _r(birth),
                "resend_cost": _r(life),
                "total": _r(birth + life),
                "share_pct": _r(100 * (birth + life) / total, 2),
            }
    rows["side_calls"] = {"total": _r(side), "share_pct": _r(100 * side / total, 2)}
    rows["unattributed"] = {
        "total": _r(unattributed),
        "share_pct": _r(100 * unattributed / total, 2),
    }
    return rows


def usage_mix(pairs: list[tuple[Session, Attribution]]) -> Json:
    """Share of recorded spend by billing component."""
    parts: dict[str, float] = defaultdict(float)
    for s, _ in pairs:
        for t in s.turns:
            parts["cache_read"] += t.cost.cache_read
            parts["cache_write"] += t.cost.cache_write
            parts["input"] += t.cost.input
            parts["output"] += t.cost.output
        for c in s.side_calls:
            parts["side_calls"] += c.cost.total
    total = sum(parts.values()) or 1.0
    return {k: _r(100 * v / total, 2) for k, v in parts.items()}


def _finding_json(f: Finding) -> Json:
    return {
        "cost": _r(f.cost),
        "tokens": f.tokens,
        "turn": f.turn,
        "summary": f.summary,
        "detail": f.detail,
    }


def detector_totals(findings: dict[str, list[Finding]], total: float, top: int) -> Json:
    out: Json = {}
    for name, found in findings.items():
        cost = sum(f.cost for f in found)
        entry: Json = {
            "count": len(found),
            "cost": _r(cost),
            "share_pct": _r(100 * cost / total if total else 0.0, 2),
            "waste": name in WASTE_DETECTORS,
            "top": [_finding_json(f) for f in found[:top]],
        }
        if name == "cache_busts":
            causes: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
            for f in found:
                c = causes[str(f.detail.get("cause", "unknown"))]
                c[0] += 1
                c[1] += f.cost
            entry["by_cause"] = {
                k: {"count": int(v[0]), "cost": _r(v[1])}
                for k, v in sorted(causes.items(), key=lambda kv: -kv[1][1])
            }
        if name == "failed_tool_calls":
            entry["retry_loops"] = sum(1 for f in found if f.detail.get("loop"))
            entry["failed_calls"] = sum(int(f.detail.get("attempts", 1)) for f in found)
        out[name] = entry
    return out


def top_items(pairs: list[tuple[Session, Attribution]], n: int) -> list[Json]:
    rows: list[Json] = []
    for s, a in pairs:
        for item_id, cost in a.items.items():
            if cost.total <= 0:
                continue
            item = s.items[item_id]
            rows.append(
                {
                    "session": s.name,
                    "id": item_id,
                    "kind": item.kind.value,
                    "label": item.label,
                    "tokens": item.tokens,
                    "token_source": item.source.value,
                    "born_turn": item.born_turn,
                    "first_turn": item.first_turn,
                    "turns_in_context": cost.turns,
                    "output_cost": _r(cost.birth),
                    "resend_cost": _r(cost.lifetime),
                    "total": _r(cost.total),
                }
            )
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows[:n]


def by_tool(pairs: list[tuple[Session, Attribution]], n: int) -> list[Json]:
    """Tool calls and their results grouped by tool name: what each tool's use cost."""
    acc: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
    total = sum(a.total for _, a in pairs) or 1.0
    for s, a in pairs:
        for item_id, cost in a.items.items():
            item = s.items[item_id]
            if item.kind not in (Kind.TOOL_CALL, Kind.TOOL_RESULT) or not item.tool_name:
                continue
            row = acc[item.tool_name]
            if item.kind == Kind.TOOL_RESULT:
                row[0] += 1
                row[1] += item.tokens
                row[2] += int(item.is_error)
                row[3] += cost.total
            else:
                row[4] += cost.total
    rows: list[Json] = [
        {
            "tool": tool,
            "results": int(r[0]),
            "result_tokens": int(r[1]),
            "errors": int(r[2]),
            "result_cost": _r(r[3]),
            "call_cost": _r(r[4]),
            "cost": _r(r[3] + r[4]),
            "share_pct": _r(100 * (r[3] + r[4]) / total, 2),
        }
        for tool, r in acc.items()
    ]
    rows.sort(key=lambda r: float(r["cost"]), reverse=True)
    return rows[:n]


def side_calls_by_purpose(pairs: list[tuple[Session, Attribution]]) -> Json:
    acc: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for s, _ in pairs:
        for c in s.side_calls:
            acc[c.purpose][0] += 1
            acc[c.purpose][1] += c.cost.total
    return {
        k: {"calls": int(v[0]), "cost": _r(v[1])}
        for k, v in sorted(acc.items(), key=lambda kv: -kv[1][1])
    }


def subagents(group: Group) -> list[Json]:
    total = group.total or 1.0
    rows = []
    for s, a in group.pairs:
        spawner = None
        parent = next((p for p, _ in group.pairs if s in p.subagents), None)
        if parent is not None and s.agent in parent.spawned_by:
            call = parent.items.get(parent.spawned_by[s.agent])
            spawner = call.born_turn if call else None
        rows.append(
            {
                "session": s.name,
                "agent": s.agent is not None,
                "turns": len(s.turns),
                "cost": _r(a.total),
                "share_pct": _r(100 * a.total / total, 2),
                "spawned_at_turn": spawner,
            }
        )
    return rows


def curve(group: Group) -> list[Json]:
    """Cumulative cost over the main session's turns; subagent and side-call spend is added
    at the first main turn that starts after it."""
    main, main_attr = group.pairs[0]
    stamps = [t.timestamp for t in main.turns]
    extra = [0.0] * (len(main.turns) + 1)

    def slot(ts: float) -> int:
        lo, hi = 0, len(stamps)
        while lo < hi:
            mid = (lo + hi) // 2
            if stamps[mid] < ts:
                lo = mid + 1
            else:
                hi = mid
        return lo

    for s, _ in group.pairs[1:]:
        for t in s.turns:
            extra[slot(t.timestamp)] += t.cost.total
    for s, _ in group.pairs:
        for c in s.side_calls:
            extra[slot(c.timestamp)] += c.cost.total
    points = []
    main_cum = 0.0
    all_cum = 0.0
    for t, tc in zip(main.turns, main_attr.turns, strict=True):
        main_cum += tc.total
        all_cum += tc.total + extra[t.index]
        points.append({"turn": t.index, "main": _r(main_cum), "all": _r(all_cum)})
    if points:
        points[-1]["all"] = _r(all_cum + extra[len(main.turns)])
    return points


def turns_table(session: Session, attr: Attribution) -> list[Json]:
    rows = []
    for t, tc in zip(session.turns, attr.turns, strict=True):
        rows.append(
            {
                "turn": t.index,
                "time": t.timestamp,
                "model": t.model,
                "prompt_tokens": t.usage.prompt,
                "cache_read": t.usage.cache_read,
                "cache_write": t.usage.cache_write,
                "input": t.usage.input,
                "output": t.usage.output,
                "cost": _r(tc.total, 5),
                "events": [e.kind for e in t.events],
                "residual_tokens": tc.residual_tokens,
                "replayed": t.replayed,
            }
        )
    return rows


def session_report(group: Group, top: int = 15) -> Json:
    root = group.root
    total = group.total
    models = Counter(t.model for s, _ in group.pairs for t in s.turns)
    return {
        "session": {
            "id": root.id,
            "harness": root.harness,
            "path": root.path,
            "turns": sum(len(s.turns) for s, _ in group.pairs),
            "subagents": len(group.pairs) - 1,
            "models": dict(models),
            "recorded_total": root.recorded_total,
            "skipped_lines": sum(s.skipped_lines for s, _ in group.pairs),
        },
        "reconciliation": reconciliation(group.pairs),
        "usage_mix_pct": usage_mix(group.pairs),
        "by_kind": by_kind(group.pairs),
        "sessions": subagents(group),
        "top_items": top_items(group.pairs, top),
        "detectors": detector_totals(group.findings, total, top),
        "whatif": [
            {"name": sc.name, "description": sc.description, "saved": _r(sc.saved)}
            for sc in scenarios(group.pairs)
        ],
        "curve": curve(group),
        "busts": [
            _finding_json(f)
            for f in group.findings.get("cache_busts", [])
            if f.detail.get("session") == root.name
        ],
        "turns": turns_table(*group.pairs[0]),
    }


def summary_report(groups: list[Group], skipped: list[tuple[str, str]], top: int = 10) -> Json:
    by_harness: dict[str, list[Group]] = defaultdict(list)
    for g in groups:
        by_harness[g.root.harness].append(g)
    harnesses: Json = {}
    for harness, gs in sorted(by_harness.items()):
        pairs = [p for g in gs for p in g.pairs]
        recorded = [g.root.recorded_total for g in gs if g.root.recorded_total is not None]
        computed = sum(g.total for g in gs if g.root.recorded_total is not None)
        harnesses[harness] = {
            "sessions": len(gs),
            "session_files": len(pairs),
            "turns": sum(len(s.turns) for s, _ in pairs),
            "reconciliation": reconciliation(pairs),
            "harness_reported_total": _r(sum(recorded)) if recorded else None,
            "computed_total_for_those": _r(computed) if recorded else None,
            "unpriced_turns": sum(
                t.cost_source == CostSource.UNPRICED for s, _ in pairs for t in s.turns
            ),
        }
    pairs = [p for g in groups for p in g.pairs]
    total = sum(a.total for _, a in pairs)
    findings: dict[str, list[Finding]] = defaultdict(list)
    for g in groups:
        for name, found in g.findings.items():
            findings[name].extend(found)
    for found in findings.values():
        found.sort(key=lambda f: f.cost, reverse=True)
    rows = [r for _, a in pairs for r in single_result_samples(a.calibration.samples)]
    errors = {
        f"chars/{tokens.CHARS_PER_TOKEN:g}": dataclasses.asdict(
            estimator_error(rows, tokens.CHARS_PER_TOKEN)
        ),
        "chars/4": dataclasses.asdict(estimator_error(rows, 4.0)),
    }
    token_sources: dict[str, float] = defaultdict(float)
    for s, a in pairs:
        for item_id, cost in a.items.items():
            if cost.lifetime > 0:
                token_sources[s.items[item_id].source.value] += cost.lifetime
    resend = sum(token_sources.values()) or 1.0
    subagent_cost = sum(a.total for g in groups for _, a in g.pairs[1:])
    return {
        "total_cost": _r(total),
        "harnesses": harnesses,
        "usage_mix_pct": usage_mix(pairs),
        "by_kind": by_kind(pairs),
        "subagent_share_pct": _r(100 * subagent_cost / total if total else 0.0, 2),
        "by_tool": by_tool(pairs, max(top, 10)),
        "side_calls_by_purpose": side_calls_by_purpose(pairs),
        "detectors": detector_totals(dict(findings), total, top),
        "whatif": [
            {
                "name": sc.name,
                "description": sc.description,
                "saved": _r(sc.saved),
                "saved_pct": _r(100 * sc.saved / total if total else 0.0, 2),
            }
            for sc in scenarios(pairs)
        ],
        "estimator_error_pct": errors,
        "resend_cost_by_token_source_pct": {
            k: _r(100 * v / resend, 2)
            for k, v in sorted(token_sources.items(), key=lambda kv: -kv[1])
        },
        "skipped_files": [{"path": p, "reason": r} for p, r in skipped],
        "skipped_lines": sum(s.skipped_lines for s, _ in pairs),
    }
