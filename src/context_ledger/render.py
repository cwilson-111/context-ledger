"""Terminal and HTML renderers.

Every string that came from a log (labels, paths, commands, model names) is untrusted: the
terminal renderer strips control characters (no escape-sequence injection) and the HTML
renderer escapes everything through ``html.escape`` and emits no script at all.
"""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from typing import Any

Json = dict[str, Any]

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")


def clean(text: object) -> str:
    return _CONTROL.sub("", str(text))


def _money(x: float) -> str:
    return f"${x:,.2f}" if abs(x) >= 0.01 or x == 0 else f"${x:,.4f}"


# -- terminal -------------------------------------------------------------------------------


def _kind_table(by_kind: Json) -> list[str]:
    lines = [f"  {'kind':<20}{'output':>11}{'resend':>11}{'total':>11}{'share':>8}"]
    for kind, row in by_kind.items():
        lines.append(
            f"  {kind:<20}{_money(row.get('output_cost', 0.0)) if 'output_cost' in row else '':>11}"
            f"{_money(row['resend_cost']) if 'resend_cost' in row else '':>11}"
            f"{_money(row['total']):>11}{row['share_pct']:>7.1f}%"
        )
    return lines


def _detector_lines(detectors: Json, show: int) -> list[str]:
    lines = []
    for name, d in sorted(detectors.items(), key=lambda kv: -kv[1]["cost"]):
        tag = "" if d["waste"] else "  (informational)"
        lines.append(
            f"  {name:<20}{d['count']:>6} found {_money(d['cost']):>11} "
            f"{d['share_pct']:>6.1f}%{tag}"
        )
        if "by_cause" in d:
            for cause, c in d["by_cause"].items():
                lines.append(f"      {clean(cause)}: {c['count']} busts, {_money(c['cost'])}")
        if "retry_loops" in d:
            lines.append(f"      {d['failed_calls']} failed calls, {d['retry_loops']} retry loops")
        for f in d["top"][:show]:
            where = f"turn {f['turn']}" if f["turn"] is not None else ""
            lines.append(f"      {_money(f['cost']):>9}  {where:<10} {clean(f['summary'])[:90]}")
    return lines


def text_report(rep: Json, top: int) -> str:
    s, rec = rep["session"], rep["reconciliation"]
    out = [
        f"context-ledger: {s['harness']} session {clean(s['id'])}",
        f"  {s['turns']} turns, {s['subagents']} subagents, models: "
        + ", ".join(f"{clean(m)} ({n})" for m, n in s["models"].items()),
        "",
        f"Total {_money(rec['total_cost'])}; attributed to items {_money(rec['attributed_cost'])}, "
        f"unattributed {_money(rec['unattributed_cost'])} ({rec['unattributed_pct']:.2f}%), "
        f"token residual {rec['abs_token_residual_pct']:.2f}% of prompt tokens",
    ]
    if s["recorded_total"] is not None:
        out.append(f"  harness-reported session total: {_money(s['recorded_total'])}")
    mix = rep["usage_mix_pct"]
    out += [
        "Spend by billing component: "
        + ", ".join(f"{k.replace('_', ' ')} {v:.1f}%" for k, v in mix.items()),
        "",
        "Spend by item kind (output = generating it, resend = carrying it in later prompts):",
        *_kind_table(rep["by_kind"]),
        "",
        f"Top {min(top, len(rep['top_items']))} items by lifetime cost:",
        f"  {'total':>9} {'tokens':>8} {'turns':>6}  {'kind':<15} {'at':>9}  label",
    ]
    for it in rep["top_items"][:top]:
        at = it["first_turn"] if it["first_turn"] is not None else it["born_turn"]
        main = it["session"] == rep["sessions"][0]["session"]
        where = f"turn {at}" if main else f"{clean(it['session'])[:10]}:{at}"
        out.append(
            f"  {_money(it['total']):>9} {it['tokens']:>8,} {it['turns_in_context']:>6}  "
            f"{it['kind']:<15} {where:>9}  {clean(it['label'])[:70]}"
        )
    out += ["", "Detectors (cost = money the pattern is responsible for):"]
    out += _detector_lines(rep["detectors"], 3)
    if len(rep["sessions"]) > 1:
        out += ["", "Main session vs subagents:"]
        for row in rep["sessions"]:
            spawn = (
                f" (spawned at main turn {row['spawned_at_turn']})"
                if row["spawned_at_turn"] is not None
                else ""
            )
            out.append(
                f"  {clean(row['session'])[:40]:<40} {row['turns']:>5} turns "
                f"{_money(row['cost']):>10} {row['share_pct']:>6.1f}%{spawn}"
            )
    out += ["", "What-if (same turns re-priced; upper bounds):"]
    for w in rep["whatif"]:
        out.append(f"  {_money(w['saved']):>10}  {w['description']}")
    return "\n".join(out) + "\n"


def text_summary(rep: Json, top: int) -> str:
    out = [f"context-ledger summary: total {_money(rep['total_cost'])}", ""]
    out.append("Per harness:")
    for name, h in rep["harnesses"].items():
        rec = h["reconciliation"]
        line = (
            f"  {name:<12} {h['sessions']:>3} sessions ({h['session_files']} files, "
            f"{h['turns']} turns) {_money(rec['total_cost']):>10}; unattributed "
            f"{rec['unattributed_pct']:.2f}%, token residual {rec['abs_token_residual_pct']:.2f}%"
        )
        out.append(line)
        if h["harness_reported_total"] is not None:
            out.append(
                f"  {'':<12} harness-reported total {_money(h['harness_reported_total'])} vs "
                f"computed {_money(h['computed_total_for_those'])}"
            )
    out += [
        "",
        "Spend by billing component: "
        + ", ".join(f"{k.replace('_', ' ')} {v:.1f}%" for k, v in rep["usage_mix_pct"].items()),
        f"Subagent share of spend: {rep['subagent_share_pct']:.1f}%",
        "",
        "Spend by item kind:",
        *_kind_table(rep["by_kind"]),
        "",
        "Spend by tool (its calls plus its results, generation and resend):",
        f"  {'tool':<28}{'results':>8}{'errors':>8}{'tokens':>13}{'cost':>11}{'share':>8}",
        *[
            f"  {clean(t['tool'])[:28]:<28}{t['results']:>8}{t['errors']:>8}"
            f"{t['result_tokens']:>13,}{_money(t['cost']):>11}{t['share_pct']:>7.1f}%"
            for t in rep["by_tool"]
        ],
        "",
        "Side calls (billed requests outside the conversation): "
        + (
            ", ".join(
                f"{clean(p)} {v['calls']}x {_money(v['cost'])}"
                for p, v in rep["side_calls_by_purpose"].items()
            )
            or "none"
        ),
        "",
        "Detectors across all sessions:",
        *_detector_lines(rep["detectors"], top),
        "",
        "What-if (same turns re-priced; upper bounds):",
    ]
    for w in rep["whatif"]:
        out.append(f"  {_money(w['saved']):>10} {w['saved_pct']:>6.1f}%  {w['description']}")
    out += ["", "Token estimator error (clean single-tool-result deltas):"]
    for name, e in rep["estimator_error_pct"].items():
        out.append(
            f"  {name:<10} n={e['samples']}: median |err| {e['median_abs_pct']:.1f}%, "
            f"p90 {e['p90_abs_pct']:.1f}%, aggregate {e['aggregate_pct']:+.1f}%"
        )
    if rep["skipped_files"] or rep["skipped_lines"]:
        out += ["", f"Skipped: {len(rep['skipped_files'])} files, {rep['skipped_lines']} lines"]
        for s in rep["skipped_files"]:
            out.append(f"  {clean(s['path'])}: {clean(s['reason'])}")
    return "\n".join(out) + "\n"


# -- HTML -----------------------------------------------------------------------------------

_CSS = """
body{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
margin:0;color:#1d232b;background:#f6f7f9}
main{max-width:1180px;margin:0 auto;padding:28px 32px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:30px 0 10px}
.sub{color:#5b6573}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}
.card{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:12px 16px;min-width:112px}
.card b{display:block;font-size:20px}.card span{color:#5b6573;font-size:12px}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e1e4e8}
th,td{padding:5px 9px;border-bottom:1px solid #eef0f2;text-align:left;vertical-align:top}
th{background:#fafbfc;font-weight:600;font-size:12px;color:#5b6573}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.lbl{font-family:ui-monospace,Menlo,monospace;font-size:12px;word-break:break-all}
.barrow{display:flex;align-items:center;gap:8px}.bar{flex:1;background:#dbe7f5;height:8px;
border-radius:4px}.bar i{display:block;height:8px;border-radius:4px;background:#2f6fbd}
tr.ev td{background:#fff7e6}tr:target td{background:#e8f1ff}
.chart{background:#fff;border:1px solid #e1e4e8;border-radius:8px;padding:8px}
a{color:#2f6fbd;text-decoration:none}a:hover{text-decoration:underline}
details{margin:6px 0}summary{cursor:pointer;font-weight:600}
.k{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
"""


def _e(x: object) -> str:
    return html.escape(clean(x), quote=True)


def _chart(points: list[Json], busts: list[Json], width: int = 1100, height: int = 280) -> str:
    if not points:
        return "<p class=sub>No billed turns.</p>"
    pad_l, pad_r, pad_t, pad_b = 60, 16, 14, 32
    n = max(points[-1]["turn"], 1)
    ymax = max(p["all"] for p in points) or 1.0

    def xy(turn: float, y: float) -> tuple[float, float]:
        x = pad_l + (width - pad_l - pad_r) * turn / n
        yy = height - pad_b - (height - pad_t - pad_b) * y / ymax
        return x, yy

    def line(key: str) -> str:
        return " ".join("{:.1f},{:.1f}".format(*xy(p["turn"], p[key])) for p in points)

    grid = []
    for i in range(5):
        v = ymax * i / 4
        _, y = xy(0, v)
        grid.append(
            f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="#eef0f2"/><text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#5b6573">{_e(_money(v))}</text>'
        )
    ticks = []
    for i in range(6):
        t = round(n * i / 5)
        x, _ = xy(t, 0)
        anchor = "start" if i == 0 else "end" if i == 5 else "middle"
        ticks.append(
            f'<text x="{x:.1f}" y="{height - 10}" text-anchor="{anchor}" font-size="11" '
            f'fill="#5b6573">turn {t}</text>'
        )
    marks = []
    for b in busts:
        if b["turn"] is None or b["turn"] > n:
            continue
        x, _ = xy(b["turn"], 0)
        marks.append(
            f'<a href="#turn-{int(b["turn"])}"><line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" '
            f'y2="{height - pad_b}" stroke="#d9534f" stroke-width="1" stroke-dasharray="3,3">'
            f"<title>{_e(b['summary'])} ({_e(_money(b['cost']))})</title></line></a>"
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="cumulative cost by turn">{"".join(grid)}{"".join(ticks)}{"".join(marks)}'
        f'<polyline fill="none" stroke="#9db8dc" stroke-width="2" points="{line("all")}"/>'
        f'<polyline fill="none" stroke="#2f6fbd" stroke-width="2" points="{line("main")}"/>'
        "</svg>"
    )


def html_report(rep: Json, top: int) -> str:
    s, rec = rep["session"], rep["reconciliation"]
    total = rec["total_cost"] or 1.0
    parts: list[str] = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        "<meta http-equiv=Content-Security-Policy content=\"default-src 'none'; "
        "style-src 'unsafe-inline'\">",
        f"<title>context-ledger: {_e(s['id'])}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>Why this session cost {_e(_money(rec['total_cost']))}</h1>",
        f"<div class=sub>{_e(s['harness'])} session <span class=lbl>{_e(s['id'])}</span> &middot; "
        f"{s['turns']} turns &middot; {s['subagents']} subagents &middot; "
        + _e(", ".join(f"{m} ({n})" for m, n in s["models"].items()))
        + "</div>",
        "<div class=cards>",
    ]
    cards = [
        (_money(rec["total_cost"]), "total cost"),
        (f"{rec['unattributed_pct']:.2f}%", "unattributed (reconciliation residual)"),
    ]
    mix = rep["usage_mix_pct"]
    cards += [(f"{v:.0f}%", k.replace("_", " ")) for k, v in mix.items()]
    for value, label in cards:
        parts.append(f"<div class=card><b>{_e(value)}</b><span>{_e(label)}</span></div>")
    parts.append("</div>")

    busts = rep.get("busts", [])
    parts += [
        "<h2>Cumulative cost</h2><div class=chart>",
        _chart(rep["curve"], busts),
        "<div class=sub style='padding:4px 8px'><span class=k style='background:#2f6fbd'></span>"
        "main session <span class=k style='background:#9db8dc;margin-left:12px'></span>"
        "including subagents and side calls <span class=k style='background:#d9534f;"
        "margin-left:12px'></span>cache bust (click to jump)</div></div>",
    ]

    parts.append(
        "<h2>Where the money went, by item kind</h2><table><tr><th>kind</th>"
        "<th class=n>generating</th><th class=n>resending</th><th class=n>total</th>"
        "<th>share</th></tr>"
    )
    for kind, row in rep["by_kind"].items():
        gen = _money(row["output_cost"]) if "output_cost" in row else ""
        resend = _money(row["resend_cost"]) if "resend_cost" in row else ""
        width = max(0.0, min(100.0, row["share_pct"]))
        parts.append(
            f"<tr><td>{_e(kind)}</td><td class=n>{_e(gen)}</td><td class=n>{_e(resend)}</td>"
            f"<td class=n>{_e(_money(row['total']))}</td><td style='width:30%'><div class=barrow>"
            f"<div class=bar><i style='width:{width:.1f}%'></i></div>"
            f"<span>{row['share_pct']:.1f}%</span></div></td></tr>"
        )
    parts.append("</table>")

    parts.append(
        f"<h2>Top {min(top, len(rep['top_items']))} items by lifetime cost</h2>"
        "<table><tr><th class=n>total</th><th class=n>generating</th><th class=n>resending</th>"
        "<th class=n>tokens</th><th class=n>turns carried</th><th>kind</th><th>first in prompt</th>"
        "<th>label</th></tr>"
    )
    main_name = rep["sessions"][0]["session"] if rep["sessions"] else ""
    for it in rep["top_items"][:top]:
        at = it["first_turn"] if it["first_turn"] is not None else it["born_turn"]
        if at is None:
            where = "&ndash;"
        elif it["session"] == main_name:
            where = f"<a href='#turn-{int(at)}'>turn {int(at)}</a>"
        else:
            where = f"{_e(it['session'])} turn {int(at)}"
        parts.append(
            f"<tr><td class=n>{_e(_money(it['total']))}</td>"
            f"<td class=n>{_e(_money(it['output_cost']))}</td>"
            f"<td class=n>{_e(_money(it['resend_cost']))}</td><td class=n>{it['tokens']:,}</td>"
            f"<td class=n>{it['turns_in_context']}</td><td>{_e(it['kind'])}</td><td>{where}</td>"
            f"<td class=lbl>{_e(it['label'])}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h2>Detectors</h2>")
    for name, d in sorted(rep["detectors"].items(), key=lambda kv: -kv[1]["cost"]):
        tag = "" if d["waste"] else " (informational)"
        parts.append(
            f"<details{' open' if d['waste'] and d['count'] else ''}><summary>{_e(name)}{tag}: "
            f"{d['count']} found, {_e(_money(d['cost']))} ({d['share_pct']:.1f}% of total)"
            "</summary><table><tr><th class=n>cost</th><th>turn</th><th>finding</th></tr>"
        )
        for f in d["top"]:
            sess = str(f["detail"].get("session", ""))
            if f["turn"] is None:
                where = "&ndash;"
            elif sess == main_name:
                where = f"<a href='#turn-{int(f['turn'])}'>{int(f['turn'])}</a>"
            else:
                where = f"{_e(sess)} {int(f['turn'])}"
            parts.append(
                f"<tr><td class=n>{_e(_money(f['cost']))}</td><td>{where}</td>"
                f"<td class=lbl>{_e(f['summary'])}</td></tr>"
            )
        parts.append("</table></details>")

    if len(rep["sessions"]) > 1:
        parts.append(
            "<h2>Main session vs subagents</h2><table><tr><th>session</th><th class=n>turns</th>"
            "<th class=n>cost</th><th class=n>share</th><th>spawned at</th></tr>"
        )
        for row in rep["sessions"]:
            spawn = row["spawned_at_turn"]
            link = (
                f"<a href='#turn-{int(spawn)}'>turn {int(spawn)}</a>" if spawn is not None else ""
            )
            parts.append(
                f"<tr><td class=lbl>{_e(row['session'])}</td><td class=n>{row['turns']}</td>"
                f"<td class=n>{_e(_money(row['cost']))}</td>"
                f"<td class=n>{row['share_pct']:.1f}%</td>"
                f"<td>{link}</td></tr>"
            )
        parts.append("</table>")

    parts.append(
        "<h2>What-if</h2><p class=sub>Same turns re-priced at what each token actually cost; "
        "upper bounds (the agent might have needed to re-read what was cut).</p>"
        "<table><tr><th class=n>saved</th><th class=n>share</th><th>policy</th></tr>"
    )
    for w in rep["whatif"]:
        parts.append(
            f"<tr><td class=n>{_e(_money(w['saved']))}</td>"
            f"<td class=n>{100 * w['saved'] / total:.1f}%</td><td>{_e(w['description'])}</td></tr>"
        )
    parts.append("</table>")

    parts.append(
        "<h2>Turns (main session)</h2><table><tr><th class=n>turn</th><th>time (UTC)</th>"
        "<th>model</th><th class=n>prompt</th><th class=n>cache read</th>"
        "<th class=n>cache write</th><th class=n>output</th><th class=n>cost</th>"
        "<th>events</th></tr>"
    )
    for t in rep["turns"]:
        when = datetime.fromtimestamp(t["time"], UTC).strftime("%Y-%m-%d %H:%M:%S")
        cls = " class=ev" if t["events"] else ""
        parts.append(
            f"<tr id='turn-{t['turn']}'{cls}><td class=n>{t['turn']}</td><td>{when}</td>"
            f"<td>{_e(t['model'])}</td><td class=n>{t['prompt_tokens']:,}</td>"
            f"<td class=n>{t['cache_read']:,}</td><td class=n>{t['cache_write']:,}</td>"
            f"<td class=n>{t['output']:,}</td><td class=n>{_e(_money(t['cost']))}</td>"
            f"<td>{_e(', '.join(t['events']))}{' (replayed)' if t['replayed'] else ''}</td></tr>"
        )
    parts.append("</table>")
    parts.append(
        "<p class=sub>Generated offline by context-ledger. Attribution: each turn's recorded "
        "cache-read, cache-write and input cost is split across the items in prompt order; "
        "output cost goes to the items it generated.</p></main></body></html>"
    )
    return "\n".join(parts)
