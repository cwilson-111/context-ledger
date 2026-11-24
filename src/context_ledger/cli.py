"""``context-ledger`` command line. Reads logs read-only; never touches the network."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from context_ledger import __version__
from context_ledger.parsers import load
from context_ledger.pricing import PriceTable
from context_ledger.render import html_report, text_report, text_summary
from context_ledger.report import Group, build_group, session_report, summary_report


def _prices(path: str | None) -> PriceTable:
    return PriceTable.from_file(Path(path)) if path else PriceTable()


def _groups(target: Path, prices: PriceTable) -> tuple[list[Group], list[tuple[str, str]]]:
    if not target.exists():
        raise SystemExit(f"context-ledger: {target}: no such file or directory")
    loaded = load(target, prices)
    groups = [build_group(s) for s in loaded.sessions if s.turns or s.subagents]
    return groups, loaded.skipped


def cmd_report(args: argparse.Namespace) -> int:
    groups, skipped = _groups(Path(args.path).expanduser(), _prices(args.prices))
    if not groups:
        print(
            "context-ledger: no omp or Claude Code sessions with billed turns found",
            file=sys.stderr,
        )
        return 2
    reports = [session_report(g, args.top) for g in groups]
    if args.html:
        out = Path(args.html)
        if len(reports) == 1:
            out.write_text(html_report(reports[0], args.top), encoding="utf-8")
            print(f"wrote {out}", file=sys.stderr)
        else:
            out.mkdir(parents=True, exist_ok=True)
            for rep in reports:
                name = "".join(c if c.isalnum() or c in "-_" else "_" for c in rep["session"]["id"])
                (out / f"{name}.html").write_text(html_report(rep, args.top), encoding="utf-8")
            print(f"wrote {len(reports)} reports to {out}/", file=sys.stderr)
    if args.json:
        payload = reports[0] if len(reports) == 1 else reports
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif not args.html or args.text:
        sys.stdout.write("\n".join(text_report(r, args.top) for r in reports))
    for path, reason in skipped:
        print(f"skipped {path}: {reason}", file=sys.stderr)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    groups, skipped = _groups(Path(args.path).expanduser(), _prices(args.prices))
    if not groups:
        print(
            "context-ledger: no omp or Claude Code sessions with billed turns found",
            file=sys.stderr,
        )
        return 2
    rep = summary_report(groups, skipped, args.top)
    if args.json:
        json.dump(rep, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(text_summary(rep, args.top))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="context-ledger",
        description="Explain why a coding-agent session cost what it did "
        "(omp and Claude Code logs; fully offline, read-only).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--top", type=int, default=15, help="rows per table (default 15)")
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument(
        "--prices", metavar="FILE", help="JSON price overrides for models without recorded cost"
    )

    rep = sub.add_parser(
        "report",
        parents=[common],
        help="attribute one session (or each session under a directory)",
    )
    rep.add_argument("path", help="session .jsonl file or a directory of them")
    rep.add_argument(
        "--html", metavar="OUT", help="write a self-contained HTML report (a directory if several)"
    )
    rep.add_argument("--text", action="store_true", help="also print the text report with --html")
    rep.set_defaults(func=cmd_report)

    summ = sub.add_parser(
        "summary", parents=[common], help="aggregate every session under a directory"
    )
    summ.add_argument("path", help="directory such as ~/.omp/agent/sessions or ~/.claude/projects")
    summ.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
