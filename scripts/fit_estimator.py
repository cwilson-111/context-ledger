"""Fit the chars-per-token constant in ``context_ledger.tokens`` on your own logs.

Uses only clean prompt-size deltas that isolate exactly one new text-only tool result (the
delta is then that result's exact token count, framing included). Sessions are split in
half by a seeded shuffle: the constant is fitted on one half (ratio of sums) and the error
is reported on the other, for both the fitted constant and the common chars/4 rule. Prints
aggregates only.

    uv run python scripts/fit_estimator.py ~/.omp/agent/sessions ~/.claude/projects
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from context_ledger.attribute import attribute
from context_ledger.calibrate import estimator_error, single_result_samples
from context_ledger.parsers import load


def main(paths: list[str]) -> None:
    by_session: dict[str, list[tuple[float, int]]] = {}
    for p in paths:
        for session in load(Path(p).expanduser()).all_sessions():
            rows = single_result_samples(attribute(session).calibration.samples)
            if rows:
                by_session[session.path] = rows
    names = sorted(by_session)
    random.Random(0).shuffle(names)
    train = [r for n in names[: len(names) // 2] for r in by_session[n]]
    test = [r for n in names[len(names) // 2 :] for r in by_session[n]]
    fitted = sum(c for c, _ in train) / sum(a for _, a in train)
    print(f"sessions with samples: {len(names)}; train samples {len(train)}, test {len(test)}")
    print(f"fitted chars/token (train half, ratio of sums): {fitted:.3f}")
    for label, k in (("chars/4", 4.0), (f"chars/{fitted:.2f}", fitted)):
        e = estimator_error(test, k)
        print(
            f"{label:>12} on held-out results >= 50 tokens (n={e.samples}): "
            f"median |err| {e.median_abs_pct:.1f}%, p90 |err| {e.p90_abs_pct:.1f}%, "
            f"median signed {e.median_signed_pct:+.1f}%, aggregate {e.aggregate_pct:+.1f}%"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or [str(Path.home() / ".omp/agent/sessions")])
