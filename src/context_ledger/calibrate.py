"""Pin item token counts to the recorded per-request prompt sizes.

Every turn records its full prompt size ``P_t = input + cache_read + cache_write``. Items
already sized at an earlier turn (and assistant output, sized by the recorded ``output``
count) are known, so ``P_t`` minus their sum, minus whatever was still unexplained at the
previous turn, is exactly the size of whatever is new in this prompt. One new item gets that
delta outright (``TokenSource.DELTA``); several share it in proportion to their estimates
(``SCALED``). After a harness event (model, system or tool change, pruning) the delta may
contain other changes, so it is only used when it looks plausible next to the estimate;
otherwise the estimate is kept and the mismatch stays visible as the turn's residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from context_ledger import tokens
from context_ledger.model import Item, Kind, Session, TokenSource, spread

# After an event, accept a delta within this factor of the estimate, or within the slack.
RATIO_LIMIT = 3.0
ABS_SLACK_TOKENS = 64
MIN_ERROR_SAMPLE_TOKENS = 50


@dataclass(frozen=True)
class EstimatorError:
    samples: int
    median_abs_pct: float
    p90_abs_pct: float
    median_signed_pct: float
    aggregate_pct: float  # (sum of estimates / sum of actual) - 1


def single_result_samples(samples: list[DeltaSample]) -> list[tuple[float, int]]:
    """``(chars, exact tokens)`` for clean deltas that isolate one text-only tool result."""
    return [
        (s.text_estimate * tokens.CHARS_PER_TOKEN, s.actual)
        for s in samples
        if s.clean and not s.has_image and s.kinds == (Kind.TOOL_RESULT,) and s.actual > 0
    ]


def estimator_error(rows: list[tuple[float, int]], chars_per_token: float) -> EstimatorError:
    """Error of ``chars / chars_per_token`` against exact counts (results >= 50 tokens)."""
    rows = [r for r in rows if r[1] >= MIN_ERROR_SAMPLE_TOKENS]
    if not rows:
        return EstimatorError(0, 0.0, 0.0, 0.0, 0.0)
    rel = sorted((chars / chars_per_token - actual) / actual for chars, actual in rows)
    absolute = sorted(abs(x) for x in rel)
    estimated = sum(chars / chars_per_token for chars, _ in rows)
    actual = sum(a for _, a in rows)
    return EstimatorError(
        samples=len(rows),
        median_abs_pct=100 * absolute[len(absolute) // 2],
        p90_abs_pct=100 * absolute[int(0.9 * len(absolute))],
        median_signed_pct=100 * rel[len(rel) // 2],
        aggregate_pct=100 * (estimated / actual - 1),
    )


@dataclass(frozen=True)
class DeltaSample:
    """A turn where everything new was estimable, so the prompt-size delta is their exact
    total. Recorded whether or not calibration accepted it, so the estimator's error can be
    measured without selection bias."""

    turn: int
    estimated: int
    actual: int
    kinds: tuple[Kind, ...]
    text_estimate: int  # estimated tokens of the text in those items (images excluded)
    has_image: bool
    clean: bool  # no harness event (model/system/tools change, pruning, ...) since last turn
    accepted: bool


@dataclass
class Calibration:
    residual_tokens: list[int] = field(default_factory=list)  # per turn: P_t - sum(sizes)
    samples: list[DeltaSample] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0


def _plausible(actual: int, estimated: int) -> bool:
    if actual <= 0:
        return False
    if abs(actual - estimated) <= ABS_SLACK_TOKENS:
        return True
    if estimated <= 0:
        return False
    ratio = actual / estimated
    return 1 / RATIO_LIMIT <= ratio <= RATIO_LIMIT


def calibrate(session: Session) -> Calibration:
    """Set ``tokens``/``source``/``first_turn`` on every item; return residuals and samples."""
    items = session.items
    cal = Calibration()
    known: set[str] = set()
    prev_system: Item | None = None
    carry = 0
    for turn in session.turns:
        t = turn.index
        new = [i for i in turn.context if i not in known]
        open_new: list[Item] = []
        for item_id in new:
            item = items[item_id]
            item.first_turn = t
            if item.kind == Kind.SYSTEM and prev_system is not None and item.est_tokens > 0:
                # A new system version: the harness count moved by a known amount, and the
                # calibrated predecessor already includes whatever that count misses.
                item.tokens = max(prev_system.tokens + item.est_tokens - prev_system.est_tokens, 0)
                item.source = TokenSource.RECORDED
                item.size_unknown = False
            if item.source == TokenSource.RECORDED and not item.size_unknown:
                continue
            open_new.append(item)
        open_ids = {i.id for i in open_new}
        known_sum = sum(items[i].size_at(t) for i in turn.context if i not in open_ids)
        # Tokens left unexplained at the previous turn are still in this prompt unless the
        # context was rebuilt; they must not be mistaken for new items.
        if turn.after_compaction:
            carry = 0
        remainder = turn.usage.prompt - known_sum - carry
        absorbers = [i for i in open_new if i.size_unknown]
        estimable = [i for i in open_new if not i.size_unknown]
        est_sum = sum(i.est_tokens for i in estimable)

        if absorbers:
            for i in estimable:
                i.tokens, i.source = i.est_tokens, TokenSource.ESTIMATE
            left = remainder - est_sum
            if left > 0:
                spread(absorbers, left, [max(i.est_tokens, 1) for i in absorbers])
                source = TokenSource.DELTA if len(absorbers) == 1 else TokenSource.SCALED
                for i in absorbers:
                    i.source = source
            else:
                for i in absorbers:
                    i.tokens, i.source = i.est_tokens, TokenSource.ESTIMATE
        elif estimable:
            # With no harness event since the last turn, everything new is these items, so a
            # positive delta is their size even when the log shows less text than was sent
            # (Claude Code logs shortened web results). After an event (model, system or
            # tool change, pruning) the delta can contain other changes, so it must also
            # look plausible next to the estimate.
            clean = not turn.events
            accepted = remainder > 0 if clean else _plausible(remainder, est_sum)
            if accepted:
                spread(estimable, remainder, [i.est_tokens for i in estimable])
                source = TokenSource.DELTA if len(estimable) == 1 else TokenSource.SCALED
                for i in estimable:
                    i.source = source
                cal.accepted += 1
            else:
                cal.rejected += 1
                for i in estimable:
                    i.tokens, i.source = i.est_tokens, TokenSource.ESTIMATE
            if t > 0:
                cal.samples.append(
                    DeltaSample(
                        turn=t,
                        estimated=est_sum,
                        actual=remainder,
                        kinds=tuple(i.kind for i in estimable),
                        text_estimate=sum(i.est_tokens - i.image_tokens for i in estimable),
                        has_image=any(i.image_tokens for i in estimable),
                        clean=not turn.events and not turn.replayed,
                        accepted=accepted,
                    )
                )
        known.update(new)
        for item_id in turn.context:
            if items[item_id].kind == Kind.SYSTEM:
                prev_system = items[item_id]
        sized = sum(items[i].size_at(t) for i in turn.context)
        cal.residual_tokens.append(turn.usage.prompt - sized)
        carry = turn.usage.prompt - sized

    for item in items.values():
        if item.first_turn is None and item.source == TokenSource.ESTIMATE and not item.tokens:
            item.tokens = item.est_tokens
    return cal
