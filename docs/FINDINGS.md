# Findings from one developer's real agent logs

Aggregate numbers from running context-ledger over every omp and Claude Code session log on
the author's machine (snapshot 2026-11-24 02:33 UTC). Only aggregates are reported: no prompts,
code, file names, paths or tool output. Tool names are the harnesses' built-in tool names.

Reproduce on your own logs:

```sh
context-ledger summary ~/.omp/agent/sessions --json
context-ledger summary ~/.claude/projects --json
uv run python scripts/fit_estimator.py ~/.omp/agent/sessions ~/.claude/projects
```

## The data

| | omp | Claude Code |
|---|---:|---:|
| `.jsonl` files found | 61 | 2 |
| session logs with billed requests | 58 (18 top-level sessions + 40 subagent transcripts) | 2 |
| billed requests ("turns") | 5,591 | 35 |
| of which replayed history (recorded at $0, see below) | 1,017 | 0 |
| spend | **$574.55** (recorded per request by omp) | **$6.23** (priced from tokens; Claude Code's own total: $6.46) |
| models | claude-opus-5-5, claude-opus-5, claude-fable-5 (+ haiku/sonnet side calls) | claude-fable-5-2, claude-opus-5-5 |

The omp total includes the session that built this tool, which was still running at snapshot
time.

**Not parsed:** 2 of the 61 omp files are file-change journals, not session logs, and were
skipped. One omp session log has no billed request (its only request failed) and is counted
but contributes nothing. No line in any file failed to parse. Content that the logs record
but context-ledger does not model as separate items (omp date/cwd reminders and injected
rules, Claude Code attachments without rendered text) still gets paid for: it lands inside
the calibrated size of the neighbouring item.

## Where the money went

### By billing component

| | omp | Claude Code |
|---|---:|---:|
| cache read | 54.4% | 14.4% |
| cache write | 29.9% | 55.0% |
| uncached input | 0.0% | 0.0% |
| output (incl. thinking) | 14.3% | 30.6% |
| side calls (titles, cache warmers, probes) | 1.3% | n/a |

So 84% of omp spend is carrying context (cache reads plus cache writes, including each item's
first write), not generating anything. The two Claude Code
sessions are short (35 turns), so cache writes dominate.

### By item kind (lifetime cost: generating the item plus every later prompt it rode in)

| item kind | omp generate | omp resend | omp total | share | Claude Code share |
|---|---:|---:|---:|---:|---:|
| tool results | n/a | $265.19 | $265.19 | **46.2%** | 38.0% |
| tool calls (arguments) | $42.71 | $97.46 | $140.18 | **24.4%** | 22.7% |
| thinking | $34.79 | $72.34 | $107.13 | **18.6%** | 26.7% |
| system prompt + tool schemas | n/a | $33.45 | $33.45 | 5.8% | 2.9% |
| assistant text | $4.63 | $9.39 | $14.02 | 2.4% | 2.8% |
| injected reminders / async results | n/a | $3.59 | $3.59 | 0.6% | 6.3% |
| compaction summaries | n/a | $1.73 | $1.73 | 0.3% | n/a |
| user messages | n/a | $1.53 | $1.53 | 0.3% | 0.7% |
| side calls | | | $7.70 | 1.3% | 0% |
| unattributed (reconciliation residual) | | | $0.05 | 0.01% | 0.00% |

Everything the model generates costs about twice as much to carry as to produce. Thinking
costs $34.79 to generate and $72.34 to resend, and the arguments of tool calls (the file
contents inside `write` and `edit` calls) cost $97.46 to resend.

### By tool (calls plus results, omp)

| tool | results | errors | result tokens | cost | share |
|---|---:|---:|---:|---:|---:|
| read | 1,769 | 17 | 2,977,656 | $128.42 | 22.4% |
| edit | 1,231 | 116 | 535,803 | $68.15 | 11.9% |
| eval | 1,147 | 55 | 806,539 | $57.97 | 10.1% |
| write | 437 | 26 | 213,162 | $44.74 | 7.8% |
| bash | 1,200 | 56 | 721,367 | $40.56 | 7.1% |
| grep | 641 | 1 | 864,050 | $31.81 | 5.5% |
| web_search | 136 | 0 | 278,216 | $7.86 | 1.4% |

For `edit` and `write`, most of the cost is the call, not the result: $42.88 and $32.76 of
argument resend and generation against $25.27 and $11.98 for results.

**Subagents** account for 23.0% of omp spend.

## Detectors

| detector | omp findings | omp cost | share | Claude Code |
|---|---:|---:|---:|---:|
| cache busts | 31 | **$77.96** | **13.6%** | 1 bust, $1.36 (21.9%) |
| huge tool outputs (cost above 5,000 tokens) | 194 results | $23.59 | 4.1% | 6, $0.66 (10.5%) |
| failed tool calls | 247 failures, 25 retry loops | $10.80 | 1.9% | 2, $0.06 (1.0%) |
| repeated identical reads | 12 | $1.11 | 0.2% | 0 |
| compactions (informational: estimated saving) | 4 | +$44.24 saved | | 0 |
| side calls (informational) | 98 calls | $7.70 | 1.3% | 0 |

Detector costs overlap (a huge output can also be a failed call), so they do not add up.

### Cache busts by likely cause (omp)

| cause | busts | cost of rewriting what should have been read |
|---|---:|---:|
| cache TTL expired (gap since last request longer than the recorded 1 h TTL) | 16 | $30.43 |
| context edited mid-prompt (harness pruned a superseded tool result) | 5 | $18.09 |
| unknown (no event, gap shorter than the TTL) | 2 | $8.88 |
| tool set changed | 2 | $8.37 |
| system prompt changed | 5 | $7.19 |
| thinking blocks stripped by the harness | 1 | $5.01 |
| **total** | **31** | **$77.96** |

The only Claude Code bust was a mid-session model switch, which rewrote 174k cached tokens
for $1.36, 22% of that harness's spend. The cache is per model.

The 5 pruning busts are a real trade-off. omp replaces a superseded read with a one-line
placeholder to save resend tokens, but the edit lands in the middle of the prompt, so
everything after it has to be re-written to cache. The rewrites were 91k to 746k tokens,
an average of $3.62 per event.

### Side calls (omp)

cache-warm 8 calls $5.36, task-completion-probe 29 calls $1.11, job-widget 22 calls $1.02,
title 20 calls $0.14, job-caption 14 calls $0.06, auto-thinking 5 calls $0.01. One of the
cache-warm calls cost $4.91 by itself.

### Compactions (omp)

4 compactions (3 remote, 1 snapshot) removed 712k, 773k, 129k and about 170k tokens. Re-priced
at the cache-read rate of every later turn until the session ended, minus the cost of
re-writing the cache once, they saved an estimated $44.24. That is an upper bound, because
without compaction those sessions would have hit the context limit.

## What to change (measured counterfactuals)

Each number re-prices the same sessions with the agent's behaviour held fixed. Every item
is charged the per-token price it actually paid at each turn. The numbers are upper bounds:
cutting context can make an agent re-read or re-run something, and that is not charged.
omp base: $574.55.

| change | saved (omp) | share | saved (Claude Code) |
|---|---:|---:|---:|
| Replace each tool result with a 20-token stub **10 turns** after it arrives | $192.61 | **33.5%** | 16.0% |
| ... after **25 turns** | $172.08 | **30.0%** | 0% (sessions too short) |
| ... after **50 turns** | $143.23 | 24.9% | 0% |
| Do not resend earlier turns' thinking blocks | $72.34 | 12.6% | 10.5% |
| Cap every tool result at **2,000 tokens** | $68.83 | 12.0% | 17.2% |
| Cap at **5,000 tokens** | $23.59 | 4.1% | 10.5% |
| Cap at **10,000 tokens** | $7.76 | 1.4% | 5.7% |
| Avoid every cache bust | $77.96 | 13.6% | 21.9% |

Concrete advice for this workload:

1. **Expire old tool output, in batches.** Tool results are 46% of spend and all of it is
   resend: stubbing results 25 turns after arrival would have cut spend by up to 30%. Each
   mid-prompt edit forces a cache rewrite, though (the 5 pruning busts above cost $18.09),
   so expire in batches: when the cache is already cold after a TTL expiry, or as part of
   compaction, not one result at a time.
2. **Read ranges, not whole files.** `read` is the most expensive tool at 22% of spend.
   Capping any single result at 2,000 tokens would have saved 12.0%; at 5,000, 4.1%. The
   difference ($45) is the tokens between the 2k and 5k marks of large results, which a
   "grep first, then read the range" instruction targets directly.
3. **Don't leave a long session idle for more than an hour.** TTL expiry caused 16 busts
   for $30.43 (5.3%). At claude-opus-5-5 rates, resuming a 500k-token context after the
   cache expired costs 500k x ($8 - $0.20)/M = $3.90 before the agent does anything.
   Compacting before a break makes the rewrite much smaller.
4. **Switch models at a session boundary, not mid-session.** One switch was 22% of the
   Claude Code spend.
5. **Thinking is 18.6% of spend and two thirds of that is resend.** If the harness or API
   can drop thinking from turns before the latest user message, that is worth up to 12.6%.
   Dropping it costs one cache rewrite at the moment it happens; omp's one logged strip
   cost $5.01.
6. **Large `write`/`edit` arguments are resent like any other context.** Tool-call arguments
   are 17.0% of spend as resend alone. Preferring small targeted edits over whole-file
   writes, or having the harness elide the arguments of a successful write, would address
   most of it.
7. Failed tool calls (1.9%) and repeated reads (0.2%) are small here. omp already prunes
   superseded reads. `edit` failed 9.4% of the time (116 of 1,231 calls), the highest rate
   of any frequently used tool.

## Reconciliation residual

For every turn, the item costs plus the unattributed remainder equal the turn's recorded cost
to floating-point precision. That is checked in the tests. The residual reported here is how
much of the cost could not be pinned to an item.

| | omp | Claude Code |
|---|---:|---:|
| unattributed cost | $0.054 (**0.009%**) | $0.000 (**0.00%**) |
| prompt tokens not matched by item sizes (sum of absolute per-turn residuals) | 1.54% | 0.71% |
| ... of which items larger than the recorded prompt (spread proportionally, no cost lost) | 1.54% | 0.71% |
| priced total vs the harness's own session total | n/a (omp records cost per request) | $6.23 vs $6.46 (-3.6%) |

- **Output and thinking are not a gap.** Output cost is charged to the parts the turn
  generated. Thinking is resent at its full generated size, which the data confirms: the
  prompt-size delta between consecutive turns equals the previous turn's output tokens
  plus the new tool results.
- **omp token residual (1.54%).** It is all overflow, from one session and its replayed
  copy. When that session was resumed after 42 days, its prompt shrank by about 20.9k tokens
  with no event in the log. The likely cause is thinking blocks stripped by an older harness
  version that did not log the transformation. The overflow is spread across that turn's
  items, so no money is lost or invented.
- **Claude Code -3.6%.** Claude Code's `cost-state` total includes $0.158 of claude-haiku-5-5
  side calls (including 11 web-search requests) and about $0.07 of usage in one session that
  are not in its transcript. Turns are priced with table rates that reproduce Claude Code's
  own per-model totals exactly for the models that appear in the transcripts.
- **Replayed history.** 1,017 omp turns are earlier sessions copied into "seance"/recall
  subagents and recorded at $0. Their cost stays $0, and their tool calls are left out of
  detector counts so that each real call is counted once.

## Token counts and the estimator

Item sizes come from recorded usage wherever the log allows it. 99.0% of omp resend cost is
carried by items whose size is pinned by recorded numbers:

| size source | share of omp resend cost | Claude Code |
|---|---:|---:|
| recorded (assistant output tokens, harness counts) | 37.1% | 31.1% |
| delta: the only new item between two turns, so the prompt-size delta is its exact size | 33.6% | 11.1% |
| scaled: a delta pins a group of new items exactly, split by estimate | 28.3% | 48.1% |
| estimate only | 1.0% | 9.7% |

Estimator: `ceil(chars / 2.24)`. The ratio was fitted (ratio of sums) on clean deltas that
isolate one text-only tool result, using a seeded half of the sessions, and evaluated on the
other half (`scripts/fit_estimator.py`; results of at least 50 tokens):

| estimator | held-out results | median abs. error | p90 abs. error | median signed | aggregate |
|---|---:|---:|---:|---:|---:|
| chars / 4 (common rule of thumb) | 1,148 | 46.0% | 55.6% | -46.0% | -44.8% |
| **chars / 2.24 (fitted)** | 1,148 | **10.1%** | **23.4%** | -3.5% | **-1.3%** |

On the 15 clean Claude Code samples (in-sample, too few for a split), the median absolute
error is 10.5% and the aggregate is +5.5%. Tool output from these agents is much denser
than prose: numbered source lines, JSON, paths and hashes come to about 2.2 characters per
token. chars/4 would undercount it by 45%.
