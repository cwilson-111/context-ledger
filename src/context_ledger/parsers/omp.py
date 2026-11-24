"""Parser for omp (Oh My Pi) session logs: ``~/.omp/agent/sessions/**/*.jsonl``.

Each line is an entry with ``id``/``parentId``; the conversation is the parent chain of an
assistant message. A ``compaction`` entry on that chain replaces everything before
``firstKeptEntryId``. ``model_usage`` entries are billed requests outside the conversation
(titles, cache warmers). Subagents are separate files whose ``session`` header names the
parent in ``parentSession``.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

from context_ledger import tokens
from context_ledger.model import (
    Compaction,
    Cost,
    Event,
    Item,
    Kind,
    Session,
    SideCall,
    Turn,
    Usage,
    split_output,
)
from context_ledger.parsers.common import (
    JsonObj,
    as_float,
    as_int,
    call_label,
    digest,
    edited_paths,
    normalize_read_path,
    parse_time,
    price_usage,
    read_jsonl,
    short,
)
from context_ledger.pricing import PriceTable

_READ_TOOLS = {"read"}
_CONTEXT_ROLES = {"user", "assistant", "toolResult", "developer", "custom"}


class OmpParseError(ValueError):
    pass


def looks_like_omp(first_objects: list[JsonObj]) -> bool:
    return any(o.get("type") == "session" and "version" in o for o in first_objects)


def _usage(raw: object) -> tuple[Usage, Cost | None, int | None]:
    u = raw if isinstance(raw, dict) else {}
    cttl = u.get("cttl") if isinstance(u.get("cttl"), dict) else {}
    assert isinstance(cttl, dict)
    write_1h = as_int(cttl.get("ephemeral1h"))
    ttl: int | None = None
    if as_int(cttl.get("ephemeral5m")) > 0:
        ttl = 300
    elif write_1h > 0 or "ephemeral1h" in cttl:
        ttl = 3600
    usage = Usage(
        input=as_int(u.get("input")),
        output=as_int(u.get("output")),
        cache_read=as_int(u.get("cacheRead")),
        cache_write=as_int(u.get("cacheWrite")),
        cache_write_1h=write_1h,
    )
    c = u.get("cost")
    recorded = None
    if isinstance(c, dict):
        recorded = Cost(
            input=as_float(c.get("input")),
            output=as_float(c.get("output")),
            cache_read=as_float(c.get("cacheRead")),
            cache_write=as_float(c.get("cacheWrite")),
        )
    return usage, recorded, ttl


def _content_text(content: object) -> tuple[str, int]:
    """Concatenated text and estimated image tokens of a message content field."""
    if isinstance(content, str):
        return content, 0
    texts: list[str] = []
    image_tokens = 0
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif part.get("type") == "image":
                data = part.get("data")
                image_tokens += tokens.estimate_image(data if isinstance(data, str) else None)
    return "\n".join(texts), image_tokens


class _Builder:
    def __init__(self, path: Path, prices: PriceTable) -> None:
        self.path = path
        self.prices = prices
        self.entries: list[JsonObj] = []
        self.by_id: dict[str, JsonObj] = {}
        self.entry_items: dict[str, list[str]] = {}
        self.call_items: dict[str, Item] = {}  # toolCallId -> tool call item
        self.prune_at: dict[str, float] = {}
        self.dropped: set[str] = set()  # thinking items the harness stripped from prompts
        self.session: Session | None = None

    def load(self) -> Session:
        header: JsonObj | None = None
        skipped = 0
        for _, obj in read_jsonl(self.path):
            if obj is None:
                skipped += 1
                continue
            self.entries.append(obj)
            entry_id = obj.get("id")
            if isinstance(entry_id, str):
                self.by_id[entry_id] = obj
            if header is None and obj.get("type") == "session":
                header = obj
        if header is None:
            raise OmpParseError(f"{self.path}: no omp session header")
        parent = header.get("parentSession")
        session = Session(
            id=str(header.get("id") or self.path.stem),
            harness="omp",
            path=str(self.path),
            agent=self.path.stem if isinstance(parent, str) and parent else None,
            parent_id=parent if isinstance(parent, str) and parent else None,
            skipped_lines=skipped,
        )
        self.session = session
        self._build()
        return session

    # -- items ---------------------------------------------------------------------------

    def _items_for(self, entry: JsonObj) -> list[str]:
        entry_id = str(entry.get("id"))
        cached = self.entry_items.get(entry_id)
        if cached is not None:
            return cached
        assert self.session is not None
        made: list[Item] = []
        etype = entry.get("type")
        if etype == "message" and isinstance(entry.get("message"), dict):
            made = self._message_items(entry_id, entry["message"])
        elif etype == "custom_message":
            text, img = _content_text(entry.get("content"))
            made = [
                Item(
                    id=entry_id,
                    kind=Kind.INJECTED,
                    label=f"injected {entry.get('customType', 'message')}",
                    entry_id=entry_id,
                    est_tokens=tokens.estimate_text(text) + img,
                    image_tokens=img,
                    content_hash=digest(text),
                )
            ]
        elif etype == "compaction":
            summary = entry.get("summary")
            text = summary if isinstance(summary, str) else ""
            made = [
                Item(
                    id=entry_id,
                    kind=Kind.SUMMARY,
                    label=f"compaction summary ({entry.get('method', 'unknown')})",
                    entry_id=entry_id,
                    est_tokens=tokens.estimate_text(text),
                    # Remote/snapshot compactions resend an opaque provider blob or images,
                    # not the logged summary text.
                    size_unknown=True,
                )
            ]
        for item in made:
            self.session.add_item(item)
        ids = [i.id for i in made]
        self.entry_items[entry_id] = ids
        return ids

    def _message_items(self, entry_id: str, msg: JsonObj) -> list[Item]:
        role = msg.get("role")
        if role not in _CONTEXT_ROLES:
            return []
        content = msg.get("content")
        if role == "assistant":
            parts: list[Item] = []
            for idx, part in enumerate(content if isinstance(content, list) else []):
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                pid = f"{entry_id}:{idx}"
                if ptype == "thinking":
                    text = str(part.get("thinking") or "")
                    parts.append(
                        Item(
                            id=pid,
                            kind=Kind.THINKING,
                            label="thinking",
                            entry_id=entry_id,
                            est_tokens=tokens.estimate_text(text),
                        )
                    )
                elif ptype == "text":
                    text = str(part.get("text") or "")
                    parts.append(
                        Item(
                            id=pid,
                            kind=Kind.ASSISTANT_TEXT,
                            label=short(text, 60) or "text",
                            entry_id=entry_id,
                            est_tokens=tokens.estimate_text(text),
                            content_hash=digest(text),
                        )
                    )
                elif ptype == "toolCall":
                    name = str(part.get("name") or "tool")
                    args = part.get("arguments")
                    args_d: dict[str, Any] = args if isinstance(args, dict) else {}
                    call_id = str(part.get("id") or pid)
                    path_arg = args_d.get("path")
                    is_read = name in _READ_TOOLS and isinstance(path_arg, str)
                    item = Item(
                        id=pid,
                        kind=Kind.TOOL_CALL,
                        label=call_label(name, args_d),
                        entry_id=entry_id,
                        est_tokens=tokens.estimate_text(name) + tokens.estimate_json(args),
                        tool_name=name,
                        tool_call_id=call_id,
                        file_path=normalize_read_path(str(path_arg)) if is_read else None,
                        read_key=str(path_arg) if is_read else None,
                        writes=tuple(edited_paths(name, args_d)),
                        content_hash=digest(name + json.dumps(args, sort_keys=True)),
                    )
                    parts.append(item)
                    self.call_items[call_id] = item
            return parts
        text, img = _content_text(content)
        est = tokens.estimate_text(text) + img
        if role == "toolResult":
            call_id = str(msg.get("toolCallId") or "")
            call = self.call_items.get(call_id)
            item = Item(
                id=entry_id,
                kind=Kind.TOOL_RESULT,
                label=call.label if call else str(msg.get("toolName") or "tool result"),
                entry_id=entry_id,
                est_tokens=est,
                image_tokens=img,
                tool_name=str(msg.get("toolName") or (call.tool_name if call else "")),
                tool_call_id=call_id,
                file_path=call.file_path if call else None,
                is_error=bool(msg.get("isError")),
                content_hash=digest(text),
            )
            pruned = parse_time(msg.get("prunedAt"))
            if pruned is not None:
                self.prune_at[entry_id] = pruned
                item.size_unknown = True
            self._link_subagents(item, msg.get("details"))
            return [item]
        kind = Kind.USER if role == "user" else Kind.INJECTED
        label = short(text, 60) if kind == Kind.USER else f"injected {role} message"
        return [
            Item(
                id=entry_id,
                kind=kind,
                label=label or role,
                entry_id=entry_id,
                est_tokens=est,
                image_tokens=img,
                content_hash=digest(text),
            )
        ]

    def _link_subagents(self, result: Item, details: object) -> None:
        if result.tool_name != "task" or not isinstance(details, dict):
            return
        assert self.session is not None
        call = self.call_items.get(result.tool_call_id or "")
        for key in ("progress", "results"):
            runs = details.get(key)
            for run in runs if isinstance(runs, list) else []:
                if isinstance(run, dict) and isinstance(run.get("id"), str):
                    self.session.spawned_by.setdefault(run["id"], call.id if call else result.id)

    # -- turns ---------------------------------------------------------------------------

    def _chain(self, entry: JsonObj) -> list[JsonObj]:
        chain: list[JsonObj] = []
        seen: set[str] = set()
        parent = entry.get("parentId")
        while isinstance(parent, str) and parent in self.by_id and parent not in seen:
            seen.add(parent)
            node = self.by_id[parent]
            chain.append(node)
            parent = node.get("parentId")
        chain.reverse()
        last_compaction = None
        for idx, node in enumerate(chain):
            if node.get("type") == "compaction":
                last_compaction = idx
        if last_compaction is None:
            return chain
        comp = chain[last_compaction]
        kept_from = last_compaction
        first_kept = comp.get("firstKeptEntryId")
        for idx in range(last_compaction):
            if chain[idx].get("id") == first_kept:
                kept_from = idx
                break
        return [comp, *chain[kept_from:last_compaction], *chain[last_compaction + 1 :]]

    def _dropped_thinking(self, ctx_entries: list[JsonObj], transforms: object) -> set[str]:
        """Item ids of thinking blocks the harness stripped from this request.

        Paths look like ``messages.N.content.M``; the API alternates user and assistant
        messages, so consecutive non-assistant entries share one API message.
        """
        if not isinstance(transforms, list) or not transforms:
            return set()
        api_messages: list[JsonObj | None] = []
        for node in ctx_entries:
            is_assistant = (
                node.get("type") == "message"
                and isinstance(node.get("message"), dict)
                and node["message"].get("role") == "assistant"
            )
            if is_assistant:
                api_messages.append(node)
            elif not api_messages or api_messages[-1] is not None:
                api_messages.append(None)
        dropped: set[str] = set()
        for t in transforms:
            if not isinstance(t, dict) or t.get("type") != "thinking_dropped":
                continue
            bits = str(t.get("path", "")).split(".")
            if len(bits) != 4 or not bits[1].isdigit() or not bits[3].isdigit():
                continue
            n, m = int(bits[1]), int(bits[3])
            if n < len(api_messages):
                target = api_messages[n]
                if target is not None:
                    dropped.add(f"{target.get('id')}:{m}")
        return dropped

    def _build(self) -> None:
        session = self.session
        assert session is not None
        pending: list[Event] = []
        system: Item | None = None
        system_version = 0
        last_digest: str | None = None
        last_tools: tuple[str, ...] | None = None
        system_prompt_est = 0
        pending_compactions: list[Compaction] = []

        for entry in self.entries:
            etype = entry.get("type")
            ts = parse_time(entry.get("timestamp")) or 0.0
            if etype == "model_change":
                pending.append(Event("model_change", ts, str(entry.get("model", ""))))
            elif etype == "thinking_level_change":
                pending.append(
                    Event("thinking_level_change", ts, str(entry.get("thinkingLevel", "")))
                )
            elif etype == "custom" and entry.get("customType") == "system-prompt-digest":
                value = str(entry.get("data"))
                if last_digest is not None and value != last_digest:
                    pending.append(Event("system_prompt_change", ts))
                last_digest = value
            elif etype == "session_init":
                prompt = entry.get("systemPrompt")
                texts = prompt if isinstance(prompt, list) else [prompt]
                system_prompt_est = sum(
                    tokens.estimate_text(t) for t in texts if isinstance(t, str)
                )
            elif etype == "compaction":
                comp = Compaction(
                    entry_id=str(entry.get("id")),
                    timestamp=ts,
                    first_turn_after=None,
                    tokens_before=as_int(entry.get("tokensBefore")) or None,
                    tokens_after=as_int(entry.get("tokensAfter")) or None,
                    method=str(entry.get("method", "unknown")),
                )
                session.compactions.append(comp)
                pending_compactions.append(comp)
                pending.append(Event("compaction", ts, comp.method))
            elif etype == "model_usage":
                usage, recorded, _ = _usage(entry.get("usage"))
                model = str(entry.get("model", ""))
                cost, source = price_usage(model, usage, recorded, self.prices)
                purpose = str(entry.get("purpose", "side call"))
                session.side_calls.append(
                    SideCall(str(entry.get("id")), purpose, model, ts, usage, cost, source)
                )
                if isinstance(entry.get("cacheRefresh"), dict):
                    pending.append(Event("cache_warm", ts, model))
            elif etype == "message" and isinstance(entry.get("message"), dict):
                msg = entry["message"]
                if msg.get("role") != "assistant":
                    continue
                usage, recorded, ttl = _usage(msg.get("usage"))
                parts = self._items_for(entry)
                if usage.prompt + usage.output == 0:
                    # Failed request (rate limit, interrupt): nothing billed. Any partial
                    # content still rides along in later prompts at its estimated size.
                    for pid in parts:
                        session.items[pid].tokens = session.items[pid].est_tokens
                    continue
                model = str(msg.get("model", ""))
                cost, source = price_usage(model, usage, recorded, self.prices)
                start = parse_time(msg.get("timestamp")) or ts
                snapshot = msg.get("contextSnapshot")
                non_message = (
                    as_int(snapshot.get("nonMessageTokens")) if isinstance(snapshot, dict) else 0
                )
                controls = msg.get("requestControls")
                tools_active: tuple[str, ...] | None = None
                if isinstance(controls, dict) and isinstance(controls.get("tools"), dict):
                    active = controls["tools"].get("active")
                    if isinstance(active, list):
                        tools_active = tuple(sorted(str(a) for a in active))
                if (
                    tools_active is not None
                    and last_tools is not None
                    and tools_active != last_tools
                ):
                    pending.append(Event("tools_change", start))
                if tools_active is not None:
                    last_tools = tools_active

                if system is None or (non_message and system.est_tokens != non_message):
                    system_version += 1
                    sid = f"system#{system_version}"
                    # omp's ``nonMessageTokens`` misses ~1-1.5k tokens the provider adds
                    # (measured: P_0 - nonMessageTokens - first message is consistently in
                    # that range), so the first version absorbs the turn-0 remainder and
                    # later versions move by the recorded difference (see calibrate).
                    system = session.add_item(
                        Item(
                            id=sid,
                            kind=Kind.SYSTEM,
                            label=f"system prompt + tool schemas (v{system_version})",
                            entry_id=sid,
                            est_tokens=non_message or system_prompt_est,
                            size_unknown=True,
                        )
                    )
                    if system_version > 1:
                        pending.append(Event("system_size_change", start))

                ctx_entries = self._chain(entry)
                dropped = self._dropped_thinking(ctx_entries, msg.get("inputTransformations"))
                if dropped - self.dropped:
                    pending.append(Event("thinking_dropped", start, str(len(dropped))))
                # Measured on real logs: once stripped, the blocks stay out of later prompts
                # even though the transformation is only logged on some requests.
                self.dropped |= dropped
                context = [system.id]
                for node in ctx_entries:
                    context.extend(i for i in self._items_for(node) if i not in self.dropped)

                index = len(session.turns)
                for pid in parts:
                    session.items[pid].born_turn = index
                split_output([session.items[p] for p in parts], usage.output)
                completed = parse_time(msg.get("completedAt"))
                turn = Turn(
                    index=index,
                    entry_id=str(entry.get("id")),
                    timestamp=start,
                    completed_at=completed,
                    model=model,
                    usage=usage,
                    cost=cost,
                    cost_source=source,
                    stop_reason=str(msg.get("stopReason", "")),
                    context=context,
                    output_items=list(parts),
                    cache_ttl_s=ttl,
                    events=pending,
                    after_compaction=bool(pending_compactions),
                    system_tokens=non_message or None,
                    replayed=recorded is not None and cost.total == 0 and usage.output > 0,
                )
                for comp in pending_compactions:
                    comp.first_turn_after = index
                pending_compactions = []
                pending = []
                session.turns.append(turn)

        self._apply_pruning()

    def _apply_pruning(self) -> None:
        session = self.session
        assert session is not None
        starts = [t.timestamp for t in session.turns]
        for item_id, when in self.prune_at.items():
            item = session.items[item_id]
            idx = bisect.bisect_left(starts, when)
            item.shrunk_tokens = item.est_tokens
            if idx < len(session.turns):
                item.shrink_turn = idx
                session.turns[idx].events.append(Event("context_edit", when, item.label))
            else:
                item.shrink_turn = None


def parse_omp(path: Path, prices: PriceTable | None = None) -> Session:
    return _Builder(path, prices or PriceTable()).load()
