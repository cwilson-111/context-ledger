"""Parser for Claude Code transcripts: ``~/.claude/projects/<project>/<session>.jsonl``.

Entries link through ``uuid``/``parentUuid``. One API response is streamed into several
``assistant`` lines (one per content block) that share ``message.id`` and repeat the same
``usage``; they are merged into one turn. Claude Code records tokens but not per-request
cost, so turns are priced from the price table; its ``cost-state`` line carries the
harness's own session total, kept as ``recorded_total`` for reconciliation. A
``compact_boundary`` system entry starts a new chain (its ``parentUuid`` is null), so the
parent walk stops there by itself. Subagent transcripts (``isSidechain``) become child
sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from context_ledger import tokens
from context_ledger.model import (
    Compaction,
    Event,
    Item,
    Kind,
    Session,
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
    parse_time,
    price_usage,
    read_jsonl,
    short,
)
from context_ledger.pricing import PriceTable

_READ_TOOLS = {"Read"}
_ENTRY_TYPES = {"user", "assistant", "attachment", "system"}


def looks_like_claude_code(first_objects: list[JsonObj]) -> bool:
    return any(
        "sessionId" in o and o.get("type") in _ENTRY_TYPES and "uuid" in o for o in first_objects
    )


@dataclass
class _Response:
    """All lines of one streamed assistant response."""

    message_id: str
    first: JsonObj
    lines: list[JsonObj] = field(default_factory=list)


def _usage(raw: object) -> tuple[Usage, int | None]:
    u = raw if isinstance(raw, dict) else {}
    creation = u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
    assert isinstance(creation, dict)
    write_1h = as_int(creation.get("ephemeral_1h_input_tokens"))
    write_5m = as_int(creation.get("ephemeral_5m_input_tokens"))
    details = u.get("output_tokens_details")
    thinking = as_int(details.get("thinking_tokens")) if isinstance(details, dict) else None
    ttl = 300 if write_5m > 0 else 3600 if write_1h > 0 else None
    return (
        Usage(
            input=as_int(u.get("input_tokens")),
            output=as_int(u.get("output_tokens")),
            cache_read=as_int(u.get("cache_read_input_tokens")),
            cache_write=as_int(u.get("cache_creation_input_tokens")),
            cache_write_1h=write_1h,
            thinking=thinking,
        ),
        ttl,
    )


def _block_text(block: object) -> tuple[str, int]:
    """Text and image-token estimate of a tool_result ``content`` value."""
    if isinstance(block, str):
        return block, 0
    texts: list[str] = []
    image_tokens = 0
    if isinstance(block, list):
        for part in block:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(str(part.get("text") or ""))
            elif part.get("type") == "image":
                source = part.get("source")
                data = source.get("data") if isinstance(source, dict) else None
                image_tokens += tokens.estimate_image(data if isinstance(data, str) else None)
    return "\n".join(texts), image_tokens


class _Builder:
    def __init__(self, path: Path, prices: PriceTable, entries: list[JsonObj]) -> None:
        self.path = path
        self.prices = prices
        self.entries = entries
        self.by_uuid: dict[str, JsonObj] = {}
        self.position: dict[str, int] = {}
        self.result_entry: dict[str, JsonObj] = {}  # tool_use_id -> user entry answering it
        self.response_lines: dict[str, list[JsonObj]] = {}  # message.id -> its streamed lines
        self.entry_items: dict[str, list[str]] = {}
        self.calls: dict[str, Item] = {}
        self.part_items: dict[str, Item] = {}  # "<uuid>:<idx>" -> assistant part item
        for pos, e in enumerate(entries):
            if not isinstance(e.get("uuid"), str):
                continue
            self.by_uuid[e["uuid"]] = e
            self.position[e["uuid"]] = pos
            msg = e.get("message")
            content = msg.get("content") if isinstance(msg, dict) else None
            if e.get("type") == "user" and isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "tool_result":
                        self.result_entry[str(part.get("tool_use_id"))] = e
            if e.get("type") == "assistant" and isinstance(msg, dict) and msg.get("id"):
                self.response_lines.setdefault(str(msg["id"]), []).append(e)

    def _items_for(self, session: Session, entry: JsonObj) -> list[str]:
        uuid = str(entry.get("uuid"))
        cached = self.entry_items.get(uuid)
        if cached is not None:
            return cached
        made: list[Item] = []
        etype = entry.get("type")
        msg = entry.get("message") if isinstance(entry.get("message"), dict) else {}
        assert isinstance(msg, dict)
        content = msg.get("content")
        if etype == "assistant":
            for idx, part in enumerate(content if isinstance(content, list) else []):
                if isinstance(part, dict):
                    item = self._assistant_part(f"{uuid}:{idx}", uuid, part)
                    if item is not None:
                        made.append(item)
        elif etype == "user" and not entry.get("isMeta"):
            made = self._user_items(uuid, entry, content)
        elif etype == "attachment":
            rendered = entry.get("rendered")
            text = (
                "\n".join(str(r.get("content", "")) for r in rendered if isinstance(r, dict))
                if isinstance(rendered, list)
                else ""
            )
            if text:
                att = entry.get("attachment")
                att_type = att.get("type", "attachment") if isinstance(att, dict) else "attachment"
                made = [
                    Item(
                        id=uuid,
                        kind=Kind.INJECTED,
                        label=f"injected {att_type}",
                        entry_id=uuid,
                        est_tokens=tokens.estimate_text(text),
                        content_hash=digest(text),
                    )
                ]
        for item in made:
            session.add_item(item)
        ids = [i.id for i in made]
        self.entry_items[uuid] = ids
        return ids

    def _assistant_part(self, pid: str, uuid: str, part: JsonObj) -> Item | None:
        ptype = part.get("type")
        if ptype in ("thinking", "redacted_thinking"):
            text = str(part.get("thinking") or "")
            item = Item(pid, Kind.THINKING, "thinking", uuid, tokens.estimate_text(text))
        elif ptype == "text":
            text = str(part.get("text") or "")
            item = Item(
                pid,
                Kind.ASSISTANT_TEXT,
                short(text, 60) or "text",
                uuid,
                tokens.estimate_text(text),
                content_hash=digest(text),
            )
        elif ptype == "tool_use":
            name = str(part.get("name") or "tool")
            args = part.get("input")
            args_d: dict[str, Any] = args if isinstance(args, dict) else {}
            call_id = str(part.get("id") or pid)
            path_arg = args_d.get("file_path")
            is_read = name in _READ_TOOLS and isinstance(path_arg, str)
            ranges = ",".join(
                f"{k}={args_d[k]}" for k in ("offset", "limit", "pages") if k in args_d
            )
            item = Item(
                pid,
                Kind.TOOL_CALL,
                call_label(name, args_d),
                uuid,
                tokens.estimate_text(name) + tokens.estimate_json(args),
                tool_name=name,
                tool_call_id=call_id,
                file_path=str(path_arg) if is_read else None,
                read_key=f"{path_arg}#{ranges}"
                if is_read and ranges
                else path_arg
                if is_read
                else None,
                writes=tuple(edited_paths(name, args_d)),
                content_hash=digest(name + json.dumps(args, sort_keys=True)),
            )
            self.calls[call_id] = item
        else:
            return None
        self.part_items[pid] = item
        return item

    def _user_items(self, uuid: str, entry: JsonObj, content: object) -> list[Item]:
        if isinstance(content, str):
            kind = Kind.SUMMARY if entry.get("isCompactSummary") else Kind.USER
            return [
                Item(
                    uuid,
                    kind,
                    "compaction summary" if kind == Kind.SUMMARY else short(content, 60) or "user",
                    uuid,
                    tokens.estimate_text(content),
                    content_hash=digest(content),
                )
            ]
        made: list[Item] = []
        texts: list[str] = []
        image_tokens = 0
        for idx, part in enumerate(content if isinstance(content, list) else []):
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_result":
                call_id = str(part.get("tool_use_id") or "")
                call = self.calls.get(call_id)
                text, img = _block_text(part.get("content"))
                made.append(
                    Item(
                        f"{uuid}:{idx}",
                        Kind.TOOL_RESULT,
                        call.label if call else "tool result",
                        uuid,
                        tokens.estimate_text(text) + img,
                        image_tokens=img,
                        tool_name=call.tool_name if call else None,
                        tool_call_id=call_id,
                        file_path=call.file_path if call else None,
                        is_error=bool(part.get("is_error")),
                        content_hash=digest(text),
                    )
                )
            elif part.get("type") == "text":
                texts.append(str(part.get("text") or ""))
            elif part.get("type") == "image":
                text, img = _block_text([part])
                image_tokens += img
        if texts or image_tokens:
            text = "\n".join(texts)
            kind = Kind.SUMMARY if entry.get("isCompactSummary") else Kind.USER
            made.insert(
                0,
                Item(
                    uuid,
                    kind,
                    short(text, 60) or "user",
                    uuid,
                    tokens.estimate_text(text) + image_tokens,
                    image_tokens=image_tokens,
                    content_hash=digest(text),
                ),
            )
        return made

    def _chain(self, parent: object, before: int) -> list[JsonObj]:
        """Entries sent with a request whose first line is at file position ``before``.

        The parent walk alone is not enough: a response's later blocks and the results of
        parallel tool calls hang off side branches. Every line of a response that has a line
        on the walked chain is added, then every result answering one of its tool calls, and
        everything is put back in file order.
        """
        chain: list[JsonObj] = []
        seen: set[str] = set()
        while isinstance(parent, str) and parent in self.by_uuid and parent not in seen:
            seen.add(parent)
            node = self.by_uuid[parent]
            chain.append(node)
            parent = node.get("parentUuid")

        def add(node: JsonObj) -> None:
            uuid = str(node.get("uuid"))
            if uuid not in seen and self.position[uuid] < before:
                seen.add(uuid)
                chain.append(node)

        responses = {
            str(n["message"].get("id"))
            for n in chain
            if n.get("type") == "assistant" and isinstance(n.get("message"), dict)
        }
        for mid in responses:
            for line in self.response_lines.get(mid, []):
                add(line)
        for node in list(chain):
            if node.get("type") != "assistant" or not isinstance(node.get("message"), dict):
                continue
            content = node["message"].get("content")
            for part in content if isinstance(content, list) else []:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    result = self.result_entry.get(str(part.get("id")))
                    if result is not None:
                        add(result)
        chain.sort(key=lambda n: self.position[str(n.get("uuid"))])
        return chain

    def build(self, session: Session, sidechain: bool) -> None:
        responses: dict[str, _Response] = {}
        order: list[_Response] = []
        for e in self.entries:
            if e.get("type") != "assistant" or bool(e.get("isSidechain")) != sidechain:
                continue
            msg = e.get("message")
            if not isinstance(msg, dict):
                continue
            mid = str(msg.get("id") or e.get("uuid"))
            resp = responses.get(mid)
            if resp is None:
                resp = responses[mid] = _Response(mid, e)
                order.append(resp)
            resp.lines.append(e)

        system: Item | None = None
        last_model: str | None = None
        compactions = [
            e
            for e in self.entries
            if e.get("type") == "system" and e.get("subtype") == "compact_boundary"
        ]
        comp_idx = 0
        for resp in order:
            first = resp.first
            msg = first["message"]
            usage, ttl = _usage(resp.lines[-1]["message"].get("usage"))
            for line in resp.lines:
                candidate, _ = _usage(line["message"].get("usage"))
                if candidate.output > usage.output:
                    usage = candidate
            start = parse_time(first.get("timestamp")) or 0.0
            events: list[Event] = []
            after_compaction = False
            while comp_idx < len(compactions):
                comp_ts = parse_time(compactions[comp_idx].get("timestamp")) or 0.0
                if comp_ts > start:
                    break
                meta = compactions[comp_idx].get("compactMetadata")
                session.compactions.append(
                    Compaction(
                        entry_id=str(compactions[comp_idx].get("uuid")),
                        timestamp=comp_ts,
                        first_turn_after=len(session.turns),
                        tokens_before=as_int(meta.get("preTokens")) or None
                        if isinstance(meta, dict)
                        else None,
                        tokens_after=None,
                        method=str(meta.get("trigger", "auto"))
                        if isinstance(meta, dict)
                        else "auto",
                    )
                )
                events.append(Event("compaction", comp_ts))
                after_compaction = True
                comp_idx += 1
            parts: list[str] = []
            for line in resp.lines:
                parts.extend(self._items_for(session, line))
            if usage.prompt + usage.output == 0:
                continue
            model = str(msg.get("model", ""))
            if last_model is not None and model != last_model:
                events.append(Event("model_change", start, model))
            last_model = model
            cost, source = price_usage(model, usage, None, self.prices)
            if system is None:
                system = session.add_item(
                    Item(
                        "system#1",
                        Kind.SYSTEM,
                        "system prompt + tool schemas",
                        "system#1",
                        0,
                        size_unknown=True,
                    )
                )
            context = [system.id]
            for node in self._chain(first.get("parentUuid"), self.position[str(first.get("uuid"))]):
                context.extend(self._items_for(session, node))
            index = len(session.turns)
            part_items = [session.items[p] for p in parts]
            for item in part_items:
                item.born_turn = index
            split_output(part_items, usage.output, usage.thinking)
            completed = parse_time(resp.lines[-1].get("timestamp"))
            session.turns.append(
                Turn(
                    index=index,
                    entry_id=str(first.get("uuid")),
                    timestamp=start,
                    completed_at=completed,
                    model=model,
                    usage=usage,
                    cost=cost,
                    cost_source=source,
                    stop_reason=str(msg.get("stop_reason", "")),
                    context=context,
                    output_items=parts,
                    cache_ttl_s=ttl,
                    events=events,
                    after_compaction=after_compaction,
                )
            )


def parse_claude_code(path: Path, prices: PriceTable | None = None) -> Session:
    table = prices or PriceTable()
    entries: list[JsonObj] = []
    skipped = 0
    recorded_total: float | None = None
    session_id: str | None = None
    agent_id: str | None = None
    for _, obj in read_jsonl(path):
        if obj is None:
            skipped += 1
            continue
        entries.append(obj)
        if obj.get("type") == "cost-state":
            recorded_total = as_float(obj.get("totalCostUSD"))
        if session_id is None and isinstance(obj.get("sessionId"), str):
            session_id = obj["sessionId"]
        if agent_id is None and isinstance(obj.get("agentId"), str):
            agent_id = obj["agentId"]
    if session_id is None:
        raise ValueError(f"{path}: no Claude Code sessionId")
    convo = [e for e in entries if e.get("type") in _ENTRY_TYPES and "uuid" in e]
    all_sidechain = bool(convo) and all(e.get("isSidechain") for e in convo)
    builder = _Builder(path, table, entries)
    if all_sidechain:
        session = Session(
            id=agent_id or path.stem,
            harness="claude-code",
            path=str(path),
            agent=path.stem,
            parent_id=session_id,
            skipped_lines=skipped,
        )
        builder.build(session, sidechain=True)
        return session
    session = Session(
        id=session_id,
        harness="claude-code",
        path=str(path),
        recorded_total=recorded_total,
        skipped_lines=skipped,
    )
    builder.build(session, sidechain=False)
    if any(e.get("isSidechain") for e in convo):
        side = Session(
            id=f"{session_id}:sidechain",
            harness="claude-code",
            path=str(path),
            agent="sidechain",
            parent_id=session_id,
        )
        _Builder(path, table, entries).build(side, sidechain=True)
        session.subagents.append(side)
    return session
