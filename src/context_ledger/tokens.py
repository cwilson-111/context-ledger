"""Token-count estimator, used only where recorded usage cannot pin an item down.

Claude's tokenizer is not public, and OpenAI's tiktoken encodings would both be the wrong
tokenizer and need a network download, so the estimate is a plain character ratio. The
ratio was fitted (ratio of sums) on prompt-size deltas that isolate exactly one new
text-only tool result, using half of the reference sessions, and checked on the other half;
``scripts/fit_estimator.py`` reproduces the fit on your own logs and ``docs/FINDINGS.md``
reports the held-out error. Agent tool output (numbered source lines, JSON, paths) is far
denser than prose, which is why the common chars/4 rule undercounts it by ~45%.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import struct

CHARS_PER_TOKEN = 2.24

# Anthropic bills an image at about width*height/750 tokens after scaling the long edge to
# at most 1568 px; unreadable images fall back to the cap.
_IMAGE_MAX_EDGE = 1568
_IMAGE_TOKEN_CAP = 1600
_IMAGE_PIXELS_PER_TOKEN = 750


def estimate_text(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_json(value: object) -> int:
    return estimate_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Width and height of a PNG, GIF, JPEG or WebP image from its header, else None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        w, h = struct.unpack("<HH", data[6:10])
        return int(w), int(h)
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8X":
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return w, h
        if chunk == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            return int(w) & 0x3FFF, int(h) & 0x3FFF
        if chunk == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        return None
    if data.startswith(b"\xff\xd8"):
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            (length,) = struct.unpack(">H", data[i + 2 : i + 4])
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return int(w), int(h)
            i += 2 + length
    return None


def estimate_image(b64: str | None) -> int:
    if not b64:
        return _IMAGE_TOKEN_CAP
    try:
        head = base64.b64decode(b64[:65536] + "=" * (-len(b64[:65536]) % 4), validate=False)
    except (binascii.Error, ValueError):
        return _IMAGE_TOKEN_CAP
    dims = image_dimensions(head)
    if dims is None or dims[0] <= 0 or dims[1] <= 0:
        return _IMAGE_TOKEN_CAP
    w, h = dims
    scale = min(1.0, _IMAGE_MAX_EDGE / max(w, h))
    tokens = math.ceil(w * scale * h * scale / _IMAGE_PIXELS_PER_TOKEN)
    return min(tokens, _IMAGE_TOKEN_CAP)
