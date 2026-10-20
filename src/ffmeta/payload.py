"""Encoding and decoding of the ffmeta payload block and its JSON envelope.

Block layout (big-endian 64-bit words):

* word 1: the 8 ASCII bytes ``FFMETA01`` (magic ``FFMETA`` + format version ``01``);
* word 2: N, the length in bytes of the JSON text;
* words 3..: the JSON text, ASCII only, zero-padded to whole words, optionally
  followed by more zero words (reserve / sector alignment).

Because every payload byte is ASCII (< 0x80) the sign bit of every word is
clear and no word can be read as NaN or Inf by a reader that interprets the
section as IEEE doubles.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import pathlib
import re
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

import numpy as np

import ffmeta

from ._layout import METADATA_SECTION, SECTOR_WORDS, WORD_SIZE
from .errors import CorruptMetadataError, NoMetadataError

MAGIC_PREFIX = b"FFMETA"
FORMAT_VERSION = 1
MAGIC = MAGIC_PREFIX + b"%02d" % FORMAT_VERSION
HEADER_WORDS = 2
HEADER_BYTES = HEADER_WORDS * WORD_SIZE
ENVELOPE_VERSION = 1
LAYOUTS = ("trailing",)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_KEYS = (
    "ffmeta",
    "tool_version",
    "section",
    "created",
    "modified",
    "original_header",
    "original_size",
    "original_sha256",
    "meta",
)


def utcnow_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def to_jsonable(obj: Any) -> Any:
    """Convert ``obj`` to plain JSON types (dict/list/str/int/float/bool/None).

    Numpy scalars and arrays, datetimes and paths are converted; anything else
    raises :class:`TypeError`.  NaN and infinities raise :class:`ValueError`.
    """
    text = json.dumps(obj, default=_json_default, allow_nan=False, ensure_ascii=True)
    return json.loads(text)


def dumps(obj: Any) -> str:
    """Canonical JSON text: ASCII only, sorted keys, compact separators, no NaN."""
    return json.dumps(
        obj,
        default=_json_default,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def make_envelope(
    meta: Mapping[str, Any],
    *,
    original_header: Mapping[str, int],
    original_size: int,
    original_sha256: str | None,
    created: str | None = None,
) -> dict[str, Any]:
    """Build a new envelope around user metadata."""
    if not isinstance(meta, Mapping):
        raise TypeError("metadata must be a JSON object (mapping)")
    now = created or utcnow_iso()
    env = {
        "ffmeta": ENVELOPE_VERSION,
        "tool_version": ffmeta.__version__,
        "section": METADATA_SECTION,
        "created": now,
        "modified": now,
        "original_header": {
            "start": int(original_header["start"]),
            "length": int(original_header["length"]),
        },
        "original_size": int(original_size),
        "original_sha256": original_sha256,
        "meta": to_jsonable(dict(meta)),
    }
    validate_envelope(env)
    return env


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_envelope(env: Any) -> None:
    """Raise :class:`CorruptMetadataError` unless ``env`` is a well-formed envelope."""

    def fail(msg: str) -> CorruptMetadataError:
        return CorruptMetadataError(f"invalid ffmeta envelope: {msg}")

    if not isinstance(env, dict):
        raise fail("not a JSON object")
    missing = [k for k in _REQUIRED_KEYS if k not in env]
    if missing:
        raise fail("missing keys " + ", ".join(missing))
    if env["ffmeta"] != ENVELOPE_VERSION or not _is_int(env["ffmeta"]):
        raise fail(f"unsupported envelope version {env['ffmeta']!r}")
    if not isinstance(env["tool_version"], str):
        raise fail("tool_version is not a string")
    if env["section"] != METADATA_SECTION:
        raise fail(f"unknown section {env['section']!r}")
    for key in ("created", "modified"):
        if not isinstance(env[key], str):
            raise fail(f"{key} is not a string")
    hdr = env["original_header"]
    if not (isinstance(hdr, dict) and _is_int(hdr.get("start")) and _is_int(hdr.get("length"))):
        raise fail("original_header must hold integer start and length")
    if not _is_int(env["original_size"]) or env["original_size"] < 0:
        raise fail("original_size must be a non-negative integer")
    sha = env["original_sha256"]
    if sha is not None and not (isinstance(sha, str) and _HEX64.match(sha)):
        raise fail("original_sha256 must be null or a 64-character hex digest")
    if not isinstance(env["meta"], dict):
        raise fail("meta is not a JSON object")


def encode(env: Mapping[str, Any]) -> bytes:
    """Encode an envelope as a block of whole words (magic, length, padded JSON)."""
    validate_envelope(dict(env))
    data = dumps(env).encode("ascii")
    pad = (-len(data)) % WORD_SIZE
    return MAGIC + len(data).to_bytes(WORD_SIZE, "big", signed=True) + data + b"\0" * pad


def pad_block(
    block: bytes, *, reserve: int = 0, start_word: int | None = None, align_sector: bool = False
) -> bytes:
    """Append zero words for ``reserve`` bytes and/or to end on a 2048-word sector boundary."""
    if reserve < 0:
        raise ValueError("reserve must be >= 0")
    total = len(block) + reserve
    total += (-total) % WORD_SIZE
    if align_sector:
        if start_word is None:
            raise ValueError("start_word is required for sector alignment")
        end_words = (start_word - 1) + total // WORD_SIZE
        total += ((-end_words) % SECTOR_WORDS) * WORD_SIZE
    return block + b"\0" * (total - len(block))


def payload_words(nbytes_json: int) -> int:
    """Number of words used by a block whose JSON text has ``nbytes_json`` bytes."""
    return HEADER_WORDS + -(-nbytes_json // WORD_SIZE)


def declared_length(head: bytes) -> int:
    """Check the magic word of a block and return the declared JSON length in bytes."""
    if len(head) < HEADER_BYTES:
        raise NoMetadataError("section is too short to hold an ffmeta block")
    magic = head[:WORD_SIZE]
    if not magic.startswith(MAGIC_PREFIX):
        raise NoMetadataError("section does not start with the ffmeta magic word")
    if magic != MAGIC:
        raise CorruptMetadataError(
            f"unsupported ffmeta format version {magic[len(MAGIC_PREFIX) :]!r} (expected {MAGIC!r})"
        )
    length = int.from_bytes(head[WORD_SIZE:HEADER_BYTES], "big", signed=True)
    if length < 2:
        raise CorruptMetadataError(f"declared payload length {length} is not plausible")
    return length


def decode(block: bytes) -> dict[str, Any]:
    """Decode a block (at least magic + length + JSON) into a validated envelope."""
    length = declared_length(block)
    if len(block) < HEADER_BYTES + length:
        available = len(block) - HEADER_BYTES
        raise CorruptMetadataError(
            f"declared payload length {length} exceeds the available {available} bytes"
        )
    text = block[HEADER_BYTES : HEADER_BYTES + length]
    try:
        env: dict[str, Any] = json.loads(text.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptMetadataError(f"payload is not valid ASCII JSON: {exc}") from exc
    validate_envelope(env)
    return env


def trailing_is_zero(block: bytes, length: int) -> bool:
    """True when every byte after the JSON text in ``block`` is zero."""
    return not any(block[HEADER_BYTES + length :])


def words_as_floats_are_finite(block: bytes) -> bool:
    """True when no word of the block reads as NaN/Inf if interpreted as IEEE doubles."""
    arr = np.frombuffer(block, dtype=">f8")
    return bool(np.all(np.isfinite(arr)))


# --- dotted-key helpers -----------------------------------------------------


def split_key(key: str) -> list[str]:
    parts = key.split(".") if key else []
    if not parts or any(p == "" for p in parts):
        raise ValueError(f"invalid dotted key {key!r}")
    return parts


def get_dotted(obj: Any, key: str) -> Any:
    """Return ``obj[a][b]...`` for ``key == "a.b..."``; raises :class:`KeyError`."""
    cur = obj
    for part in split_key(key):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        elif (
            isinstance(cur, list)
            and part.lstrip("-").isdigit()
            and -len(cur) <= int(part) < len(cur)
        ):
            cur = cur[int(part)]
        else:
            raise KeyError(key)
    return cur


def set_dotted(obj: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set ``obj[a][b]... = value`` creating intermediate objects."""
    parts = split_key(key)
    cur: MutableMapping[str, Any] = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def unset_dotted(obj: MutableMapping[str, Any], key: str) -> bool:
    """Delete ``obj[a][b]...``; returns False (without error) if the key is missing."""
    parts = split_key(key)
    cur: Any = obj
    for part in parts[:-1]:
        if not isinstance(cur, Mapping) or part not in cur:
            return False
        cur = cur[part]
    if isinstance(cur, MutableMapping) and parts[-1] in cur:
        del cur[parts[-1]]
        return True
    return False


def deep_merge(base: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``changes`` merged recursively (changes win)."""
    out: dict[str, Any] = dict(base)
    for key, value in changes.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_set_value(text: str) -> Any:
    """Parse ``--set`` values: JSON if possible, otherwise the string itself."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(value, float) and not math.isfinite(value):
        return text
    return value


def apply_sets(meta: MutableMapping[str, Any], pairs: Iterable[str]) -> None:
    """Apply ``KEY=VALUE`` assignments with dotted keys to ``meta``."""
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        set_dotted(meta, key.strip(), parse_set_value(value))
