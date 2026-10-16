"""Tests for ffmeta.payload."""

from __future__ import annotations

import datetime as dt
import pathlib

import numpy as np
import pytest

from ffmeta import CorruptMetadataError, NoMetadataError
from ffmeta import payload as p
from ffmeta._layout import IMDI, WORD_SIZE


def env_for(meta, **overrides):
    env = p.make_envelope(
        meta,
        section="extra_constants",
        original_header={"start": IMDI, "length": IMDI},
        original_size=65536,
        original_sha256="0" * 64,
    )
    env.update(overrides)
    return env


def test_magic_word_is_ascii_and_positive():
    assert p.MAGIC == b"FFMETA01"
    assert p.MAGIC_WORD > 0
    assert p.MAGIC_WORD.to_bytes(8, "big") == p.MAGIC


def test_encode_decode_roundtrip(sample_meta):
    env = env_for(sample_meta)
    block = p.encode(env)
    assert len(block) % WORD_SIZE == 0
    assert block[:8] == p.MAGIC
    assert p.decode(block) == env
    assert p.decode(block)["meta"]["tags"][1] == "ünïcode"
    # JSON text is ASCII only (word 2 is a binary length, so it is excluded)
    n = int.from_bytes(block[8:16], "big")
    assert all(b < 0x80 for b in block[16 : 16 + n])


def test_encoded_words_contain_no_nan_or_inf(sample_meta):
    block = p.pad_block(p.encode(env_for(sample_meta)), reserve=100)
    assert p.words_as_floats_are_finite(block)
    assert np.all(np.frombuffer(block, dtype=">i8") >= 0)


def test_padding_and_alignment(sample_meta):
    block = p.encode(env_for(sample_meta))
    padded = p.pad_block(block, reserve=13)
    assert len(padded) >= len(block) + 13 and len(padded) % WORD_SIZE == 0
    assert p.trailing_is_zero(padded, p.declared_length(padded))
    aligned = p.pad_block(block, start_word=4097 + 5, align_sector=True)
    assert ((4097 + 5 - 1) + len(aligned) // WORD_SIZE) % 2048 == 0
    with pytest.raises(ValueError):
        p.pad_block(block, align_sector=True)
    with pytest.raises(ValueError):
        p.pad_block(block, reserve=-1)


def test_unknown_magic_is_no_metadata(sample_meta):
    block = bytearray(p.encode(env_for(sample_meta)))
    block[:8] = b"SOMETHIN"
    with pytest.raises(NoMetadataError):
        p.decode(bytes(block))
    with pytest.raises(NoMetadataError):
        p.declared_length(b"short")


def test_bad_version_is_corrupt(sample_meta):
    block = bytearray(p.encode(env_for(sample_meta)))
    block[6:8] = b"99"
    with pytest.raises(CorruptMetadataError, match="version"):
        p.decode(bytes(block))


def test_bad_length_is_corrupt(sample_meta):
    block = bytearray(p.encode(env_for(sample_meta)))
    block[8:16] = (10**9).to_bytes(8, "big")
    with pytest.raises(CorruptMetadataError, match="exceeds"):
        p.decode(bytes(block))
    block[8:16] = (0).to_bytes(8, "big")
    with pytest.raises(CorruptMetadataError, match="plausible"):
        p.decode(bytes(block))


def test_bad_json_and_schema_are_corrupt(sample_meta):
    block = bytearray(p.encode(env_for(sample_meta)))
    block[16] = ord("x")
    with pytest.raises(CorruptMetadataError, match="JSON"):
        p.decode(bytes(block))
    for key, value in [
        ("ffmeta", 2),
        ("section", "lookup"),
        ("layout", "weird"),
        ("original_sha256", "zz"),
        ("original_size", -1),
        ("meta", []),
        ("original_header", {"start": "a", "length": 1}),
    ]:
        env = env_for(sample_meta)
        env[key] = value
        with pytest.raises(CorruptMetadataError):
            p.validate_envelope(env)
    env = env_for(sample_meta)
    del env["created"]
    with pytest.raises(CorruptMetadataError, match="missing"):
        p.validate_envelope(env)
    with pytest.raises(CorruptMetadataError):
        p.validate_envelope([1, 2])


def test_to_jsonable_conversions():
    value = p.to_jsonable(
        {
            "i": np.int64(3),
            "f": np.float32(1.5),
            "b": np.bool_(True),
            "arr": np.arange(3),
            "dt": dt.datetime(2026, 10, 16, 1, 0),
            "d": dt.date(2026, 1, 2),
            "p": pathlib.Path("/tmp/x"),
        }
    )
    assert value == {
        "i": 3,
        "f": 1.5,
        "b": True,
        "arr": [0, 1, 2],
        "dt": "2026-10-16T01:00:00",
        "d": "2026-01-02",
        "p": "/tmp/x",
    }
    with pytest.raises(TypeError):
        p.to_jsonable({"obj": object()})
    with pytest.raises(ValueError):
        p.to_jsonable({"nan": float("nan")})
    with pytest.raises(TypeError):
        p.make_envelope(
            ["not", "a", "dict"],
            section="extra_constants",  # type: ignore[arg-type]
            original_header={"start": 1, "length": 1},
            original_size=0,
            original_sha256=None,
        )


def test_dotted_key_helpers():
    d = {}
    p.set_dotted(d, "a.b.c", 1)
    p.set_dotted(d, "a.x", [1, 2])
    assert d == {"a": {"b": {"c": 1}, "x": [1, 2]}}
    assert p.get_dotted(d, "a.b.c") == 1
    assert p.get_dotted(d, "a.x.1") == 2
    with pytest.raises(KeyError):
        p.get_dotted(d, "a.nope")
    assert p.unset_dotted(d, "a.b.c") is True
    assert p.unset_dotted(d, "a.b.c") is False
    assert p.unset_dotted(d, "zz.y") is False
    assert d == {"a": {"b": {}, "x": [1, 2]}}
    with pytest.raises(ValueError):
        p.split_key("a..b")
    merged = p.deep_merge({"a": {"b": 1, "c": 2}, "k": 1}, {"a": {"b": 9}, "n": 3})
    assert merged == {"a": {"b": 9, "c": 2}, "k": 1, "n": 3}


def test_parse_set_value_and_apply_sets():
    assert p.parse_set_value("3") == 3
    assert p.parse_set_value("2.5") == 2.5
    assert p.parse_set_value("true") is True
    assert p.parse_set_value("null") is None
    assert p.parse_set_value('{"a": 1}') == {"a": 1}
    assert p.parse_set_value("hello world") == "hello world"
    assert p.parse_set_value("NaN") == "NaN"
    meta = {}
    p.apply_sets(meta, ["a.b=3", "name=Some Name", "list=[1,2]"])
    assert meta == {"a": {"b": 3}, "name": "Some Name", "list": [1, 2]}
    with pytest.raises(ValueError):
        p.apply_sets(meta, ["novalue"])
