"""Tests for add / read / edit / remove (ffmeta.core)."""

from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pytest

import ffmeta
from ffmeta import core as core_module
from ffmeta import payload as p
from ffmeta._layout import COMPONENT_BY_NAME, METADATA_SECTION, WORD_SIZE, Region, inspect
from ffmeta.core import _Transaction, plan_add, plan_edit, plan_remove
from fixtures import IMDI, byteswap_file, make_um_file, read_word, to_32bit_file, write_word


def pointer_words(section):
    comp = COMPONENT_BY_NAME[section]
    return comp.start_word, comp.length_word


def test_add_read_remove_roundtrip(um_file, sample_meta):
    original = um_file.read_bytes()
    size_words = len(original) // WORD_SIZE
    section = "temp_historyfile"
    start_w, len_w = pointer_words(section)

    out = ffmeta.add(um_file, sample_meta)
    assert out == um_file
    assert ffmeta.has_metadata(um_file)
    assert ffmeta.read(um_file) == sample_meta
    assert read_word(um_file, 130) == IMDI

    modified = um_file.read_bytes()
    assert len(modified) > len(original)
    # only the two pointer words changed in the original span
    a = np.frombuffer(original, dtype=">i8")
    b = np.frombuffer(modified[: len(original)], dtype=">i8")
    changed = set((np.nonzero(a != b)[0] + 1).tolist())
    assert changed == {start_w, len_w}
    assert read_word(um_file, start_w) == size_words + 1
    assert read_word(um_file, len_w) == (len(modified) - len(original)) // WORD_SIZE

    env = ffmeta.read(um_file, envelope=True)
    assert env["section"] == section
    assert env["original_header"] == {"start": IMDI, "length": IMDI}
    assert env["original_size"] == len(original)
    assert env["original_sha256"] == hashlib.sha256(original).hexdigest()
    assert env["tool_version"] == ffmeta.__version__
    assert env["created"] == env["modified"]

    ffmeta.remove(um_file)
    assert um_file.read_bytes() == original
    assert not ffmeta.has_metadata(um_file)


def test_add_with_output_leaves_source_untouched(ff_file, tmp_path, sample_meta):
    original = ff_file.read_bytes()
    out = ffmeta.add(ff_file, sample_meta, output=tmp_path / "copy.ff")
    assert out == tmp_path / "copy.ff"
    assert ff_file.read_bytes() == original
    assert ffmeta.read(out) == sample_meta
    out2 = ffmeta.edit(out, {"x": 1}, output=tmp_path / "copy2.ff")
    assert ffmeta.read(out) == sample_meta
    assert ffmeta.read(out2)["x"] == 1
    out3 = ffmeta.remove(out2, output=tmp_path / "copy3.ff")
    assert out3.read_bytes() == original
    assert ffmeta.read(out2)["x"] == 1


def test_output_same_path_is_in_place(ff_file):
    out = ffmeta.add(ff_file, {"a": 1}, output=ff_file)
    assert out == ff_file
    assert ffmeta.read(ff_file) == {"a": 1}


def test_add_output_removed_on_failure(ff_file, tmp_path):
    ffmeta.add(ff_file, {"a": 1})
    target = tmp_path / "fail.ff"
    with pytest.raises(ffmeta.MetadataExistsError):
        ffmeta.add(ff_file, {"a": 2}, output=target)
    assert not target.exists()


def test_output_cleanup_suppresses_unlink_error(ff_file, tmp_path, monkeypatch):
    target = tmp_path / "failed.ff"

    def fail_unlink(self, *args, **kwargs):
        raise OSError("cannot unlink")

    monkeypatch.setattr(type(target), "unlink", fail_unlink)
    with pytest.raises(TypeError):
        ffmeta.add(ff_file, {"bad": object()}, output=target)


def test_no_hash(ff_file):
    ffmeta.add(ff_file, {"a": 1}, with_hash=False)
    assert ffmeta.read(ff_file, envelope=True)["original_sha256"] is None


def test_read_without_metadata(um_file):
    with pytest.raises(ffmeta.NoMetadataError):
        ffmeta.read(um_file)
    assert not ffmeta.has_metadata(um_file)
    with pytest.raises(ffmeta.NoMetadataError):
        ffmeta.remove(um_file)
    with pytest.raises(ffmeta.NoMetadataError):
        ffmeta.edit(um_file, {"a": 1})


def test_metadata_section_is_fixed(ff_file):
    ffmeta.add(ff_file, {"a": 1})
    assert ffmeta.read(ff_file) == {"a": 1}
    with pytest.raises(TypeError):
        ffmeta.add(ff_file, {"a": 1}, section="extra_constants")
    with pytest.raises(TypeError):
        ffmeta.read(ff_file, section="extra_constants")


def test_add_requires_mapping(ff_file):
    with pytest.raises(TypeError):
        ffmeta.add(ff_file, ["list"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ffmeta.add(ff_file, {"obj": object()})
    assert not ffmeta.has_metadata(ff_file)


def test_numpy_values_are_converted(ff_file):
    ffmeta.add(ff_file, {"n": np.int32(4), "arr": np.array([1.5, 2.5])})
    assert ffmeta.read(ff_file) == {"n": 4, "arr": [1.5, 2.5]}


# --- edit -------------------------------------------------------------------


def test_edit_keep_size_in_place(um_file):
    section = "temp_historyfile"
    ffmeta.add(um_file, {"a": 1, "b": {"c": 2}} | {"pad": "x" * 500})
    size = um_file.stat().st_size
    start, length = (read_word(um_file, w) for w in pointer_words(section))
    ffmeta.edit(um_file, {"b": {"d": 3}}, unset=["a", "pad", "missing.key"])
    assert ffmeta.read(um_file) == {"b": {"c": 2, "d": 3}}
    assert um_file.stat().st_size == size
    assert (
        read_word(um_file, pointer_words(section)[0]),
        read_word(um_file, pointer_words(section)[1]),
    ) == (start, length)
    env = ffmeta.read(um_file, envelope=True)
    assert env["created"] <= env["modified"]


def test_edit_replace(ff_file):
    ffmeta.add(ff_file, {"a": 1, "b": 2})
    ffmeta.edit(ff_file, {"c": 3}, replace={"z": 0}, unset=["z"])
    assert ffmeta.read(ff_file) == {"c": 3}
    with pytest.raises(TypeError):
        ffmeta.edit(ff_file, replace=[1])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ffmeta.edit(ff_file, changes=[1])  # type: ignore[arg-type]


def test_edit_grow_trailing_block(um_file):
    section = "temp_historyfile"
    original = um_file.read_bytes()
    ffmeta.add(um_file, {"a": 1})
    size1 = um_file.stat().st_size
    big = {"big": "x" * 5000}
    ffmeta.edit(um_file, big)
    assert um_file.stat().st_size > size1
    assert ffmeta.read(um_file) == {"a": 1, **big}
    start_w, len_w = pointer_words(section)
    assert read_word(um_file, start_w) == len(original) // WORD_SIZE + 1
    assert read_word(um_file, len_w) == (um_file.stat().st_size - len(original)) // WORD_SIZE
    ffmeta.remove(um_file)
    assert um_file.read_bytes() == original


def test_edit_shrink_keeps_block_size(ff_file):
    ffmeta.add(ff_file, {"a": 1, "pad": "y" * 3000})
    size = ff_file.stat().st_size
    ffmeta.edit(ff_file, unset=["pad"])
    assert ff_file.stat().st_size == size
    assert ffmeta.read(ff_file) == {"a": 1}


def test_align_sector(um_file):
    ffmeta.add(um_file, {"a": 1}, align_sector=True)
    assert (um_file.stat().st_size // WORD_SIZE) % 2048 == 0
    ffmeta.remove(um_file)


def fake_original_size(path):
    """Change the stored original size in place, as if the file was rewritten after add."""
    size = ffmeta.read(path, envelope=True)["original_size"]
    old, new = f'"original_size":{size}'.encode(), f'"original_size":{size + 1}'.encode()
    assert len(old) == len(new)
    path.write_bytes(path.read_bytes().replace(old, new))


def test_remove_warns_when_file_was_rewritten(ff_file):
    ffmeta.add(ff_file, {"a": 1})
    fake_original_size(ff_file)
    assert "not byte-identical" in core_module.plan_remove(ff_file).describe()
    with pytest.warns(ffmeta.FfmetaWarning, match="not byte-identical"):
        ffmeta.remove(ff_file)
    assert not ffmeta.has_metadata(ff_file)


# --- refusals ---------------------------------------------------------------


def test_add_refuses_existing_metadata(ff_file):
    ffmeta.add(ff_file, {"a": 1})
    before = ff_file.read_bytes()
    with pytest.raises(ffmeta.MetadataExistsError):
        ffmeta.add(ff_file, {"b": 2})
    with pytest.raises(ffmeta.MetadataExistsError):
        ffmeta.add(ff_file, {"b": 2})
    assert ff_file.read_bytes() == before


def test_add_refuses_section_in_use(tmp_path, section):
    other = np.arange(10, dtype=">i8").tobytes()
    path = make_um_file(tmp_path / "inuse.ff", **{section: other})
    before = path.read_bytes()
    with pytest.raises(ffmeta.SectionInUseError):
        ffmeta.add(path, {"a": 1})
    assert path.read_bytes() == before


def test_add_refuses_damaged_block(ff_file):
    ffmeta.add(ff_file, {"a": 1})
    start = read_word(ff_file, 135)
    write_word(ff_file, start + 1, 0)  # bad length word
    with pytest.raises(ffmeta.MetadataExistsError, match="damaged"):
        ffmeta.add(ff_file, {"b": 2})
    with pytest.raises(ffmeta.CorruptMetadataError):
        ffmeta.read(ff_file)


def test_read_block_size_and_region_guards():
    with pytest.raises(ffmeta.NoMetadataError, match="too short"):
        core_module._read_block(io.BytesIO(b""), Region(METADATA_SECTION, 1, 1))

    header = p.MAGIC + (1000).to_bytes(8, "big", signed=True)
    with pytest.raises(ffmeta.CorruptMetadataError, match="holds only"):
        core_module._read_block(io.BytesIO(header), Region(METADATA_SECTION, 1, 2))

    huge_length = (1 << 26) + 1
    header = p.MAGIC + huge_length.to_bytes(8, "big", signed=True)
    with pytest.raises(ffmeta.CorruptMetadataError, match="implausibly large"):
        core_module._read_block(io.BytesIO(header), Region(METADATA_SECTION, 1, 9_000_000))


def test_locate_rejects_section_beyond_eof(ff_file):
    write_word(ff_file, 135, ff_file.stat().st_size // WORD_SIZE + 100)
    with open(ff_file, "rb") as fh:
        info = inspect(fh, strict=False)
        with pytest.raises(ffmeta.FormatError, match="extends past the end"):
            core_module._locate(fh, info)


def test_add_refuses_wrong_formats(ff_file, tmp_path):
    le = byteswap_file(ff_file, tmp_path / "le.ff")
    with pytest.raises(ffmeta.FormatError, match="little-endian"):
        ffmeta.add(le, {"a": 1})
    be32 = to_32bit_file(ff_file, tmp_path / "be32.ff")
    with pytest.raises(ffmeta.FormatError, match="32-bit"):
        ffmeta.add(be32, {"a": 1})
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\0" * 4096)
    with pytest.raises(ffmeta.FormatError):
        ffmeta.add(junk, {"a": 1})


def test_add_refuses_truncated_file_unless_forced(ff_file):
    data = ff_file.read_bytes()
    ff_file.write_bytes(data[: -2048 * 8])
    before = ff_file.read_bytes()
    with pytest.raises(ffmeta.FormatError, match="truncated"):
        ffmeta.add(ff_file, {"a": 1})
    assert ff_file.read_bytes() == before
    ffmeta.add(ff_file, {"a": 1}, force=True)
    assert ffmeta.read(ff_file) == {"a": 1}


def test_add_refuses_inconsistent_pointers(ff_file):
    write_word(ff_file, 150, 10**7)
    with pytest.raises(ffmeta.FormatError):
        ffmeta.add(ff_file, {"a": 1})


# --- rollback ---------------------------------------------------------------


def test_rollback_on_failure_mid_write(ff_file, monkeypatch):
    before = ff_file.read_bytes()
    calls = {"n": 0}
    real_write = _Transaction.write

    def flaky(self, offset, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk on fire")
        real_write(self, offset, data)

    monkeypatch.setattr(_Transaction, "write", flaky)
    with pytest.raises(OSError, match="fire"):
        ffmeta.add(ff_file, {"a": 1})
    assert ff_file.read_bytes() == before


def test_rollback_on_fsync_failure(ff_file, monkeypatch):
    before = ff_file.read_bytes()
    real_fsync = os.fsync
    state = {"fail": True}

    def flaky(fd):
        if state["fail"]:
            state["fail"] = False
            raise OSError("fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", flaky)
    with pytest.raises(OSError, match="fsync"):
        ffmeta.add(ff_file, {"a": 1})
    assert ff_file.read_bytes() == before


def test_rollback_on_self_check_failure(ff_file, monkeypatch):
    before = ff_file.read_bytes()

    def broken_locate(*args, **kwargs):
        raise ffmeta.NoMetadataError("simulated")

    monkeypatch.setattr("ffmeta.core._locate", broken_locate)
    with pytest.raises(ffmeta.FfmetaError):
        ffmeta.add(ff_file, {"a": 1})
    assert ff_file.read_bytes() == before


def test_remove_rollback_on_failure(ff_file, monkeypatch):
    ffmeta.add(ff_file, {"a": 1})
    before = ff_file.read_bytes()
    real_truncate = _Transaction.truncate

    def flaky(self, new_size):
        real_truncate(self, new_size)
        raise OSError("boom")

    monkeypatch.setattr(_Transaction, "truncate", flaky)
    with pytest.raises(OSError, match="boom"):
        ffmeta.remove(ff_file)
    assert ff_file.read_bytes() == before
    assert ffmeta.read(ff_file) == {"a": 1}


def test_removed_self_check_rejects_inconsistent_results(ff_file, tmp_path):
    ffmeta.add(ff_file, {"a": 1})
    plan = plan_remove(ff_file)
    check = core_module._check_removed(plan)
    with pytest.raises(ffmeta.FfmetaError, match="file size"):
        check(ff_file)

    clean = tmp_path / "clean.ff"
    clean.write_bytes(ff_file.read_bytes())
    ffmeta.remove(clean)
    write_word(clean, 135, read_word(clean, 100))
    with pytest.raises(ffmeta.FfmetaError, match="pointer words"):
        check(clean)

    still_present = tmp_path / "still-present.ff"
    still_present.write_bytes(ff_file.read_bytes())
    plan.size_after = still_present.stat().st_size
    plan.envelope["original_header"] = {
        "start": read_word(still_present, 135),
        "length": read_word(still_present, 136),
    }
    with pytest.raises(ffmeta.FfmetaError, match="metadata still present"):
        core_module._check_removed(plan)(still_present)


# --- block not at the end of the file ---------------------------------------


def canonical_block(meta, section="temp_historyfile"):
    env = p.make_envelope(
        meta,
        original_header={"start": IMDI, "length": IMDI},
        original_size=0,
        original_sha256=None,
    )
    return p.pad_block(p.encode(env))


@pytest.fixture
def middle_file(tmp_path, section):
    """A file whose metadata block sits in its canonical position before LOOKUP."""
    block = canonical_block({"where": "middle"}, section=section)
    return make_um_file(tmp_path / "middle.ff", **{section: block})


def test_middle_read(middle_file, section):
    assert ffmeta.read(middle_file) == {"where": "middle"}
    info = inspect(middle_file)
    region = info.flh.component_region(section)
    assert region is not None and not info.is_trailing(region)


def test_middle_edit_and_remove_refused(middle_file):
    before = middle_file.read_bytes()
    with pytest.raises(ffmeta.FormatError, match="not at the end"):
        ffmeta.edit(middle_file, {"x": 1})
    with pytest.raises(ffmeta.FormatError, match="not at the end"):
        ffmeta.remove(middle_file)
    assert middle_file.read_bytes() == before


# --- plans / dry run ----------------------------------------------------------


def test_plans_do_not_write(ff_file):
    before = ff_file.read_bytes()
    plan = plan_add(ff_file, {"a": 1})
    assert plan.operation == "add" and "append" in plan.describe()
    assert ff_file.read_bytes() == before
    ffmeta.add(ff_file, {"a": 1})
    after = ff_file.read_bytes()
    text = plan_edit(ff_file, {"b": 2}).describe()
    assert "edit" in text and ff_file.read_bytes() == after
    text = plan_remove(ff_file).describe()
    assert "truncate" in text and ff_file.read_bytes() == after


def test_add_self_check_rejects_wrong_size_and_envelope(ff_file):
    plan = plan_add(ff_file, {"a": 1})
    check = core_module._check_envelope_written(plan)
    with pytest.raises(ffmeta.FfmetaError, match="file size"):
        check(ff_file)

    ffmeta.add(ff_file, {"a": 1})
    plan.envelope["meta"] = {"wrong": True}
    with pytest.raises(ffmeta.FfmetaError, match="read back differs"):
        check(ff_file)


# --- from_mule ---------------------------------------------------------------


class _FakeComponent:
    def __init__(self, words):
        self.raw = np.array([None] + list(words), dtype=object)


class _FakeUMFile:
    def __init__(self, **components):
        self.extra_constants = components.get("extra_constants")
        self.temp_historyfile = components.get("temp_historyfile")


def test_from_mule_with_fake_object(ff_file, sample_meta, section):
    ffmeta.add(ff_file, sample_meta)
    info = inspect(ff_file)
    region = info.flh.component_region(section)
    raw = ff_file.read_bytes()[region.offset : region.offset + region.nbytes]
    words = np.frombuffer(raw, dtype=">i8").tolist()
    fake = _FakeUMFile(**{section: _FakeComponent(words)})
    assert ffmeta.from_mule(fake) == sample_meta
    with pytest.raises(TypeError):
        ffmeta.from_mule(fake, section)
    other = _FakeUMFile(extra_constants=_FakeComponent([1, 2, 3]))
    with pytest.raises(ffmeta.NoMetadataError, match="PR #41"):
        ffmeta.from_mule(other)
    with pytest.raises(ffmeta.NoMetadataError):
        ffmeta.from_mule(_FakeUMFile())


@pytest.mark.parametrize("bad_words", [[1, 2], [None, "bad", 3], [None, 1 << 80, 3]])
def test_from_mule_skips_invalid_component_values(bad_words):
    fake = _FakeUMFile(temp_historyfile=_FakeComponent(bad_words))
    with pytest.raises(ffmeta.NoMetadataError):
        ffmeta.from_mule(fake)


def test_from_mule_skips_short_raw_component():
    fake = _FakeUMFile(temp_historyfile=_FakeComponent([None]))
    with pytest.raises(ffmeta.NoMetadataError):
        ffmeta.from_mule(fake)
