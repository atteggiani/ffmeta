"""Tests for ffmeta.verify."""

from __future__ import annotations

import shutil

import pytest

import ffmeta
from ffmeta import payload as p
from ffmeta._layout import WORD_SIZE
from ffmeta._verify import FAIL, PASS, SKIP, WARN, Report, mule_seeks_by_version, parse_version
from fixtures import IMDI, make_um_file, read_word, write_bytes_at, write_word


def by_name(report):
    return {c.name: c for c in report.checks}


@pytest.fixture
def pair(tmp_path, kind, section):
    """(original, with-metadata) files of the parametrised kind and section."""
    original = make_um_file(tmp_path / "orig.ff", kind=kind)
    target = tmp_path / "meta.ff"
    shutil.copyfile(original, target)
    ffmeta.add(target, {"a": 1, "nested": {"b": [1, 2]}}, section=section)
    return original, target


def test_all_pass(pair):
    original, target = pair
    before = target.read_bytes()
    report = ffmeta.verify(target, original=original)
    checks = by_name(report)
    assert report.ok, str(report)
    for name in ("header", "section", "overlap", "payload", "hash", "diff", "reversible"):
        assert checks[name].status == PASS, checks[name]
    assert "last component" in checks["section"].message
    assert report.count(PASS) == 7 and report.count(FAIL) == 0
    assert target.read_bytes() == before  # read-only
    assert original.exists()
    assert str(report).endswith("7 passed, 0 warnings, 0 failed, 0 skipped")


def test_report_api():
    report = Report()
    report.add(PASS, "a", "ok")
    report.add(WARN, "b", "hmm")
    assert report.ok and len(report) == 2
    report.add(FAIL, "c", "bad")
    assert not report.ok
    assert report.count(FAIL) == 1
    assert report.to_list()[2] == {"status": "FAIL", "name": "c", "message": "bad"}
    assert "[FAIL] c: bad" in str(report)
    with pytest.raises(ValueError):
        report.add("MAYBE", "d", "")


def test_no_metadata_fails_section(ff_file):
    report = ffmeta.verify(ff_file)
    checks = by_name(report)
    assert checks["header"].status == PASS
    assert checks["section"].status == FAIL
    assert checks["payload"].status == SKIP
    assert not report.ok


def test_not_a_um_file(tmp_path):
    junk = tmp_path / "junk"
    junk.write_bytes(b"\0" * 4096)
    report = ffmeta.verify(junk)
    assert by_name(report)["header"].status == FAIL
    assert by_name(report)["section"].status == SKIP
    report = ffmeta.verify(tmp_path / "missing")
    assert by_name(report)["header"].status == FAIL
    (tmp_path / "odd").write_bytes(b"\0" * 4097)
    assert by_name(ffmeta.verify(tmp_path / "odd"))["header"].status == FAIL


def test_corrupted_payload(pair):
    original, target = pair
    start = read_word(target, 130) if read_word(target, 130) > 0 else read_word(target, 135)
    write_bytes_at(target, (start - 1 + 2) * WORD_SIZE, b"xxxxxxxx")
    report = ffmeta.verify(target, original=original)
    checks = by_name(report)
    assert checks["section"].status == FAIL
    assert "cannot be decoded" in checks["section"].message
    assert not report.ok


def test_bad_version_payload(pair):
    _, target = pair
    start = read_word(target, 130) if read_word(target, 130) > 0 else read_word(target, 135)
    write_bytes_at(target, (start - 1) * WORD_SIZE + 6, b"77")
    report = ffmeta.verify(target)
    assert by_name(report)["section"].status == FAIL


def test_trailing_nonzero_words_warn(ff_file):
    ffmeta.add(ff_file, {"a": 1}, reserve=64)
    start, length = read_word(ff_file, 130), read_word(ff_file, 131)
    write_word(ff_file, start + length - 1, 42)
    checks = by_name(ffmeta.verify(ff_file))
    assert checks["payload"].status == WARN
    assert "not all zero" in checks["payload"].message


def test_overlap_detected(pair):
    original, target = pair
    # point the first field's LBEGIN into the metadata block
    lookup_start = read_word(target, 150)
    block_start = read_word(target, 130) if read_word(target, 130) > 0 else read_word(target, 135)
    write_word(target, lookup_start + 29 - 1, block_start - 1)
    checks = by_name(ffmeta.verify(target))
    assert checks["overlap"].status == FAIL
    assert "field[0]" in checks["overlap"].message


def test_modified_data_gives_hash_warn(pair):
    original, target = pair
    data_start = read_word(target, 160)
    write_word(target, data_start, 123456)
    report = ffmeta.verify(target, original=original)
    checks = by_name(report)
    assert checks["hash"].status == WARN
    assert "modified since" in checks["hash"].message
    assert checks["diff"].status == FAIL
    assert "unexpected difference" in checks["diff"].message
    assert checks["reversible"].status == FAIL


def test_no_hash_skips(ff_file):
    ffmeta.add(ff_file, {"a": 1}, with_hash=False)
    assert by_name(ffmeta.verify(ff_file))["hash"].status == SKIP


def test_wrong_original(pair, tmp_path):
    original, target = pair
    wrong = make_um_file(tmp_path / "wrong.ff", nfields=2)
    checks = by_name(ffmeta.verify(target, original=wrong))
    assert checks["diff"].status in (FAIL, SKIP)
    assert checks["reversible"].status == FAIL
    checks = by_name(ffmeta.verify(target, original=tmp_path / "nope"))
    assert checks["diff"].status == FAIL


def test_block_in_the_middle(tmp_path, section):
    env = p.make_envelope(
        {"m": 1},
        section=section,
        original_header={"start": IMDI, "length": IMDI},
        original_size=1234,
        original_sha256="a" * 64,
    )
    block = p.pad_block(p.encode(env), reserve=16)
    target = make_um_file(tmp_path / "mid.ff", **{section: block})
    original = make_um_file(tmp_path / "orig.ff")
    report = ffmeta.verify(target, original=original)
    checks = by_name(report)
    assert checks["section"].status == PASS
    assert "rewritten by another tool" in checks["section"].message
    assert checks["overlap"].status == PASS
    assert checks["payload"].status == PASS
    assert checks["hash"].status == SKIP
    assert checks["diff"].status == SKIP
    assert checks["reversible"].status == SKIP
    assert report.ok


def test_mule_missing_is_fail_when_requested(pair):
    try:
        import mule  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("mule is installed")
    original, target = pair
    checks = by_name(ffmeta.verify(target, mule=True))
    assert checks["mule"].status == FAIL
    assert "cannot be imported" in checks["mule"].message


def test_iris_missing_is_fail_when_requested(pair):
    try:
        import iris  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("iris is installed")
    _, target = pair
    checks = by_name(ffmeta.verify(target, iris=True))
    assert checks["iris"].status == FAIL


def test_mule_version_parsing():
    assert parse_version("2026.09.1") == (2026, 9, 1)
    assert parse_version("v2026.10.1.dev0") == (2026, 10, 1)
    assert parse_version("garbage") is None
    assert mule_seeks_by_version("2026.09.1") is False
    assert mule_seeks_by_version("2026.10.1") is True
    assert mule_seeks_by_version("2027.1.0") is True
    assert mule_seeks_by_version("??") is None
