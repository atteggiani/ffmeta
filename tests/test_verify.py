"""Tests for ffmeta.verify."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import ffmeta
from ffmeta import _verify as v
from ffmeta import payload as p
from ffmeta._layout import WORD_SIZE
from ffmeta._verify import FAIL, PASS, SKIP, WARN, Report, mule_seeks_by_version, parse_version
from fixtures import IMDI, make_um_file, read_word, write_bytes_at, write_word


def by_name(report):
    return {c.name: c for c in report.checks}


@pytest.fixture
def pair(tmp_path, kind):
    """(original, with-metadata) files for the parametrised file kind."""
    original = make_um_file(tmp_path / "orig.ff", kind=kind)
    target = tmp_path / "meta.ff"
    shutil.copyfile(original, target)
    ffmeta.add(target, {"a": 1, "nested": {"b": [1, 2]}})
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
    start = read_word(target, 135)
    write_bytes_at(target, (start - 1 + 2) * WORD_SIZE, b"xxxxxxxx")
    report = ffmeta.verify(target, original=original)
    checks = by_name(report)
    assert checks["section"].status == FAIL
    assert "cannot be decoded" in checks["section"].message
    assert not report.ok


def test_bad_version_payload(pair):
    _, target = pair
    start = read_word(target, 135)
    write_bytes_at(target, (start - 1) * WORD_SIZE + 6, b"77")
    report = ffmeta.verify(target)
    assert by_name(report)["section"].status == FAIL


def test_trailing_nonzero_words_warn(ff_file):
    ffmeta.add(ff_file, {"a": 1}, reserve=64)
    start, length = read_word(ff_file, 135), read_word(ff_file, 136)
    write_word(ff_file, start + length - 1, 42)
    checks = by_name(ffmeta.verify(ff_file))
    assert checks["payload"].status == WARN
    assert "not all zero" in checks["payload"].message


def test_overlap_detected(pair):
    original, target = pair
    # point the first field's LBEGIN into the metadata block
    lookup_start = read_word(target, 150)
    block_start = read_word(target, 135)
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


def test_block_in_the_middle(tmp_path):
    env = p.make_envelope(
        {"m": 1},
        original_header={"start": IMDI, "length": IMDI},
        original_size=1234,
        original_sha256="a" * 64,
    )
    block = p.pad_block(p.encode(env), reserve=16)
    target = make_um_file(tmp_path / "mid.ff", temp_historyfile=block)
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


def test_verify_missing_original_and_non_file(tmp_path, ff_file):
    report = ffmeta.verify(ff_file, original=tmp_path / "gone.ff")
    assert by_name(report)["diff"].status == FAIL
    missing = ffmeta.verify(tmp_path / "missing.ff")
    assert by_name(missing)["header"].status == FAIL


def test_verify_header_pointer_and_corrupt_section(ff_file):
    write_word(ff_file, 100, 1)
    checks = by_name(ffmeta.verify(ff_file))
    assert checks["header"].status == FAIL

    valid = make_um_file(ff_file)
    ffmeta.add(valid, {"a": 1})
    write_bytes_at(valid, (read_word(valid, 135) - 1) * WORD_SIZE, b"notmagic")
    checks = by_name(ffmeta.verify(valid))
    assert checks["section"].status == FAIL


def test_guarded_and_header_unexpected_paths(monkeypatch, ff_file):
    report = v.Report()
    v._guarded(report, "known", lambda: (_ for _ in ()).throw(ffmeta.FfmetaError("known")))
    v._guarded(report, "unexpected", lambda: (_ for _ in ()).throw(RuntimeError("unexpected")))
    assert [check.status for check in report] == [FAIL, FAIL]

    fake_info = SimpleNamespace(
        problems=["bad pointer"], flh=SimpleNamespace(dataset_type=999), size_words=10
    )
    monkeypatch.setattr(v, "inspect", lambda *args, **kwargs: fake_info)
    report = v.Report()
    assert v._check_header(report, ff_file) is None
    assert "bad pointer" in report.checks[0].message


def test_check_payload_nonfinite_warning(ff_file, monkeypatch):
    ffmeta.add(ff_file, {"a": 1})
    info = v.inspect(ff_file)
    with open(ff_file, "rb") as fh:
        found = v._locate(fh, info)
    monkeypatch.setattr(v._payload, "words_as_floats_are_finite", lambda block: False)
    report = v.Report()
    v._check_payload(report, ff_file, info, found)
    assert report.checks[0].status == WARN
    assert "NaN/Inf" in report.checks[0].message


def test_check_diff_size_mismatch(ff_file, tmp_path):
    original = tmp_path / "original.ff"
    shutil.copyfile(ff_file, original)
    ffmeta.add(ff_file, {"a": 1})
    ff_file.write_bytes(ff_file.read_bytes() + b"\0" * WORD_SIZE)
    info = v.inspect(ff_file)
    with open(ff_file, "rb") as fh:
        found = v._locate(fh, info)
    report = v.Report()
    v._check_diff(report, ff_file, original, info, found)
    assert report.checks[0].status == FAIL
    assert "file length" in report.checks[0].message


def test_values_and_arrays_compare_edge_cases():
    assert not v._values_equal([1], [])
    assert not v._values_equal([None], [0])
    assert v._values_equal([None], [None])
    assert v._values_equal([float("nan")], [np.float64("nan")])
    assert not v._values_equal([1], [2])
    assert not v._arrays_equal(np.zeros((1,)), np.zeros((2,)))
    assert v._arrays_equal(np.array([np.nan]), np.array([np.nan]))
    assert not v._arrays_equal(np.array([1]), np.array([2]))


def test_close_mule_swallows_errors_and_closes_source():
    class CloseRaises:
        def close(self):
            raise OSError("close failed")

    v._close_mule(CloseRaises())

    source = SimpleNamespace(close=lambda: (_ for _ in ()).throw(OSError("close failed")))
    v._close_mule(SimpleNamespace(_source=source))


def test_mule_compare_pass_and_failure_shapes():
    raw_header = [None] + [0] * 256

    class Component:
        raw = [None, 1, 2]

    class Field:
        lbrel = 3
        raw = [None, 1]

        def __init__(self, data=None, error=False):
            self.data = data
            self.error = error

        def get_data(self):
            if self.error:
                raise ValueError("packed")
            return self.data

    base = SimpleNamespace(
        fixed_length_header=SimpleNamespace(raw=raw_header),
        temp_historyfile=Component(),
        integer_constants=Component(),
        fields=[Field(np.array([1.0, np.nan]))],
    )
    report = v.Report()
    v._mule_compare(report, base, base, "temp_historyfile")
    assert all(check.status == v.PASS for check in report)

    other = SimpleNamespace(
        fixed_length_header=SimpleNamespace(raw=[None] + [1] * 256),
        temp_historyfile=Component(),
        integer_constants=None,
        fields=[Field(np.array([2.0]))],
    )
    report = v.Report()
    v._mule_compare(report, other, base, "temp_historyfile")
    statuses = {check.name: check.status for check in report}
    assert statuses["mule_header"] == FAIL
    assert statuses["mule_components"] == FAIL
    assert statuses["mule_data"] == FAIL

    unreadable = SimpleNamespace(
        fixed_length_header=SimpleNamespace(raw=raw_header),
        temp_historyfile=Component(),
        integer_constants=Component(),
        fields=[Field(error=True)],
    )
    report = v.Report()
    v._mule_compare(report, unreadable, base, "temp_historyfile")
    assert by_name(report)["mule_data"].status == WARN

    blank = SimpleNamespace(lbrel=-99, raw=[None])
    ff = SimpleNamespace(
        fixed_length_header=SimpleNamespace(raw=raw_header),
        temp_historyfile=Component(),
        integer_constants=Component(),
        fields=[blank],
    )
    v._mule_compare(v.Report(), ff, ff, "temp_historyfile")

    fewer_fields = SimpleNamespace(
        fixed_length_header=SimpleNamespace(raw=raw_header),
        temp_historyfile=Component(),
        integer_constants=Component(),
        fields=[],
    )
    report = v.Report()
    v._mule_compare(report, base, fewer_fields, "temp_historyfile")
    assert by_name(report)["mule_lookup"].status == FAIL

    different_lookup = SimpleNamespace(
        fixed_length_header=SimpleNamespace(raw=raw_header),
        temp_historyfile=Component(),
        integer_constants=Component(),
        fields=[SimpleNamespace(raw=[None, 9], lbrel=-99)],
    )
    report = v.Report()
    v._mule_compare(report, different_lookup, base, "temp_historyfile")
    assert by_name(report)["mule_lookup"].status == FAIL


@pytest.mark.parametrize(
    ("write_error", "read_result", "expected"),
    [
        (True, {"a": 1}, WARN),
        (False, None, FAIL),
        (False, {"a": 2}, FAIL),
        (False, {"a": 1}, v.PASS),
    ],
)
def test_mule_roundtrip_outcomes(monkeypatch, write_error, read_result, expected):
    report = v.Report()

    class FakeMuleFile:
        def to_file(self, path):
            if write_error:
                raise OSError("write failed")

    if not write_error:
        if read_result is None:

            def fake_read(path):
                raise ffmeta.NoMetadataError("missing")
        else:

            def fake_read(path):
                return read_result

        monkeypatch.setattr(v, "read", fake_read)
    v._mule_roundtrip(report, FakeMuleFile(), {"a": 1})
    assert report.checks[-1].status == expected


def test_mule_version_check_all_outcomes():
    for version, seeks, decoded, status in [
        ("2026.10.1", True, True, v.PASS),
        ("2026.09.1", False, True, v.PASS),
        ("unknown", None, False, WARN),
        ("2026.09.1", False, False, WARN),
    ]:
        report = v.Report()
        v._mule_version_check(report, version, seeks, decoded=decoded)
        assert report.checks[0].status == status


def test_iris_checks_with_fake_module(monkeypatch, tmp_path):
    class Cube:
        units = "K"
        shape = (1,)

        def __init__(self, name, value):
            self._name = name
            self.value = value

        def name(self):
            return self._name

        def __eq__(self, other):
            return self.name() == other.name() and self.value == other.value

    cube = Cube("temperature", 1)
    original = tmp_path / "original.ff"
    original.touch()
    iris = ModuleType("iris")
    iris.load = lambda path: [cube]
    monkeypatch.setitem(sys.modules, "iris", iris)
    report = v.Report()
    v._iris_checks(report, tmp_path / "input.ff", None)
    assert report.checks[0].name == "iris_load"
    report = v.Report()
    v._iris_checks(report, tmp_path / "input.ff", original)
    assert report.checks[-1].status == v.PASS

    iris.load = lambda path: [Cube("temperature", 2)] if path == str(original) else [cube]
    report = v.Report()
    v._iris_checks(report, tmp_path / "input.ff", original)
    assert by_name(report)["iris_compare"].status == FAIL

    iris.load = lambda path: [] if path == str(original) else [cube]
    report = v.Report()
    v._iris_checks(report, tmp_path / "input.ff", original)
    assert by_name(report)["iris_compare"].status == FAIL

    iris.load = lambda path: (_ for _ in ()).throw(RuntimeError("load failed"))
    report = v.Report()
    v._iris_checks(report, tmp_path / "input.ff", None)
    assert by_name(report)["iris_load"].status == FAIL


def test_verify_optional_integration_wrappers_with_fakes(monkeypatch, tmp_path):
    target = make_um_file(tmp_path / "metadata.ff")
    original = make_um_file(tmp_path / "original.ff")
    ffmeta.add(target, {"a": 1})
    info = v.inspect(target)
    with open(target, "rb") as fh:
        located = v._locate(fh, info)
    raw_block = target.read_bytes()[
        located.region.offset : located.region.offset + located.region.nbytes
    ]
    meta_words = [None, *np.frombuffer(raw_block, dtype=">i8").tolist()]

    class MuleComponent:
        def __init__(self, raw):
            self.raw = raw

    class FakeMuleFile:
        def __init__(self, source, with_meta):
            self.source = source
            self.fields = []
            header = np.frombuffer(source.read_bytes()[:2048], dtype=">i8").tolist()
            self.fixed_length_header = MuleComponent([None, *header])
            self.temp_historyfile = MuleComponent(meta_words) if with_meta else None

        def to_file(self, output):
            shutil.copyfile(target, output)

        def close(self):
            pass

    mule = ModuleType("mule")
    mule.__version__ = "2026.10.1"
    mule.load_umfile = lambda path: FakeMuleFile(Path(path), Path(path) == target)
    monkeypatch.setitem(sys.modules, "mule", mule)
    report = ffmeta.verify(target, original=original, mule=True)
    checks = by_name(report)
    assert checks["mule_metadata"].status == v.PASS
    assert checks["mule_header"].status == v.PASS
    assert checks["mule_roundtrip"].status == v.PASS

    different_env = p.make_envelope(
        {"different": True},
        original_header={"start": -32768, "length": -32768},
        original_size=0,
        original_sha256=None,
    )
    different_words = [None, *np.frombuffer(p.encode(different_env), dtype=">i8").tolist()]

    def different_metadata(path):
        loaded = FakeMuleFile(Path(path), True)
        loaded.temp_historyfile = MuleComponent(different_words)
        return loaded

    mule.load_umfile = different_metadata
    report = ffmeta.verify(target, mule=True)
    assert by_name(report)["mule_metadata"].status == FAIL

    def fail_original(path):
        if Path(path) == original:
            raise OSError("original load failed")
        return FakeMuleFile(Path(path), True)

    mule.load_umfile = fail_original
    report = ffmeta.verify(target, original=original, mule=True)
    assert by_name(report)["mule_header"].status == FAIL

    mule.load_umfile = lambda path: (_ for _ in ()).throw(OSError("load failure"))
    report = ffmeta.verify(target, mule=True)
    assert by_name(report)["mule_load"].status == FAIL

    class NoMetadataMuleFile(FakeMuleFile):
        def __init__(self, source, with_meta):
            super().__init__(source, False)

        def to_file(self, output):
            shutil.copyfile(original, output)

    mule.load_umfile = lambda path: NoMetadataMuleFile(Path(path), False)
    report = ffmeta.verify(target, mule=True)
    checks = by_name(report)
    assert checks["mule_metadata"].status == FAIL
    assert checks["mule_roundtrip"].status == FAIL

    iris = ModuleType("iris")
    iris.load = lambda path: []
    monkeypatch.setitem(sys.modules, "iris", iris)
    report = ffmeta.verify(target, iris=True)
    assert by_name(report)["iris_load"].status == v.PASS

    iris.load = lambda path: [SimpleNamespace(name=lambda: "one", units="K", shape=(1,))]
    report = ffmeta.verify(target, original=original, iris=True)
    assert by_name(report)["iris_compare"].status == FAIL

    iris.load = lambda path: (
        (_ for _ in ()).throw(OSError("original Iris load failed")) if path == str(original) else []
    )
    report = ffmeta.verify(target, original=original, iris=True)
    assert by_name(report)["iris_compare"].status == FAIL


def test_optional_integration_import_errors(monkeypatch, tmp_path):
    target = make_um_file(tmp_path / "plain.ff")
    ffmeta.add(target, {"a": 1})
    info = v.inspect(target)
    with open(target, "rb") as fh:
        found = v._locate(fh, info)

    monkeypatch.setitem(sys.modules, "mule", None)
    report = v.Report()
    v._mule_checks(report, target, None, found)
    assert by_name(report)["mule"].status == FAIL

    monkeypatch.setitem(sys.modules, "iris", None)
    report = v.Report()
    v._iris_checks(report, target, None)
    assert by_name(report)["iris"].status == FAIL
