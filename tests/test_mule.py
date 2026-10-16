"""Mule integration tests (skipped unless ``mule`` is importable).

The session fixture ``mule_seeks`` probes the installed Mule: with MetOffice/mule
PR #41 (expected in 2026.10.1) Mule seeks to each component's start address
and sees the metadata; older releases read components sequentially and do not.
"""

from __future__ import annotations

import shutil
import warnings

import numpy as np
import pytest

import ffmeta
from ffmeta._verify import FAIL, MULE_SEQUENTIAL_MESSAGE, PASS, mule_seeks_by_version
from fixtures import make_um_file

mule = pytest.importorskip("mule")


def by_name(report):
    return {c.name: c for c in report.checks}


def load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mule.load_umfile(str(path))


def close(umf):
    closer = getattr(umf, "close", None)
    if callable(closer):
        closer()


@pytest.fixture
def pair(tmp_path, kind, section):
    original = make_um_file(tmp_path / "orig.ff", kind=kind)
    target = tmp_path / "meta.ff"
    shutil.copyfile(original, target)
    meta = {"experiment": "u-ab123", "levels": [1, 2, 3], "nested": {"ok": True}}
    ffmeta.add(target, meta, section=section)
    return original, target, meta


def test_fixture_loads_and_validates(um_file):
    umf = load(um_file)
    try:
        umf.validate()
        assert len(umf.fields) == 3
        data = umf.fields[0].get_data()
        assert data.shape == (3, 4)
    finally:
        close(umf)


def test_fields_and_lookup_identical(pair):
    original, target, _ = pair
    a, b = load(target), load(original)
    try:
        assert len(a.fields) == len(b.fields)
        for fa, fb in zip(a.fields, b.fields, strict=True):
            assert list(fa.raw)[1:] == list(fb.raw)[1:]
            assert np.array_equal(fa.get_data(), fb.get_data(), equal_nan=True)
        assert np.array_equal(
            np.asarray(a.integer_constants.raw[1:]), np.asarray(b.integer_constants.raw[1:])
        )
    finally:
        close(a)
        close(b)


def test_version_detection_matches_behaviour(mule_seeks):
    expected = mule_seeks_by_version(mule.__version__)
    if expected is None:
        pytest.skip(f"unparseable Mule version {mule.__version__!r}")
    if mule_seeks and not expected:
        # an unreleased PR #41 checkout still carries the previous release's version string
        pytest.skip(f"Mule {mule.__version__} seeks but predates the release that includes PR #41")
    assert expected == mule_seeks


# --- seeking Mule (PR #41) ---------------------------------------------------


def test_metadata_visible_to_seeking_mule(pair, mule_seeks):
    if not mule_seeks:
        pytest.skip("this Mule does not seek to component start addresses")
    original, target, meta = pair
    umf = load(target)
    try:
        assert ffmeta.from_mule(umf) == meta
    finally:
        close(umf)


def test_metadata_survives_to_file(pair, tmp_path, mule_seeks):
    if not mule_seeks:
        pytest.skip("this Mule does not seek to component start addresses")
    original, target, meta = pair
    umf = load(target)
    try:
        out = tmp_path / "rewritten.ff"
        umf.to_file(str(out))
    finally:
        close(umf)
    assert ffmeta.read(out) == meta
    # after the rewrite the block is in its canonical position: still editable/removable
    report = ffmeta.verify(out)
    assert report.ok, str(report)
    assert "rewritten by another tool" in by_name(report)["section"].message
    ffmeta.edit(out, {"after": "mule"})
    assert ffmeta.read(out)["after"] == "mule"
    with pytest.warns(ffmeta.FfmetaWarning):
        ffmeta.remove(out)
    assert not ffmeta.has_metadata(out)
    again = load(out)
    try:
        assert len(again.fields) == 3
    finally:
        close(again)


def test_verify_mule_passes_with_seeking_mule(pair, mule_seeks):
    if not mule_seeks:
        pytest.skip("this Mule does not seek to component start addresses")
    original, target, _ = pair
    report = ffmeta.verify(target, original=original, mule=True)
    checks = by_name(report)
    assert report.ok, str(report)
    for name in (
        "mule_load",
        "mule_metadata",
        "mule_header",
        "mule_components",
        "mule_lookup",
        "mule_data",
        "mule_roundtrip",
    ):
        assert checks[name].status == PASS, checks[name]


# --- non-seeking Mule (<= 2026.09.1) ----------------------------------------


def test_verify_mule_fails_with_sequential_mule(pair, mule_seeks):
    if mule_seeks:
        pytest.skip("this Mule seeks to component start addresses")
    original, target, _ = pair
    report = ffmeta.verify(target, original=original, mule=True)
    checks = by_name(report)
    assert checks["mule_load"].status == PASS
    assert checks["mule_metadata"].status == FAIL
    assert checks["mule_metadata"].message == MULE_SEQUENTIAL_MESSAGE
    assert checks["mule_data"].status == PASS  # the file itself still loads fine
    assert not report.ok


def test_from_mule_message_with_sequential_mule(pair, mule_seeks):
    if mule_seeks:
        pytest.skip("this Mule seeks to component start addresses")
    _, target, _ = pair
    umf = load(target)
    try:
        with pytest.raises(ffmeta.NoMetadataError, match="PR #41"):
            ffmeta.from_mule(umf)
    finally:
        close(umf)
