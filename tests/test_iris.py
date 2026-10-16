"""Iris integration tests (skipped unless ``iris`` is importable)."""

from __future__ import annotations

import shutil
import warnings

import pytest

import ffmeta
from ffmeta._verify import PASS
from fixtures import make_um_file

iris = pytest.importorskip("iris")


def load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return iris.load(str(path))


@pytest.fixture
def pair(tmp_path, kind):
    original = make_um_file(tmp_path / "orig.ff", kind=kind)
    target = tmp_path / "meta.ff"
    shutil.copyfile(original, target)
    ffmeta.add(target, {"experiment": "u-ab123"})
    return original, target


def test_fieldsfile_fixture_loads_as_one_cube(ff_file):
    cubes = load(ff_file)
    assert len(cubes) == 1
    assert cubes[0].shape == (3, 3, 4)


def test_cubes_identical_before_and_after(pair):
    original, target = pair
    before, after = load(original), load(target)
    assert len(before) == len(after)
    for a, b in zip(
        sorted(before, key=lambda c: c.name()), sorted(after, key=lambda c: c.name()), strict=True
    ):
        assert a == b


def test_verify_iris(pair):
    original, target = pair
    report = ffmeta.verify(target, original=original, iris=True)
    checks = {c.name: c for c in report.checks}
    assert report.ok, str(report)
    assert checks["iris_load"].status == PASS
    assert checks["iris_compare"].status == PASS
