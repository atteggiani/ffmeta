"""Shared fixtures.  Mule and Iris tests are skipped when the libraries are missing."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

import ffmeta
from fixtures import make_um_file

# Mule checkouts predate NumPy 2; shim the removed helpers for the tests only.
if importlib.util.find_spec("mule") is not None:
    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]
    _fromstring = np.fromstring

    def _fs(s, dtype=float, count=-1, sep="", **kw):  # type: ignore[no-untyped-def]
        if sep == "":
            return np.frombuffer(s, dtype=dtype, count=count).copy()
        return _fromstring(s, dtype=dtype, count=count, sep=sep, **kw)

    np.fromstring = _fs  # type: ignore[assignment]


@pytest.fixture(params=["fieldsfile", "ancillary"])
def kind(request):
    return request.param


@pytest.fixture(params=["extra_constants", "temp_historyfile"])
def section(request):
    return request.param


@pytest.fixture
def um_file(tmp_path, kind):
    """A fresh synthetic file of the parametrised kind."""
    return make_um_file(tmp_path / f"{kind}.ff", kind=kind)


@pytest.fixture
def ff_file(tmp_path):
    return make_um_file(tmp_path / "fields.ff", kind="fieldsfile")


@pytest.fixture
def ancil_file(tmp_path):
    return make_um_file(tmp_path / "ancil.anc", kind="ancillary")


@pytest.fixture
def sample_meta():
    return {
        "experiment": "u-ab123",
        "source": {"doi": "10.1234/xyz", "model": "ACCESS-ESM1.6"},
        "tags": ["test", "ünïcode", 3, 4.5, True, None],
        "note": 'line1\nline2 "quoted"',
    }


@pytest.fixture(scope="session")
def mule_seeks():
    """Probe the installed Mule: does it read components from their start addresses?"""
    mule = pytest.importorskip("mule")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = make_um_file(Path(tmp) / "probe.ff")
        ffmeta.add(path, {"probe": True})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            umf = mule.load_umfile(str(path))
        try:
            ffmeta.from_mule(umf)
        except ffmeta.NoMetadataError:
            return False
        finally:
            close = getattr(umf, "close", None)
            if callable(close):
                close()
        return True
