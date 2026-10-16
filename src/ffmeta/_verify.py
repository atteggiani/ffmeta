"""Read-only verification of a file carrying ffmeta metadata."""

from __future__ import annotations

import contextlib
import filecmp
import math
import re
import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import payload as _payload
from ._layout import (
    COMPONENT_BY_NAME,
    DATASET_TYPES,
    FLH_BYTES,
    WORD_SIZE,
    FileInfo,
    PathLike,
    detect_format,
    format_error_for,
    inspect,
    iter_chunks,
    sha256_range,
    word_offset,
    words_to_bytes,
)
from .core import Located, _locate, _raw_values, from_mule, read, remove
from .errors import CorruptMetadataError, FfmetaError, NoMetadataError

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
STATUSES = (PASS, WARN, FAIL, SKIP)

#: First Mule release expected to seek to component start addresses (MetOffice/mule PR #41).
MULE_SEEKING_VERSION = (2026, 10, 1)
MULE_SEQUENTIAL_MESSAGE = (
    "this Mule reads components sequentially instead of seeking to their start addresses "
    "(fixed by MetOffice/mule PR #41), so it cannot see the metadata and a Mule rewrite "
    "would drop it."
)

_HEADER_COMPONENTS = tuple(
    c.name for c in COMPONENT_BY_NAME.values() if c.name not in ("lookup", "data")
)


@dataclass(frozen=True)
class Check:
    """Outcome of one verification step."""

    status: str
    name: str
    message: str

    def __str__(self) -> str:
        return f"[{self.status}] {self.name}: {self.message}"


class Report:
    """An ordered list of :class:`Check` results."""

    def __init__(self, checks: Iterable[Check] = ()):
        self.checks: list[Check] = list(checks)

    def add(self, status: str, name: str, message: str) -> Check:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        check = Check(status, name, message)
        self.checks.append(check)
        return check

    @property
    def ok(self) -> bool:
        """True when no check failed."""
        return not any(c.status == FAIL for c in self.checks)

    def count(self, status: str) -> int:
        return sum(1 for c in self.checks if c.status == status)

    def to_list(self) -> list[dict[str, str]]:
        return [{"status": c.status, "name": c.name, "message": c.message} for c in self.checks]

    def summary(self) -> str:
        return (
            f"{self.count(PASS)} passed, {self.count(WARN)} warnings, "
            f"{self.count(FAIL)} failed, {self.count(SKIP)} skipped"
        )

    def __str__(self) -> str:
        return "\n".join([str(c) for c in self.checks] + [self.summary()])

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.checks)

    def __len__(self) -> int:
        return len(self.checks)


# --- helpers ----------------------------------------------------------------


def _guarded(report: Report, name: str, func: Callable[[], None]) -> None:
    """Run a check, turning unexpected exceptions into FAIL entries."""
    try:
        func()
    except FfmetaError as exc:
        report.add(FAIL, name, str(exc))
    except Exception as exc:  # noqa: BLE001 - verify must never raise
        report.add(FAIL, name, f"unexpected error: {exc.__class__.__name__}: {exc}")


def _pointer_offsets(section: str) -> tuple[int, int]:
    comp = COMPONENT_BY_NAME[section]
    return word_offset(comp.start_word), word_offset(comp.length_word)


def _files_equal(a: Path, b: Path) -> bool:
    return filecmp.cmp(str(a), str(b), shallow=False)


# --- tier 1 -----------------------------------------------------------------


def _check_header(report: Report, path: Path) -> FileInfo | None:
    size = path.stat().st_size
    if size % WORD_SIZE:
        report.add(FAIL, "header", f"file size {size} is not a multiple of {WORD_SIZE} bytes")
        return None
    with open(path, "rb") as fh:
        head = fh.read(FLH_BYTES)
    label = detect_format(head)
    if label != "be64":
        report.add(FAIL, "header", str(format_error_for(label)))
        return None
    info = inspect(path, strict=True)
    kind = DATASET_TYPES.get(info.flh.dataset_type, "unknown")
    if info.problems:
        report.add(FAIL, "header", "inconsistent header pointers: " + "; ".join(info.problems))
        return None
    report.add(
        PASS,
        "header",
        f"64-bit big-endian UM file, dataset type {info.flh.dataset_type} ({kind}), "
        f"{info.size_words} words",
    )
    return info


def _check_section(report: Report, path: Path, info: FileInfo) -> Located | None:
    try:
        with open(path, "rb") as fh:
            found = _locate(fh, info)
    except NoMetadataError as exc:
        report.add(FAIL, "section", str(exc))
        return None
    except CorruptMetadataError as exc:
        report.add(FAIL, "section", f"ffmeta block found but it cannot be decoded: {exc}")
        return None
    region = found.region
    if info.is_trailing(region):
        where = "the block is the last component of the file"
    else:
        where = (
            "the block is in the middle of the file (file was probably rewritten by another tool)"
        )
    report.add(PASS, "section", f"metadata found in {region}; {where}")
    return found


def _check_overlap(report: Report, info: FileInfo, found: Located) -> None:
    block = found.region
    others = info.regions(exclude=("data", found.section))
    clashes = [r for r in others if r.overlaps(block)]
    n_comp = len(info.flh.present_components(exclude=("data", found.section, "lookup")))
    n_fields = len(info.field_regions())
    n_noaddr = info.fields_without_address()
    if clashes:
        report.add(FAIL, "overlap", "metadata block overlaps " + ", ".join(str(c) for c in clashes))
        return
    report.add(
        PASS,
        "overlap",
        f"no overlap with {n_comp} header components, the lookup table and {n_fields} "
        f"fields ({n_noaddr} fields without a data address)",
    )


def _check_payload(report: Report, path: Path, info: FileInfo, found: Located) -> None:
    region = found.region
    with open(path, "rb") as fh:
        block = b"".join(iter_chunks(fh, region.offset, region.offset + region.nbytes))
    env = _payload.decode(block)  # raises CorruptMetadataError -> FAIL via _guarded
    length = _payload.declared_length(block)
    details = (
        f"magic {_payload.MAGIC.decode()}, {length}-byte JSON in {found.used_words} of "
        f"{region.length} words, envelope v{env['ffmeta']} written by ffmeta "
        f"{env['tool_version']}"
    )
    if not _payload.trailing_is_zero(block, length):
        report.add(
            WARN, "payload", details + "; trailing words after the JSON text are not all zero"
        )
        return
    if not _payload.words_as_floats_are_finite(block):
        report.add(WARN, "payload", details + "; some words read as NaN/Inf when viewed as doubles")
        return
    report.add(PASS, "payload", details)


def _check_hash(report: Report, path: Path, info: FileInfo, found: Located) -> None:
    env = found.envelope
    digest = env["original_sha256"]
    if digest is None:
        report.add(SKIP, "hash", "no original SHA-256 stored (added with --no-hash)")
        return
    original_size = env["original_size"]
    if found.region.offset != original_size or original_size > info.size:
        report.add(
            SKIP,
            "hash",
            "block is not where add put it (file rewritten by another tool); "
            "the original file cannot be reconstructed",
        )
        return
    start_off, len_off = _pointer_offsets(found.section)
    patches = {
        start_off: words_to_bytes([env["original_header"]["start"]]),
        len_off: words_to_bytes([env["original_header"]["length"]]),
    }
    with open(path, "rb") as fh:
        actual = sha256_range(fh, 0, original_size, patches)
    if actual == digest:
        report.add(
            PASS, "hash", "SHA-256 of the file without the metadata matches the stored digest"
        )
    else:
        report.add(WARN, "hash", "file modified since the metadata was added; re-add if intended")


# --- tier 2 -----------------------------------------------------------------


def _check_diff(report: Report, path: Path, original: Path, info: FileInfo, found: Located) -> None:
    orig_size = original.stat().st_size
    region = found.region
    if region.offset != orig_size:
        report.add(
            SKIP,
            "diff",
            f"block starts at byte {region.offset} but the original is "
            f"{orig_size} bytes long: the block does not directly follow the original content",
        )
        return
    if info.size != orig_size + region.nbytes:
        report.add(
            FAIL, "diff", f"file length {info.size} != original {orig_size} + block {region.nbytes}"
        )
        return
    allowed = set(_pointer_offsets(found.section))
    with open(path, "rb") as fa, open(original, "rb") as fb:
        pos = 0
        for ca, cb in zip(
            iter_chunks(fa, 0, orig_size), iter_chunks(fb, 0, orig_size), strict=True
        ):
            if ca != cb:
                a, b = np.frombuffer(ca, dtype=">i8"), np.frombuffer(cb, dtype=">i8")
                for idx in np.nonzero(a != b)[0]:
                    offset = pos + int(idx) * WORD_SIZE
                    if offset not in allowed:
                        report.add(
                            FAIL,
                            "diff",
                            f"unexpected difference at byte {offset} "
                            f"(word {offset // WORD_SIZE + 1}): {int(b[idx])} -> {int(a[idx])}",
                        )
                        return
            pos += len(ca)
    report.add(
        PASS,
        "diff",
        f"only FLH words {COMPONENT_BY_NAME[found.section].start_word} and "
        f"{COMPONENT_BY_NAME[found.section].length_word} and the appended "
        f"{region.length}-word tail differ from the original",
    )


def _check_reversible(
    report: Report, path: Path, original: Path, info: FileInfo, found: Located
) -> None:
    if not info.is_trailing(found.region):
        report.add(
            SKIP,
            "reversible",
            "block is not at the end of the file; remove would zero-fill "
            "it rather than restore the original bytes",
        )
        return
    with tempfile.TemporaryDirectory(prefix="ffmeta-verify-") as tmp:
        copy = Path(tmp) / path.name
        shutil.copyfile(path, copy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            remove(copy)
        if _files_equal(copy, original):
            report.add(PASS, "reversible", "remove on a copy reproduces the original byte for byte")
        else:
            report.add(FAIL, "reversible", "remove on a copy does not reproduce the original")


# --- tier 3: Mule -----------------------------------------------------------


def parse_version(text: str) -> tuple[int, ...] | None:
    """Parse ``2026.09.1``-style version strings into a tuple of ints."""
    match = re.match(r"^\s*v?(\d+(?:\.\d+)*)", str(text))
    if not match:
        return None
    return tuple(int(p) for p in match.group(1).split("."))


def mule_seeks_by_version(version: str) -> bool | None:
    """Whether a Mule version is expected to seek to component starts (None if unknown)."""
    parsed = parse_version(version)
    if parsed is None:
        return None
    return parsed >= MULE_SEEKING_VERSION


def _close_mule(obj: Any) -> None:
    for attr in ("close",):
        method = getattr(obj, attr, None)
        if callable(method):
            with contextlib.suppress(Exception):
                method()
            return
    source = getattr(obj, "_source", None)
    if source is not None and hasattr(source, "close"):
        with contextlib.suppress(Exception):
            source.close()


def _values_equal(a: Sequence[Any], b: Sequence[Any]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x is None or y is None:
            if x is not y:
                return False
            continue
        both_float = isinstance(x, (float, np.floating)) and isinstance(y, (float, np.floating))
        if both_float and math.isnan(float(x)) and math.isnan(float(y)):
            continue
        if x != y:
            return False
    return True


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _mule_compare(report: Report, ff: Any, orig: Any, section: str) -> None:
    comp = COMPONENT_BY_NAME[section]
    skip = {comp.start_word, comp.length_word}
    flh_a, flh_b = _raw_values(ff.fixed_length_header), _raw_values(orig.fixed_length_header)
    diff_words = [
        i + 1
        for i, (x, y) in enumerate(zip(flh_a, flh_b, strict=True))
        if x != y and i + 1 not in skip
    ]
    if len(flh_a) != len(flh_b) or diff_words:
        report.add(FAIL, "mule_header", f"fixed length header differs at words {diff_words}")
    else:
        report.add(
            PASS, "mule_header", "fixed length header identical apart from the section pointers"
        )

    bad = []
    for name in _HEADER_COMPONENTS:
        if name == section:
            continue
        ca, cb = getattr(ff, name, None), getattr(orig, name, None)
        if (
            (ca is None) != (cb is None)
            or ca is not None
            and not _values_equal(_raw_values(ca), _raw_values(cb))
        ):
            bad.append(name)
    if bad:
        report.add(FAIL, "mule_components", "components differ: " + ", ".join(bad))
    else:
        report.add(PASS, "mule_components", "all other header components identical")

    if len(ff.fields) != len(orig.fields):
        report.add(
            FAIL, "mule_lookup", f"{len(ff.fields)} fields vs {len(orig.fields)} in the original"
        )
        return
    bad_lookup = [
        i
        for i, (fa, fb) in enumerate(zip(ff.fields, orig.fields, strict=True))
        if not _values_equal(_raw_values(fa), _raw_values(fb))
    ]
    if bad_lookup:
        report.add(FAIL, "mule_lookup", f"lookup entries differ for fields {bad_lookup[:10]}")
    else:
        report.add(PASS, "mule_lookup", f"all {len(ff.fields)} lookup entries identical")

    differing, unreadable = [], []
    for i, (fa, fb) in enumerate(zip(ff.fields, orig.fields, strict=True)):
        if getattr(fa, "lbrel", None) == -99:
            continue
        try:
            if not _arrays_equal(fa.get_data(), fb.get_data()):
                differing.append(i)
        except Exception:  # noqa: BLE001 - packed fields may need optional libraries
            unreadable.append(i)
    if differing:
        report.add(FAIL, "mule_data", f"field data differ for fields {differing[:10]}")
    elif unreadable:
        report.add(
            WARN,
            "mule_data",
            f"{len(unreadable)} fields could not be unpacked and were "
            f"not compared; all others identical",
        )
    else:
        report.add(PASS, "mule_data", "field data identical for all fields (NaN-aware)")


def _mule_roundtrip(report: Report, ff: Any, meta: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="ffmeta-mule-") as tmp:
        out = Path(tmp) / "roundtrip.ff"
        try:
            ff.to_file(str(out))
        except Exception as exc:  # noqa: BLE001 - a Mule limitation, not an ffmeta failure
            report.add(
                WARN,
                "mule_roundtrip",
                f"Mule could not write the file ({exc.__class__.__name__}: "
                f"{exc}); round trip not tested",
            )
            return
        try:
            again = read(out)
        except NoMetadataError:
            report.add(
                FAIL,
                "mule_roundtrip",
                "metadata lost after Mule rewrote the file. " + MULE_SEQUENTIAL_MESSAGE,
            )
            return
        if again == meta:
            report.add(PASS, "mule_roundtrip", "metadata survives a Mule load / to_file round trip")
        else:
            report.add(
                FAIL, "mule_roundtrip", "metadata differs after a Mule load / to_file round trip"
            )


def _mule_version_check(report: Report, version: str, seeks: bool | None, *, decoded: bool) -> None:
    """Report what the Mule version says, reconciled with the observed behaviour."""
    release = ".".join(map(str, MULE_SEEKING_VERSION))
    if seeks:
        report.add(
            PASS,
            "mule_version",
            f"Mule {version} seeks to component start addresses (includes MetOffice/mule PR #41)",
        )
    elif decoded:
        report.add(
            PASS,
            "mule_version",
            f"Mule {version} seeks to component start addresses (behaves like PR #41, "
            f"although its version predates the {release} release)",
        )
    elif seeks is None:
        report.add(
            WARN,
            "mule_version",
            f"cannot tell from Mule version {version!r} whether it "
            f"seeks to component start addresses",
        )
    else:
        report.add(
            WARN,
            "mule_version",
            f"Mule {version} predates PR #41 ({release}); " + MULE_SEQUENTIAL_MESSAGE,
        )


def _mule_checks(report: Report, path: Path, original: Path | None, found: Located) -> None:
    try:
        import mule
    except ImportError:
        report.add(FAIL, "mule", "Mule was requested (--mule) but cannot be imported")
        return
    version = getattr(mule, "__version__", "unknown")
    seeks = mule_seeks_by_version(version)
    ff = orig = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ff = mule.load_umfile(str(path))
            except Exception as exc:  # noqa: BLE001
                report.add(FAIL, "mule_load", f"mule.load_umfile failed: {exc}")
                return
            report.add(PASS, "mule_load", f"mule.load_umfile succeeded ({len(ff.fields)} fields)")
            try:
                meta = from_mule(ff, found.section)
            except NoMetadataError:
                meta = None
            _mule_version_check(report, version, seeks, decoded=meta is not None)
            if meta is None:
                report.add(FAIL, "mule_metadata", MULE_SEQUENTIAL_MESSAGE)
            elif meta == found.envelope["meta"]:
                report.add(
                    PASS,
                    "mule_metadata",
                    f"metadata decodes from the Mule {found.section} component",
                )
            else:
                report.add(
                    FAIL,
                    "mule_metadata",
                    "metadata decoded from Mule differs from the block in the file",
                )
            if original is not None:
                try:
                    orig = mule.load_umfile(str(original))
                except Exception as exc:  # noqa: BLE001
                    report.add(
                        FAIL, "mule_header", f"mule.load_umfile failed on the original: {exc}"
                    )
                else:
                    _mule_compare(report, ff, orig, found.section)
            _mule_roundtrip(report, ff, found.envelope["meta"] if meta is None else meta)
    finally:
        for obj in (ff, orig):
            if obj is not None:
                _close_mule(obj)


# --- tier 3: Iris -----------------------------------------------------------


def _iris_checks(report: Report, path: Path, original: Path | None) -> None:
    try:
        import iris
    except ImportError:
        report.add(FAIL, "iris", "Iris was requested (--iris) but cannot be imported")
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cubes = iris.load(str(path))
        except Exception as exc:  # noqa: BLE001
            report.add(FAIL, "iris_load", f"iris.load failed: {exc}")
            return
        report.add(PASS, "iris_load", f"iris.load succeeded ({len(cubes)} cubes)")
        if original is None:
            return
        try:
            orig_cubes = iris.load(str(original))
        except Exception as exc:  # noqa: BLE001
            report.add(FAIL, "iris_compare", f"iris.load failed on the original: {exc}")
            return
        if len(cubes) != len(orig_cubes):
            report.add(
                FAIL, "iris_compare", f"{len(cubes)} cubes vs {len(orig_cubes)} in the original"
            )
            return
        key = lambda c: (c.name(), str(c.units), c.shape)  # noqa: E731
        pairs = zip(sorted(cubes, key=key), sorted(orig_cubes, key=key), strict=True)
        unequal = [a.name() for a, b in pairs if a != b]
        if unequal:
            report.add(
                FAIL, "iris_compare", "cubes differ from the original: " + ", ".join(unequal)
            )
        else:
            report.add(PASS, "iris_compare", f"all {len(cubes)} cubes equal to the original's")


# --- entry point ------------------------------------------------------------


def verify(
    path: PathLike, *, original: PathLike | None = None, mule: bool = False, iris: bool = False
) -> Report:
    """Check a file carrying ffmeta metadata without modifying anything.

    Tier 1 inspects the file alone; tier 2 (``original``) compares with the
    file as it was before ``add``; tier 3 loads the file with Mule and/or
    Iris when requested.
    """
    report = Report()
    target = Path(path)
    orig_path = Path(original) if original is not None else None
    if not target.is_file():
        report.add(FAIL, "header", f"{target} is not a file")
        return report
    if orig_path is not None and not orig_path.is_file():
        report.add(FAIL, "diff", f"original {orig_path} is not a file")
        orig_path = None

    info: FileInfo | None = None

    def header() -> None:
        nonlocal info
        info = _check_header(report, target)

    _guarded(report, "header", header)
    if info is None:
        for name in ("section", "overlap", "payload", "hash"):
            report.add(SKIP, name, "header check failed")
        return report
    assert info is not None
    live_info: FileInfo = info

    found: Located | None = None

    def section() -> None:
        nonlocal found
        found = _check_section(report, target, live_info)

    _guarded(report, "section", section)
    if found is None:
        for name in ("overlap", "payload", "hash"):
            report.add(SKIP, name, "no metadata block to check")
        return report
    live_found: Located = found

    _guarded(report, "overlap", lambda: _check_overlap(report, live_info, live_found))
    _guarded(report, "payload", lambda: _check_payload(report, target, live_info, live_found))
    _guarded(report, "hash", lambda: _check_hash(report, target, live_info, live_found))

    if orig_path is not None:
        orig = orig_path
        _guarded(report, "diff", lambda: _check_diff(report, target, orig, live_info, live_found))
        _guarded(
            report,
            "reversible",
            lambda: _check_reversible(report, target, orig, live_info, live_found),
        )
    if mule:
        _guarded(report, "mule", lambda: _mule_checks(report, target, orig_path, live_found))
    if iris:
        _guarded(report, "iris", lambda: _iris_checks(report, target, orig_path))
    return report
