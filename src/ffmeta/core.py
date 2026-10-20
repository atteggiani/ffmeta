"""Add, read, edit and remove ffmeta metadata in UM files.

The metadata block is always appended after the last byte of the file and the
two pointer words for ``temp_historyfile`` are updated. Reading works purely by
pointer; editing and removing require the block to still be the last thing in
the file (it may not be after another tool, e.g. Mule, has rewritten the file).
"""

from __future__ import annotations

import contextlib
import copy
import os
import shutil
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from . import payload as _payload
from ._layout import (
    COMPONENT_BY_NAME,
    METADATA_SECTION,
    WORD_SIZE,
    FileInfo,
    PathLike,
    Region,
    inspect,
    sha256_range,
    word_offset,
    words_to_bytes,
)
from .errors import (
    CorruptMetadataError,
    FfmetaError,
    FfmetaWarning,
    FormatError,
    MetadataExistsError,
    NoMetadataError,
    SectionInUseError,
)

__all__ = [
    "add",
    "read",
    "edit",
    "remove",
    "has_metadata",
    "from_mule",
    "plan_add",
    "plan_edit",
    "plan_remove",
    "Plan",
]

_READ_LIMIT = 1 << 26  # refuse to read blocks larger than 64 MiB in one go


# --- plans ------------------------------------------------------------------


@dataclass(frozen=True)
class Write:
    """A single byte-range write."""

    offset: int
    data: bytes
    note: str


@dataclass
class Plan:
    """Everything an operation is going to do to a file, computed before any write."""

    operation: str
    path: Path
    section: str
    size_before: int
    size_after: int
    writes: list[Write] = field(default_factory=list)
    truncate_to: int | None = None
    notes: list[str] = field(default_factory=list)
    envelope: dict[str, Any] | None = None

    def describe(self) -> str:
        lines = [
            f"{self.operation}: {self.path} (section {self.section})",
            f"  file size: {self.size_before} -> {self.size_after} bytes",
        ]
        for w in self.writes:
            lines.append(f"  write {len(w.data):>8} bytes at offset {w.offset:>12}: {w.note}")
        if self.truncate_to is not None:
            lines.append(f"  truncate file to {self.truncate_to} bytes")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


# --- transactional writer ---------------------------------------------------


class _Transaction:
    """Apply writes/truncation to an open file, remembering how to undo them."""

    def __init__(self, fh: BinaryIO, size: int):
        self._fh = fh
        self._size = size
        self._undo: list[tuple[int, bytes]] = []

    def write(self, offset: int, data: bytes) -> None:
        if offset < self._size:
            self._fh.seek(offset)
            self._undo.append((offset, self._fh.read(min(len(data), self._size - offset))))
        self._fh.seek(offset)
        self._fh.write(data)

    def truncate(self, new_size: int) -> None:
        self._fh.seek(0, os.SEEK_END)
        current = self._fh.tell()
        if new_size < current:
            self._fh.seek(new_size)
            self._undo.append((new_size, self._fh.read(current - new_size)))
        self._fh.truncate(new_size)

    def commit(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def rollback(self) -> None:
        for offset, old in reversed(self._undo):
            self._fh.seek(offset)
            self._fh.write(old)
        self._fh.truncate(self._size)
        self._fh.flush()
        os.fsync(self._fh.fileno())


def _apply(plan: Plan, self_check: Callable[[Path], None]) -> None:
    with open(plan.path, "r+b") as fh:
        tx = _Transaction(fh, plan.size_before)
        try:
            for w in plan.writes:
                tx.write(w.offset, w.data)
            if plan.truncate_to is not None:
                tx.truncate(plan.truncate_to)
            tx.commit()
            self_check(plan.path)
        except BaseException:
            tx.rollback()
            raise


def _prepare_target(path: PathLike, output: PathLike | None) -> tuple[Path, bool]:
    src = Path(path)
    if output is None:
        return src, False
    dst = Path(output)
    if dst.resolve() == src.resolve():
        return src, False
    shutil.copyfile(src, dst)
    return dst, True


def _unlink_quiet(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _run(
    path: PathLike,
    output: PathLike | None,
    planner: Callable[[Path], Plan],
    checker: Callable[[Plan], Callable[[Path], None]],
) -> Plan:
    target, created = _prepare_target(path, output)
    try:
        plan = planner(target)
        _apply(plan, checker(plan))
    except BaseException:
        if created:
            _unlink_quiet(target)
        raise
    return plan


# --- locating an existing block --------------------------------------------


@dataclass(frozen=True)
class Located:
    """An ffmeta block found in a file."""

    section: str
    region: Region
    envelope: dict[str, Any]
    json_length: int

    @property
    def used_words(self) -> int:
        return _payload.payload_words(self.json_length)


def _read_block(fh: BinaryIO, region: Region) -> tuple[dict[str, Any], int]:
    """Decode the block at ``region`` reading only as much as the payload declares."""
    if region.length < _payload.HEADER_WORDS:
        raise NoMetadataError(f"{region} is too short to hold an ffmeta block")
    fh.seek(region.offset)
    head = fh.read(_payload.HEADER_BYTES)
    length = _payload.declared_length(head)
    need = _payload.payload_words(length)
    if need > region.length:
        raise CorruptMetadataError(
            f"payload declares {length} bytes but the section holds only {region.length} words"
        )
    if need * WORD_SIZE > _READ_LIMIT:
        raise CorruptMetadataError(f"payload of {length} bytes is implausibly large")
    fh.seek(region.offset)
    block = fh.read(need * WORD_SIZE)
    return _payload.decode(block), length


def _locate(fh: BinaryIO, info: FileInfo) -> Located:
    region = info.flh.component_region(METADATA_SECTION)
    if region is None:
        raise NoMetadataError(f"no ffmeta metadata found in {METADATA_SECTION} of {info.path}")
    if region.end > info.end_word:
        raise FormatError(f"{region} extends past the end of the file")
    try:
        env, length = _read_block(fh, region)
    except NoMetadataError:
        raise NoMetadataError(
            f"no ffmeta metadata found in {METADATA_SECTION} of {info.path}"
        ) from None
    return Located(METADATA_SECTION, region, env, length)


def _pointer_write(info: FileInfo, word: int, value: int, label: str) -> Write:
    old = info.flh[word]
    return Write(
        word_offset(word), words_to_bytes([value]), f"FLH word {word} ({label}): {old} -> {value}"
    )


def _section_pointer_words(section: str) -> tuple[int, int]:
    comp = COMPONENT_BY_NAME[section]
    return comp.start_word, comp.length_word


def _require_trailing(info: FileInfo, found: Located) -> None:
    if not info.is_trailing(found.region):
        raise FormatError(
            f"the metadata block ({found.region}) is not at the end of {info.path}; "
            "the file was probably rewritten by another tool, so it cannot be changed safely"
        )


# --- add --------------------------------------------------------------------


def _check_can_add(fh: BinaryIO, info: FileInfo, force: bool) -> None:
    if not force:
        bad = info.truncated_fields()
        if bad:
            f = bad[0]
            region = f.region
            assert region is not None
            raise FormatError(
                f"file looks truncated or damaged: field {f.index} claims words "
                f"{region.start}-{region.end - 1} but the file has {info.size_words} words "
                f"(use force=True / --force to override)"
            )
    region = info.flh.component_region(METADATA_SECTION)
    if region is None:
        return
    try:
        _read_block(fh, region)
    except NoMetadataError:
        raise SectionInUseError(
            f"section {METADATA_SECTION} already holds {region.length} words of other data "
            f"(starting at word {region.start})"
        ) from None
    except CorruptMetadataError as exc:
        raise MetadataExistsError(
            f"section {METADATA_SECTION} holds a damaged ffmeta block ({exc}); remove it first"
        ) from exc
    raise MetadataExistsError(f"metadata already present in section {METADATA_SECTION}")


def plan_add(
    path: PathLike,
    meta: Mapping[str, Any],
    *,
    with_hash: bool = True,
    align_sector: bool = False,
    force: bool = False,
) -> Plan:
    """Compute, without writing, what :func:`add` would do."""
    if not isinstance(meta, Mapping):
        raise TypeError("meta must be a mapping (JSON object)")
    target = Path(path)
    with open(target, "rb") as fh:
        info = inspect(fh, path=target)
        _check_can_add(fh, info, force)
        section = METADATA_SECTION
        start_w, len_w = _section_pointer_words(section)
        digest = sha256_range(fh, 0, info.size) if with_hash else None
    env = _payload.make_envelope(
        meta,
        original_header={"start": info.flh[start_w], "length": info.flh[len_w]},
        original_size=info.size,
        original_sha256=digest,
    )
    block = _payload.pad_block(
        _payload.encode(env), start_word=info.end_word, align_sector=align_sector
    )
    n_words = len(block) // WORD_SIZE
    writes = [
        Write(info.size, block, f"append {n_words}-word metadata block at word {info.end_word}"),
        _pointer_write(info, start_w, info.end_word, f"{section} start"),
        _pointer_write(info, len_w, n_words, f"{section} length"),
    ]
    return Plan("add", target, section, info.size, info.size + len(block), writes, envelope=env)


def _check_envelope_written(plan: Plan) -> Callable[[Path], None]:
    def check(path: Path) -> None:
        with open(path, "rb") as fh:
            info = inspect(fh, path=path)
            if info.size != plan.size_after:
                raise FfmetaError(f"self-check failed: file size {info.size} != {plan.size_after}")
            found = _locate(fh, info)
        if found.envelope != plan.envelope:
            raise FfmetaError("self-check failed: metadata read back differs from what was written")

    return check


def add(
    path: PathLike,
    meta: Mapping[str, Any],
    *,
    output: PathLike | None = None,
    with_hash: bool = True,
    align_sector: bool = False,
    force: bool = False,
) -> Path:
    """Embed ``meta`` (a JSON object) in ``path``.

    The block is always appended at the end of the file in ``temp_historyfile``;
    only its two pointer words change. With
    ``output`` the file is copied first and the original is untouched.
    ``align_sector`` pads the block to end on a 2048-word boundary; ``force``
    skips the "file looks truncated" check.

    Returns the path of the modified file.
    """
    plan = _run(
        path,
        output,
        lambda target: plan_add(
            target,
            meta,
            with_hash=with_hash,
            align_sector=align_sector,
            force=force,
        ),
        _check_envelope_written,
    )
    return plan.path


# --- read -------------------------------------------------------------------


def read(path: PathLike, *, envelope: bool = False) -> dict[str, Any]:
    """Return the user metadata (or the whole envelope) stored in ``path``.

    Raises :class:`NoMetadataError` when there is none.
    """
    target = Path(path)
    with open(target, "rb") as fh:
        info = inspect(fh, path=target)
        found = _locate(fh, info)
    return found.envelope if envelope else found.envelope["meta"]


def has_metadata(path: PathLike) -> bool:
    """True if ``path`` holds decodable ffmeta metadata."""
    try:
        read(path)
    except NoMetadataError:
        return False
    return True


# --- edit -------------------------------------------------------------------


def _new_meta(
    current: Mapping[str, Any],
    changes: Mapping[str, Any] | None,
    replace: Mapping[str, Any] | None,
    unset: Iterable[str],
) -> dict[str, Any]:
    if replace is not None and not isinstance(replace, Mapping):
        raise TypeError("replace must be a mapping (JSON object)")
    if changes is not None and not isinstance(changes, Mapping):
        raise TypeError("changes must be a mapping (JSON object)")
    meta: dict[str, Any] = copy.deepcopy(dict(replace if replace is not None else current))
    if changes:
        meta = _payload.deep_merge(meta, _payload.to_jsonable(dict(changes)))
    for key in unset:
        _payload.unset_dotted(meta, key)
    result: dict[str, Any] = _payload.to_jsonable(meta)
    return result


def plan_edit(
    path: PathLike,
    changes: Mapping[str, Any] | None = None,
    *,
    replace: Mapping[str, Any] | None = None,
    unset: Iterable[str] = (),
) -> Plan:
    """Compute, without writing, what :func:`edit` would do."""
    target = Path(path)
    with open(target, "rb") as fh:
        info = inspect(fh, path=target)
        found = _locate(fh, info)
    _require_trailing(info, found)
    env = dict(found.envelope)
    env["meta"] = _new_meta(env["meta"], changes, replace, unset)
    env["modified"] = _payload.utcnow_iso()
    block = _payload.encode(env)
    region = found.region
    _, len_w = _section_pointer_words(found.section)
    new_words = len(block) // WORD_SIZE
    plan = Plan("edit", target, found.section, info.size, info.size, envelope=env)
    if new_words <= region.length:
        padded = block + b"\0" * (region.nbytes - len(block))
        plan.writes.append(
            Write(
                region.offset,
                padded,
                f"rewrite metadata block in place ({new_words} of {region.length} words)",
            )
        )
        plan.notes.append("new payload fits in the existing block; pointers unchanged")
    else:
        plan.writes.append(
            Write(
                region.offset,
                block,
                f"rewrite trailing metadata block ({region.length} -> {new_words} words)",
            )
        )
        plan.writes.append(_pointer_write(info, len_w, new_words, f"{found.section} length"))
        plan.size_after = region.offset + len(block)
        plan.notes.append("block is at the end of the file; it grows in place")
    return plan


def edit(
    path: PathLike,
    changes: Mapping[str, Any] | None = None,
    *,
    replace: Mapping[str, Any] | None = None,
    unset: Iterable[str] = (),
    output: PathLike | None = None,
) -> Path:
    """Modify the stored metadata.

    ``replace`` swaps the whole metadata object, then ``changes`` is
    deep-merged into it, then the dotted keys in ``unset`` are deleted
    (missing keys are ignored).  The block must be the last thing in the file
    (otherwise :class:`FormatError` is raised); it is rewritten in place and
    grows at the end of the file when the new payload is larger.
    """
    plan = _run(
        path,
        output,
        lambda target: plan_edit(target, changes, replace=replace, unset=unset),
        _check_envelope_written,
    )
    return plan.path


# --- remove -----------------------------------------------------------------


def plan_remove(path: PathLike) -> Plan:
    """Compute, without writing, what :func:`remove` would do."""
    target = Path(path)
    with open(target, "rb") as fh:
        info = inspect(fh, path=target)
        found = _locate(fh, info)
    _require_trailing(info, found)
    region = found.region
    start_w, len_w = _section_pointer_words(found.section)
    orig = found.envelope["original_header"]
    plan = Plan("remove", target, found.section, info.size, info.size, envelope=found.envelope)
    plan.writes.append(_pointer_write(info, start_w, orig["start"], f"{found.section} start"))
    plan.writes.append(_pointer_write(info, len_w, orig["length"], f"{found.section} length"))
    plan.truncate_to = region.offset
    plan.size_after = region.offset
    plan.notes.append("block is at the end of the file; it is truncated away")
    if region.offset != found.envelope["original_size"]:
        plan.notes.append(
            "the file content before the block differs in size from the pre-add file "
            "(it was rewritten by another tool), so the result is not byte-identical "
            "to the file before the metadata was added"
        )
    return plan


def _check_removed(plan: Plan) -> Callable[[Path], None]:
    def check(path: Path) -> None:
        with open(path, "rb") as fh:
            info = inspect(fh, path=path)
            if info.size != plan.size_after:
                raise FfmetaError(f"self-check failed: file size {info.size} != {plan.size_after}")
            assert plan.envelope is not None
            start_w, len_w = _section_pointer_words(plan.section)
            orig = plan.envelope["original_header"]
            if (info.flh[start_w], info.flh[len_w]) != (orig["start"], orig["length"]):
                raise FfmetaError("self-check failed: pointer words were not restored")
            try:
                _locate(fh, info)
            except NoMetadataError:
                return
        raise FfmetaError("self-check failed: metadata still present after remove")

    return check


def remove(path: PathLike, *, output: PathLike | None = None) -> Path:
    """Remove the metadata block and restore the section's pointer words.

    The block must be the last thing in the file (otherwise :class:`FormatError`
    is raised) and is truncated away. If the file was rewritten after
    :func:`add`, the result differs from the pre-add file and a
    :class:`FfmetaWarning` is issued.
    """
    plan = _run(path, output, plan_remove, _check_removed)
    assert plan.envelope is not None
    if plan.size_after != plan.envelope["original_size"]:
        warnings.warn(
            f"{plan.path}: the file was rewritten after the metadata was added, so the "
            "result is not byte-identical to the pre-add file",
            FfmetaWarning,
            stacklevel=2,
        )
    return plan.path


# --- Mule -------------------------------------------------------------------


def _raw_values(component: Any) -> list[Any]:
    """Values of a Mule header component without the leading ``None`` of 1-based ``.raw``."""
    raw = getattr(component, "raw", component)
    values = list(np.asarray(raw, dtype=object).ravel())
    if values and values[0] is None:
        values = values[1:]
    return values


def from_mule(umfile: Any) -> dict[str, Any]:
    """Decode ffmeta metadata from a loaded Mule ``UMFile`` object.

    Raises :class:`NoMetadataError` when no supported section of the object
    holds an ffmeta block.
    """
    component = getattr(umfile, METADATA_SECTION, None)
    if component is not None:
        values = _raw_values(component)
        if len(values) >= _payload.HEADER_WORDS:
            try:
                block = words_to_bytes(int(v) for v in values)
                meta: dict[str, Any] = _payload.decode(block)["meta"]
                return meta
            except (NoMetadataError, TypeError, ValueError, OverflowError):
                pass
    raise NoMetadataError(
        "no ffmeta metadata in the Mule object. Note that Mule versions without "
        "MetOffice/mule PR #41 read header components sequentially instead of seeking to "
        "their start addresses, so they cannot see a block stored elsewhere in the file."
    )
