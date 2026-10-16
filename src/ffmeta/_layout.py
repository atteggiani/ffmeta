"""File-layout facts for UM fieldsfiles and region arithmetic.

Everything here follows UMDP F03 (UM vn13.2 edition).  Word numbers are
1-based as in that document; byte offsets are 0-based.

Facts that F03 leaves open and the assumptions taken here:

* An empty component "has a negative start address" (F03 ch. 2).  A start of
  0 is not a valid 1-based address either, so ``start <= 0`` is treated as
  "component not present".
* F03 describes LBEGIN (lookup word 29) as a "disk address (word number)
  relative to start of file".  Mule and the UM seek to byte ``LBEGIN * 8``,
  i.e. LBEGIN is a 0-based word offset, and the first field of a file has
  ``LBEGIN == FLH(160) - 1``.  That convention is used here.
* LBNREC (lookup word 30) is the on-disk length of a field in words for files
  with well-formed records and 0 otherwise; when it is 0 the extent of the
  field is taken from LBLREC (word 15).
* ``LBREL == -99`` marks a blank lookup entry (UM/Mule convention, not in F03).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np

from .errors import FormatError

WORD_SIZE = 8
SECTOR_WORDS = 2048
FLH_WORDS = 256
FLH_BYTES = FLH_WORDS * WORD_SIZE
IMDI = -32768
WORD_DTYPE = np.dtype(">i8")

# Fixed length header words (1-based).
FLH_DATASET_TYPE = 5
FLH_GRID_STAGGERING = 9

DATASET_TYPES: Mapping[int, str] = {
    1: "instantaneous dump",
    2: "mean dump",
    3: "fieldsfile",
    4: "ancillary dataset",
    5: "lateral boundary condition dataset",
    6: "AC observation file",
    7: "VAR observation file",
    8: "Cx file",
    9: "covariance file",
    10: "OPS obstore file",
    11: "GPCS fieldsfile",
}

# Lookup entry words (1-based within an entry).
LB_LBLREC = 15
LB_LBPACK = 21
LB_LBREL = 22
LB_LBEGIN = 29
LB_LBNREC = 30
LBREL_BLANK = -99
LOOKUP_ENTRY_LENGTHS = (64, 128)

PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class Component:
    """A component of the file located by pointer words in the FLH."""

    name: str
    start_word: int
    dim_words: tuple[int, ...]

    @property
    def length_word(self) -> int:
        """The FLH word used by ffmeta as the component length (1-D sections only)."""
        return self.dim_words[0]


COMPONENTS: tuple[Component, ...] = (
    Component("integer_constants", 100, (101,)),
    Component("real_constants", 105, (106,)),
    Component("level_dependent_constants", 110, (111, 112)),
    Component("row_dependent_constants", 115, (116, 117)),
    Component("column_dependent_constants", 120, (121, 122)),
    Component("additional_parameters", 125, (126, 127)),
    Component("extra_constants", 130, (131,)),
    Component("temp_historyfile", 135, (136,)),
    Component("compressed_field_index1", 140, (141,)),
    Component("compressed_field_index2", 142, (143,)),
    Component("compressed_field_index3", 144, (145,)),
    Component("lookup", 150, (151, 152)),
    Component("data", 160, (161,)),
)
COMPONENT_BY_NAME: Mapping[str, Component] = {c.name: c for c in COMPONENTS}

#: Sections that may hold ffmeta metadata, in the order they are searched.
METADATA_SECTIONS: tuple[str, ...] = ("extra_constants", "temp_historyfile")
DEFAULT_SECTION = "extra_constants"


def word_offset(word: int) -> int:
    """Byte offset of a 1-based word number."""
    return (word - 1) * WORD_SIZE


def words_to_bytes(values: Iterable[int]) -> bytes:
    """Encode integers as big-endian 64-bit words."""
    return np.asarray(list(values), dtype=WORD_DTYPE).tobytes()


def bytes_to_words(raw: bytes) -> np.ndarray:
    """Decode big-endian 64-bit words."""
    words: np.ndarray = np.frombuffer(raw, dtype=WORD_DTYPE)
    return words


@dataclass(frozen=True)
class Region:
    """A contiguous run of words in the file (``start`` is a 1-based word number)."""

    name: str
    start: int
    length: int

    @property
    def end(self) -> int:
        """First 1-based word after the region."""
        return self.start + self.length

    @property
    def offset(self) -> int:
        return word_offset(self.start)

    @property
    def nbytes(self) -> int:
        return self.length * WORD_SIZE

    def overlaps(self, other: Region) -> bool:
        return (
            self.length > 0
            and other.length > 0
            and (self.start < other.end and other.start < self.end)
        )

    def __str__(self) -> str:
        return f"{self.name} (words {self.start}-{self.end - 1}, {self.length} words)"


class FixedLengthHeader:
    """The 256-word fixed length header with 1-based indexing."""

    def __init__(self, words: Sequence[int] | np.ndarray):
        arr = np.array(words, dtype=WORD_DTYPE)
        if arr.shape != (FLH_WORDS,):
            raise FormatError(f"fixed length header must have {FLH_WORDS} words, got {arr.size}")
        self._words = arr

    @classmethod
    def from_bytes(cls, raw: bytes) -> FixedLengthHeader:
        if len(raw) < FLH_BYTES:
            raise FormatError("file is too short to contain a fixed length header")
        return cls(bytes_to_words(raw[:FLH_BYTES]))

    def to_bytes(self) -> bytes:
        return self._words.tobytes()

    def copy(self) -> FixedLengthHeader:
        return FixedLengthHeader(self._words.copy())

    def __getitem__(self, word: int) -> int:
        if not 1 <= word <= FLH_WORDS:
            raise IndexError(f"FLH word {word} out of range 1-{FLH_WORDS}")
        return int(self._words[word - 1])

    def __setitem__(self, word: int, value: int) -> None:
        if not 1 <= word <= FLH_WORDS:
            raise IndexError(f"FLH word {word} out of range 1-{FLH_WORDS}")
        self._words[word - 1] = value

    @property
    def words(self) -> np.ndarray:
        copy: np.ndarray = self._words.copy()
        return copy

    @property
    def dataset_type(self) -> int:
        return self[FLH_DATASET_TYPE]

    @property
    def grid_staggering(self) -> int:
        return self[FLH_GRID_STAGGERING]

    def pointer_words(self, name: str) -> tuple[int, ...]:
        """Raw values of the pointer words (start, dims...) of a component."""
        comp = COMPONENT_BY_NAME[name]
        return (self[comp.start_word],) + tuple(self[w] for w in comp.dim_words)

    def component_region(self, name: str) -> Region | None:
        """Region of a component, or ``None`` when it is not present."""
        comp = COMPONENT_BY_NAME[name]
        start = self[comp.start_word]
        if start <= 0:
            return None
        length = 1
        for w in comp.dim_words:
            dim = self[w]
            length *= max(dim, 0)
        return Region(name, start, length)

    def present_components(self, exclude: Iterable[str] = ()) -> list[Region]:
        skip = set(exclude)
        regions = []
        for comp in COMPONENTS:
            if comp.name in skip:
                continue
            region = self.component_region(comp.name)
            if region is not None:
                regions.append(region)
        return regions


@dataclass(frozen=True)
class FieldExtent:
    """The part of a lookup entry needed to locate a field's data on disk."""

    index: int
    lbrel: int
    lblrec: int
    lbpack: int
    lbegin: int
    lbnrec: int

    @property
    def is_blank(self) -> bool:
        return self.lbrel == LBREL_BLANK

    @property
    def has_address(self) -> bool:
        return not self.is_blank and self.lbegin > 0

    @property
    def region(self) -> Region | None:
        """On-disk extent of the field, or ``None`` when it has no data address."""
        if not self.has_address:
            return None
        length = self.lbnrec if self.lbnrec > 0 else max(self.lblrec, 0)
        return Region(f"field[{self.index}]", self.lbegin + 1, length)


def detect_format(head: bytes) -> str:
    """Classify the first bytes of a file as ``be64``, ``le64``, ``be32``, ``le32`` or ``unknown``.

    The test is a plausibility check on the dataset type (word 5) and the
    pointers to the integer constants and lookup table (words 100 and 150).
    """

    def plausible(words: np.ndarray) -> bool:
        if words.size < FLH_WORDS:
            return False
        dataset_type = int(words[FLH_DATASET_TYPE - 1])
        ptrs = (int(words[99]), int(words[149]))
        return dataset_type in DATASET_TYPES and all(p <= 0 or p > FLH_WORDS for p in ptrs)

    for label, dtype, nbytes in (
        ("be64", ">i8", FLH_BYTES),
        ("le64", "<i8", FLH_BYTES),
        ("be32", ">i4", FLH_WORDS * 4),
        ("le32", "<i4", FLH_WORDS * 4),
    ):
        if len(head) >= nbytes and plausible(np.frombuffer(head[:nbytes], dtype=dtype)):
            return label
    return "unknown"


def format_error_for(label: str) -> FormatError:
    hints = {
        "le64": "the file looks like a little-endian 64-bit file; ffmeta supports only "
        "big-endian 64-bit UM files",
        "be32": "the file looks like a 32-bit big-endian file; ffmeta supports only "
        "big-endian 64-bit UM files",
        "le32": "the file looks like a 32-bit little-endian file; ffmeta supports only "
        "big-endian 64-bit UM files",
    }
    return FormatError(hints.get(label, "not a recognisable 64-bit big-endian UM file"))


@dataclass
class FileInfo:
    """Everything ffmeta needs to know about a file before touching it."""

    path: Path
    size: int
    flh: FixedLengthHeader
    fields: list[FieldExtent]
    problems: list[str]

    @property
    def size_words(self) -> int:
        return self.size // WORD_SIZE

    @property
    def end_word(self) -> int:
        """First 1-based word after the end of the file."""
        return self.size_words + 1

    def lookup_region(self) -> Region | None:
        return self.flh.component_region("lookup")

    def field_regions(self) -> list[Region]:
        return [f.region for f in self.fields if f.region is not None]

    def fields_without_address(self) -> int:
        return sum(1 for f in self.fields if f.region is None)

    def regions(self, exclude: Iterable[str] = ("data",)) -> list[Region]:
        """Regions of all present components (DATA excluded by default) plus field extents."""
        return self.flh.present_components(exclude=exclude) + self.field_regions()

    def truncated_fields(self) -> list[FieldExtent]:
        """Fields whose nominal extent runs past the end of the file."""
        return [f for f in self.fields if f.region is not None and f.region.end > self.end_word]

    def is_trailing(self, region: Region) -> bool:
        return region.end == self.end_word


def _read_exact(fh: BinaryIO, offset: int, nbytes: int) -> bytes:
    fh.seek(offset)
    data = fh.read(nbytes)
    if len(data) != nbytes:
        raise FormatError(
            f"unexpected end of file reading {nbytes} bytes at offset {offset} "
            f"(file is probably truncated)"
        )
    return data


def read_words(fh: BinaryIO, region: Region) -> np.ndarray:
    """Read a region as big-endian 64-bit integers."""
    return bytes_to_words(_read_exact(fh, region.offset, region.nbytes))


def _read_lookup(fh: BinaryIO, flh: FixedLengthHeader, size: int) -> list[FieldExtent]:
    region = flh.component_region("lookup")
    if region is None:
        return []
    entry_len, count = flh[151], flh[152]
    if entry_len not in LOOKUP_ENTRY_LENGTHS:
        raise FormatError(f"lookup entry length (FLH word 151) is {entry_len}, expected 64 or 128")
    if count < 0 or region.offset + region.nbytes > size:
        raise FormatError("lookup table points outside the file")
    if count == 0:
        return []
    table = read_words(fh, region).reshape(count, entry_len)
    return [
        FieldExtent(
            index=i,
            lbrel=int(row[LB_LBREL - 1]),
            lblrec=int(row[LB_LBLREC - 1]),
            lbpack=int(row[LB_LBPACK - 1]),
            lbegin=int(row[LB_LBEGIN - 1]),
            lbnrec=int(row[LB_LBNREC - 1]),
        )
        for i, row in enumerate(table)
    ]


def _pointer_problems(flh: FixedLengthHeader, size_words: int) -> list[str]:
    problems = []
    for comp in COMPONENTS:
        region = flh.component_region(comp.name)
        if region is None:
            continue
        if region.start <= FLH_WORDS:
            problems.append(f"{comp.name} starts at word {region.start}, inside the FLH")
        elif comp.name == "data":
            if region.start > size_words + 1:
                problems.append(f"data starts at word {region.start}, past the end of the file")
        elif region.end > size_words + 1:
            problems.append(f"{region} extends past the end of the file ({size_words} words)")
    return problems


def inspect(
    source: PathLike | BinaryIO, *, path: PathLike | None = None, strict: bool = True
) -> FileInfo:
    """Read the FLH and lookup table and sanity-check the file.

    With ``strict`` the function raises :class:`FormatError` for any pointer
    inconsistency; otherwise the problems are listed in ``FileInfo.problems``.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as fh:
            return _inspect_fh(fh, Path(source), strict)
    return _inspect_fh(
        source, Path(path) if path is not None else Path(getattr(source, "name", "?")), strict
    )


def _inspect_fh(fh: BinaryIO, path: Path, strict: bool) -> FileInfo:
    size = os.fstat(fh.fileno()).st_size
    if size % WORD_SIZE:
        raise FormatError(f"file size {size} is not a multiple of {WORD_SIZE} bytes")
    if size < FLH_BYTES:
        raise FormatError("file is too short to contain a fixed length header")
    head = _read_exact(fh, 0, FLH_BYTES)
    label = detect_format(head)
    if label != "be64":
        raise format_error_for(label)
    flh = FixedLengthHeader.from_bytes(head)
    problems = _pointer_problems(flh, size // WORD_SIZE)
    if problems and strict:
        raise FormatError("inconsistent header pointers: " + "; ".join(problems))
    fields: list[FieldExtent] = []
    try:
        fields = _read_lookup(fh, flh, size)
    except FormatError as exc:
        if strict:
            raise
        problems.append(str(exc))
    return FileInfo(path=path, size=size, flh=flh, fields=fields, problems=problems)


def iter_chunks(fh: BinaryIO, start: int, end: int, chunk: int = 1 << 20) -> Iterator[bytes]:
    """Yield the bytes in ``[start, end)`` in chunks."""
    fh.seek(start)
    pos = start
    while pos < end:
        data = fh.read(min(chunk, end - pos))
        if not data:
            raise FormatError(f"unexpected end of file at offset {pos}")
        pos += len(data)
        yield data


def sha256_range(
    fh: BinaryIO, start: int, end: int, patches: Mapping[int, bytes] | None = None
) -> str:
    """Streaming SHA-256 of ``[start, end)`` with optional byte patches applied."""
    digest = hashlib.sha256()
    patches = dict(patches or {})
    for pos, data in _positioned_chunks(fh, start, end):
        if patches:
            buf = bytearray(data)
            for offset, value in patches.items():
                lo, hi = offset, offset + len(value)
                if lo < pos + len(buf) and hi > pos:
                    a = max(lo, pos) - pos
                    b = min(hi, pos + len(buf)) - pos
                    buf[a:b] = value[max(lo, pos) - lo : min(hi, pos + len(buf)) - lo]
            data = bytes(buf)
        digest.update(data)
    return digest.hexdigest()


def _positioned_chunks(fh: BinaryIO, start: int, end: int) -> Iterator[tuple[int, bytes]]:
    pos = start
    for data in iter_chunks(fh, start, end):
        yield pos, data
        pos += len(data)
