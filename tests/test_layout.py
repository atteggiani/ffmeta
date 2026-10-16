"""Tests for the header/layout parsing in ffmeta._layout."""

from __future__ import annotations

import numpy as np
import pytest

from ffmeta import FormatError
from ffmeta._layout import (
    COMPONENTS,
    FLH_BYTES,
    IMDI,
    FixedLengthHeader,
    Region,
    detect_format,
    inspect,
    sha256_range,
)
from fixtures import byteswap_file, make_um_file, read_word, to_32bit_file, write_word


def test_component_table_matches_umdp_f03():
    table = {c.name: (c.start_word, c.dim_words) for c in COMPONENTS}
    assert table["integer_constants"] == (100, (101,))
    assert table["real_constants"] == (105, (106,))
    assert table["level_dependent_constants"] == (110, (111, 112))
    assert table["row_dependent_constants"] == (115, (116, 117))
    assert table["column_dependent_constants"] == (120, (121, 122))
    assert table["additional_parameters"] == (125, (126, 127))
    assert table["extra_constants"] == (130, (131,))
    assert table["temp_historyfile"] == (135, (136,))
    assert table["compressed_field_index1"] == (140, (141,))
    assert table["compressed_field_index3"] == (144, (145,))
    assert table["lookup"] == (150, (151, 152))
    assert table["data"] == (160, (161,))


def test_flh_indexing_is_one_based(ff_file):
    raw = ff_file.read_bytes()[:FLH_BYTES]
    flh = FixedLengthHeader.from_bytes(raw)
    assert flh[5] == 3
    assert flh[9] == 6
    assert flh[100] == 257
    assert flh.component_region("integer_constants") == Region("integer_constants", 257, 46)
    assert flh.component_region("extra_constants") is None
    assert flh.pointer_words("extra_constants") == (IMDI, IMDI)
    with pytest.raises(IndexError):
        flh[0]
    with pytest.raises(IndexError):
        flh[257]


def test_two_dimensional_component_length(ff_file):
    info = inspect(ff_file)
    ldc = info.flh.component_region("level_dependent_constants")
    assert ldc is not None and ldc.length == 3 * 8
    assert info.flh.component_region("real_constants").start == ldc.start - 38


def test_inspect_reads_lookup_extents(ff_file):
    info = inspect(ff_file)
    assert len(info.fields) == 3
    data_start = info.flh[160]
    assert (data_start - 1) % 2048 == 0
    first = info.fields[0].region
    assert first is not None and first.start == data_start and first.length == 2048
    assert info.fields[2].region.end == info.end_word
    assert info.truncated_fields() == []
    assert info.fields_without_address() == 0


def test_blank_lookup_entries_have_no_address(tmp_path):
    path = make_um_file(tmp_path / "blank.ff", blank_entries=2)
    info = inspect(path)
    assert len(info.fields) == 5
    assert info.fields_without_address() == 2
    assert info.fields[4].is_blank


def test_ancillary_header_sizes(ancil_file):
    info = inspect(ancil_file)
    assert info.flh.dataset_type == 4
    assert info.flh.component_region("integer_constants").length == 15
    assert info.flh.component_region("real_constants").length == 6
    assert info.flh.component_region("level_dependent_constants") is None


def test_detect_format(ff_file, tmp_path):
    assert detect_format(ff_file.read_bytes()[:FLH_BYTES]) == "be64"
    le = byteswap_file(ff_file, tmp_path / "le.ff")
    assert detect_format(le.read_bytes()[:FLH_BYTES]) == "le64"
    be32 = to_32bit_file(ff_file, tmp_path / "be32.ff")
    assert detect_format(be32.read_bytes()[:FLH_BYTES]) == "be32"
    assert detect_format(b"\0" * FLH_BYTES) == "unknown"
    assert detect_format(b"abc") == "unknown"


def test_inspect_refuses_wrong_formats(ff_file, tmp_path):
    le = byteswap_file(ff_file, tmp_path / "le.ff")
    with pytest.raises(FormatError, match="little-endian"):
        inspect(le)
    be32 = to_32bit_file(ff_file, tmp_path / "be32.ff")
    with pytest.raises(FormatError, match="32-bit"):
        inspect(be32)
    (tmp_path / "odd.ff").write_bytes(ff_file.read_bytes() + b"x")
    with pytest.raises(FormatError, match="multiple of 8"):
        inspect(tmp_path / "odd.ff")
    (tmp_path / "short.ff").write_bytes(b"\0" * 64)
    with pytest.raises(FormatError, match="too short"):
        inspect(tmp_path / "short.ff")


def test_inspect_detects_truncation(ff_file):
    data = ff_file.read_bytes()
    ff_file.write_bytes(data[: len(data) - 2048 * 8])
    info = inspect(ff_file)
    assert [f.index for f in info.truncated_fields()] == [2]


def test_inspect_strict_pointer_problems(ff_file):
    write_word(ff_file, 100, read_word(ff_file, 160) + 10_000_000)
    with pytest.raises(FormatError, match="integer_constants"):
        inspect(ff_file)
    info = inspect(ff_file, strict=False)
    assert info.problems


def test_region_overlap():
    a = Region("a", 10, 5)
    b = Region("b", 14, 3)
    c = Region("c", 15, 3)
    assert a.overlaps(b) and b.overlaps(a)
    assert not a.overlaps(c)
    assert not a.overlaps(Region("empty", 12, 0))
    assert a.end == 15 and a.offset == 72 and a.nbytes == 40


def test_sha256_range_with_patches(tmp_path):
    path = tmp_path / "x.bin"
    payload = bytes(range(256)) * 20
    path.write_bytes(payload)
    import hashlib

    with open(path, "rb") as fh:
        assert sha256_range(fh, 0, len(payload)) == hashlib.sha256(payload).hexdigest()
        patched = bytearray(payload)
        patched[8:16] = b"\xff" * 8
        patched[len(payload) - 8 :] = b"\x01" * 8
        expect = hashlib.sha256(bytes(patched)).hexdigest()
        got = sha256_range(fh, 0, len(payload), {8: b"\xff" * 8, len(payload) - 8: b"\x01" * 8})
        assert got == expect
    # patch straddling a chunk boundary
    big = tmp_path / "big.bin"
    data = np.zeros(1 << 18, dtype=np.uint8).tobytes() * 5  # 1.25 MiB
    big.write_bytes(data)
    patched = bytearray(data)
    patched[(1 << 20) - 4 : (1 << 20) + 4] = b"\x07" * 8
    with open(big, "rb") as fh:
        assert (
            sha256_range(fh, 0, len(data), {(1 << 20) - 4: b"\x07" * 8})
            == hashlib.sha256(bytes(patched)).hexdigest()
        )
