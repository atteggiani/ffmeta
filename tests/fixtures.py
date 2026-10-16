"""Numpy-only generator of small synthetic UM fieldsfiles and ancillary files.

The layout follows UMDP F03: a 256-word FLH, integer/real/level dependent
constants, optional row/column dependent constants, optional extra
constants / temp historyfile sections (so files with a metadata block in its
canonical position can be built by hand), a LOOKUP table of 64-word entries
and DATA starting on a 2048-word sector boundary with one sector per field.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WORD = 8
SECTOR = 2048
IMDI = -32768
RMDI = -1073741824.0
FLH_WORDS = 256
LOOKUP_LEN = 64


def _sector_ceil(nwords: int) -> int:
    return -(-nwords // SECTOR) * SECTOR


def _ints(n: int, fill: int = IMDI) -> np.ndarray:
    return np.full(n, fill, dtype=">i8")


def _reals(n: int, fill: float = RMDI) -> np.ndarray:
    return np.full(n, fill, dtype=">f8")


def make_um_file(
    path,
    *,
    kind: str = "fieldsfile",
    nx: int = 4,
    ny: int = 3,
    nfields: int = 3,
    num_p_levels: int = 2,
    row_col: bool = False,
    blank_entries: int = 0,
    extra_constants: bytes | None = None,
    temp_historyfile: bytes | None = None,
    data_seed: int = 0,
) -> Path:
    """Write a synthetic UM file and return its path.

    ``kind`` is ``"fieldsfile"`` (dataset type 3, 46/38 constants, level
    dependent constants of shape (num_p_levels + 1, 8)) or ``"ancillary"``
    (dataset type 4, 15/6 constants, no level dependent constants).
    ``extra_constants`` / ``temp_historyfile`` are raw bytes (whole words) to
    store in those sections at their canonical position before LOOKUP.
    """
    if kind not in ("fieldsfile", "ancillary"):
        raise ValueError(kind)
    is_ff = kind == "fieldsfile"
    dx, dy = 360.0 / nx, 180.0 / (ny + 1)
    start_lon, start_lat = 0.0, -90.0 + dy

    flh = _ints(FLH_WORDS)
    flh[0] = 20  # dataset format version (MASS)
    flh[1] = 1  # atmosphere
    flh[2] = 1  # hybrid vertical coordinate
    flh[3] = 0  # global grid
    flh[4] = 3 if is_ff else 4  # dataset type
    flh[5] = 1 if is_ff else IMDI  # run identifier
    flh[7] = 1  # Gregorian calendar
    flh[8] = 6  # ENDGame grid staggering
    flh[9] = IMDI if is_ff else 0  # ancillary: single time
    flh[10] = 1  # regular lon/lat grid
    flh[11] = 1302  # vn13.2
    first = [2000, 1, 1, 0, 0, 0, 1]
    flh[20:27] = first
    flh[27:34] = first
    flh[34:41] = [2026, 10, 16, 0, 0, 0, 290] if is_ff else [0] * 7

    pos = FLH_WORDS + 1  # next free 1-based word
    parts: dict[int, bytes] = {}

    def place(start_word: int, dims, raw: bytes) -> None:
        nonlocal pos
        flh[start_word - 1] = pos
        for w, d in zip(range(start_word + 1, start_word + 1 + len(dims)), dims, strict=True):
            flh[w - 1] = d
        parts[pos] = raw
        pos += len(raw) // WORD

    # integer constants
    ic = _ints(46 if is_ff else 15)
    ic[5], ic[6] = nx, ny
    ic[7] = num_p_levels if is_ff else 1
    if is_ff:
        ic[8] = num_p_levels
        ic[9] = 4
        ic[10] = num_p_levels
        ic[12] = num_p_levels
        ic[16] = 2
        ic[20] = IMDI
        ic[23] = 1
        ic[27] = 4
    else:
        ic[2] = 1
        ic[14] = 1
    place(100, [ic.size], ic.tobytes())

    # real constants
    rc = _reals(38 if is_ff else 6)
    rc[0], rc[1], rc[2], rc[3], rc[4], rc[5] = dx, dy, start_lat, start_lon, 90.0, 0.0
    if is_ff:
        rc[15] = 80000.0
        rc[28] = RMDI
    place(105, [rc.size], rc.tobytes())

    # level dependent constants (fieldsfiles only), Fortran order (levels vary fastest)
    if is_ff:
        nlev = num_p_levels + 1
        ldc = np.full((8, nlev), RMDI, dtype=">f8")
        eta_theta = np.linspace(0.0, 1.0, nlev)
        eta_rho = np.append((eta_theta[:-1] + eta_theta[1:]) / 2, RMDI)
        ldc[0] = eta_theta
        ldc[1] = eta_rho
        ldc[2] = 0.9
        soil = [0.1, 0.25, 0.65, 2.0][:nlev]
        ldc[3, : len(soil)] = soil
        ldc[4] = eta_theta * 80000.0
        ldc[5] = 0.0
        ldc[6] = np.where(eta_rho == RMDI, RMDI, eta_rho * 80000.0)
        ldc[7] = np.where(eta_rho == RMDI, RMDI, 0.0)
        place(110, [nlev, 8], ldc.tobytes())

    if row_col:
        rdc = np.full((2, ny + 1), RMDI, dtype=">f8")
        rdc[0] = np.linspace(start_lat, start_lat + ny * dy, ny + 1)
        rdc[1] = rdc[0] - dy / 2
        place(115, [ny + 1, 2], rdc.tobytes())
        cdc = np.full((2, nx), RMDI, dtype=">f8")
        cdc[0] = start_lon + dx * np.arange(nx)
        cdc[1] = cdc[0] - dx / 2
        place(120, [nx, 2], cdc.tobytes())

    if extra_constants is not None:
        place(130, [len(extra_constants) // WORD], extra_constants)
    if temp_historyfile is not None:
        place(135, [len(temp_historyfile) // WORD], temp_historyfile)

    # lookup
    nentries = nfields + blank_entries
    lookup_start = pos
    flh[149], flh[150], flh[151] = lookup_start, LOOKUP_LEN, nentries
    pos += LOOKUP_LEN * nentries
    data_start = _sector_ceil(pos - 1) + 1
    flh[159], flh[160] = data_start, nfields * SECTOR

    rng = np.random.default_rng(data_seed)
    lookup = np.zeros((nentries, LOOKUP_LEN), dtype=">i8")
    data = bytearray()
    for i in range(nfields):
        lbegin = (data_start - 1) + i * SECTOR
        ints = _ints(45, 0)
        ints[0:6] = [2000, 1, 1, 0, 0, 0]
        ints[6:12] = [2000, 1, 1, 0, 0, 0]
        ints[12] = 1  # LBTIM: validity time only, Gregorian
        ints[13] = 0
        ints[14] = nx * ny  # LBLREC
        ints[15] = 1  # LBCODE regular lat/lon
        ints[16] = 0  # LBHEM global
        ints[17], ints[18] = ny, nx
        ints[19] = 0  # LBEXT
        ints[20] = 0  # LBPACK: unpacked
        ints[21] = 3  # LBREL
        ints[22] = 16 if is_ff else 73  # LBFC
        ints[25] = 8 if is_ff else 129  # LBVC pressure / surface
        ints[28] = lbegin  # LBEGIN (0-based word offset)
        ints[29] = SECTOR  # LBNREC (sector-rounded disk length)
        ints[30] = 900  # LBPROJ
        ints[32] = (i + 1) if is_ff else 9999  # LBLEV
        ints[37] = 13021111  # LBSRCE
        ints[38] = 1  # LBUSER1: real data
        ints[39] = lbegin + 1  # LBUSER2: start address relative to FLH
        ints[41] = 16203 if is_ff else 30 + i  # STASH code
        ints[44] = 1  # LBUSER7: atmosphere
        reals = _reals(19, 0.0)
        reals[5] = -99.0  # BACC
        reals[6] = [1000.0, 850.0, 500.0, 250.0, 100.0][i % 5] if is_ff else 0.0  # BLEV
        reals[10], reals[11] = 90.0, 0.0  # BPLAT / BPLON
        reals[13] = start_lat - dy  # BZY
        reals[14] = dy  # BDY
        reals[15] = start_lon - dx  # BZX
        reals[16] = dx  # BDX
        reals[17] = RMDI  # BMDI
        reals[18] = 1.0  # BMKS
        lookup[i, :45] = ints
        lookup[i, 45:] = np.frombuffer(reals.tobytes(), dtype=">i8")
        field = (rng.random((ny, nx)) * 10 + 100 * (i + 1)).astype(">f8")
        data += field.tobytes()
        data += b"\0" * ((SECTOR - nx * ny) * WORD)
    if blank_entries:
        lookup[nfields:, :] = -99
        lookup[nfields:, 45:] = np.frombuffer(_reals(19, -99.0).tobytes(), dtype=">i8")

    total_words = (data_start - 1) + nfields * SECTOR
    buf = bytearray(total_words * WORD)
    buf[: FLH_WORDS * WORD] = flh.tobytes()
    for start, raw in parts.items():
        off = (start - 1) * WORD
        buf[off : off + len(raw)] = raw
    off = (lookup_start - 1) * WORD
    buf[off : off + lookup.nbytes] = lookup.tobytes()
    off = (data_start - 1) * WORD
    buf[off : off + len(data)] = data

    out = Path(path)
    out.write_bytes(bytes(buf))
    return out


def read_word(path, word: int) -> int:
    """Return the 1-based ``word`` of a file as a Python int."""
    with open(path, "rb") as fh:
        fh.seek((word - 1) * WORD)
        return int.from_bytes(fh.read(WORD), "big", signed=True)


def write_word(path, word: int, value: int) -> None:
    """Overwrite the 1-based ``word`` of a file."""
    with open(path, "r+b") as fh:
        fh.seek((word - 1) * WORD)
        fh.write(int(value).to_bytes(WORD, "big", signed=True))


def write_bytes_at(path, offset: int, data: bytes) -> None:
    with open(path, "r+b") as fh:
        fh.seek(offset)
        fh.write(data)


def byteswap_file(src, dst, *, word_size: int = 8) -> Path:
    """Write a byte-swapped copy of ``src`` (used to fake little-endian files)."""
    raw = Path(src).read_bytes()
    arr = np.frombuffer(raw, dtype=f">i{word_size}").astype(f"<i{word_size}")
    Path(dst).write_bytes(arr.tobytes())
    return Path(dst)


def to_32bit_file(src, dst) -> Path:
    """Write a 32-bit big-endian version of the header words of ``src`` (fake 32-bit file)."""
    raw = Path(src).read_bytes()
    words = np.frombuffer(raw, dtype=">i8")
    clipped = np.clip(words, -(2**31), 2**31 - 1).astype(">i4")
    Path(dst).write_bytes(clipped.tobytes())
    return Path(dst)
