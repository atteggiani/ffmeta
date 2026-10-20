# ffmeta

Embed, read, edit, remove and verify free-text metadata in Unified Model
fieldsfiles and ancillary files, without disturbing anything the UM, Mule or
Iris read.

UM files (UMDP F03) have no place for free-text metadata. `ffmeta` stores a
small, self-describing JSON block in the unused `TEMP_HISTORYFILE` header
section as a post-processing step on an already-written file. The block is
appended at the end of the file; the only words of the original file that
change are the two header words that point at the section. `remove` restores
the original file byte for byte.

The only runtime dependency is NumPy. Mule and Iris are optional and are only
imported by `verify` and `from_mule`.

## Install

```sh
pip install .            # or: pip install -e .[dev]
```

Mule is not on PyPI. To run the Mule checks and tests, clone it and put
`mule/lib` on `PYTHONPATH` (see [Compatibility](#compatibility)).

## Command line

```sh
ffmeta add run.ff --set experiment=u-ab123 --set source.doi=10.1234/xyz
ffmeta add run.ff --json '{"project": "ACCESS", "contact": "me@example.org"}'
ffmeta add run.ff --json-file meta.json -o run_with_meta.ff
ffmeta read run.ff                       # pretty JSON
ffmeta read run.ff --key source.doi --raw
ffmeta read run.ff --envelope            # the full stored envelope
ffmeta edit run.ff --set version=2 --unset contact
ffmeta edit run.ff                       # open the metadata in $VISUAL / $EDITOR
ffmeta remove run.ff
ffmeta verify run.ff -O run_original.ff --mule --iris
```

- `--set a.b=3` creates nested keys; the value is parsed as JSON when possible
  and kept as a string otherwise. `--set` can be combined with `--json` /
  `--json-file` (`--set` wins). `--json-file -` reads stdin.
- `--align-sector` pads the block to a 2048-word sector boundary.
- `-o OUT` copies the file first and leaves the source untouched; `--dry-run`
  prints what would change and writes nothing.
- Exit status: 0 success; 1 for `read` without metadata or any failed `verify`
  check; 2 for errors (`ffmeta: error: ...` on stderr, no traceback).

## Python API

```python
import ffmeta

ffmeta.add("run.ff", {"experiment": "u-ab123", "source": {"doi": "10.1234/xyz"}})
ffmeta.read("run.ff")  # -> {'experiment': ..., 'source': {...}}
ffmeta.read("run.ff", envelope=True)  # the whole stored envelope
ffmeta.edit("run.ff", {"version": 2}, unset=["source.doi"])
ffmeta.has_metadata("run.ff")  # True
report = ffmeta.verify("run.ff", original="run_original.ff", mule=True, iris=True)
print(report)
report.ok
ffmeta.remove("run.ff")  # byte-identical to the original again

import mule

ffmeta.from_mule(mule.load_umfile("run.ff"))  # decode from a loaded Mule object
```

Signatures:

```python
add(path, meta, *, output=None, with_hash=True, align_sector=False,
  force=False, dry_run=False) -> Path
read(path, *, envelope=False) -> dict
edit(path, changes=None, *, replace=None, unset=(), output=None) -> Path
remove(path, *, output=None) -> Path
has_metadata(path) -> bool
from_mule(umfile) -> dict
verify(path, *, original=None, mule=False, iris=False) -> Report
```

Exceptions derive from `FfmetaError`: `FormatError` (not a 64-bit big-endian
UM file, truncated, inconsistent pointers), `NoMetadataError`,
`MetadataExistsError`, `SectionInUseError`, `CorruptMetadataError`.
`remove` issues an `FfmetaWarning` when the result cannot be byte-identical to
the pre-`add` file (it was rewritten after `add`, see below).

## What changes in the file

```
before add                              after add
word 1    ┌──────────────────────┐      word 1    ┌──────────────────────┐
          │ fixed length header  │                │ fixed length header  │  words 135-136
          │  (135,136) = (-32768,│                │  (135,136) = (S, L)  │  are the ONLY
          │             -32768)  │                │                      │  changed words
word 257  ├──────────────────────┤      word 257  ├──────────────────────┤
          │ integer constants    │                │ integer constants    │
          │ real constants       │                │ real constants       │
          │ level dep. constants │                │ level dep. constants │  byte-identical
          │ ...                  │                │ ...                  │
          │ LOOKUP               │                │ LOOKUP               │
          │ (sector padding)     │                │ (sector padding)     │
word 160  ├──────────────────────┤      word 160  ├──────────────────────┤
          │ DATA                 │                │ DATA                 │
          └──────────────────────┘ EOF   word S   ├──────────────────────┤
                                                  │ FFMETA01 | N | JSON  │  L words,
                                                  │ ... zero padding ... │  appended
                                                  └──────────────────────┘ EOF
```

`S = file_size_in_bytes / 8 + 1`. The block is stored in `TEMP_HISTORYFILE`.
Every component, every LOOKUP entry (so every `LBEGIN`/`LBNREC`) and all field
data stay where they were.

If another tool rewrites the file (Mule's `to_file` puts the section in its
canonical position before LOOKUP), the block is no longer last. `read` and
`verify` still work by pointer, but `edit` and `remove` refuse to change the
file (`FormatError`), because the block is only ever written at the end.

### Safety

Every write validates the file first, keeps the original bytes of everything
it overwrites or truncates in memory, `flush`es and `fsync`s, then re-opens the
file and checks that the metadata decodes to what was written and that the
pointers are consistent. On any failure the original bytes are restored (and
the file truncated back) before the exception is re-raised. Field data is
never loaded; hashing is streaming.

## Payload format

Big-endian 64-bit words:

| word | content |
|---|---|
| 1 | magic + version: the 8 ASCII bytes `FFMETA01` |
| 2 | N, the length in bytes of the JSON text |
| 3.. | the JSON text, ASCII only (`ensure_ascii`, sorted keys, compact, no NaN/Inf), zero-padded to whole words, then optional zero words (`--align-sector`) |

Because the text is ASCII, no word read as a float is NaN or Inf. The JSON is
an envelope:

```json
{"ffmeta": 1, "tool_version": "0.1.0", "section": "extra_constants",
 "created": "2026-10-16T01:00:00Z", "modified": "2026-10-16T01:00:00Z",
 "original_header": {"start": -32768, "length": -32768},
 "original_size": 65536, "original_sha256": "<hex or null>",
 "meta": {"...": "user metadata, any JSON object"}}
```

`original_sha256` is the SHA-256 of the file before `add` (what `remove`
restores); `--no-hash` stores `null`. The envelope is validated strictly on
read: an unknown magic prefix means the section holds something else
(`NoMetadataError`); a known prefix with a bad version, length, JSON or schema
is `CorruptMetadataError`. NumPy scalars/arrays, `datetime`s and `pathlib`
paths in user metadata are converted to JSON types; anything else raises
`TypeError`.

## `verify`

Read-only. Prints one `[PASS]/[WARN]/[FAIL]/[SKIP] name: message` line per
check and a summary; exits 1 on any FAIL.

| tier | check | what it tests |
|---|---|---|
| file alone | `header` | size multiple of 8, 64-bit big-endian UM file, plausible dataset type |
| | `section` | metadata found, pointers inside the file, block last or "not at the end" |
| | `overlap` | block overlaps no component, no LOOKUP, no field (bounded by `LBEGIN`/`LBNREC`, not word 161) |
| | `payload` | magic/version, declared length, JSON, envelope schema, trailing words zero (WARN) |
| | `hash` | SHA-256 of the virtually removed file equals the stored digest (mismatch is WARN) |
| `-O ORIGINAL` | `diff` | only the two pointer words and the appended tail differ |
| | `reversible` | `remove` on a copy equals the original byte for byte |
| `--mule` | `mule_version`, `mule_load`, `mule_metadata`, `mule_header`, `mule_components`, `mule_lookup`, `mule_data`, `mule_roundtrip` | see Compatibility |
| `--iris` | `iris_load`, `iris_compare` | `iris.load` works; same cubes as the original |

## Compatibility

**Mule.** Released Mule (up to 2026.09.1) reads header components
*sequentially* from the end of the fixed length header, ignoring their start
addresses; only LOOKUP is located by its pointer. Such a Mule loads an
`ffmeta`-processed file fine and sees identical field data, but it cannot see
the metadata (it reads unrelated words into `temp_historyfile`) and it drops
the block when it rewrites the file with `to_file`. MetOffice/mule
[PR #41](https://github.com/MetOffice/mule/pull/41) makes Mule seek to each
component's start address; with it the metadata is visible in
`umfile.temp_historyfile` (`ffmeta.from_mule`) and
survives a `to_file` round trip. `verify --mule` checks this behaviourally and
fails with an explicit message when the Mule in use cannot see the metadata.

**Iris.** Unaffected: Iris reads LOOKUP and the data and never looks at the
section. Cubes are identical before and after `add`.

**The UM.** This tool does not run the UM. The file stays well-formed under
UMDP F03 and nothing the UM reads has moved, but whether a particular UM
build is happy with a non-empty `TEMP_HISTORYFILE` section
placed after DATA has not been tested here. Try a real UM run with a processed
file before relying on it.

**What destroys the metadata:** the UM rewriting the file, conversion to PP or
netCDF, and Mule versions without PR #41 rewriting the file. Appending a block
to a file that is later rewritten by a seeking Mule keeps the metadata, but
moves it to the canonical position (see "What changes in the file").

**Versions the tests ran against** (synthetic fieldsfile and ancillary
fixtures, Python 3.12, NumPy 2.5.4):

- Mule `atteggiani/mule` branch `40-move-buffer-to-component-start`
  (PR #41, reports `__version__ == "2026.09.1"`): metadata visible, survives
  `to_file`, all checks pass.
- Mule `MetOffice/mule` release 2026.09.1: file loads, data identical,
  metadata not visible; `verify --mule` fails `mule_metadata` as designed.
- Iris 3.16.2: cubes identical before/after.

### Assumptions where UMDP F03 is silent

- `LBEGIN` (lookup word 29) is treated as a 0-based word offset from the start
  of the file (byte offset `LBEGIN * 8`), as Mule and the UM I/O layer do.
- A field's on-disk extent is `LBNREC` (word 30) words when positive, else
  `LBLREC` (word 15). Entries with `LBREL == -99` or `LBEGIN <= 0` have no data
  address and are counted, not failed.
- A component with start address `<= 0` is "not present" (F03 says negative;
  0 is not a valid 1-based address). Whatever the original pointer words
  were, they are stored in the envelope and restored exactly by `remove`.
- F03 does not require components to be in canonical order or to precede
  LOOKUP/DATA; placing the section after DATA relies on readers honouring the
  start address in the fixed length header.
- Ancillary fixtures use 15 integer and 6 real constants (F03 §3.2–3.3), with
  no level dependent constants (Appendix E).

## Development

```sh
pip install -e .[dev]
pytest                     # includes pytest-cov coverage reporting
PYTHONPATH=/path/to/mule/lib pytest   # with a Mule checkout
ruff check . && ruff format --check .
```

Package versions are derived from Git tags by setuptools-scm.

`tests/fixtures.py` generates synthetic fieldsfiles/ancillaries with NumPy
only; the `mule_seeks` session fixture probes the installed Mule behaviourally
so the Mule tests adapt to either variant.

## Licence

Apache License 2.0; see `LICENSE`.
