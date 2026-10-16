# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-16

### Added

- `ffmeta add/read/edit/remove/verify` CLI and the matching Python API
  (`add`, `read`, `edit`, `remove`, `has_metadata`, `from_mule`, `verify`).
- `FFMETA01` payload format: magic word, byte length, ASCII-only JSON envelope
  with the original pointer words, original size and SHA-256.
- Trailing layout: the block is appended at the end of the file and only the
  two pointer words of `extra_constants` (default) or `temp_historyfile`
  change; `remove` restores the original file byte for byte.
- Pointer-based `read`/`edit`/`remove` for blocks that are no longer last
  (e.g. after a Mule rewrite).
- Transactional writes: pre-validation, in-memory undo, `fsync`, re-open
  self-check and rollback; `--dry-run`; `-o/--output`.
- `verify` with file-only, `--original`, `--mule` and `--iris` tiers.
- Synthetic NumPy-only fieldsfile/ancillary generator and a `mule_seeks`
  fixture for the test suite.
