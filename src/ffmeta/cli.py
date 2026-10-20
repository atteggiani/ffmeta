"""Command line interface: ``ffmeta add|read|edit|remove|verify``."""

from __future__ import annotations
import ffmeta

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import payload as _payload
from ._verify import verify
from .core import add, edit, plan_add, plan_edit, plan_remove, read, remove
from .errors import FfmetaError, FfmetaWarning, NoMetadataError

EXIT_OK, EXIT_NOT_FOUND, EXIT_ERROR = 0, 1, 2

_EXAMPLES = """\
examples:
  ffmeta add um_file.ff --set experiment=abc123 --set source.doi=10.1234/xyz
  ffmeta add um_file.ff --json '{"project": "ACCESS", "contact": "me@example.org"}'
  ffmeta add um_file.ff --json-file metadata.json -o run_with_meta.ff
  ffmeta read um_file.ff
  ffmeta read um_file.ff --key source.doi --raw
  ffmeta edit um_file.ff --set version=2 --unset contact
  ffmeta edit um_file.ff                       # open the metadata in $VISUAL / $EDITOR
  ffmeta remove um_file.ff
  ffmeta verify um_file.ff -O run_original.ff --mule --iris
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffmeta",
        description="Add, read, edit, remove and verify metadata in UM files.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"ffmeta {ffmeta.__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    def meta_inputs(p: argparse.ArgumentParser, required_hint: str) -> None:
        group = p.add_mutually_exclusive_group()
        group.add_argument("--json", metavar="TEXT", help="metadata as a JSON object")
        group.add_argument(
            "--json-file", metavar="PATH", help="read the JSON object from PATH ('-' for stdin)"
        )
        p.add_argument(
            "--set",
            metavar="KEY=VALUE",
            action="append",
            default=[],
            dest="sets",
            help="set a (dotted) key; VALUE is parsed as JSON when possible, otherwise "
            "kept as a string. Wins over --json/--json-file. " + required_hint,
        )

    p_add = sub.add_parser(
        "add",
        help="embed metadata in a file",
        description="Append metadata to FILE and point TEMP_HISTORYFILE at it.",
    )
    p_add.add_argument("file", metavar="FILE")
    meta_inputs(p_add, "")
    p_add.add_argument(
        "--no-hash", action="store_true", help="do not store the SHA-256 of the original file"
    )
    p_add.add_argument(
        "--align-sector",
        action="store_true",
        help="pad the block so the file ends on a 2048-word sector boundary",
    )
    p_add.add_argument(
        "--force", action="store_true", help="skip the 'file looks truncated or damaged' check"
    )
    p_add.add_argument(
        "-o", "--output", metavar="OUT", help="write to OUT and leave FILE untouched"
    )
    p_add.add_argument(
        "--dry-run", action="store_true", help="print what would change; write nothing"
    )
    p_add.set_defaults(func=cmd_add)

    p_read = sub.add_parser("read", help="print the metadata")
    p_read.add_argument("file", metavar="FILE")
    p_read.add_argument("--key", metavar="DOTTED.KEY", help="print only this key")
    p_read.add_argument("--raw", "-r", action="store_true", help="print strings without JSON quoting")
    p_read.add_argument("--compact", "-c", action="store_true", help="single-line JSON")
    p_read.add_argument("--envelope", action="store_true", help="print the whole envelope")
    p_read.set_defaults(func=cmd_read)

    p_edit = sub.add_parser(
        "edit",
        help="change the metadata",
        description="Without --json/--json-file/--set/--unset the metadata is "
        "opened in $VISUAL or $EDITOR.",
    )
    p_edit.add_argument("file", metavar="FILE")
    meta_inputs(p_edit, "")
    p_edit.add_argument(
        "--unset",
        metavar="DOTTED.KEY",
        action="append",
        default=[],
        dest="unsets",
        help="delete a (dotted) key; missing keys are ignored",
    )
    p_edit.add_argument(
        "--replace",
        action="store_true",
        help="replace the whole metadata with --json/--json-file instead of merging",
    )
    p_edit.add_argument(
        "-o", "--output", metavar="OUT", help="write to OUT and leave FILE untouched"
    )
    p_edit.add_argument(
        "--dry-run", action="store_true", help="print what would change; write nothing"
    )
    p_edit.set_defaults(func=cmd_edit)

    p_rm = sub.add_parser("remove", help="remove the metadata from the file")
    p_rm.add_argument("file", metavar="FILE")
    p_rm.add_argument("-o", "--output", metavar="OUT", help="write to OUT and leave FILE untouched")
    p_rm.add_argument(
        "--dry-run", action="store_true", help="print what would change; write nothing"
    )
    p_rm.set_defaults(func=cmd_remove)

    p_ver = sub.add_parser("verify", help="check a file carrying metadata (read-only)")
    p_ver.add_argument("file", metavar="FILE")
    p_ver.add_argument(
        "-O",
        "--original",
        metavar="ORIGINAL",
        help="the file as it was before 'add', for byte-level comparison",
    )
    p_ver.add_argument("--mule", action="store_true", help="also load the file with Mule")
    p_ver.add_argument("--iris", action="store_true", help="also load the file with Iris")
    p_ver.add_argument(
        "--json", action="store_true", dest="as_json", help="print the report as JSON"
    )
    p_ver.set_defaults(func=cmd_verify)
    return parser


# --- helpers ----------------------------------------------------------------


def _load_json_object(text: str, what: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{what} must be a JSON object")
    return value


def _collect_meta(args: argparse.Namespace) -> dict[str, Any] | None:
    """Metadata from --json / --json-file / --set, or None when none were given."""
    meta: dict[str, Any] | None = None
    if args.json is not None:
        meta = _load_json_object(args.json, "--json")
    elif args.json_file is not None:
        text = sys.stdin.read() if args.json_file == "-" else Path(args.json_file).read_text()
        meta = _load_json_object(text, "--json-file")
    if args.sets:
        meta = meta if meta is not None else {}
        _payload.apply_sets(meta, args.sets)
    return meta


def _dump(value: Any, compact: bool) -> str:
    if compact:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)


def _warn_to_stderr(
    message: Any, category: Any, filename: str, lineno: int, file: Any = None, line: Any = None
) -> None:
    prefix = (
        "ffmeta: warning: " if issubclass(category, FfmetaWarning) else f"{category.__name__}: "
    )
    print(prefix + str(message), file=sys.stderr)


# --- commands ---------------------------------------------------------------


def cmd_add(args: argparse.Namespace) -> int:
    meta = _collect_meta(args)
    if meta is None:
        raise ValueError("no metadata given: use --json, --json-file or --set")
    kwargs = dict(
        with_hash=not args.no_hash,
        align_sector=args.align_sector,
        force=args.force,
    )
    if args.dry_run:
        print(plan_add(args.file, meta, **kwargs).describe())
        return EXIT_OK
    out = add(args.file, meta, output=args.output, **kwargs)
    print(f"metadata added to {out}")
    return EXIT_OK


def cmd_read(args: argparse.Namespace) -> int:
    try:
        value: Any = read(args.file, envelope=args.envelope)
    except NoMetadataError as exc:
        print(f"ffmeta: {exc}", file=sys.stderr)
        return EXIT_NOT_FOUND
    if args.key:
        try:
            value = _payload.get_dotted(value, args.key)
        except KeyError:
            print(f"ffmeta: key {args.key!r} not found", file=sys.stderr)
            return EXIT_NOT_FOUND
    if args.raw and isinstance(value, str):
        print(value)
    else:
        print(_dump(value, args.compact))
    return EXIT_OK


def _edit_interactively(path: str) -> int:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        raise ValueError("no edit options given and neither $VISUAL nor $EDITOR is set")
    current = read(path)
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="ffmeta-", delete=False) as tmp:
        tmp.write(_dump(current, compact=False) + "\n")
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run([*shlex.split(editor), str(tmp_path)], check=False)
        if result.returncode != 0:
            raise ValueError(f"editor exited with status {result.returncode}; nothing changed")
        new_meta = _load_json_object(tmp_path.read_text(), "edited metadata")
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
    if new_meta == current:
        print("metadata unchanged")
        return EXIT_OK
    out = edit(path, replace=new_meta)
    print(f"metadata updated in {out}")
    return EXIT_OK


def cmd_edit(args: argparse.Namespace) -> int:
    meta = _collect_meta(args)
    if meta is None and not args.unsets:
        if args.dry_run or args.output:
            raise ValueError("nothing to change: use --json, --json-file, --set or --unset")
        return _edit_interactively(args.file)
    replace = meta if args.replace else None
    changes = None if args.replace else meta
    if args.dry_run:
        print(plan_edit(args.file, changes, replace=replace, unset=args.unsets).describe())
        return EXIT_OK
    out = edit(args.file, changes, replace=replace, unset=args.unsets, output=args.output)
    print(f"metadata updated in {out}")
    return EXIT_OK


def cmd_remove(args: argparse.Namespace) -> int:
    if args.dry_run:
        print(plan_remove(args.file).describe())
        return EXIT_OK
    out = remove(args.file, output=args.output)
    print(f"metadata removed from {out}")
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    report = verify(args.file, original=args.original, mule=args.mule, iris=args.iris)
    if args.as_json:
        print(json.dumps({"ok": report.ok, "checks": report.to_list()}, indent=2))
    else:
        print(str(report))
    return EXIT_OK if report.ok else EXIT_NOT_FOUND


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the exit status (0 ok, 1 not found / failed checks, 2 error)."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    with warnings.catch_warnings():
        warnings.simplefilter("always", FfmetaWarning)
        warnings.showwarning = _warn_to_stderr
        try:
            return int(args.func(args))
        except (FfmetaError, OSError, ValueError, TypeError) as exc:
            print(f"ffmeta: error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        except KeyboardInterrupt:
            print("ffmeta: error: interrupted", file=sys.stderr)
            return EXIT_ERROR


__all__: list[str] = ["main", "build_parser"]
