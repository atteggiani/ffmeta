"""Tests for the command line interface."""

from __future__ import annotations

import io
import json
import os
import stat
import sys

import pytest

import ffmeta
from ffmeta.cli import main
from fixtures import make_um_file, read_word


def run(capsys, *argv):
    code = main([str(a) for a in argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert f"ffmeta {ffmeta.__version__}" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "examples:" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_add_read_edit_remove(ff_file, capsys):
    original = ff_file.read_bytes()
    code, out, err = run(
        capsys,
        "add",
        ff_file,
        "--set",
        "a.b=3",
        "--set",
        "name=Some Name",
        "--json",
        '{"project": "x", "a": {"c": 1}}',
    )
    assert code == 0 and "metadata added" in out and err == ""
    assert ffmeta.read(ff_file) == {"project": "x", "a": {"b": 3, "c": 1}, "name": "Some Name"}

    code, out, _ = run(capsys, "read", ff_file)
    assert code == 0
    assert json.loads(out) == ffmeta.read(ff_file)
    code, out, _ = run(capsys, "read", ff_file, "--compact")
    assert code == 0 and "\n" not in out.strip()
    code, out, _ = run(capsys, "read", ff_file, "--key", "a.b")
    assert code == 0 and out.strip() == "3"
    code, out, _ = run(capsys, "read", ff_file, "--key", "name", "--raw")
    assert code == 0 and out.strip() == "Some Name"
    code, out, err = run(capsys, "read", ff_file, "--key", "nope")
    assert code == 1 and "not found" in err
    code, out, _ = run(capsys, "read", ff_file, "--envelope")
    assert code == 0 and json.loads(out)["section"] == "extra_constants"
    code, out, err = run(capsys, "read", ff_file, "--section", "temp_historyfile")
    assert code == 1 and "no ffmeta metadata" in err

    code, out, _ = run(
        capsys, "edit", ff_file, "--set", "a.b=4", "--unset", "project", "--unset", "missing"
    )
    assert code == 0 and "updated" in out
    assert ffmeta.read(ff_file) == {"a": {"b": 4, "c": 1}, "name": "Some Name"}
    code, out, _ = run(capsys, "edit", ff_file, "--replace", "--json", '{"only": true}')
    assert code == 0 and ffmeta.read(ff_file) == {"only": True}

    code, out, _ = run(capsys, "remove", ff_file)
    assert code == 0 and "removed" in out
    assert ff_file.read_bytes() == original


def test_add_from_json_file_and_stdin(ff_file, tmp_path, capsys, monkeypatch):
    jf = tmp_path / "m.json"
    jf.write_text('{"k": [1, 2, 3]}')
    code, _, _ = run(capsys, "add", ff_file, "--json-file", jf, "-o", tmp_path / "out.ff")
    assert code == 0 and ffmeta.read(tmp_path / "out.ff") == {"k": [1, 2, 3]}
    assert not ffmeta.has_metadata(ff_file)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"from": "stdin"}'))
    code, _, _ = run(
        capsys,
        "add",
        ff_file,
        "--json-file",
        "-",
        "--section",
        "temp_historyfile",
        "--reserve",
        "100",
        "--align-sector",
        "--no-hash",
    )
    assert code == 0
    assert ffmeta.read(ff_file) == {"from": "stdin"}
    assert read_word(ff_file, 135) > 0
    assert (ff_file.stat().st_size // 8) % 2048 == 0
    assert ffmeta.read(ff_file, envelope=True)["original_sha256"] is None


def test_error_exit_codes_without_tracebacks(ff_file, tmp_path, capsys):
    code, out, err = run(capsys, "add", ff_file)
    assert code == 2 and err.startswith("ffmeta: error:") and "Traceback" not in err
    code, out, err = run(capsys, "add", ff_file, "--json", "not json")
    assert code == 2 and "not valid JSON" in err
    code, out, err = run(capsys, "add", ff_file, "--json", "[1]")
    assert code == 2 and "JSON object" in err
    code, out, err = run(capsys, "add", ff_file, "--set", "novalue")
    assert code == 2 and "KEY=VALUE" in err
    code, out, err = run(capsys, "read", ff_file)
    assert code == 1 and "no ffmeta metadata" in err and "Traceback" not in err
    code, out, err = run(capsys, "remove", ff_file)
    assert code == 2 and "ffmeta: error:" in err
    code, out, err = run(capsys, "read", tmp_path / "missing.ff")
    assert code == 2 and "ffmeta: error:" in err
    junk = tmp_path / "junk.ff"
    junk.write_bytes(b"\0" * 4096)
    code, out, err = run(capsys, "add", junk, "--set", "a=1")
    assert code == 2 and "UM file" in err
    ffmeta.add(ff_file, {"a": 1})
    code, out, err = run(capsys, "add", ff_file, "--set", "a=1")
    assert code == 2 and "already present" in err
    code, out, err = run(capsys, "edit", ff_file, "--dry-run")
    assert code == 2 and "nothing to change" in err


def test_dry_run_writes_nothing(ff_file, capsys):
    before = ff_file.read_bytes()
    code, out, _ = run(capsys, "add", ff_file, "--set", "a=1", "--dry-run")
    assert code == 0 and "append" in out and "FLH word 130" in out
    assert ff_file.read_bytes() == before
    ffmeta.add(ff_file, {"a": 1})
    after = ff_file.read_bytes()
    code, out, _ = run(capsys, "edit", ff_file, "--set", "b=2", "--dry-run")
    assert code == 0 and "edit" in out and ff_file.read_bytes() == after
    code, out, _ = run(capsys, "remove", ff_file, "--dry-run")
    assert code == 0 and "truncate" in out and ff_file.read_bytes() == after


def test_verify_exit_codes(ff_file, tmp_path, capsys):
    original = tmp_path / "orig.ff"
    original.write_bytes(ff_file.read_bytes())
    code, out, _ = run(capsys, "verify", ff_file)
    assert code == 1 and "[FAIL] section" in out
    ffmeta.add(ff_file, {"a": 1})
    code, out, _ = run(capsys, "verify", ff_file, "-O", original)
    assert code == 0
    assert "[PASS] header" in out and "[PASS] reversible" in out and "0 failed" in out
    code, out, _ = run(capsys, "verify", ff_file, "--json")
    data = json.loads(out)
    assert code == 0 and data["ok"] is True
    assert {c["name"] for c in data["checks"]} >= {
        "header",
        "section",
        "overlap",
        "payload",
        "hash",
    }


def test_remove_warning_goes_to_stderr(tmp_path, capsys):
    from ffmeta import payload as p
    from ffmeta._layout import IMDI

    env = p.make_envelope(
        {"m": 1},
        section="extra_constants",
        original_header={"start": IMDI, "length": IMDI},
        original_size=0,
        original_sha256=None,
    )
    path = make_um_file(tmp_path / "mid.ff", extra_constants=p.encode(env))
    code, out, err = run(capsys, "remove", path)
    assert code == 0 and "removed" in out
    assert "ffmeta: warning:" in err and "not byte-identical" in err


def test_interactive_edit_with_editor(ff_file, tmp_path, capsys, monkeypatch):
    ffmeta.add(ff_file, {"a": 1})
    script = tmp_path / "editor.py"
    script.write_text(
        "import json, sys\n"
        "p = sys.argv[1]\n"
        "d = json.load(open(p))\n"
        "d['edited'] = True\n"
        "json.dump(d, open(p, 'w'))\n"
    )
    monkeypatch.setenv("VISUAL", f"{sys.executable} {script}")
    code, out, err = run(capsys, "edit", ff_file)
    assert code == 0 and "updated" in out, err
    assert ffmeta.read(ff_file) == {"a": 1, "edited": True}
    # editor that changes nothing
    noop = tmp_path / "noop.py"
    noop.write_text("import sys\n")
    monkeypatch.setenv("VISUAL", f"{sys.executable} {noop}")
    code, out, _ = run(capsys, "edit", ff_file)
    assert code == 0 and "unchanged" in out
    # failing editor
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nsys.exit(3)\n")
    monkeypatch.setenv("VISUAL", f"{sys.executable} {bad}")
    code, out, err = run(capsys, "edit", ff_file)
    assert code == 2 and "status 3" in err
    monkeypatch.delenv("VISUAL")
    monkeypatch.delenv("EDITOR", raising=False)
    code, out, err = run(capsys, "edit", ff_file)
    assert code == 2 and "EDITOR" in err


def test_python_m_entry_point(monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["ffmeta"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("ffmeta", run_name="__main__", alter_sys=True)
    assert exc.value.code == 2  # no command given -> argparse usage error


@pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX-specific")
def test_unwritable_file_is_an_error(ff_file, capsys):
    ff_file.chmod(stat.S_IRUSR)
    try:
        code, out, err = run(capsys, "add", ff_file, "--set", "a=1")
    finally:
        ff_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    if os.geteuid() == 0:
        pytest.skip("running as root")
    assert code == 2 and "ffmeta: error:" in err
    assert not ffmeta.has_metadata(ff_file)
