"""M2 tests: SS mapping + RSA normalization (pure), and a live mkdssp run on TEM-1."""
import pathlib
import shutil
import tempfile

import pytest

from foldenv import dssp as D


def test_ss8_to_ss3_mapping():
    assert D.SS8_TO_SS3["H"] == "H"
    assert D.SS8_TO_SS3["G"] == "H"
    assert D.SS8_TO_SS3["I"] == "H"
    assert D.SS8_TO_SS3["E"] == "E"
    assert D.SS8_TO_SS3["B"] == "E"
    for coil in ("T", "S", "-", "P", " "):
        assert D.SS8_TO_SS3[coil] == "C"


def test_max_asa_tables_present_and_complete():
    for name, table in D.MAX_ASA_TABLES.items():
        assert set(table) == set("ACDEFGHIKLMNPQRSTVWY"), name


def test_max_asa_lookup_and_unknown():
    assert D.max_asa("A", "tien2013_theoretical") == 129.0
    assert D.max_asa("X", "tien2013_theoretical") is None  # non-standard → None


def test_tables_differ_between_references():
    # sanity: Sander & Tien give different MaxASA for the same residue
    assert D.max_asa("A", "sander_rost1994") != D.max_asa("A", "tien2013_theoretical")


def test_run_dssp_rejects_unknown_table():
    with pytest.raises(ValueError):
        D.run_dssp("nonexistent.cif", max_asa_table="not_a_table")


def test_missing_dssp_binary_says_what_to_install():
    """An absent DSSP must name itself, not surface as a Biopython traceback.

    DSSP is an external program, so this is the one missing dependency `pip install` cannot
    fix -- which makes the error text the entire remedy. Runs whether or not mkdssp is
    installed: the executable name is one this package would never find.
    """
    with pytest.raises(FileNotFoundError) as exc:
        D.run_dssp("nonexistent.cif", exe="mkdssp-does-not-exist")
    msg = str(exc.value)
    assert "mkdssp-does-not-exist" in msg           # names the binary it looked for
    assert "could not be launched" in msg
    assert "pip cannot install it" in msg           # says why there is no extra to name
    # One install line per platform. Matched on the distinctive part of each command rather
    # than the tool name: "conda" is a substring of "secondary", which appears in the first
    # sentence, so `"conda" in msg` is true however the message is edited.
    assert "brew install" in msg
    assert "conda-forge" in msg
    assert "apt-get install" in msg
    assert "dssp.executable" in msg                 # the config leaf for a non-standard name
    assert "get_structural_context" in msg          # an entry point that stops working
    assert "get_dssp" in msg                        # the most obviously DSSP-dependent one
    assert "tool.call" in msg                       # public, and under `invoke` rather than it
    assert "functional_site_stats" in msg           # public, and under `summarize` rather than it
    assert "get_contacts" in msg                    # and one that keeps working
    # The package supports 3.x -- `_detect_version` returns the parsed version so Biopython can
    # use the legacy command form -- so the message must not tell the reader v4 is required.
    assert "2.x/3.x work" in msg


@pytest.mark.parametrize("mode", ["absent", "not_executable", "directory", "bad_shebang",
                                  "through_a_file", "wrong_architecture"])
def test_unlaunchable_dssp_raises_without_warning(tmp_path, recwarn, mode):
    """A binary that cannot be launched gives the actionable error, and no warning.

    `_detect_version` warns that it assumed mkdssp 4.x when it cannot parse a version banner.
    None of these cases produces a banner, so that warning would send the reader after output
    that was never printed. They raise four different `OSError` subclasses -- ENOENT for an
    absent binary and for a missing interpreter behind a shebang, EACCES for a non-executable
    file and for a directory, ENOTDIR for a path leading through a regular file, and ENOEXEC
    (plain `OSError`) for a binary of the wrong architecture. A handler naming only
    `FileNotFoundError` and `PermissionError` would let the last two fall through to the warning,
    so it catches `OSError`, and these six modes are what hold it there.

    Asserts the *absence of any RuntimeWarning* rather than the warning's text, so rewording
    the warning cannot turn this green without fixing anything.
    """
    exe = str(tmp_path / f"dssp_{mode}")
    if mode == "not_executable":
        pathlib.Path(exe).write_text("x")
        pathlib.Path(exe).chmod(0o644)
    elif mode == "directory":
        pathlib.Path(exe).mkdir()
    elif mode == "bad_shebang":
        pathlib.Path(exe).write_text("#!/nonexistent/interpreter\n")
        pathlib.Path(exe).chmod(0o755)
    elif mode == "through_a_file":
        # `dssp.executable` pointing at a path *through* a regular file -- ENOTDIR, not ENOENT.
        plain = tmp_path / "a_regular_file"
        plain.write_text("x")
        exe = str(plain / "mkdssp")
    elif mode == "wrong_architecture":
        # Present and executable, but not a loadable binary: ENOEXEC, which is a plain OSError
        # and so escapes any handler that names subclasses instead.
        pathlib.Path(exe).write_bytes(b"\x00\x01\x02not a binary")
        pathlib.Path(exe).chmod(0o755)

    with pytest.raises(FileNotFoundError, match="could not be launched"):
        D.run_dssp("nonexistent.cif", exe=exe)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_detect_version_alone_also_raises_the_actionable_error():
    """Reached directly, not only through run_dssp's preflight -- the two must agree."""
    with pytest.raises(FileNotFoundError) as exc:
        D._detect_version("mkdssp-does-not-exist")
    assert "pip cannot install it" in str(exc.value)


# --- live: needs the mkdssp binary + network -------------------------------------------

needs_mkdssp = pytest.mark.skipif(shutil.which("mkdssp") is None, reason="mkdssp not installed")


def _online():
    import requests
    from foldenv import config
    try:
        requests.get(config.load()["alphafold"]["api_base"] + "/P62593", timeout=15)
        return True
    except Exception:
        return False


@needs_mkdssp
@pytest.mark.skipif(not _online(), reason="AlphaFold-DB unreachable")
def test_dssp_on_tem1():
    pytest.importorskip("Bio")
    from foldenv.fetch import fetch_structure

    with tempfile.TemporaryDirectory() as tmp:
        rec = fetch_structure("P62593", cache_dir=tmp)
        res = D.run_dssp(rec.cif_path, exe="mkdssp", max_asa_table="tien2013_theoretical")

    assert len(res) == 286  # TEM-1 canonical length
    assert all(r.ss3 in ("H", "E", "C") for r in res.values())
    assert all(0.0 <= r.rsa <= 1.0 for r in res.values())
    # residues are keyed by canonical (UniProt) number, contiguous from 1
    assert min(res) == 1 and max(res) == 286
    # regression: DSSP lowercases disulfide-bonded cysteines (a,b,c,…); run_dssp must
    # normalize them back to canonical uppercase 'C' (TEM-1 has a Cys–Cys bridge).
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    assert all(r.aa in canonical for r in res.values())
    assert any(r.aa == "C" for r in res.values())
