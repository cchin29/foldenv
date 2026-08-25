"""Release-metadata consistency: one version number, declared in three places.

`foldenv.__version__`, `pyproject.toml` and `CITATION.cff` are written by hand and drift
independently — a release that bumps one and forgets another ships a package whose citation
cites a different version than the code reports. This is the regression test for that whole
class of bug, so it compares all three rather than any one pair.

Neither packaging file ships inside the installed wheel, so the tests locate them relative to
this file and skip (never fail) when the suite runs against an installed copy.
"""
import re
from pathlib import Path

import pytest

import foldenv

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_CITATION = _ROOT / "CITATION.cff"
_CHANGELOG = _ROOT / "CHANGELOG.md"

pytestmark = pytest.mark.skipif(
    not (_PYPROJECT.is_file() and _CITATION.is_file()),
    reason="pyproject.toml / CITATION.cff are source-tree only, not shipped in the wheel",
)


def _toml_load(path: Path) -> dict:
    """Parse TOML with the stdlib when it exists, else `tomli`, else skip.

    `tomllib` is stdlib only from 3.11, and `requires-python` still admits 3.9/3.10, where
    `tomli` is not guaranteed to be installed either — a missing TOML parser is an untestable
    environment, not a failing invariant.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover — 3.9/3.10 path
        tomllib = pytest.importorskip("tomli", reason="no TOML parser (tomllib/tomli) available")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _cff_scalar(path: Path, key: str) -> str:
    """Top-level scalar from a CITATION.cff.

    Read line-wise instead of through a YAML parser: the only keys this file needs are
    top-level scalars, and matching at column 0 cannot pick up a same-named key nested inside
    the file's `references` block.
    """
    pat = re.compile(rf"^{re.escape(key)}:\s*(.+?)\s*$")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            return m.group(1).strip().strip("\"'")
    raise AssertionError(f"{path.name} has no top-level {key!r} key")


def _imported_from_repo() -> bool:
    try:
        Path(foldenv.__file__).resolve().relative_to(_ROOT)
    except ValueError:
        return False
    return True


def test_version_is_a_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", foldenv.__version__), foldenv.__version__


def test_pyproject_version_matches_package():
    if not _imported_from_repo():
        pytest.skip("imported foldenv is not this source tree; the comparison is meaningless")
    assert _toml_load(_PYPROJECT)["project"]["version"] == foldenv.__version__


def test_citation_version_matches_package():
    if not _imported_from_repo():
        pytest.skip("imported foldenv is not this source tree; the comparison is meaningless")
    assert _cff_scalar(_CITATION, "version") == foldenv.__version__


def _changelog_latest_version(path: Path) -> str:
    """The version of the topmost `## [x.y.z]` release heading in a Keep a Changelog file.

    Deliberately reads only the first such heading: it is the release being cut, and the older
    entries below it are history that must not move.
    """
    m = re.search(r"^## \[(\d+\.\d+\.\d+)\]", path.read_text(), re.MULTILINE)
    assert m, f"{path.name} has no `## [x.y.z]` release heading"
    return m.group(1)


def test_changelog_documents_this_version():
    """The release being tagged has an entry, and it is the newest one.

    Same drift class as the three version declarations: a bumped package with no changelog
    entry ships a release nobody can read the notes for, and it is invisible until someone
    goes looking after the fact.
    """
    if not _imported_from_repo():
        pytest.skip("imported foldenv is not this source tree; the comparison is meaningless")
    if not _CHANGELOG.is_file():
        pytest.skip("no CHANGELOG.md in this tree")
    assert _changelog_latest_version(_CHANGELOG) == foldenv.__version__


def test_all_three_declarations_agree():
    # The pairwise tests above localize a mismatch; this one states the invariant itself, so a
    # future fourth location can join the tuple without inventing a new pairing.
    if not _imported_from_repo():
        pytest.skip("imported foldenv is not this source tree; the comparison is meaningless")
    declared = {
        "foldenv.__version__": foldenv.__version__,
        "pyproject.toml": _toml_load(_PYPROJECT)["project"]["version"],
        "CITATION.cff": _cff_scalar(_CITATION, "version"),
    }
    assert len(set(declared.values())) == 1, declared


def test_citation_release_date_is_iso():
    # Not pinned to a literal date (that rots every release); the failure this catches is a
    # date left in a shape no citation manager can read.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", _cff_scalar(_CITATION, "date-released"))


def _declared_python_minors(project: dict) -> set:
    """The `3.x` minors claimed by the `Programming Language :: Python :: 3.x` classifiers."""
    return {
        int(c.rsplit(".", 1)[1])
        for c in project["classifiers"]
        if c.startswith("Programming Language :: Python :: 3.")
    }


def _requires_python_bounds(spec: str):
    """`(floor_minor, ceiling_minor_or_None)` parsed from a `requires-python` string.

    Written to survive the ceiling being absent, which is the intended state: `requires-python`
    is a resolver gate, so foldenv asserts only the floor it has actually tested downwards to.
    A ceiling is still parsed if one is ever added, so the checks below keep working either way.
    """
    lo = re.search(r">=\s*3\.(\d+)", spec)
    hi = re.search(r"<\s*3\.(\d+)", spec)
    assert lo, f"no parseable >=3.x floor in requires-python: {spec!r}"
    return int(lo.group(1)), (int(hi.group(1)) if hi else None)


def test_python_classifiers_agree_with_requires_python():
    """The classifiers are the statement of the tested range; `requires-python` is the gate.

    Deliberately not a two-way equality against a hardcoded range: with no ceiling there is no
    upper bound to compare to, and hardcoding one would have to be edited for every future
    CPython — the maintenance this project avoids by not capping in the first place. What must
    hold is that the two never contradict: no classifier below the floor (a version the resolver
    refuses but the metadata advertises), and none above a ceiling if one is ever added.
    """
    if not _imported_from_repo():
        pytest.skip("imported foldenv is not this source tree; the comparison is meaningless")
    project = _toml_load(_PYPROJECT)["project"]
    floor, ceiling = _requires_python_bounds(project["requires-python"])
    declared = _declared_python_minors(project)

    assert declared, "no `Programming Language :: Python :: 3.x` classifiers"
    assert min(declared) == floor, {
        "requires-python floor": f"3.{floor}",
        "lowest classifier": f"3.{min(declared)}",
        "why": "the floor must be advertised, and nothing below it may be",
    }
    if ceiling is not None:
        assert max(declared) < ceiling, {
            "requires-python ceiling": f"<3.{ceiling}",
            "highest classifier": f"3.{max(declared)}",
        }


def test_python_classifiers_have_no_gaps():
    """The claimed minors are contiguous.

    Catches the drift this guards in practice: adding a new CPython to the classifiers or the CI
    matrix and missing the one before it. A genuine hole in support is not expressible here, but
    for a pure-Python package it is not a real state either.
    """
    if not _imported_from_repo():
        pytest.skip("imported foldenv is not this source tree; the comparison is meaningless")
    declared = _declared_python_minors(_toml_load(_PYPROJECT)["project"])
    expected = set(range(min(declared), max(declared) + 1))
    assert declared == expected, {"missing": sorted(expected - declared)}


def test_ci_matrix_matches_classifiers():
    """Every advertised Python is actually built, and nothing is built that is not advertised.

    A version this package claims to support but never runs on is a claim nothing checks — the
    exact gap that let the matrix skip 3.10 while the metadata advertised it. Reads the matrix
    rather than restating it, so adding a version means editing one list and not two.
    """
    if not _imported_from_repo():
        pytest.skip("imported foldenv is not this source tree; the comparison is meaningless")
    ci = _ROOT / ".github" / "workflows" / "ci.yml"
    if not ci.is_file():
        pytest.skip("no CI workflow in this tree")
    yaml = pytest.importorskip("yaml")

    matrix = yaml.safe_load(ci.read_text())["jobs"]["test"]["strategy"]["matrix"]
    # `include:` entries add combinations (a version pinned to a specific runner image), so they
    # count as built versions just as much as the base list does.
    built = {str(v) for v in matrix.get("python-version", [])}
    built |= {str(e["python-version"]) for e in matrix.get("include", []) if "python-version" in e}

    declared = {f"3.{m}" for m in _declared_python_minors(_toml_load(_PYPROJECT)["project"])}
    assert built == declared, {
        "advertised but never built": sorted(declared - built),
        "built but not advertised": sorted(built - declared),
    }
