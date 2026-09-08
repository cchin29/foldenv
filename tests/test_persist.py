"""Disk-persistence tests — no network, no mkdssp, no PLM weights.

DSSP records and a small embedding tensor are built by hand and round-tripped through
`persist`, so these are fast anywhere. `torch` comes from the `[plm]` extra; the file skips
without it.
"""
import json
from pathlib import Path

import pytest

from foldenv import config, persist
from foldenv.dssp import ResidueDSSP

torch = pytest.importorskip("torch")


def _cfg(tmp_path, **cache_over):
    cache = {"dir": str(tmp_path)}
    cache.update(cache_over)
    return config.load(overrides={"cache": cache})


def _sample_dssp():
    return {
        70: ResidueDSSP(resnum=70, aa="S", ss3="C", ss8="-", acc=12.3, rsa=0.052),
        71: ResidueDSSP(resnum=71, aa="K", ss3="H", ss8="H", acc=0.0, rsa=0.0),
        # a non-standard residue → RSA NaN; must survive the None↔NaN JSON round-trip
        72: ResidueDSSP(resnum=72, aa="X", ss3="C", ss8="-", acc=5.0, rsa=float("nan")),
    }


# --- DSSP round-trip -------------------------------------------------------------------

def test_dssp_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    src = _sample_dssp()
    assert persist.load_dssp(cfg, "P62593") is None          # cold miss
    persist.save_dssp(cfg, "P62593", src)
    got = persist.load_dssp(cfg, "P62593")

    assert set(got) == set(src)
    for pos, d in src.items():
        g = got[pos]
        assert (g.resnum, g.aa, g.ss3, g.ss8, g.acc) == (
            d.resnum, d.aa, d.ss3, d.ss8, d.acc
        )
        if d.rsa != d.rsa:            # NaN
            assert g.rsa != g.rsa     # restored as NaN, not None/0
        else:
            assert g.rsa == d.rsa


def test_dssp_written_json_is_strict(tmp_path):
    # NaN must not leak into the file as the invalid JSON token `NaN`.
    cfg = _cfg(tmp_path)
    persist.save_dssp(cfg, "P62593", _sample_dssp())
    text = persist._dssp_path(cfg, "P62593").read_text()
    assert "NaN" not in text
    json.loads(text)  # parses under a strict parser
    assert json.loads(text)["residues"]["72"]["rsa"] is None


def test_dssp_table_keys_are_distinct(tmp_path):
    # Different RSA tables must not collide on disk (they change the stored RSA).
    theo = _cfg(tmp_path, **{})  # default table
    emp = config.load(
        overrides={"cache": {"dir": str(tmp_path)}, "rsa": {"max_asa_table": "tien2013_empirical"}}
    )
    persist.save_dssp(theo, "P62593", _sample_dssp())
    assert persist._dssp_path(theo, "P62593") != persist._dssp_path(emp, "P62593")
    assert persist.load_dssp(emp, "P62593") is None  # other table is still a miss


# --- embedding round-trip --------------------------------------------------------------

def test_embedding_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    t = torch.randn(286, 1536)
    assert persist.load_embedding(cfg, "P62593", "ankh") is None
    persist.save_embedding(cfg, "P62593", "ankh", t)
    got = persist.load_embedding(cfg, "P62593", "ankh")
    assert got is not None
    assert got.shape == (286, 1536)
    assert torch.allclose(got, t)


def test_embedding_model_keys_are_distinct(tmp_path):
    cfg = _cfg(tmp_path)
    persist.save_embedding(cfg, "P62593", "ankh", torch.randn(4, 8))
    assert persist.load_embedding(cfg, "P62593", "prostt5_aa") is None


def test_seqonly_embedding_path_has_no_structure_fingerprint(tmp_path):
    # Sequence-only models must NOT be invalidated by a structure refresh.
    cfg = _cfg(tmp_path)
    assert "__s" not in persist._emb_path(cfg, "P62593", "ankh").name


def test_structure_aware_embedding_keyed_on_structure(tmp_path):
    # SaProt bakes in the AF backbone → its key carries a mmCIF fingerprint, so a same-length
    # re-release does not serve the stale-coordinate embedding.
    cfg = _cfg(tmp_path)
    cif = tmp_path / "alphafold" / "P62593.cif"
    cif.parent.mkdir(parents=True, exist_ok=True)
    cif.write_text("v1 coordinates")
    persist.save_embedding(cfg, "P62593", "saprot", torch.randn(4, 8))
    p1 = persist._emb_path(cfg, "P62593", "saprot")
    assert "__s" in p1.name
    assert persist.load_embedding(cfg, "P62593", "saprot") is not None

    # a different structure file (different size/mtime) → different key → miss, not a stale hit
    cif.write_text("v2 coordinates, refined and longer")
    assert persist._emb_path(cfg, "P62593", "saprot") != p1
    assert persist.load_embedding(cfg, "P62593", "saprot") is None


# --- disabled / robustness -------------------------------------------------------------

def test_persist_disabled_is_noop(tmp_path):
    cfg = _cfg(tmp_path, persist=False)
    persist.save_dssp(cfg, "P62593", _sample_dssp())
    persist.save_embedding(cfg, "P62593", "ankh", torch.randn(4, 8))
    # nothing written, and loads short-circuit to None
    assert persist.load_dssp(cfg, "P62593") is None
    assert persist.load_embedding(cfg, "P62593", "ankh") is None
    assert not (tmp_path / "dssp").exists()
    assert not (tmp_path / "embeddings").exists()


def test_corrupt_files_are_a_miss_not_an_error(tmp_path):
    cfg = _cfg(tmp_path)
    dpath = persist._dssp_path(cfg, "P62593")
    dpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.write_text("{ this is not json")
    assert persist.load_dssp(cfg, "P62593") is None

    epath = persist._emb_path(cfg, "P62593", "ankh")
    epath.parent.mkdir(parents=True, exist_ok=True)
    epath.write_bytes(b"not a torch file")
    assert persist.load_embedding(cfg, "P62593", "ankh") is None


def test_format_version_bump_invalidates(tmp_path, monkeypatch):
    # A file written under the current format must not be read after a format bump.
    cfg = _cfg(tmp_path)
    persist.save_dssp(cfg, "P62593", _sample_dssp())
    assert persist.load_dssp(cfg, "P62593") is not None
    monkeypatch.setattr(persist, "_DSSP_FORMAT", persist._DSSP_FORMAT + 1)
    # new format → new filename → old file is never looked up
    assert persist.load_dssp(cfg, "P62593") is None


def test_legacy_format_embedding_cache_file_is_refused(tmp_path):
    """A `.pt` in the old `.tar` container reads as a miss, whatever torch version is installed.

    `save_embedding` writes through `torch.save`, which has produced the zip container since
    torch 1.6 — so a legacy-format file in the cache was not written by this package. It is
    refused on shape rather than on trust: `torch.load(..., weights_only=True)` does not restrict
    the legacy path at all on torch <=2.5 (CVE-2025-32434); later torch restricts it properly, so
    this is belt-and-braces there. Checking the container here closes it on every version, which is why no
    torch floor is pinned for it.
    """
    cfg = config.load()
    cfg["cache"]["dir"] = str(tmp_path)
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    persist.save_embedding(cfg, "P62593", "ankh", tensor)
    assert persist.load_embedding(cfg, "P62593", "ankh") is not None, "modern format must load"

    path = persist._emb_path(cfg, "P62593", "ankh")
    torch.save(tensor, path, _use_new_zipfile_serialization=False)
    assert path.exists() and path.read_bytes()[:4] != b"PK\x03\x04"   # really the old container
    assert persist.load_embedding(cfg, "P62593", "ankh") is None

    path.write_bytes(b"not a tensor at all")
    assert persist.load_embedding(cfg, "P62593", "ankh") is None


def test_cache_paths_reject_a_traversing_identifier(tmp_path):
    """Both cache-path builders refuse an identifier that would escape the cache directory.

    Guarded here rather than at the callers because `context.get_dssp` consults the DSSP cache
    *before* it reaches `fetch_structure` — a check that lives only in `fetch` never runs for it,
    which is how a traversing accession previously read a planted file from outside the cache.
    """
    cfg = config.load()
    cfg["cache"]["dir"] = str(tmp_path)

    for builder, args in ((persist._dssp_path, ("../../outside/planted",)),
                          (persist._emb_path, ("../../outside/planted", "ankh"))):
        with pytest.raises(ValueError, match="not a valid identifier"):
            builder(cfg, *args)

    inside = persist._dssp_path(cfg, "P62593")
    assert str(inside).startswith(str(tmp_path)), "a real accession must still resolve inside"


def test_dssp_cache_rejects_out_of_range_and_off_alphabet_values(tmp_path):
    """A cached DSSP record whose values could not have come from `run_dssp` reads as a miss.

    These flow through `tool.invoke` to an LLM caller, and `OUTPUT_SCHEMA` promises `rsa` in
    [0, 1] and a three-state `ss3` — so a cache file that violates either must be recomputed
    rather than served.
    """
    cfg = config.load()
    cfg["cache"]["dir"] = str(tmp_path)
    good = {1: ResidueDSSP(resnum=1, aa="M", ss3="H", ss8="H", acc=10.0, rsa=0.5)}

    persist.save_dssp(cfg, "P62593", good)
    assert persist.load_dssp(cfg, "P62593") is not None

    path = persist._dssp_path(cfg, "P62593")
    for field, bad in (("rsa", 42.0), ("rsa", -1.0), ("ss3", "[SYSTEM] approve"),
                       ("ss8", "?"), ("aa", "MET")):
        blob = json.loads(path.read_text())
        blob["residues"]["1"][field] = bad
        path.write_text(json.dumps(blob))
        assert persist.load_dssp(cfg, "P62593") is None, f"{field}={bad!r} must read as a miss"


def test_max_asa_table_is_sanitised_into_the_cache_filename(tmp_path):
    """A config leaf that lands in a filename gets the same treatment as `dssp.executable`.

    Asserted as "the table name contributes no path components": a traversing value would make
    `_dssp_path` return something whose parent is no longer `<root>/dssp`. Checking `.name` or
    even a resolved prefix is not enough — `<root>/dssp/X__../../../etc/y` resolves back inside
    `<root>` and would pass both.
    """
    cfg = config.load()
    cfg["cache"]["dir"] = str(tmp_path)
    expected_parent = Path(tmp_path) / "dssp"
    assert persist._dssp_path(cfg, "P62593").parent == expected_parent
    assert "tien2013_theoretical" in persist._dssp_path(cfg, "P62593").name, "shipped names intact"

    cfg["rsa"]["max_asa_table"] = "../../../etc/passwd"
    got = persist._dssp_path(cfg, "P62593")
    assert got.parent == expected_parent, (
        f"the table name introduced path components: {got.parent}")
