"""L2 disk-persistence tests (L2 persistence) — no network, no mkdssp, no PLM weights.

We construct DSSP records and a small embedding tensor by hand and round-trip them through
`persist`, so these run fast in any `.venv-structctx`. `torch` is already a repo dep.
"""
import json

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
