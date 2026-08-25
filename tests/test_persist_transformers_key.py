"""The embedding cache key carries the transformers version; the DSSP key does not.

Ankh3 tokenization changed at transformers 4.50, so the same (protein, model) pair yields
numerically different embeddings on either side of that boundary. Keying the persisted
embedding on the accession and model alone would let two versions' tensors collide on one
path, and a later run would silently read the wrong one. DSSP is mkdssp plus coordinates, so
it must stay untouched by any of this — a transformers upgrade must not throw away a DSSP
cache that took mkdssp runs to build.
"""
import re

import pytest

from foldenv import config, persist
from foldenv.dssp import ResidueDSSP

torch = pytest.importorskip("torch")

# The transformers component of a cache key, e.g. `__t4.57__`.
_TAG_PART = re.compile(r"__t\d+\.\d+(?:__|\.)")


def _cfg(tmp_path, **cache_over):
    cache = {"dir": str(tmp_path)}
    cache.update(cache_over)
    return config.load(overrides={"cache": cache})


def _pretend_transformers(monkeypatch, tag):
    # `_transformers_tag` is lru_cached (the installed version cannot change mid-process);
    # replacing the module attribute is what lets one test span two versions.
    monkeypatch.setattr(persist, "_transformers_tag", lambda: tag)


def _sample_dssp():
    return {70: ResidueDSSP(resnum=70, aa="S", ss3="C", ss8="-", acc=12.3, rsa=0.052)}


# --- the tag ---------------------------------------------------------------------------

def test_transformers_tag_is_major_minor_or_na():
    assert re.fullmatch(r"\d+\.\d+|na", persist._transformers_tag())


def test_transformers_tag_matches_the_installed_version():
    pytest.importorskip("transformers")
    from importlib.metadata import version

    major, minor = version("transformers").split(".")[:2]
    assert persist._transformers_tag() == f"{major}.{minor}"


def test_patch_releases_share_a_tag(monkeypatch):
    # Patch-level keying would evict a good cache on every point release; the differences that
    # make two tensors incomparable land on minors. `_transformers_tag` reads the metadata at
    # call time, so patching `version` there is enough once the lru_cache is cleared.
    from importlib import metadata

    seen = set()
    try:
        for raw in ("4.57.0", "4.57.6"):
            monkeypatch.setattr(metadata, "version", lambda _n, raw=raw: raw)
            persist._transformers_tag.cache_clear()
            seen.add(persist._transformers_tag())
    finally:
        monkeypatch.undo()
        persist._transformers_tag.cache_clear()  # never leave a fake tag cached
    assert seen == {"4.57"}


def test_unreadable_transformers_version_degrades_to_na(monkeypatch):
    from importlib import metadata

    try:
        monkeypatch.setattr(metadata, "version", lambda _n: "some-vendor-build")
        persist._transformers_tag.cache_clear()
        assert persist._transformers_tag() == "na"
    finally:
        monkeypatch.undo()
        persist._transformers_tag.cache_clear()


# --- the embedding key -------------------------------------------------------------------

def test_embedding_key_carries_the_transformers_version(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _pretend_transformers(monkeypatch, "4.57")
    assert "__t4.57__" in persist._emb_path(cfg, "P62593", "ankh").name


def test_embedding_key_is_stable_within_one_version(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _pretend_transformers(monkeypatch, "4.44")
    first = persist._emb_path(cfg, "P62593", "ankh3_large")
    second = persist._emb_path(cfg, "P62593", "ankh3_large")
    assert first == second  # a cache that never hits is not a cache


def test_embedding_key_differs_across_transformers_versions(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _pretend_transformers(monkeypatch, "4.44")
    pre = persist._emb_path(cfg, "P62593", "ankh3_large")
    _pretend_transformers(monkeypatch, "4.50")
    post = persist._emb_path(cfg, "P62593", "ankh3_large")
    assert pre != post  # the 4.50 tokenization change must not read back as a hit


def test_cache_written_under_one_version_is_a_miss_under_another(tmp_path, monkeypatch):
    # The behaviour the key exists for, end to end through save/load.
    cfg = _cfg(tmp_path)
    t = torch.randn(4, 8)
    _pretend_transformers(monkeypatch, "4.44")
    persist.save_embedding(cfg, "P62593", "ankh3_large", t)
    assert persist.load_embedding(cfg, "P62593", "ankh3_large") is not None

    _pretend_transformers(monkeypatch, "4.50")
    assert persist.load_embedding(cfg, "P62593", "ankh3_large") is None  # miss, not stale hit

    _pretend_transformers(monkeypatch, "4.44")
    back = persist.load_embedding(cfg, "P62593", "ankh3_large")
    assert back is not None and torch.allclose(back, t)  # and the original is still there


def test_both_versions_can_coexist_on_disk(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    old, new = torch.randn(4, 8), torch.randn(4, 8)
    _pretend_transformers(monkeypatch, "4.44")
    persist.save_embedding(cfg, "P62593", "ankh3_large", old)
    _pretend_transformers(monkeypatch, "4.50")
    persist.save_embedding(cfg, "P62593", "ankh3_large", new)

    assert torch.allclose(persist.load_embedding(cfg, "P62593", "ankh3_large"), new)
    _pretend_transformers(monkeypatch, "4.44")
    assert torch.allclose(persist.load_embedding(cfg, "P62593", "ankh3_large"), old)
    assert len(list((tmp_path / "embeddings").glob("*.pt"))) == 2


def test_sdk_models_are_not_keyed_on_transformers(tmp_path, monkeypatch):
    # ESM C via the `esm` SDK never goes through transformers, so a transformers upgrade must
    # not invalidate its cache.
    cfg = _cfg(tmp_path)
    _pretend_transformers(monkeypatch, "4.44")
    p1 = persist._emb_path(cfg, "P62593", "esmc_600m")
    assert _TAG_PART.search(p1.name) is None
    persist.save_embedding(cfg, "P62593", "esmc_600m", torch.randn(4, 8))
    _pretend_transformers(monkeypatch, "5.15")
    assert persist._emb_path(cfg, "P62593", "esmc_600m") == p1
    assert persist.load_embedding(cfg, "P62593", "esmc_600m") is not None


def test_structure_aware_key_keeps_both_components(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _pretend_transformers(monkeypatch, "4.57")
    name = persist._emb_path(cfg, "P62593", "saprot").name
    assert "__t4.57__" in name and "__s" in name


def test_model_key_still_separates_within_one_version(tmp_path, monkeypatch):
    # The new component must not have collapsed the old ones.
    cfg = _cfg(tmp_path)
    _pretend_transformers(monkeypatch, "4.57")
    assert persist._emb_path(cfg, "P62593", "ankh") != persist._emb_path(
        cfg, "P62593", "ankh3_large"
    )
    assert persist._emb_path(cfg, "P62593", "ankh") != persist._emb_path(
        cfg, "Q00987", "ankh"
    )


# --- the DSSP key is untouched -----------------------------------------------------------

def test_dssp_key_ignores_the_transformers_version(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _pretend_transformers(monkeypatch, "4.44")
    pre = persist._dssp_path(cfg, "P62593")
    _pretend_transformers(monkeypatch, "5.15")
    assert persist._dssp_path(cfg, "P62593") == pre
    # matched as a whole key component: the RSA table name also starts with "__t".
    assert _TAG_PART.search(pre.name) is None


def test_dssp_cache_survives_a_transformers_upgrade(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _pretend_transformers(monkeypatch, "4.44")
    persist.save_dssp(cfg, "P62593", _sample_dssp())
    _pretend_transformers(monkeypatch, "5.15")
    got = persist.load_dssp(cfg, "P62593")
    assert got is not None and got[70].rsa == pytest.approx(0.052)
