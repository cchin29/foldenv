"""M5 assembler tests. The default embedding (Ankh, ~2 GB) is skipped by using
`embedding.model: none`; the real embedding path is an opt-in heavy test."""
import os
import tempfile

import pytest

from foldenv import config, context

pytest.importorskip("Bio")


def _online():
    import requests
    try:
        requests.get(config.load()["alphafold"]["api_base"] + "/P62593", timeout=15)
        return True
    except Exception:
        return False


import shutil  # noqa: E402

live = pytest.mark.skipif(
    not _online() or shutil.which("mkdssp") is None,
    reason="needs AlphaFold-DB + mkdssp",
)

# Config with embedding disabled so assembly runs without downloading PLM weights.
NO_EMB = {"embedding": {"model": "none"}}


@pytest.fixture(autouse=True)
def _clear():
    context.clear_cache()
    yield
    context.clear_cache()


@live
def test_sequence_from_structure_tem1():
    seq = context.get_sequence("P62593", config.load())
    assert len(seq) == 286
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWYX")


@live
def test_assemble_full_dict_tem1():
    cfg = config.load(overrides=NO_EMB)
    out = context.get_structural_context("P62593", 150, config=cfg)

    assert out["uniprot_id"] == "P62593"
    assert out["position"] == 150
    # wildtype identity must agree across the sequence and the DSSP record (same structure)
    assert out["wildtype_aa"] == context.get_sequence("P62593", cfg)[149]
    assert out["wildtype_aa"] == context.get_dssp("P62593", cfg)[150].aa
    assert out["secondary_structure"] in ("H", "E", "C")
    assert 0.0 <= out["rsa"] <= 1.0
    assert out["contact_count"] == 13                   # matches M3 smoke check
    assert len(out["nearest_contacts"]) <= 5
    assert all({"resnum", "aa", "distance"} <= set(c) for c in out["nearest_contacts"])
    assert 0.0 <= out["plddt"] <= 100.0
    # embedding disabled in this config
    assert out["embedding"] is None
    assert out["embedding_model"] is None
    # strict JSON: no NaN/Inf tokens leak into the dict
    import json
    json.dumps(out, allow_nan=False)


@live
def test_gap_position_degrades_gracefully(monkeypatch):
    # a position present in the sequence but absent from the structure (an 'X' gap-fill) must
    # null out the structural fields, not raise — extend the reported sequence past the model.
    cfg = config.load(overrides=NO_EMB)
    real_seq = context.get_sequence("P62593", cfg)
    monkeypatch.setattr(context, "get_sequence", lambda uid, c=None: real_seq + "XXX")
    out = context.get_structural_context("P62593", len(real_seq) + 2, config=cfg)
    assert out["wildtype_aa"] == "X"
    assert out["rsa"] is None
    assert out["secondary_structure"] is None
    assert out["contact_count"] == 0
    assert out["nearest_contacts"] == []
    assert out["plddt"] is None
    import json
    json.dumps(out, allow_nan=False)


@live
def test_gap_position_nulls_embedding(monkeypatch):
    # C1: at a gap-fill position the embedding must degrade to null WITH the other structural
    # fields, not return the 'X' placeholder's vector. Fake the embedder (non-null) so a
    # regression would surface as a non-None embedding.
    real_seq = context.get_sequence("P62593", config.load(overrides=NO_EMB))
    monkeypatch.setattr(context, "get_sequence", lambda uid, c=None: real_seq + "XXX")
    monkeypatch.setattr(
        context, "_protein_embedding",
        lambda uid, cfg: ([[0.0] * 8] * (len(real_seq) + 3), "fake-model"),
    )
    cfg = config.load()  # default model, but _protein_embedding is stubbed → no weights
    out = context.get_structural_context("P62593", len(real_seq) + 2, config=cfg)
    assert out["embedding"] is None
    assert out["embedding_model"] is None
    # a real (in-structure) position still gets the fake embedding, proving the stub is live
    real = context.get_structural_context("P62593", 150, config=cfg)
    assert real["embedding"] is not None and real["embedding_model"] == "fake-model"


@live
def test_position_out_of_range_raises():
    cfg = config.load(overrides=NO_EMB)
    with pytest.raises(IndexError):
        context.get_structural_context("P62593", 10_000, config=cfg)
    with pytest.raises(IndexError):
        context.get_structural_context("P62593", 0, config=cfg)


@live
def test_nearest_contacts_consistent_with_count():
    cfg = config.load(overrides=NO_EMB)
    out = context.get_structural_context("P62593", 1, config=cfg)  # N-terminus
    # count is the full shell; nearest is the truncated, sorted head
    dists = [c["distance"] for c in out["nearest_contacts"]]
    assert dists == sorted(dists)
    assert len(out["nearest_contacts"]) <= out["contact_count"]


# --- heavy: real default embedding (Ankh) end-to-end -----------------------------------

@pytest.mark.skipif(os.environ.get("RUN_HEAVY_EMB") != "1", reason="set RUN_HEAVY_EMB=1")
@live
def test_assemble_with_real_embedding():
    out = context.get_structural_context("P62593", 150)  # default cfg → Ankh
    assert out["embedding_model"] == "ankh"
    assert isinstance(out["embedding"], list)
    assert len(out["embedding"]) == 1536
    # second call for another position is served from the cached forward pass
    out2 = context.get_structural_context("P62593", 151)
    assert len(out2["embedding"]) == 1536
