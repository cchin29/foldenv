"""Embedding-model registry + pure-logic tests (no weight downloads).

Weight-loading / forward-pass paths are covered by the (skipped-by-default) heavy tests at
the bottom; the fast tests here exercise the registry, SaProt SA-string construction, 3Di
gap-filling, and the ESM-C-off-MPS device routing.
"""
import numpy as np
import pytest

from foldenv import embedding as E


def test_registry_has_expected_models():
    assert set(E.EMBEDDING_MODELS) == {
        "ankh", "ankh3_large", "ankh3_xl", "prostt5_aa",
        "saprot", "saprot_1.3b", "esm2_3b", "esm2_650m", "esmc_600m", "esmc_6b",
    }
    expected_dim = {
        "ankh": 1536, "ankh3_large": 1536, "ankh3_xl": 2560, "prostt5_aa": 1024,
        "saprot": 1280, "saprot_1.3b": 1280, "esm2_3b": 2560, "esm2_650m": 1280,
        "esmc_600m": 1152, "esmc_6b": 2560,
    }
    for name, dim in expected_dim.items():
        assert E.embedding_dim(name) == dim, name
    # structure-aware = the SaProt family only
    assert {n for n in E.EMBEDDING_MODELS if E.get_model_spec(n).structure_aware} == {
        "saprot", "saprot_1.3b"
    }


def test_ankh_is_default():
    from foldenv import config
    assert config.load()["embedding"]["model"] == "ankh"


def test_saprot_is_structure_aware_and_others_not():
    assert E.get_model_spec("saprot").structure_aware is True
    for name in ("ankh", "prostt5_aa", "esmc_6b"):
        assert E.get_model_spec(name).structure_aware is False


def test_esmc_flagged_mps_incompatible():
    assert E.get_model_spec("esmc_6b").mps_ok is False
    assert E.get_model_spec("ankh").mps_ok is True


def test_esmc_600m_is_sdk_backed_and_mps_ok():
    # 600M runs locally via the `esm` SDK (MPS-friendly); 6B stays transformers/cluster-only.
    spec = E.get_model_spec("esmc_600m")
    assert spec.sdk is True and spec.mps_ok is True
    assert E.get_model_spec("esmc_6b").sdk is False
    assert E.get_model_spec("ankh").sdk is False


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        E.get_model_spec("not_a_model")


def test_sa_sequence_interleaves_and_masks_noncanonical():
    # canonical residue -> AA(upper)+3Di(lower); 'X' -> '#' AA-mask half
    assert E.sa_sequence("AC", "dp") == "AdCp"
    assert E.sa_sequence("X", "d") == "#d"
    # length is 2 chars per residue
    assert len(E.sa_sequence("ACDE", "dddd")) == 8


def test_esmc_reroutes_off_mps(monkeypatch):
    import torch

    # Simulate a Mac: get_device() returns MPS, no CUDA.
    monkeypatch.setattr("foldenv.plm.get_device", lambda: torch.device("mps"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning):
        dev = E.resolve_device("esmc_6b", device=None)
    assert dev.type == "cpu"  # rerouted off MPS


def test_mps_ok_model_keeps_mps(monkeypatch):
    import torch

    monkeypatch.setattr("foldenv.plm.get_device", lambda: torch.device("mps"))
    dev = E.resolve_device("ankh", device=None)
    assert dev.type == "mps"


def test_explicit_device_honored():
    dev = E.resolve_device("ankh", device="cpu")
    assert dev.type == "cpu"


# --- 3Di from a tiny synthetic structure (mini3di; no network) -------------------------

def _tiny_structure(n_res: int):
    """Build an in-memory Biopython structure with n_res glycines on a straight backbone."""
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom

    s = Structure("t")
    m = Model(0)
    c = Chain("A")
    for i in range(1, n_res + 1):
        res = Residue((" ", i, " "), "GLY", "")
        base = np.array([i * 3.8, 0.0, 0.0])
        for name, off in (("N", [-1, 0, 0]), ("CA", [0, 0, 0]), ("C", [1, 0, 0])):
            res.add(Atom(name, base + np.array(off, float), 0.0, 1.0, " ", name, i, "C"))
        c.add(res)
    m.add(c)
    s.add(m)
    return s


def test_structure_to_3di_length_matches_seq():
    pytest.importorskip("mini3di")
    pytest.importorskip("Bio")
    n = 8
    s = _tiny_structure(n)
    tdi = E.structure_to_3di(s, seq_len=n)
    assert len(tdi) == n
    assert tdi == tdi.lower()  # 3Di is lowercase


# --- heavy: real weights + forward pass (opt in with RUN_HEAVY_EMB=1) ------------------

heavy = pytest.mark.skipif(
    __import__("os").environ.get("RUN_HEAVY_EMB") != "1",
    reason="set RUN_HEAVY_EMB=1 to download weights and run the forward pass",
)


@heavy
def test_ankh_forward_shape():
    seq = "MKTAYIAKQR"
    per_res = E.embed_protein("ankh", seq, device="cpu")
    assert per_res.shape == (len(seq), 1536)


@heavy
def test_embedding_alignment_mutation_sensitive():
    """Row↔residue alignment + mutation sensitivity: a point mutation must change the
    embedding MOST at the mutated row (proves per_res[pos-1] indexes the right residue).
    """
    import numpy as np

    from foldenv import config, context

    context.clear_cache()
    seq = context.get_sequence("P62593", config.load())
    p = 150
    model, tok = E.load_embedder("ankh", "cpu")
    mut = seq[: p - 1] + ("A" if seq[p - 1] != "A" else "G") + seq[p:]
    e_wt = E.embed_protein("ankh", seq, model=model, tokenizer=tok)
    e_mut = E.embed_protein("ankh", mut, model=model, tokenizer=tok)

    d = (e_wt - e_mut).norm(dim=1).cpu().numpy()
    assert np.isfinite(d).all()
    assert int(d.argmax()) + 1 == p                 # largest change at the mutated residue
    assert d[p - 1] > 5 * np.median(d)              # and it dominates


@heavy
def test_ankh3_length_alignment_regression():
    """Regression for the ankh3 off-by-one: embed_sequence must return exactly len(seq) rows
    (the [NLU] prefix strip is anchored on the residue count, not hardcoded). Also alignment.
    """
    import numpy as np

    from foldenv import config, context

    context.clear_cache()
    seq = context.get_sequence("P62593", config.load())
    model, tok = E.load_embedder("ankh3_large", "cpu")
    e_wt = E.embed_protein("ankh3_large", seq, model=model, tokenizer=tok)
    assert e_wt.shape == (len(seq), 1536)          # correct length (was L-1 before the fix)
    p = 150
    mut = seq[: p - 1] + "A" + seq[p:]
    e_mut = E.embed_protein("ankh3_large", mut, model=model, tokenizer=tok)
    d = (e_wt - e_mut).norm(dim=1).cpu().numpy()
    assert int(d.argmax()) + 1 == p                # aligned: mutation localizes to its row


@heavy
def test_saprot_structure_aware_forward_and_alignment():
    """SaProt path: 3Di from the AF backbone (mini3di) → SA tokens → EsmForMaskedLM → [L,1280].
    A point mutation flips the SA token's AA half, so the embedding changes most at that row.
    """
    import numpy as np

    from foldenv import config, context

    context.clear_cache()
    cfg = config.load(overrides={"embedding": {"model": "saprot"}})
    seq = context.get_sequence("P62593", cfg)
    structure = context.get_structure("P62593", cfg).structure

    model, tok = E.load_embedder("saprot", "cpu")
    e_wt = E.embed_protein("saprot", seq, model=model, tokenizer=tok, structure=structure)
    assert e_wt.shape == (len(seq), 1280)           # structure-aware model dim
    assert np.isfinite(e_wt.numpy()).all()

    p = 150
    mut = seq[: p - 1] + "A" + seq[p:]
    e_mut = E.embed_protein("saprot", mut, model=model, tokenizer=tok, structure=structure)
    d = (e_wt - e_mut).norm(dim=1).cpu().numpy()
    assert int(d.argmax()) + 1 == p
    assert d[p - 1] > 5 * np.median(d)
