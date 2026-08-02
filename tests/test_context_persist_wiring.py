"""L2 read-through *wiring* tests for context.py (PLAN_0709 §4).

test_persist.py round-trips persist.py in isolation; this file verifies that
`context.get_dssp` and `context._protein_embedding` actually consult the disk cache
*before* the expensive path (mkdssp / mmCIF parse / PLM load+forward). We poison those
callables to raise, so a disk hit is proven by the fact that they are never reached.
No network, no mkdssp, no PLM weights.
"""
import pytest

from foldenv import config, context, persist
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
        72: ResidueDSSP(resnum=72, aa="X", ss3="C", ss8="-", acc=5.0, rsa=float("nan")),
    }


def _boom(*a, **k):
    raise AssertionError("expensive path invoked despite a disk hit")


@pytest.fixture(autouse=True)
def _clear():
    context.clear_cache()
    yield
    context.clear_cache()


# (a) DSSP disk hit is used by context.get_dssp (skips run_dssp AND get_structure).
def test_get_dssp_uses_disk_hit(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    persist.save_dssp(cfg, "P62593", _sample_dssp())
    context.clear_cache()  # drop L1 so the disk cache is the only source
    monkeypatch.setattr(context, "run_dssp", _boom)
    monkeypatch.setattr(context, "get_structure", _boom)

    got = context.get_dssp("P62593", cfg)
    assert set(got) == {70, 71, 72}
    assert got[70].rsa == 0.052
    assert got[71].ss3 == "H"
    assert got[72].rsa != got[72].rsa  # NaN restored, not None/0


# (b) Embedding disk hit avoids the embedder (never loads/forwards the PLM).
def test_protein_embedding_uses_disk_hit(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    model = cfg["embedding"]["model"]
    t = torch.randn(120, 16)
    persist.save_embedding(cfg, "P62593", model, t)
    context.clear_cache()
    monkeypatch.setattr(context, "load_embedder", _boom)
    monkeypatch.setattr(context, "embed_protein", _boom)
    monkeypatch.setattr(context, "get_structure", _boom)
    monkeypatch.setattr(context, "get_sequence", lambda uid, c=None: "A" * 120)

    per_res, mname = context._protein_embedding("P62593", cfg)
    assert mname == model
    assert tuple(per_res.shape) == (120, 16)
    assert torch.allclose(per_res, t)


# (c) Length guard rejects a stale disk tensor whose row count != sequence length.
def test_protein_embedding_length_guard_rejects_stale_tensor(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    model = cfg["embedding"]["model"]
    persist.save_embedding(cfg, "P62593", model, torch.randn(10, 16))  # stale rows
    context.clear_cache()
    monkeypatch.setattr(context, "load_embedder", _boom)
    monkeypatch.setattr(context, "embed_protein", _boom)
    monkeypatch.setattr(context, "get_structure", _boom)
    monkeypatch.setattr(context, "get_sequence", lambda uid, c=None: "A" * 20)

    with pytest.raises(ValueError):
        context._protein_embedding("P62593", cfg)


# (d) persist=False makes both read-through paths ignore any file on disk.
def test_persist_false_ignores_disk(tmp_path, monkeypatch):
    seed_cfg = _cfg(tmp_path)                # write files with persist ON
    off_cfg = _cfg(tmp_path, persist=False)  # same dir, persist OFF
    model = seed_cfg["embedding"]["model"]
    persist.save_dssp(seed_cfg, "P62593", _sample_dssp())
    persist.save_embedding(seed_cfg, "P62593", model, torch.randn(120, 16))
    assert persist._dssp_path(seed_cfg, "P62593").exists()
    assert persist._emb_path(seed_cfg, "P62593", model).exists()
    context.clear_cache()

    # With persist OFF both must fall through to the (poisoned) recompute path.
    monkeypatch.setattr(context, "run_dssp", _boom)
    monkeypatch.setattr(context, "load_embedder", _boom)
    monkeypatch.setattr(context, "embed_protein", _boom)
    monkeypatch.setattr(context, "get_structure", _boom)
    monkeypatch.setattr(context, "get_sequence", lambda uid, c=None: "A" * 120)

    with pytest.raises(AssertionError):  # reached get_structure/run_dssp => disk ignored
        context.get_dssp("P62593", off_cfg)
    with pytest.raises(AssertionError):  # reached embedder => disk ignored
        context._protein_embedding("P62593", off_cfg)
