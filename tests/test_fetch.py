"""M1 fetch/parse tests. The live tests hit AlphaFold-DB and self-skip if offline."""
import tempfile

import pytest

from foldenv import config
from foldenv.fetch import NoAlphaFoldModelError, fetch_structure

pytest.importorskip("Bio", reason="Biopython not installed")
requests = pytest.importorskip("requests")

TEM1 = "P62593"  # TEM-1 β-lactamase — the plan's primary test protein


def _online() -> bool:
    try:
        requests.get(config.load()["alphafold"]["api_base"] + f"/{TEM1}", timeout=15)
        return True
    except requests.RequestException:
        return False


needs_network = pytest.mark.skipif(not _online(), reason="AlphaFold-DB unreachable")


@needs_network
def test_fetch_tem1_parses():
    with tempfile.TemporaryDirectory() as tmp:
        rec = fetch_structure(TEM1, cache_dir=tmp)
        assert rec.accession == TEM1
        assert rec.cif_path.exists()
        residues = [r for r in rec.structure.get_residues()]
        assert len(residues) > 100  # TEM-1 is ~286 aa
        # AF stores pLDDT in the B-factor column: values live on the 0–100 scale.
        bfactors = [a.get_bfactor() for a in rec.structure.get_atoms()]
        assert max(bfactors) <= 100.0 and min(bfactors) >= 0.0


@needs_network
def test_fetch_is_cached():
    with tempfile.TemporaryDirectory() as tmp:
        rec1 = fetch_structure(TEM1, cache_dir=tmp)
        mtime = rec1.cif_path.stat().st_mtime
        rec2 = fetch_structure(TEM1, cache_dir=tmp)  # should reuse the cached file
        assert rec2.cif_path.stat().st_mtime == mtime


def test_select_model_entry_single():
    from foldenv.fetch import _select_model_entry
    entries = [{"uniprotAccession": "P62593", "cifUrl": "x"}]
    entry, own = _select_model_entry(entries, "P62593")
    assert entry is entries[0]
    assert len(own) == 1


def test_select_model_entry_filters_isoforms():
    # TP53-shaped: canonical + 8 isoforms → pick canonical, own has just the 1 (no fragment warn)
    from foldenv.fetch import _select_model_entry
    entries = [{"uniprotAccession": "P04637", "cifUrl": "canon"}] + [
        {"uniprotAccession": f"P04637-{i}", "cifUrl": f"iso{i}"} for i in range(2, 10)
    ]
    entry, own = _select_model_entry(entries, "P04637")
    assert entry["cifUrl"] == "canon"
    assert len(own) == 1  # isoforms excluded → no spurious multi-fragment warning


def test_select_model_entry_true_fragments():
    # genuine length-fragments share the accession (F1/F2) → own keeps both
    from foldenv.fetch import _select_model_entry
    entries = [
        {"uniprotAccession": "Q8WZ42", "cifUrl": "F1"},
        {"uniprotAccession": "Q8WZ42", "cifUrl": "F2"},
    ]
    _, own = _select_model_entry(entries, "Q8WZ42")
    assert len(own) == 2


def test_select_model_entry_warns_on_fallback():
    import warnings
    from foldenv.fetch import _select_model_entry
    entries = [{"uniprotAccession": "P62593", "cifUrl": "x"}]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        entry, own = _select_model_entry(entries, "P99999")  # no exact match → fallback
        assert any("falling back" in str(x.message) for x in w)
        assert entry["cifUrl"] == "x"


@needs_network
def test_unknown_accession_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(NoAlphaFoldModelError):
            fetch_structure("Q0000FAKE0", cache_dir=tmp)
