"""M2 tests: SS mapping + RSA normalization (pure), and a live mkdssp run on TEM-1."""
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
