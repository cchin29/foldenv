"""M6 Tier-1: crystal cross-check (AF vs experimental) on TEM-1 (P62593 vs 1BTL).

Observed on calibration: SS3 agreement 0.996, RSA Pearson 0.984, MAE 0.027 over 263
residues. Thresholds below sit well under those, leaving margin for DSSP/version variation
across machines while still catching a broken numbering-alignment or a regressed pipeline.
"""
import shutil

import pytest

from foldenv import config, context, validation

pytest.importorskip("Bio")


def _online():
    import requests
    try:
        requests.get(config.load()["alphafold"]["api_base"] + "/P62593", timeout=15)
        # also need RCSB for the experimental structure
        requests.get("https://files.rcsb.org/download/1BTL.cif", timeout=15, stream=True)
        return True
    except Exception:
        return False


live = pytest.mark.skipif(
    not _online() or shutil.which("mkdssp") is None,
    reason="needs AlphaFold-DB + RCSB + mkdssp",
)


@pytest.fixture(autouse=True)
def _clear():
    context.clear_cache()
    yield
    context.clear_cache()


@live
def test_crystal_crosscheck_tem1():
    rep = validation.crystal_crosscheck("P62593", "1BTL")
    assert rep.chain_id == "A"
    assert rep.n_compared > 200          # 1BTL orders ~263 of TEM-1's residues
    assert rep.ss3_agreement > 0.85      # observed 0.996
    assert rep.rsa_pearson > 0.85        # observed 0.984
    assert rep.rsa_mae < 0.10            # observed 0.027


@live
def test_crosscheck_numbering_alignment_not_identity():
    # sanity: the mapping must reconcile Ambler↔UniProt numbering, not assume equality.
    # If it wrongly assumed exp_resnum == af_pos, agreement would collapse.
    rep = validation.crystal_crosscheck("P62593", "1BTL")
    assert rep.ss3_agreement > 0.85 and rep.rsa_pearson > 0.85


def test_crosscheck_report_to_dict_is_strict_json():
    """`to_dict` survives `allow_nan=False`, including the under-two-residues NaN case.

    Pure: builds the report directly rather than running the cross-check, because what is being
    pinned is the serialisation contract, not the measurement. `json.dumps` writes a bare `NaN`
    token by default -- invalid JSON that many parsers reject -- so a report that reaches an
    agent framework has to carry `null` instead.
    """
    import json

    r = validation.CrosscheckReport(
        uniprot_id="P62593", pdb_id="1BTL", chain_id="A",
        n_compared=263, ss3_agreement=0.996, rsa_pearson=0.984, rsa_mae=0.027,
    )
    assert json.loads(json.dumps(r.to_dict(), allow_nan=False))["n_compared"] == 263

    nan = validation.CrosscheckReport(
        uniprot_id="P62593", pdb_id="1BTL", chain_id="A",
        n_compared=1, ss3_agreement=0.0, rsa_pearson=float("nan"), rsa_mae=float("nan"),
    )
    out = json.loads(json.dumps(nan.to_dict(), allow_nan=False))
    assert out["rsa_pearson"] is None and out["rsa_mae"] is None
