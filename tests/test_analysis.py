"""M7 Tier-2 tests: functional-site structural signatures on TEM-1 and TP53.

These encode the biological finding (observed, with margin): TEM-1 catalytic residues are
buried; TP53's hotspots split into a buried/packed structural hotspot (R175) vs an exposed
DNA-contact hotspot (R248). Live (network + mkdssp)."""
import shutil

import pytest

from foldenv import analysis, config, context

pytest.importorskip("Bio")


def _online():
    import requests
    try:
        requests.get(config.load()["alphafold"]["api_base"] + "/P62593", timeout=15)
        return True
    except Exception:
        return False


live = pytest.mark.skipif(
    not _online() or shutil.which("mkdssp") is None, reason="needs AlphaFold-DB + mkdssp"
)


@pytest.fixture(autouse=True)
def _clear():
    context.clear_cache()
    yield
    context.clear_cache()


@live
def test_structural_profile_shape_and_consistency():
    cfg = config.load()
    prof = context.structural_profile("P62593", cfg)
    assert len(prof) == 286
    # contact_count must match a direct get_contacts call at a sample position
    assert prof[150]["contact_count"] == context.get_contacts("P62593", 150, cfg).contact_count
    # fields present, rsa is None or in [0,1]
    for p in prof.values():
        assert p["rsa"] is None or 0.0 <= p["rsa"] <= 1.0
        assert p["ss3"] in ("H", "E", "C", None)


@live
def test_tem1_catalytic_all_buried():
    stats = analysis.functional_site_stats("P62593")
    assert {s.position for s in stats} == {68, 71, 164}
    # residue identity confirms the Ambler→UniProt mapping (S70/K73/E166 → 68/71/164)
    assert {(s.position, s.aa) for s in stats} == {(68, "S"), (71, "K"), (164, "E")}
    assert all(s.buried for s in stats)          # active-site cleft → all buried
    assert all(s.rsa < 0.15 for s in stats)      # observed max ~0.05


@live
def test_numbering_mismatch_raises():
    # wrong expected identity at a real position must be caught (guards a bad offset)
    with pytest.raises(ValueError, match="expected"):
        analysis.functional_site_stats("P62593", sites={68: "W"})  # 68 is S, not W


@live
def test_out_of_range_site_raises():
    with pytest.raises(ValueError, match="out of range"):
        analysis.functional_site_stats("P62593", sites={99999: "A"})


@live
def test_tp53_hotspots_split_structural_vs_contact():
    stats = {s.position: s for s in analysis.functional_site_stats("P04637")}
    r175, r248 = stats[175], stats[248]
    # R175: buried structural hotspot — low RSA, high packing percentile
    assert r175.buried and r175.rsa < 0.10
    assert r175.contact_percentile > 0.9
    # R248: exposed DNA-contact hotspot — clearly not buried
    assert not r248.buried and r248.rsa > 0.4
    # the two are structurally distinguishable on the burial axis
    assert r248.rsa > r175.rsa + 0.3


@live
def test_summarize_fields():
    s = analysis.summarize("P62593")
    assert s["n_sites"] == 3 and s["all_buried"] is True
    assert 0.0 <= s["mean_contact_percentile"] <= 1.0
