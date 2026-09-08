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


def test_site_stat_to_dict_is_strict_json():
    """`SiteStat.to_dict` survives `allow_nan=False`; a NaN percentile becomes `null`.

    `_percentile` returns NaN when there is nothing to rank against, so the non-finite path is
    reachable rather than hypothetical.
    """
    import json

    s = analysis.SiteStat(position=68, aa="S", rsa=0.052, rsa_percentile=float("nan"),
                          contact_count=11, contact_percentile=0.3, buried=True)
    out = json.loads(json.dumps(s.to_dict(), allow_nan=False))
    assert out["rsa_percentile"] is None and out["position"] == 68 and out["buried"] is True


def test_summarize_top_level_floats_are_strict_json(monkeypatch):
    """`summarize`'s own values are JSON-safe, not just the `SiteStat`s inside it.

    The README documents converting the sites with `.to_dict()`; that alone is not enough,
    because `mean_rsa` is NaN when no site has an RSA. Driven through `summarize` itself rather
    than through `_json_safe`, so removing the wrapper fails this test.
    """
    import json

    gapped = [analysis.SiteStat(position=1, aa="X", rsa=None, rsa_percentile=None,
                                contact_count=0, contact_percentile=0.0, buried=False)]
    monkeypatch.setattr(analysis, "functional_site_stats", lambda *a, **k: gapped)

    out = analysis.summarize("P62593")
    out["sites"] = [s.to_dict() for s in out["sites"]]
    assert out["mean_rsa"] is None, "an undefined mean must serialise as null, not NaN"
    json.dumps(out, allow_nan=False)
