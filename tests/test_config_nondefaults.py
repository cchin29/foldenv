"""Coverage for NON-default decisions.yaml choices, driven through the config/assembler
(the defaults are exercised throughout the rest of the suite). All live (network + mkdssp),
embedding disabled so no weights download."""
import shutil

import pytest

from foldenv import config, context

pytest.importorskip("Bio")

NO_EMB = {"embedding": {"model": "none"}}


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


# --- D3: non-default RSA MaxASA table, end-to-end + table-keyed cache ------------------

@live
def test_nondefault_rsa_table_changes_rsa_and_cache_is_table_keyed():
    theo = config.load(overrides={"rsa": {"max_asa_table": "tien2013_theoretical"}, **NO_EMB})
    sand = config.load(overrides={"rsa": {"max_asa_table": "sander_rost1994"}, **NO_EMB})

    d_theo = context.get_dssp("P62593", theo)
    d_sand = context.get_dssp("P62593", sand)

    # same protein/structure → same SS, but the two MaxASA tables give a different RSA scale
    assert d_theo[150].ss3 == d_sand[150].ss3
    assert any(
        abs(d_theo[n].rsa - d_sand[n].rsa) > 1e-4
        for n in d_theo
        if d_theo[n].rsa == d_theo[n].rsa and d_sand[n].rsa == d_sand[n].rsa  # both non-NaN
    ), "non-default RSA table did not change any residue's RSA"

    # the assembled dict reflects the selected table
    out_theo = context.get_structural_context("P62593", 286, config=theo)  # C-term, exposed
    out_sand = context.get_structural_context("P62593", 286, config=sand)
    assert out_theo["rsa"] != out_sand["rsa"]

    # DSSP cache is keyed by table → both coexist, neither clobbers the other
    assert context.get_dssp("P62593", theo) is d_theo
    assert context.get_dssp("P62593", sand) is d_sand
    assert d_theo is not d_sand


# --- D4: caching off --------------------------------------------------------------------

@live
def test_in_memory_false_is_uncached_but_correct():
    cfg = config.load(overrides={"cache": {"in_memory": False}, **NO_EMB})

    a = context.get_structure("P62593", cfg)
    b = context.get_structure("P62593", cfg)
    assert a is not b                      # not cached → a fresh parse each call
    assert len(context._STRUCTURE_CACHE) == 0
    assert len(context._DSSP_CACHE) == 0

    # output is still correct with caching disabled
    out = context.get_structural_context("P62593", 150, config=cfg)
    assert out["contact_count"] == 13
    assert out["secondary_structure"] in ("H", "E", "C")
    assert len(context._DSSP_CACHE) == 0   # still nothing cached after a full assemble


# --- D2: non-default pLDDT mask threshold, through the config ---------------------------

@live
def test_nondefault_plddt_mask_through_config():
    # threshold above the 0–100 pLDDT range masks every partner → empty shell
    masked = config.load(overrides={"plddt": {"mask_below": 200.0}, **NO_EMB})
    assert context.get_contacts("P62593", 150, masked).contact_count == 0

    # default mask (50) leaves a normal shell (sanity contrast)
    default = config.load(overrides=NO_EMB)
    assert context.get_contacts("P62593", 150, default).contact_count > 0
