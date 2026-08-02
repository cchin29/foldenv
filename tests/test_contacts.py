"""M3 tests: deterministic synthetic geometry + live TEM-1 invariants."""
import shutil
import tempfile

import numpy as np
import pytest

from foldenv import contacts as Ct

pytest.importorskip("Bio")


def _res(resnum, resname, ca_x, bfactor=99.0, cb=False):
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue

    r = Residue((" ", resnum, " "), resname, "")
    r.add(Atom("CA", np.array([ca_x, 0.0, 0.0]), bfactor, 1.0, " ", "CA", resnum, "C"))
    if cb:
        r.add(Atom("CB", np.array([ca_x, 1.5, 0.0]), bfactor, 1.0, " ", "CB", resnum, "C"))
    return r


def _structure(residues):
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure

    s, m, c = Structure("t"), Model(0), Chain("A")
    for r in residues:
        c.add(r)
    m.add(c)
    s.add(m)
    return s


def _line_scene(bfactors=None):
    # CA atoms on the x-axis at 0,4,7,12,16 Å (all Ala unless overridden)
    xs = [0.0, 4.0, 7.0, 12.0, 16.0]
    bf = bfactors or [99.0] * 5
    return _structure([_res(i + 1, "ALA", x, bfactor=b, cb=True) for i, (x, b) in enumerate(zip(xs, bf))])


def test_ca_contact_count_and_ordering():
    s = _line_scene()
    res = Ct.compute_contacts(s, 1, mode="ca", ca_cutoff=8.0, n_nearest=5)
    # from x=0: neighbors within 8 Å are x=4 (d4) and x=7 (d7); x=12,16 excluded
    assert res.contact_count == 2
    assert [c.resnum for c in res.nearest_contacts] == [2, 3]
    assert [round(c.distance) for c in res.nearest_contacts] == [4, 7]
    assert res.atom_mode == "CA" and res.cutoff == 8.0


def test_self_excluded():
    s = _line_scene()
    res = Ct.compute_contacts(s, 2, mode="ca", ca_cutoff=8.0)
    assert all(c.resnum != 2 for c in res.nearest_contacts)


def test_n_nearest_truncates():
    s = _line_scene()
    res = Ct.compute_contacts(s, 1, mode="ca", ca_cutoff=100.0, n_nearest=2)
    assert res.contact_count == 4          # all other residues within 100 Å
    assert len(res.nearest_contacts) == 2  # but only the 2 closest returned


def test_plddt_mask_drops_low_confidence_partner():
    # res 3 (x=7) gets pLDDT 30 (<50) → dropped; only res 2 remains within 8 Å of res 1
    s = _line_scene(bfactors=[99, 99, 30, 99, 99])
    res = Ct.compute_contacts(s, 1, mode="ca", ca_cutoff=8.0, plddt_mask_below=50.0)
    assert res.contact_count == 1
    assert [c.resnum for c in res.nearest_contacts] == [2]


def test_glycine_cb_fallback_to_ca():
    # a glycine (no CB) queried in cb-mode must fall back to CA, not crash
    s = _structure([_res(1, "GLY", 0.0, cb=False), _res(2, "ALA", 3.0, cb=True)])
    res = Ct.compute_contacts(s, 1, mode="cb", cb_cutoff=5.0, glycine_cb_fallback="ca")
    assert res.atom_mode == "CA"
    assert res.contact_count == 1


def test_prebuilt_index_matches_oneoff():
    # passing a cached index must give the identical result to building per-call
    s = _line_scene()
    idx = Ct.build_contact_index(s, mode="ca", plddt_mask_below=50.0)
    a = Ct.compute_contacts(s, 1, mode="ca", ca_cutoff=8.0)
    b = Ct.compute_contacts(s, 1, mode="ca", ca_cutoff=8.0, index=idx)
    assert a.contact_count == b.contact_count
    assert [c.resnum for c in a.nearest_contacts] == [c.resnum for c in b.nearest_contacts]


def test_index_self_excluded():
    # the shared tree includes the query residue's own atom; it must still be excluded (by
    # residue id, so it holds even when the index came from a different structure instance)
    s = _line_scene()
    idx = Ct.build_contact_index(s, mode="ca", plddt_mask_below=50.0)
    res = Ct.compute_contacts(s, 2, mode="ca", ca_cutoff=8.0, index=idx)
    assert all(c.resnum != 2 for c in res.nearest_contacts)


def test_missing_residue_raises():
    s = _line_scene()
    with pytest.raises(KeyError):
        Ct.compute_contacts(s, 999, mode="ca")


def test_bad_mode_raises():
    s = _line_scene()
    with pytest.raises(ValueError):
        Ct.compute_contacts(s, 1, mode="xx")


# --- live TEM-1 invariants -------------------------------------------------------------

def _online():
    import requests
    from foldenv import config
    try:
        requests.get(config.load()["alphafold"]["api_base"] + "/P62593", timeout=15)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _online(), reason="AlphaFold-DB unreachable")
def test_contacts_on_tem1_invariants():
    from foldenv.fetch import fetch_structure

    with tempfile.TemporaryDirectory() as tmp:
        rec = fetch_structure("P62593", cache_dir=tmp)

    res = Ct.compute_contacts(rec.structure, 150, mode="ca", ca_cutoff=8.0, n_nearest=5)
    assert res.contact_count > 0
    # nearest are a subset, ascending, within cutoff, self excluded
    dists = [c.distance for c in res.nearest_contacts]
    assert dists == sorted(dists)
    assert all(d <= 8.0 for d in dists)
    assert all(c.resnum != 150 for c in res.nearest_contacts)
    assert len(res.nearest_contacts) <= 5
    assert res.contact_count < 40  # sane upper bound for an 8 Å Cα shell


@pytest.mark.skipif(not _online(), reason="AlphaFold-DB unreachable")
def test_contacts_symmetric_tem1():
    # Tier-1 invariant: contacts are geometrically symmetric. Disable the pLDDT mask so the
    # relation is pure distance (masking a low-confidence partner can break symmetry by design).
    from foldenv.fetch import fetch_structure

    with tempfile.TemporaryDirectory() as tmp:
        rec = fetch_structure("P62593", cache_dir=tmp)

    def contacts_of(pos):
        r = Ct.compute_contacts(rec.structure, pos, mode="ca", ca_cutoff=8.0,
                                n_nearest=100, plddt_mask_below=0.0)
        return {c.resnum for c in r.nearest_contacts}

    partners = contacts_of(150)
    assert partners  # non-empty
    for p in partners:
        assert 150 in contacts_of(p), f"asymmetry: 150→{p} but not {p}→150"
