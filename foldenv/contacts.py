"""Contacts via KD-tree (Biopython NeighborSearch).

For a queried residue: count neighbors under the D1 cutoff (Cα–Cα ≤ 8 Å primary, or Cβ–Cβ ≤
5 Å), excluding self, and return the top-N nearest as `nearest_contacts`. Partners whose
pLDDT is below the D2 mask threshold are dropped as candidates (AF disordered loops otherwise
produce spurious contacts). Glycine has no Cβ → fall back to Cα (config
`contacts.glycine_cb_fallback`). pLDDT lives in the AF B-factor column.

`contact_count` is a burial/packing proxy: buried residues ~10–20 Cα-neighbors, surface ~4–8.
"""
from __future__ import annotations

from dataclasses import dataclass

from .constants import three2one


@dataclass
class Contact:
    resnum: int          # partner residue number (UniProt/AF numbering)
    aa: str              # 1-letter amino acid
    distance: float      # Å between representative atoms


@dataclass
class ContactResult:
    contact_count: int                 # neighbors under cutoff (self excluded, mask applied)
    nearest_contacts: list[Contact]    # up to n_nearest, ascending distance
    atom_mode: str                     # "CA" or "CB" — which atom actually anchored the query
    cutoff: float                      # Å cutoff used


def _one_letter(resname: str) -> str:
    return three2one.get(resname.strip().upper(), "X")


def _plddt(residue) -> float | None:
    """AF pLDDT for a residue = its B-factor (shared across the residue's atoms)."""
    if "CA" in residue:
        return float(residue["CA"].get_bfactor())
    for atom in residue:
        return float(atom.get_bfactor())
    return None


def _rep_atom(residue, mode: str, glycine_cb_fallback: str):
    """Representative atom for contacts: CA in ca-mode; CB in cb-mode (→ CA for glycine /
    any residue missing CB when `glycine_cb_fallback == 'ca'`). Returns (atom, atom_name)."""
    if mode == "ca":
        return (residue["CA"], "CA") if "CA" in residue else (None, None)
    if "CB" in residue:
        return residue["CB"], "CB"
    if glycine_cb_fallback == "ca" and "CA" in residue:
        return residue["CA"], "CA"
    return None, None


def _is_amino_acid(residue) -> bool:
    """Standard residue (not HETATM/water). AF models are all-standard, but be defensive."""
    return not residue.id[0].strip()


def _chain(structure, chain_id: str | None):
    model = next(structure.get_models())
    return model[chain_id] if chain_id is not None else next(model.get_chains())


def build_contact_index(
    structure,
    *,
    chain_id: str | None = None,
    mode: str = "ca",
    plddt_mask_below: float = 50.0,
    glycine_cb_fallback: str = "ca",
):
    """One KD-tree of representative atoms for a chain, reusable across positions.

    The tree covers every standard residue passing the pLDDT mask (self is NOT excluded
    here — `compute_contacts` filters the query residue out per call by object identity). The
    agent queries many positions of one protein, so caching this (per protein+mode+mask) turns
    per-position work from O(N log N) tree builds into a single search. Returns a Biopython
    `NeighborSearch`, or None if no atoms qualify.
    """
    from Bio.PDB import NeighborSearch

    atoms = []
    for r in _chain(structure, chain_id):
        if not _is_amino_acid(r):
            continue
        plddt = _plddt(r)
        if plddt is not None and plddt < plddt_mask_below:
            continue
        atom, _ = _rep_atom(r, mode, glycine_cb_fallback)
        if atom is not None:
            atoms.append(atom)
    return NeighborSearch(atoms) if atoms else None


def compute_contacts(
    structure,
    resnum: int,
    *,
    chain_id: str | None = None,
    mode: str = "ca",
    ca_cutoff: float = 8.0,
    cb_cutoff: float = 5.0,
    n_nearest: int = 5,
    plddt_mask_below: float = 50.0,
    glycine_cb_fallback: str = "ca",
    index=None,
) -> ContactResult:
    """Contacts of residue `resnum` (1-based UniProt numbering) in one chain.

    Args:
        mode: "ca" (Cα–Cα ≤ ca_cutoff) or "cb" (Cβ–Cβ ≤ cb_cutoff).
        plddt_mask_below: drop partner residues with pLDDT below this (D2).
        chain_id: chain to search; None → first chain (AF monomers are single-chain).
        index: a prebuilt `build_contact_index` result to reuse (must share mode / mask /
            glycine_cb_fallback). If None, a one-off tree is built for this call.

    Note: in cb-mode the glycine fallback measures a glycine partner at its Cα, so
    `nearest_contacts` distances can mix Cβ–Cβ and Cβ–Cα; `atom_mode` reflects only the query
    atom. Default config is ca-mode, where this does not arise.

    Raises:
        KeyError: `resnum` not present in the chain.
        ValueError: query residue lacks a representative atom, or bad `mode`.
    """
    if mode not in ("ca", "cb"):
        raise ValueError(f"mode must be 'ca' or 'cb', got {mode!r}")

    chain = _chain(structure, chain_id)
    cutoff = ca_cutoff if mode == "ca" else cb_cutoff

    query_res = next(
        (r for r in chain if _is_amino_acid(r) and r.id[1] == resnum), None
    )
    if query_res is None:
        raise KeyError(f"residue {resnum} not found in chain {chain.id}")
    q_atom, q_name = _rep_atom(query_res, mode, glycine_cb_fallback)
    if q_atom is None:
        raise ValueError(f"residue {resnum} has no {mode.upper()} atom for contacts")

    if index is None:
        index = build_contact_index(
            structure, chain_id=chain_id, mode=mode,
            plddt_mask_below=plddt_mask_below, glycine_cb_fallback=glycine_cb_fallback,
        )

    contacts: list[Contact] = []
    if index is not None:
        for atom in index.search(q_atom.coord, cutoff, level="A"):
            r = atom.get_parent()
            # Exclude self by residue id (het, resnum, icode) rather than object identity: the
            # cached KD-tree can be built from a different structure *instance* than query_res
            # (e.g. cache.in_memory=false), so `is` would miss it and count self as a contact.
            # Comparing .id still distinguishes insertion codes within the chain.
            if r.id == query_res.id:
                continue
            contacts.append(
                Contact(resnum=r.id[1], aa=_one_letter(r.resname), distance=float(atom - q_atom))
            )
        contacts.sort(key=lambda c: c.distance)

    return ContactResult(
        contact_count=len(contacts),
        nearest_contacts=contacts[:n_nearest],
        atom_mode=q_name,
        cutoff=cutoff,
    )
