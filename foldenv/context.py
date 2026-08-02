"""M5 — the public `get_structural_context` API + per-protein in-memory cache.

`get_structural_context(uniprot_id, position)` assembles the residue's structural
neighborhood from the M1–M4 stages: fetch/parse (M1) → RSA + secondary structure (M2) →
contacts (M3) → PLM embedding (M4). Per D4, everything derived per protein (parsed structure,
DSSP, the contact KD-tree, the one PLM forward pass) is cached in memory so repeated position
queries on the same protein are cheap.
"""
from __future__ import annotations

from typing import Any

from .constants import three2one

from . import config as _config
from . import persist
from .contacts import ContactResult, build_contact_index, compute_contacts
from .dssp import ResidueDSSP, run_dssp
from .embedding import embed_protein, load_embedder
from .fetch import StructureRecord, fetch_structure

# Per-protein caches (D4, in-memory): accession → derived object.
_STRUCTURE_CACHE: dict[str, StructureRecord] = {}
# DSSP result depends on the accession, the RSA normalization table, and the DSSP executable,
# so key on all three — lets a comparison across MaxASA tables coexist rather than returning
# the first table's RSA.
_DSSP_CACHE: dict[tuple[str, str, str], dict[int, ResidueDSSP]] = {}
# Contact KD-tree, reused across all position queries of a protein. Keyed by everything that
# changes the tree's atom set: accession, atom mode, pLDDT mask, glycine fallback.
_CONTACT_INDEX_CACHE: dict[tuple, object] = {}
# Canonical sequence (from the AF structure) per accession.
_SEQUENCE_CACHE: dict[str, str] = {}
# Loaded PLM (model, tokenizer) per model name, and the one per-protein [L, dim] forward pass.
_EMBEDDER_CACHE: dict[str, tuple] = {}
_PROTEIN_EMBEDDING_CACHE: dict[tuple[str, str], Any] = {}


def get_structure(uniprot_id: str, cfg: dict | None = None) -> StructureRecord:
    """Fetch+parse (M1) with the per-protein in-memory cache (D4)."""
    cfg = cfg or _config.load()
    if cfg["cache"].get("in_memory") and uniprot_id in _STRUCTURE_CACHE:
        return _STRUCTURE_CACHE[uniprot_id]
    record = fetch_structure(
        uniprot_id,
        cache_dir=cfg["cache"]["dir"],
        api_base=cfg["alphafold"]["api_base"],
    )
    if cfg["cache"].get("in_memory"):
        _STRUCTURE_CACHE[uniprot_id] = record
    return record


def get_dssp(uniprot_id: str, cfg: dict | None = None) -> dict[int, ResidueDSSP]:
    """DSSP (M2): {resnum → ResidueDSSP} for the protein, cached per protein (D4)."""
    cfg = cfg or _config.load()
    # Key on everything run_dssp depends on: accession, RSA table, and the DSSP executable.
    key = (uniprot_id, cfg["rsa"]["max_asa_table"], cfg["dssp"]["executable"])
    if cfg["cache"].get("in_memory") and key in _DSSP_CACHE:
        return _DSSP_CACHE[key]
    # L2 disk cache: a hit skips both the mmCIF parse and the mkdssp shell-out.
    result = persist.load_dssp(cfg, uniprot_id)
    if result is None:
        record = get_structure(uniprot_id, cfg)
        result = run_dssp(
            record.cif_path,
            exe=cfg["dssp"]["executable"],
            max_asa_table=cfg["rsa"]["max_asa_table"],
        )
        persist.save_dssp(cfg, uniprot_id, result)
    if cfg["cache"].get("in_memory"):
        _DSSP_CACHE[key] = result
    return result


def _contact_index(uniprot_id: str, cfg: dict):
    """Cached contact KD-tree for a protein (D4) — built once, reused across positions."""
    c, mask = cfg["contacts"], cfg["plddt"]["mask_below"]
    key = (uniprot_id, c["primary"], mask, c["glycine_cb_fallback"])
    if cfg["cache"].get("in_memory") and key in _CONTACT_INDEX_CACHE:
        return _CONTACT_INDEX_CACHE[key]
    record = get_structure(uniprot_id, cfg)
    index = build_contact_index(
        record.structure,
        mode=c["primary"],
        plddt_mask_below=mask,
        glycine_cb_fallback=c["glycine_cb_fallback"],
    )
    if cfg["cache"].get("in_memory"):
        _CONTACT_INDEX_CACHE[key] = index
    return index


def get_contacts(
    uniprot_id: str, position: int, cfg: dict | None = None
) -> ContactResult:
    """Contacts (M3) for `position`, reusing the cached structure + KD-tree + config (D1/D2)."""
    cfg = cfg or _config.load()
    record = get_structure(uniprot_id, cfg)
    c = cfg["contacts"]
    return compute_contacts(
        record.structure,
        position,
        mode=c["primary"],
        ca_cutoff=c["ca_cutoff"],
        cb_cutoff=c["cb_cutoff"],
        n_nearest=c["n_nearest"],
        plddt_mask_below=cfg["plddt"]["mask_below"],
        glycine_cb_fallback=c["glycine_cb_fallback"],
        index=_contact_index(uniprot_id, cfg),
    )


def _sequence_from_structure(structure, chain_id: str | None = None) -> str:
    """Canonical AA sequence from the AF structure (residue i ↔ index i-1).

    AF monomer models are full-length and numbered to UniProt canonical numbering, so the
    string reads directly at `position-1`. Any numbering gap is filled with 'X' to keep that
    index alignment (F1-only for now; long multi-fragment proteins are a documented TODO).
    """
    chain = structure[0][chain_id] if chain_id is not None else next(structure[0].get_chains())
    by_num = {
        r.id[1]: three2one.get(r.resname.strip().upper(), "X")
        for r in chain
        if not r.id[0].strip()
    }
    if not by_num:
        return ""
    return "".join(by_num.get(i, "X") for i in range(1, max(by_num) + 1))


def get_sequence(uniprot_id: str, cfg: dict | None = None) -> str:
    """Canonical sequence for the protein (from the AF structure), cached per protein."""
    cfg = cfg or _config.load()
    if cfg["cache"].get("in_memory") and uniprot_id in _SEQUENCE_CACHE:
        return _SEQUENCE_CACHE[uniprot_id]
    seq = _sequence_from_structure(get_structure(uniprot_id, cfg).structure)
    if cfg["cache"].get("in_memory"):
        _SEQUENCE_CACHE[uniprot_id] = seq
    return seq


def _protein_embedding(uniprot_id: str, cfg: dict):
    """One per-protein PLM forward pass (M4), cached (D4). Returns ([L, dim] tensor, model).

    `embedding.model == "none"` skips embedding (→ (None, None)) — used to run the structural
    fields without downloading PLM weights. The loaded model is cached across proteins.
    """
    model_name = cfg["embedding"]["model"]
    if model_name == "none":
        return None, None
    key = (uniprot_id, model_name)  # output is device-independent
    if cfg["cache"].get("in_memory") and key in _PROTEIN_EMBEDDING_CACHE:
        return _PROTEIN_EMBEDDING_CACHE[key], model_name

    seq = get_sequence(uniprot_id, cfg)  # cheap (cached); also the length-guard reference

    # L2 disk cache: a hit skips both the forward pass AND loading ~2 GB of
    # PLM weights, so it must be checked before touching the embedder.
    per_res = persist.load_embedding(cfg, uniprot_id, model_name)
    from_disk = per_res is not None
    if per_res is None:
        device = cfg["embedding"].get("device", "auto")
        # Key the loaded model by (name, device) so a device switch doesn't silently reuse the
        # first-loaded placement (clear_cache keeps this cache, so it would otherwise persist).
        embedder_key = (model_name, str(device))
        if embedder_key not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[embedder_key] = load_embedder(model_name, device)
        model, tokenizer = _EMBEDDER_CACHE[embedder_key]
        per_res = embed_protein(
            model_name,
            seq,
            model=model,
            tokenizer=tokenizer,
            structure=get_structure(uniprot_id, cfg).structure,  # SaProt needs the backbone
        )
        persist.save_embedding(cfg, uniprot_id, model_name, per_res)

    # Guard the position↔row alignment the assembler relies on: any length mismatch (a model
    # that truncated a long sequence, OR a stale disk tensor from a different structure) must
    # fail loudly, not silently misalign every high position.
    if hasattr(per_res, "shape") and per_res.shape[0] != len(seq):
        # If the bad tensor came from the disk cache (vs a fresh forward pass), evict it so
        # the next run recomputes rather than re-loading the same poisoned/stale file forever.
        if from_disk:
            persist.evict_embedding(cfg, uniprot_id, model_name)
        raise ValueError(
            f"{model_name} returned {per_res.shape[0]} rows for a length-{len(seq)} sequence "
            f"({uniprot_id}); long/multi-fragment proteins are not yet supported."
        )
    if cfg["cache"].get("in_memory"):
        _PROTEIN_EMBEDDING_CACHE[key] = per_res
    return per_res, model_name


def structural_profile(uniprot_id: str, cfg: dict | None = None) -> dict[int, dict]:
    """Per-residue structural fields for the whole protein (embedding-free), cheap.

    Returns {position: {aa, rsa, ss3, contact_count, plddt}}. Reuses the cached DSSP + contact
    KD-tree, so it's one pass over positions with no re-fetch/re-DSSP. Handy for M7 analysis and
    for an agent that wants a protein-wide burial/packing map.
    """
    cfg = cfg or _config.load()
    seq = get_sequence(uniprot_id, cfg)
    dssp = get_dssp(uniprot_id, cfg)
    index = _contact_index(uniprot_id, cfg)
    structure = get_structure(uniprot_id, cfg).structure
    chain = next(structure[0].get_chains())
    res_by_num = {r.id[1]: r for r in chain if not r.id[0].strip()}
    c = cfg["contacts"]

    profile: dict[int, dict] = {}
    for pos in range(1, len(seq) + 1):
        ro = res_by_num.get(pos)
        d = dssp.get(pos)
        if ro is not None:
            cc = compute_contacts(
                structure, pos, mode=c["primary"], ca_cutoff=c["ca_cutoff"],
                cb_cutoff=c["cb_cutoff"], n_nearest=c["n_nearest"],
                plddt_mask_below=cfg["plddt"]["mask_below"],
                glycine_cb_fallback=c["glycine_cb_fallback"], index=index,
            ).contact_count
            plddt = float(ro["CA"].get_bfactor()) if "CA" in ro else None
        else:
            cc, plddt = 0, None
        rsa = d.rsa if (d is not None and d.rsa == d.rsa) else None  # None, not NaN
        profile[pos] = {
            "aa": seq[pos - 1],
            "rsa": rsa,
            "ss3": d.ss3 if d is not None else None,
            "contact_count": cc,
            "plddt": plddt,
        }
    return profile


def clear_cache() -> None:
    """Drop the in-memory per-protein caches (on-disk mmCIF cache is untouched).

    Leaves the loaded-PLM cache (`_EMBEDDER_CACHE`) in place — weights are expensive to
    reload and independent of which protein is queried.
    """
    _STRUCTURE_CACHE.clear()
    _DSSP_CACHE.clear()
    _CONTACT_INDEX_CACHE.clear()
    _SEQUENCE_CACHE.clear()
    _PROTEIN_EMBEDDING_CACHE.clear()


def get_structural_context(
    uniprot_id: str,
    position: int,
    *,
    config: dict | None = None,
) -> dict[str, Any]:
    """Return the structural neighborhood of residue `position` (1-based) in `uniprot_id`.

    Keys:
        uniprot_id, position, wildtype_aa,
        rsa (0–1, or None if DSSP has no value for the residue),
        secondary_structure (H/E/C, or None if absent),
        contact_count (int), nearest_contacts (list of {resnum, aa, distance}),
        embedding (per-residue PLM vector as a list, or None if embedding.model == "none"),
        embedding_model (name, or None), plddt (0–100, or None).

    All fields are native-Python / strict-JSON-safe. The embedding field is model-agnostic
    (default Ankh, not ProstT5). Raises IndexError if `position` is outside 1..len(sequence).
    A position that exists in the sequence but not in the structure (an 'X' gap-fill) degrades
    to null structural fields rather than raising.
    """
    import math

    cfg = config or _config.load()
    seq = get_sequence(uniprot_id, cfg)
    if not (1 <= position <= len(seq)):
        raise IndexError(
            f"position {position} out of range 1..{len(seq)} for {uniprot_id}"
        )

    residue = get_dssp(uniprot_id, cfg).get(position)

    # Locate the residue in the structure once — drives both pLDDT and whether contacts are
    # defined. Absent (a sequence gap-fill) → structural fields degrade to null together.
    structure = get_structure(uniprot_id, cfg).structure
    chain = next(structure[0].get_chains())
    res_obj = next(
        (r for r in chain if not r.id[0].strip() and r.id[1] == position), None
    )

    # A position absent from the structure (a sequence 'X' gap-fill) has no structural data —
    # ALL structure-derived fields (contacts, pLDDT, and the embedding) degrade to null together,
    # rather than reporting the placeholder residue's embedding. (Unreachable on AF F1 monomers,
    # which are gap-free; matters once multi-fragment / non-1-start numbering lands.)
    if res_obj is not None:
        contacts = get_contacts(uniprot_id, position, cfg)
        contact_count = contacts.contact_count
        nearest = [
            {"resnum": c.resnum, "aa": c.aa, "distance": round(c.distance, 3)}
            for c in contacts.nearest_contacts
        ]
        plddt = float(res_obj["CA"].get_bfactor()) if "CA" in res_obj else None
        per_res, embedding_model = _protein_embedding(uniprot_id, cfg)
        embedding = None
        if per_res is not None:
            vec = per_res[position - 1]
            embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    else:
        contact_count, nearest, plddt = 0, [], None
        embedding, embedding_model = None, None

    rsa = residue.rsa if residue is not None else float("nan")
    rsa = None if math.isnan(rsa) else rsa  # None, not NaN, for strict JSON

    return {
        "uniprot_id": uniprot_id,
        "position": position,
        "wildtype_aa": seq[position - 1],
        "rsa": rsa,
        "secondary_structure": residue.ss3 if residue is not None else None,
        "contact_count": contact_count,
        "nearest_contacts": nearest,
        "embedding": embedding,
        "embedding_model": embedding_model,
        "plddt": plddt,
    }
