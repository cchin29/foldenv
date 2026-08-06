"""Biological analysis: do RSA + contacts separate functional from structural?

For a protein and a set of known functional residues, report each site's burial (RSA) and
packing (contact_count) together with its **percentile rank** within the protein, plus a
descriptive verdict. This is deliberately descriptive, not a benchmark — curated functional
sets are tiny (a handful of residues), so percentiles and medians are more honest than an AUC.

Key finding this exists to make explicit: enzyme catalytic
residues sit in **buried active-site clefts**, so they look structurally like buried-structural
"spandrels" on RSA/contacts alone — the structural axis is *complementary* to conservation,
not a standalone functional-site classifier.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from . import config as _config
from . import context as _ctx

# Known functional residues in UniProt numbering (map literature/Ambler numbers first!).
# Example data for the descriptive helper below, not a curated benchmark — pass `sites=` for
# anything else. The three TP53 entries are not one category, and the difference matters here.
#
# TEM-1 β-lactamase (P62593): the class A catalytic triad, S70/K73/E166 in *Ambler* numbering
# (Ambler et al. 1991) → UniProt 68/71/164. Ambler numbering carries insertions relative to the
# sequence, so the offset is not derivable from the sequence alone; the identity self-check in
# functional_site_stats is what stops a wrong one from reporting a neighbouring residue.
#
# TP53 (P04637): already UniProt numbering. **R248 and R273 are DNA-contact residues; R175 is
# not** — it is a conformational mutant that distorts the DNA-binding domain fold rather than
# losing a contact (the contact-vs-structural split from the DNA-binding domain crystal
# structure). All three are cancer *hotspots*, which is a mutation-frequency claim from the TP53
# mutation databases rather than a structural one.
#
# R175 therefore sits on both sides of the comparison this module exists to make: the finding in
# the docstring is that catalytic residues look structurally like buried spandrels, and R175 is a
# functional site whose mechanism *is* structural. Kept, because it is a genuine hotspot and this
# helper is descriptive — but read its row knowing that, and do not treat the set as three
# equivalent "DNA-contact" sites.
KNOWN_FUNCTIONAL_SITES: dict[str, dict[int, str]] = {
    "P62593": {68: "S", 71: "K", 164: "E"},
    "P04637": {175: "R", 248: "R", 273: "R"},
}


@dataclass
class SiteStat:
    position: int
    aa: str
    rsa: float | None
    rsa_percentile: float | None    # fraction of residues as-or-more buried (RSA ≤ this)
    contact_count: int
    contact_percentile: float       # fraction of residues with ≤ this many contacts
    buried: bool                    # rsa < config buried_threshold


def _percentile(sorted_vals: list, v) -> float:
    """Fraction of values ≤ v (0..1); ties inclusive (bisect_right)."""
    return bisect.bisect_right(sorted_vals, v) / len(sorted_vals) if sorted_vals else float("nan")


def functional_site_stats(
    uniprot_id: str,
    sites: dict[int, str] | None = None,
    cfg: dict | None = None,
) -> list[SiteStat]:
    """Structural signature (RSA + contacts, with percentile ranks) for known functional sites."""
    cfg = cfg or _config.load()
    sites = sites if sites is not None else KNOWN_FUNCTIONAL_SITES.get(uniprot_id, {})
    if not sites:
        raise ValueError(f"no known functional sites for {uniprot_id}; pass `sites=`")

    profile = _ctx.structural_profile(uniprot_id, cfg)
    rsas = sorted(p["rsa"] for p in profile.values() if p["rsa"] is not None)
    ccs = sorted(p["contact_count"] for p in profile.values())
    buried_thr = cfg["rsa"]["buried_threshold"]

    out = []
    for pos, expect in sorted(sites.items()):
        if pos not in profile:
            raise ValueError(
                f"position {pos} out of range 1..{len(profile)} for {uniprot_id}"
            )
        p = profile[pos]
        # Self-check the numbering mapping: the residue in the structure must match the
        # expected identity. Guards against a wrong Ambler→UniProt offset silently reporting
        # a neighboring residue (the central M7 risk). Pass expect="" / "X" to skip.
        if expect and expect not in ("X", "?") and p["aa"] != expect:
            raise ValueError(
                f"{uniprot_id} site {pos}: expected {expect} but structure has {p['aa']} — "
                "check the functional-site numbering (literature vs UniProt)."
            )
        rsa = p["rsa"]
        out.append(
            SiteStat(
                position=pos,
                aa=p["aa"],
                rsa=rsa,
                rsa_percentile=_percentile(rsas, rsa) if rsa is not None else None,
                contact_count=p["contact_count"],
                contact_percentile=_percentile(ccs, p["contact_count"]),
                buried=(rsa is not None and rsa < buried_thr),
            )
        )
    return out


def summarize(uniprot_id: str, cfg: dict | None = None) -> dict:
    """Descriptive summary: are the known functional sites buried, and how packed vs typical?"""
    stats = functional_site_stats(uniprot_id, cfg=cfg)
    n_buried = sum(s.buried for s in stats)
    present_rsa = [s.rsa for s in stats if s.rsa is not None]
    return {
        "uniprot_id": uniprot_id,
        "n_sites": len(stats),
        "n_buried": n_buried,
        "all_buried": n_buried == len(stats),
        # mean over sites that actually have a DSSP RSA (don't fold None → 0.0)
        "mean_rsa": (sum(present_rsa) / len(present_rsa)) if present_rsa else float("nan"),
        "mean_contact_percentile": sum(s.contact_percentile for s in stats) / len(stats),
        "sites": stats,
    }
