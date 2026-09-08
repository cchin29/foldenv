"""Fetch an AlphaFold-DB model for a UniProt accession and parse it with Biopython.

Flow: `/api/prediction/{accession}` → pick the model file(s) → download the mmCIF once,
cache on disk by accession → parse to a Biopython `Structure`. The API is queried (rather
than hardcoding the `-F1-model_v4.cif` URL) so we track the current file version.

Edge cases handled / flagged:
  * 404 (no model for that accession) → `NoAlphaFoldModelError`.
  * Multi-fragment proteins (>2,700 aa split into F1/F2/…): the API returns one entry per
    fragment. foldenv uses fragment **F1** and records the rest in `StructureRecord.fragments`;
    per-fragment residue-number stitching is not implemented, so positions beyond F1 are
    unavailable rather than silently misnumbered.
  * Isoforms renumber vs the canonical sequence — the caller must pass a canonical accession.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# Biopython emits noisy warnings on AF mmCIF files (e.g. missing header fields); quiet them
# at parse time only. Imported lazily inside functions so importing this module doesn't hard-
# require Biopython (keeps `config`/tooling usable before the env is fully set up).

_TIMEOUT = 30  # seconds, per HTTP request


class NoAlphaFoldModelError(RuntimeError):
    """No AlphaFold-DB prediction exists for the requested accession (API 404 / empty)."""


@dataclass
class StructureRecord:
    """A fetched, parsed AlphaFold model plus provenance for reproducibility."""

    accession: str
    cif_path: Path
    structure: Any                      # Bio.PDB.Structure.Structure
    version: str | None = None          # AF file version reported by the API (e.g. "4")
    fragments: list[dict] = field(default_factory=list)  # all API entries (F1, F2, …)


def _prediction_metadata(accession: str, api_base: str) -> list[dict]:
    """Query the AlphaFold prediction API; return the list of model entries (one per fragment)."""
    url = f"{api_base.rstrip('/')}/{accession}"
    resp = requests.get(url, timeout=_TIMEOUT)
    # 404 = well-formed accession with no model; 400 = accession the API can't resolve
    # (malformed / unknown). Either way there's no model to return.
    if resp.status_code in (400, 404):
        raise NoAlphaFoldModelError(
            f"AlphaFold-DB has no model for accession {accession!r} "
            f"(HTTP {resp.status_code})"
        )
    resp.raise_for_status()
    entries = resp.json()
    if not entries:
        raise NoAlphaFoldModelError(f"AlphaFold-DB returned no entries for {accession!r}")
    return entries


def _select_model_entry(entries: list[dict], accession: str):
    """Pick the model entry for the *exact* accession; return (entry, entries_for_accession).

    The prediction API bundles in the protein's other isoforms as separate entries with
    suffixed accessions (e.g. P04637, P04637-2, …-9 for TP53). Keep only entries whose
    accession matches the request so we don't accidentally serve an isoform, and so the
    "multi-fragment" signal reflects *true* length-fragments (same accession, F1/F2/…), not
    isoform count. Falls back to all entries if none match (e.g. an isoform was requested).
    """
    own = [
        e for e in entries
        if str(e.get("uniprotAccession", "")).upper() == accession.upper()
    ]
    if not own:
        # No entry carries the requested accession — serve the first entry but warn, since
        # this can silently return the canonical model for a mistyped/direct-isoform request
        # (positions would then index the wrong sequence).
        warnings.warn(
            f"No AlphaFold entry matches accession {accession!r} exactly; falling back to "
            f"{entries[0].get('uniprotAccession', '?')!r}. Check the accession.",
            stacklevel=2,
        )
        own = entries
    return own[0], own


#: Ceiling on a single downloaded structure. The largest AlphaFold mmCIF fragments are a few MB;
#: this is loose enough never to fire on real data and tight enough that a hostile or misconfigured
#: endpoint cannot fill the disk.
_MAX_DOWNLOAD_BYTES = 256 << 20  # 256 MiB


#: A structure identifier is interpolated into a cache path and into a URL, so it is checked
#: here -- at the point the path is built -- rather than at any one caller. `tool.invoke` also
#: validates, but `tool.call`, `get_structural_context`, `get_dssp`, `get_sequence`,
#: `structural_profile` and `fetch_structure` are all public and reach this code directly; a guard
#: that lives only in `invoke` protects whichever entry point the integrator did not choose.
#: Deliberately narrow: alphanumerics, with an optional `-N` isoform suffix. That admits every
#: UniProt accession and every PDB id while excluding the separators (`/`, `\`, `.`) that make a
#: traversal, and the NUL and newline that make a surprising filename.
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9]{1,12}(-[0-9]{1,3})?\Z")


def _check_identifier(value: str, kind: str) -> str:
    """Return `value` if it is a usable structure identifier, else raise `ValueError`."""
    if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
        # `repr` first, then truncate: interpolating `value[:60]` directly raises TypeError on a
        # non-subscriptable value, which is the one thing this function promises not to do.
        raise ValueError(
            f"{kind} {repr(value)[:60]} is not a valid identifier: expected alphanumerics "
            f"(with an optional -N isoform suffix), which is what can safely become a cache "
            f"filename and a URL path segment."
        )
    return value


def _download(url: str, dest: Path) -> None:
    """Download `url` to `dest` atomically (write to a unique temp sibling, then replace).

    The temp file comes from `mkstemp` rather than a fixed `<dest>.part`, for the reason
    `persist._write_atomic` gives: a predictable name in a predictable cache directory is both a
    collision between two processes fetching the same accession and a symlink a local attacker can
    plant ahead of time, which `open(..., "wb")` would then follow and overwrite.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with requests.get(url, timeout=_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            written = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    written += len(chunk)
                    if written > _MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"{url} exceeded {_MAX_DOWNLOAD_BYTES} bytes; refusing to continue"
                        )
                    fh.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace


def parse_structure(cif_path: Path, accession: str):
    """Parse an mmCIF file into a Biopython Structure (quieting AF header warnings)."""
    from Bio.PDB import MMCIFParser  # lazy import (see module note)
    from Bio import BiopythonWarning

    parser = MMCIFParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        return parser.get_structure(accession, str(cif_path))


def fetch_structure(
    accession: str,
    cache_dir: str | Path,
    api_base: str = "https://alphafold.ebi.ac.uk/api/prediction",
    *,
    force: bool = False,
) -> StructureRecord:
    """Fetch + parse the AlphaFold model for `accession`, caching the mmCIF by accession.

    Args:
        accession: UniProt accession (canonical), e.g. "P62593".
        cache_dir: directory for the on-disk mmCIF cache.
        api_base: AlphaFold prediction API base URL.
        force: re-download even if a cached file exists.

    Raises:
        NoAlphaFoldModelError: no prediction for the accession.
    """
    accession = _check_identifier(accession, "accession")
    cache_dir = Path(cache_dir)
    cif_path = cache_dir / "alphafold" / f"{accession}.cif"
    # Provenance sidecar: version + fragment list are only known at download time, but
    # StructureRecord must carry them on cache hits too (else version is None and the
    # multi-fragment warning never re-fires — a long protein silently uses F1). Persist
    # them next to the mmCIF so a cache hit restores them without another API round-trip.
    meta_path = cif_path.with_suffix(".meta.json")

    version: str | None = None
    fragments: list[dict] = []

    if force or not cif_path.exists():
        entry, fragments = _select_model_entry(
            _prediction_metadata(accession, api_base), accession
        )
        version = str(entry.get("latestVersion") or entry.get("modelVersion") or "") or None
        cif_url = entry.get("cifUrl")
        if not cif_url:
            raise NoAlphaFoldModelError(
                f"AlphaFold entry for {accession!r} has no cifUrl: {entry!r}"
            )
        # `cif_url` comes from the API response and is fetched without a host allowlist. That is
        # a deliberate call, not an oversight: AlphaFold-DB serves mmCIFs from a different host
        # than the prediction API, so an allowlist narrow enough to be worth having would break
        # real downloads whenever EBI moves the file host, while the attack it prevents needs the
        # API itself (or `FOLDENV_ALPHAFOLD_API_BASE`) to be hostile. Bounded rather than
        # blocked: `_download` caps the body at `_MAX_DOWNLOAD_BYTES` and writes only to a path
        # built locally from the validated accession, so a redirected fetch can waste bandwidth
        # but cannot choose where the bytes land or how many arrive.
        _download(cif_url, cif_path)
        try:
            with open(meta_path, "w") as f:
                json.dump({"version": version, "fragments": fragments}, f)
        except (OSError, TypeError):
            pass  # provenance is best-effort; a missing sidecar just degrades to None/[]
    else:
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            version = meta.get("version")
            fragments = meta.get("fragments") or []
        except (OSError, ValueError):
            version, fragments = None, []  # older cache w/o sidecar → pre-fix behavior

    # Warn on BOTH fresh and cached reads (fires whenever fragment count is known), so a
    # cached multi-fragment protein doesn't silently use F1 without notice.
    if len(fragments) > 1:  # true length-fragments (isoforms already filtered out)
        warnings.warn(
            f"{accession} is split into {len(fragments)} AlphaFold length-fragments "
            "(F1/F2/…); foldenv uses F1 only — residue-number stitching across fragments is "
            "not implemented, so positions beyond F1 are unavailable.",
            stacklevel=2,
        )

    structure = parse_structure(cif_path, accession)
    return StructureRecord(
        accession=accession,
        cif_path=cif_path,
        structure=structure,
        version=version,
        fragments=fragments,
    )


def fetch_experimental_structure(pdb_id: str, cache_dir: str | Path):
    """Fetch + parse an experimental structure from RCSB (mmCIF), cached by PDB id.

    Used only by the M6 crystal cross-check (AF-vs-experimental). Returns (cif_path,
    Biopython Structure). Note experimental "author" residue numbering usually differs from
    UniProt numbering — the cross-check reconciles it by sequence alignment, not by assuming.
    """
    pdb_id = _check_identifier(pdb_id, "pdb_id").upper()
    cif_path = Path(cache_dir) / "pdb" / f"{pdb_id}.cif"
    if not cif_path.exists():
        _download(f"https://files.rcsb.org/download/{pdb_id}.cif", cif_path)
    return cif_path, parse_structure(cif_path, pdb_id)
