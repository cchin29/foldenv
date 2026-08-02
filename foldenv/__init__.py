"""foldenv — `get_structural_context(uniprot_id, position)`.

A residue's physical neighborhood in the AlphaFold-predicted fold — RSA, secondary
structure, packing/contacts, a PLM embedding, and a pLDDT confidence flag — to help
downstream tools separate buried structural residues (spandrels) from functional surface
residues.

Decisions and their rationale live in `decisions.yaml`.
"""
import os as _os
import sys as _sys

# PLM embedding on Apple MPS relies on CPU fallback for a few ops the Metal backend does
# not implement. This must be set before torch initializes its MPS backend, so we do it at
# import — but only on macOS (where MPS exists) and only via setdefault, so non-Mac hosts
# and callers who never touch torch see no global env mutation.
if _sys.platform == "darwin":
    _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from . import analysis, config, tool, validation
from .context import (
    clear_cache,
    get_contacts,
    get_dssp,
    get_sequence,
    get_structural_context,
    get_structure,
    structural_profile,
)
from .fetch import NoAlphaFoldModelError, StructureRecord, fetch_structure

__all__ = [
    "config",
    "validation",
    "analysis",
    "tool",
    "get_structural_context",
    "get_structure",
    "get_dssp",
    "get_contacts",
    "get_sequence",
    "structural_profile",
    "clear_cache",
    "fetch_structure",
    "StructureRecord",
    "NoAlphaFoldModelError",
]
