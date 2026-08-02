"""A framework-neutral tool wrapper for `get_structural_context`.

Exposes the deliverable as a self-describing callable that any agent framework can register:
  * `TOOL_NAME` / `TOOL_DESCRIPTION` / `INPUT_SCHEMA` / `OUTPUT_SCHEMA` — plain JSON-Schema.
  * `tool_spec(style=...)` — the descriptor shaped for "anthropic" (`input_schema`),
    "openai" (function `parameters`), or "plain".
  * `invoke(arguments)` — the single entry point; takes a dict or JSON string of arguments,
    returns a strict-JSON-serializable result dict. This is what an MCP server / OpenAI
    function-call / plain Python registry all wrap.

Design note: the raw per-residue PLM embedding (1024–2560 floats) is for downstream ML, not
for an LLM's context window, so it is **omitted by default** (`include_embedding=false`) — which
also skips the expensive PLM forward pass. Agents reason over the interpretable fields (rsa,
secondary_structure, contact_count, nearest_contacts, plddt).
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

from . import config as _config
from .context import get_structural_context

TOOL_NAME = "get_structural_context"

TOOL_DESCRIPTION = (
    "Return the physical neighborhood of a residue in its AlphaFold-predicted 3D fold, to "
    "reason about whether the residue is structural (buried, tightly packed — holding the fold "
    "together) or functional (surface-exposed — doing a job). Given a UniProt accession and a "
    "1-based residue position, returns relative solvent accessibility (rsa, 0=buried..1=exposed), "
    "secondary structure (H/E/C), packing (contact_count and nearest_contacts), and an AlphaFold "
    "per-residue confidence (plddt, 0-100). Use it to tell buried structural residues apart from "
    "functional surface residues; pair it with conservation, which the structural signal does not "
    "replace (buried active-site residues look structural on geometry alone)."
)

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "uniprot_id": {
            "type": "string",
            "description": "UniProt accession (canonical), e.g. 'P62593'.",
            "pattern": "^[A-Za-z0-9]+(-[0-9]+)?$",
        },
        "position": {
            "type": "integer",
            "minimum": 1,
            "description": "1-based residue position in the canonical UniProt sequence.",
        },
        "include_embedding": {
            "type": "boolean",
            "default": False,
            "description": "Include the raw per-residue PLM embedding vector (large; default "
            "false — it is for downstream ML, not agent reasoning).",
        },
    },
    "required": ["uniprot_id", "position"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "uniprot_id": {"type": "string"},
        "position": {"type": "integer"},
        "wildtype_aa": {"type": "string", "description": "1-letter residue at the position."},
        "rsa": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
            "description": "Relative solvent accessibility; ≲0.2 buried, ≳0.25 exposed.",
        },
        "secondary_structure": {
            "type": ["string", "null"],
            "enum": ["H", "E", "C", None],
            "description": "H=helix, E=strand, C=coil.",
        },
        "contact_count": {"type": "integer", "description": "Cα neighbors ≤8 Å (packing/burial proxy)."},
        "nearest_contacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "resnum": {"type": "integer"},
                    "aa": {"type": "string"},
                    "distance": {"type": "number"},
                },
            },
        },
        "embedding": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "description": "Per-residue PLM vector; null unless include_embedding=true.",
        },
        "embedding_model": {"type": ["string", "null"]},
        "plddt": {"type": ["number", "null"], "description": "AlphaFold confidence 0-100."},
    },
    "required": [
        "uniprot_id", "position", "wildtype_aa", "rsa", "secondary_structure",
        "contact_count", "nearest_contacts", "embedding", "embedding_model", "plddt",
    ],
}

_ACCESSION_RE = re.compile(INPUT_SCHEMA["properties"]["uniprot_id"]["pattern"])


def tool_spec(style: str = "anthropic") -> dict[str, Any]:
    """The tool descriptor in a framework's expected shape.

    style: "anthropic" → {name, description, input_schema}; "openai" → OpenAI function-call
    {type:function, function:{name, description, parameters}}; "plain" → all four fields.
    """
    if style == "anthropic":
        return {"name": TOOL_NAME, "description": TOOL_DESCRIPTION, "input_schema": INPUT_SCHEMA}
    if style == "openai":
        return {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "parameters": INPUT_SCHEMA,
            },
        }
    if style == "plain":
        return {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "input_schema": INPUT_SCHEMA,
            "output_schema": OUTPUT_SCHEMA,
        }
    raise ValueError(f"unknown style {style!r}; use 'anthropic', 'openai', or 'plain'")


def call(
    uniprot_id: str,
    position: int,
    *,
    include_embedding: bool = False,
    config: dict | None = None,
) -> dict[str, Any]:
    """Typed convenience entry point. Skips the PLM forward pass unless include_embedding."""
    cfg = copy.deepcopy(config or _config.load())
    if not include_embedding:
        cfg["embedding"]["model"] = "none"  # skip the forward pass; embedding fields → null
    return get_structural_context(uniprot_id, position, config=cfg)


def invoke(arguments: dict | str) -> dict[str, Any]:
    """Framework-neutral dispatch: JSON string or dict of arguments → result dict.

    Validates required args + types (so a framework gets a clear error, not a deep stack).
    """
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise TypeError("arguments must be a JSON object / dict")

    missing = [k for k in ("uniprot_id", "position") if k not in arguments]
    if missing:
        raise ValueError(f"missing required argument(s): {', '.join(missing)}")
    uniprot_id = arguments["uniprot_id"]
    position = arguments["position"]
    if not isinstance(uniprot_id, str):
        raise TypeError("uniprot_id must be a string")
    if not _ACCESSION_RE.match(uniprot_id):
        raise ValueError(f"uniprot_id {uniprot_id!r} is not a valid UniProt accession")
    if not isinstance(position, int) or isinstance(position, bool):
        raise TypeError("position must be an integer")
    if position < 1:
        raise ValueError(f"position must be >= 1 (1-based), got {position}")
    # A stringy "false" must not silently enable the (expensive) embedding path, so require
    # an actual boolean rather than coercing with bool().
    include_embedding = arguments.get("include_embedding", False)
    if not isinstance(include_embedding, bool):
        raise TypeError("include_embedding must be a boolean")

    return call(uniprot_id, position, include_embedding=include_embedding)


def write_spec(path: str) -> None:
    """Write the plain tool spec (input + output JSON Schema) to `path` as JSON."""
    with open(path, "w") as f:
        json.dump(tool_spec("plain"), f, indent=2)
        f.write("\n")
