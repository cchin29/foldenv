"""Per-residue PLM embedding, multi-model.

Default is **Ankh-large** (1536-d), inherited from MuLAN, which foldenv was extracted from work
with. It is a default rather than a recommendation: this package neither trains nor evaluates a
downstream model, so it makes no claim about which encoder is best. Also available: **ProstT5** in amino-acid mode (1024-d), **SaProt**
(structure-aware AA+3Di, 1280-d — its 3Di half is computed from the AlphaFold backbone with
mini3di), **ESM2**, and **ESM C** (up to 2560-d).

Ankh / ProstT5 / ESM C reuse the shared `foldenv.plm` helpers (loader + `embed_sequence`, which already
handles each model's prefixes and per-residue slicing). SaProt needs a structure-aware
input, so it has a dedicated path here.

One forward pass per protein; the caller slices at `position` and caches per protein.

torch/transformers are an optional extra (`pip install "foldenv[plm]"`), so nothing here may
import them at module level: `get_structural_context` with `embedding.model = "none"` must keep
working on a bare install. Every torch import in this module goes through `_import_torch`, and
every backend import is guarded so a missing piece names the extra that supplies it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import constants as C

# --- model registry --------------------------------------------------------------------
# name → PLM key in constants.PLM_ENCODERS, output dim, whether it needs 3Di structure,
# and whether it may run on Apple MPS. `mps_ok=False` on ESM C 6B is an untested precaution, not
# a measured incompatibility: its transformers path does not load on any release this package has
# measured (see plm.check_transformers_version), so it has never reached a device on any host and
# there is nothing to have observed. Kept off MPS on the reasoning that a 6B-parameter checkpoint
# is the least likely to behave there, but note bf16 weights are ~12 GB against a working set that
# is typically tens of GB on Apple Silicon, so memory is unlikely to be the obstacle if the load
# path is ever fixed. `esmc_600m`, which goes through the `esm` SDK, is the ESM C route that works.


@dataclass(frozen=True)
class EmbeddingModelSpec:
    plm_key: str
    dim: int
    structure_aware: bool
    mps_ok: bool = True
    sdk: bool = False  # loaded via the `esm` SDK (ESMC.from_pretrained), not transformers


EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    "ankh": EmbeddingModelSpec("ankh", 1536, False),           # Ankh-large (default)
    "ankh3_large": EmbeddingModelSpec("ankh3_large", 1536, False),  # Ankh3-large ([NLU] prefix)
    "ankh3_xl": EmbeddingModelSpec("ankh3_xl", 2560, False),   # Ankh3-XL ([NLU] prefix)
    "prostt5_aa": EmbeddingModelSpec("prostt5", 1024, False),  # bilingual, amino-acid mode
    "saprot": EmbeddingModelSpec("saprot", 1280, True),        # SaProt 650M, AA+3Di
    "saprot_1.3b": EmbeddingModelSpec("saprot_1.3b", 1280, True),  # SaProt 1.3B (deeper, 1280-d)
    "esm2_3b": EmbeddingModelSpec("esm", 2560, False),         # ESM2-3B (widely used)
    "esm2_650m": EmbeddingModelSpec("esm_650M", 1280, False),  # ESM2-650M (lighter)
    # ESM Cambrian via the `esm` SDK (transformers-independent). 600M is MPS-friendly (bf16,
    # ~1.4 s/protein) and locally loadable. The 6B is registered for its dimension only: neither
    # load path reaches it (see plm.py), and running it means hand-building the module from the
    # safetensors shards, which is outside this package.
    "esmc_600m": EmbeddingModelSpec("esmc_600m", 1152, False, mps_ok=True, sdk=True),
    "esmc_6b": EmbeddingModelSpec("esmc_6b", 2560, False, mps_ok=False),  # no load path; see plm.py
}

# SaProt structure-aware vocab tokens: each residue is one AA char plus one 3Di char.
_AA_MASK = "#"               # SaProt AA-half mask for non-canonical residues
_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
_GAP_3DI = "d"               # valid Foldseek/mini3di state for un-encodable residues


def _import_torch():
    """Return the `torch` module, or foldenv's actionable optional-dependency error.

    Importing `.plm` runs `require_plm_dependencies("torch")` at its module level, so routing
    every torch import in this module through here turns a bare `pip install foldenv` into the
    message naming `foldenv[plm]` instead of ModuleNotFoundError('torch'). The import stays
    inside the functions that need it — a module-level one would make the structural path
    (which never embeds anything) depend on the whole deep-learning stack.
    """
    from . import plm  # noqa: F401  — its module-level guard is the point of this indirection

    import torch

    return torch


def get_model_spec(model_name: str) -> EmbeddingModelSpec:
    spec = EMBEDDING_MODELS.get(model_name)
    if spec is None:
        raise ValueError(
            f"Unknown embedding model {model_name!r}; choose from "
            f"{sorted(EMBEDDING_MODELS)} or 'none'."
        )
    return spec


def embedding_dim(model_name: str) -> int:
    """Output dimension for a model name (for shape assertions / field docs)."""
    return get_model_spec(model_name).dim


def resolve_device(model_name: str, device: Any = None):
    """Pick the torch device for a model, keeping MPS-incompatible models off MPS.

    Explicit `device` (including the string "mps") is honored but warned about for a model
    flagged `mps_ok=False`. With `device=None`/"auto", falls back to the shared
    `get_device()`, then reroutes an MPS pick to CUDA (if present) or CPU for such models —
    so ESM C 6B lands on CPU or CUDA. That flag is a precaution, not a measured result: see the
    registry comment above for why nothing has ever been observed either way.
    """
    torch = _import_torch()

    spec = get_model_spec(model_name)
    if device is not None and str(device) != "auto":
        dev = torch.device(device)
        if dev.type == "mps" and not spec.mps_ok:
            warnings.warn(
                f"{model_name} is flagged as untested on MPS. Run it on CPU or CUDA instead, "
                "or set the flag if you have measured it working here.",
                stacklevel=2,
            )
        return dev

    from .plm import get_device

    dev = get_device()
    if dev.type == "mps" and not spec.mps_ok:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        warnings.warn(
            f"{model_name} is not run on MPS — routing to {dev.type.upper()}. "
            "This is a precaution rather than a measured failure; see decisions.yaml D5.",
            stacklevel=2,
        )
    return dev


# --- structure → 3Di (SaProt only) -----------------------------------------------------


def structure_to_3di(structure, seq_len: int, chain_id: str | None = None) -> str:
    """Per-residue 3Di string (lowercase, length `seq_len`) from an AF structure via mini3di.

    Encodes the first (or named) chain's N/CA/C/CB backbone. Residues that can't be encoded
    (missing backbone atoms) are filled with a valid placeholder state so the string stays
    per-residue parallel to the AA sequence. Glycine's absent CB is passed as NaN — mini3di
    handles it. Assumes AF canonical numbering (residue id i ↔ sequence index i-1).
    """
    import mini3di

    model = next(structure.get_models())
    chain = model[chain_id] if chain_id is not None else next(model.get_chains())

    def _xyz(res, atom):
        return tuple(res[atom].get_coord()) if atom in res else (np.nan, np.nan, np.nan)

    keep, coords = [], {"N": [], "CA": [], "C": [], "CB": []}
    for res in chain:
        het, resnum, _ = res.id
        if het.strip():  # skip HETATM / waters
            continue
        if not all(a in res for a in ("N", "CA", "C")):
            continue
        if not (1 <= resnum <= seq_len):
            continue
        keep.append(resnum)
        for atom in ("N", "CA", "C", "CB"):
            coords[atom].append(_xyz(res, atom))

    out = [_GAP_3DI] * seq_len
    if keep:
        enc = mini3di.Encoder()
        states = enc.encode_atoms(
            ca=np.array(coords["CA"], float),
            cb=np.array(coords["CB"], float),
            n=np.array(coords["N"], float),
            c=np.array(coords["C"], float),
        )
        s3di = enc.build_sequence(states).lower()
        for resnum, ch in zip(keep, s3di):
            out[resnum - 1] = ch
    return "".join(out)


def sa_sequence(aa: str, tdi: str) -> str:
    """Interleave AA(upper)+3Di(lower) into SaProt's 2-char-per-residue SA string.

    Non-canonical AAs map to the '#' AA-mask (they have no valid SA token); this keeps the
    tokenization per-residue aligned.
    """
    return "".join(
        (a.upper() if a.upper() in _CANONICAL else _AA_MASK) + d.lower()
        for a, d in zip(aa, tdi)
    )


# --- loading + forward pass ------------------------------------------------------------


def load_embedder(model_name: str, device: Any = None):
    """Load (model, tokenizer) for an embedding model. Downloads weights on first use.

    `device` may be a torch device/string or None/"auto"; it is resolved via
    `resolve_device` so ESM C 6B never lands on MPS.
    """
    spec = get_model_spec(model_name)
    if spec.sdk:  # ESM C via the `esm` SDK — no transformers tokenizer (tokens are internal)
        # Checked before `resolve_device`, which reaches torch: on an install with neither, the
        # SDK's own extra is the one to name, and naming [plm] first would cost the caller a
        # multi-gigabyte download before telling them they still need [esmc].
        try:
            from esm.models.esmc import ESMC
        except ModuleNotFoundError as exc:  # a separate extra from [plm]: name that one
            raise ImportError(
                f"{model_name} loads through the `esm` SDK, which is not installed — "
                'install it with: pip install "foldenv[esmc]"'
            ) from exc

        device = resolve_device(model_name, device)
        model = ESMC.from_pretrained(spec.plm_key, device=device).eval()
        return model, None
    device = resolve_device(model_name, device)
    if spec.structure_aware:  # SaProt: EsmForMaskedLM, structure-aware hidden states
        from .plm import require_plm_dependencies

        require_plm_dependencies("transformers")  # this branch bypasses plm's own loader
        from transformers import AutoTokenizer, EsmForMaskedLM

        model_id = C.PLM_ENCODERS[spec.plm_key]
        tok = AutoTokenizer.from_pretrained(model_id)
        model = EsmForMaskedLM.from_pretrained(model_id, output_hidden_states=True)
        return model.to(device).eval(), tok
    # Ankh / ProstT5 / ESM C — the shared loader knows each architecture.
    from .plm import load_pretrained_plm

    return load_pretrained_plm(spec.plm_key, device=device)


def _embed_protein_sdk(model, sequence: str):
    """Per-residue `[L, dim]` embedding via the `esm` SDK (ESM C).

    The SDK uses `encode` → `logits(return_embeddings=True)` rather than a transformers
    forward. Output is `[1, L+2, dim]` with leading BOS / trailing EOS; we strip both, cast
    off MPS bfloat16 to float32, and move to CPU to match the transformers path's contract.
    """
    torch = _import_torch()
    from esm.sdk.api import ESMProtein, LogitsConfig

    seq = sequence.upper()
    with torch.inference_mode():
        tensor = model.encode(ESMProtein(sequence=seq))
        out = model.logits(tensor, LogitsConfig(return_embeddings=True))
    per_res = out.embeddings[0, 1:-1, :]  # drop BOS/EOS → [L, dim]
    if per_res.shape[0] != len(seq):
        raise RuntimeError(
            f"ESM C SDK returned {per_res.shape[0]} residue rows for a length-{len(seq)} "
            "sequence; tokenization is not 1:1 as assumed."
        )
    return per_res.float().contiguous().cpu()


def embed_protein(
    model_name: str,
    sequence: str,
    *,
    model=None,
    tokenizer=None,
    structure=None,
    device=None,
):
    """Return the per-residue embedding `[L, dim]` for `sequence` under `model_name`.

    For SaProt, `structure` (a Biopython Structure) is required to build the 3Di half.
    `model`/`tokenizer` may be supplied to reuse an already-loaded pair (per-protein cache);
    otherwise they are loaded on `device` (resolved to keep ESM C 6B off MPS).
    """
    spec = get_model_spec(model_name)
    if spec.sdk:  # ESM C SDK path — no transformers tokenizer, so guard on the model only
        # torch is deliberately not imported above this dispatch: on an install with neither
        # stack, the SDK path's missing piece is `esm`, and reaching _import_torch() first
        # would name [plm] and cost the caller a multi-gigabyte download for the wrong extra.
        if model is None:
            model, _ = load_embedder(model_name, device)
        return _embed_protein_sdk(model, sequence)

    torch = _import_torch()
    if model is None or tokenizer is None:
        model, tokenizer = load_embedder(model_name, device)

    if spec.structure_aware:
        if structure is None:
            raise ValueError(f"{model_name} is structure-aware and needs `structure`.")
        tdi = structure_to_3di(structure, seq_len=len(sequence))
        sa = sa_sequence(sequence, tdi)
        with torch.inference_mode():
            enc = tokenizer(
                sa, return_tensors="pt", add_special_tokens=True,
                return_special_tokens_mask=True,
            ).to(model.device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hidden = out.hidden_states[-1]
            per_res = hidden[~enc["special_tokens_mask"].bool()]
        return per_res.contiguous().cpu()

    # Ankh / ProstT5 / ESM C via the shared util (handles prefixes + slicing).
    from .plm import embed_sequence

    emb = embed_sequence(model, tokenizer, sequence)  # [1, L, dim]
    return emb.squeeze(0).contiguous().cpu()


def embed_residue(
    model_name: str,
    sequence: str,
    position: int,
    *,
    model=None,
    tokenizer=None,
    structure=None,
    device=None,
):
    """The `[dim]` embedding of residue `position` (1-based) — slices `embed_protein`."""
    per_res = embed_protein(
        model_name, sequence, model=model, tokenizer=tokenizer,
        structure=structure, device=device,
    )
    if not (1 <= position <= per_res.shape[0]):
        raise IndexError(
            f"position {position} out of range for sequence length {per_res.shape[0]}"
        )
    return per_res[position - 1]
