"""PLM loading + per-residue embedding helpers.

Vendored so ``foldenv`` is self-contained: ``load_pretrained_plm`` builds the
(model, tokenizer) pair for a supported encoder, and ``embed_sequence`` runs one forward
pass and returns the per-residue embedding, handling each model's prefixes and slicing.

``torch``/``transformers`` are an optional extra (``pip install "foldenv[plm]"``), not a core
dependency: the structural fields (RSA, secondary structure, contacts, pLDDT) never reach this
module, so a caller who only wants those should not pay for the deep-learning stack. Importing
this module therefore requires ``torch`` only; ``transformers`` is imported per checkpoint
inside ``load_pretrained_plm``, which is also where each checkpoint's transformers-version
window is enforced.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple, Union

import importlib.util
import os
import re
import warnings

from . import constants as C

# --- optional-dependency guard -----------------------------------------------------------
# A missing torch/transformers is an *expected* state here (they are gigabytes of wheels for a
# path many callers never take), not a broken install — so it must read as a choice the caller
# can reverse, and `ModuleNotFoundError: No module named 'torch'` names neither the cause nor
# the cure. Every entry point into the PLM stack goes through the guard below instead.
_PLM_EXTRA_HINT = 'install with: pip install "foldenv[plm]"'


def require_plm_dependencies(*modules: str) -> None:
    """Raise an actionable ImportError if any of `modules` is missing.

    Uses ``importlib.util.find_spec`` rather than a try/except around a real import: the check
    stays cheap (no module initialization) and reports *all* the missing pieces at once, so a
    caller with neither torch nor transformers does not fix them one traceback at a time.
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise ImportError(
            f"foldenv's PLM embedding path needs {', '.join(missing)}, which "
            f"{'is' if len(missing) == 1 else 'are'} not installed. As of foldenv 0.2.0 these "
            f"are the [plm] extra rather than core dependencies — {_PLM_EXTRA_HINT}. "
            "The structural fields (RSA, secondary structure, contacts, pLDDT) need neither: "
            'run them with embedding.model = "none".'
        )


# Guarded before the import, so `import foldenv.plm` on a bare install fails with the message
# above rather than a bare ModuleNotFoundError. `transformers` is deliberately NOT required at
# module scope: it is imported per checkpoint in `load_pretrained_plm`, which keeps `get_device`
# and the transformers-free ESM C SDK path usable on a torch-only install.
require_plm_dependencies("torch")

import torch  # noqa: E402  — deliberately after the guard above

if TYPE_CHECKING:  # annotation-only; a module-level transformers import would undo the above
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

TorchDevice = Union[str, torch.device]


# --- per-model transformers requirements ---------------------------------------------------
# Enforced at load time rather than as an install pin because no single pin satisfies every
# checkpoint. prot_bert and the unanchored Ankh checkpoints need < 5; ESM C 6B documents
# >= 4.57, which is not itself in conflict (the ranges overlap on 4.57-4.99) but is moot,
# because its transformers path loads on no measured release at all — so a pin cannot serve it
# under any bound. The pin therefore stays wide (>=4.27,<5, verified across 4.44–4.57.6) and
# each checkpoint declares its own window here.
#
# Two severities, because the two failure modes deserve opposite treatment:
#   hard=True  — the checkpoint cannot be loaded at all, or loads misaligned, so raising is the
#                only honest outcome (prot_bert's tokenizer does not instantiate on 5.x; no
#                measured release registers the `esmc` model_type).
#   hard=False — the checkpoint loads, every row still lands on the right residue, and only the
#                *values* differ from another transformers version. That is a comparability
#                problem, not a correctness one, and a caller who embeds one dataset end-to-end
#                under one version is fine — so warn, and let the embedding cache key
#                (persist.py) keep the versions' tensors from colliding on disk.
#
# Alignment is the dividing line, and it is checkpoint-specific. Measured on transformers
# 5.15.1: the Ankh tokenizers prepend an <unk> that `special_tokens_mask` leaves unflagged —
# ankh-large and ankh-base both tokenize "MKTVRQ" to ['<unk>','M','K','T','V','R','Q','</s>']
# with mask [0,0,0,0,0,0,0,1], i.e. 7 non-special rows for 6 residues. (The class still reports
# `is_fast=True`, so this is not the slow-tokenizer fallback it first looks like.) What that
# costs depends on the checkpoint: `embed_sequence` re-anchors on the residue count only inside
# its `is_ankh3` branch, so the Ankh3 pair drops the stray row and stays aligned (values differ,
# rows do not — soft), while ankh/ankh_base drop non-residue rows through `special_tokens_mask`
# alone, keep it, and shift every residue by one. That last case is silent corruption, so the
# two checkpoints without the anchor are hard, not soft.
_ANKH_ON_5X = (
    "transformers 5.x prepends an <unk> that special_tokens_mask does not flag, and this "
    "checkpoint has no residue-count anchor to drop it — every residue's embedding is shifted "
    "by one position"
)
_ANKH3_ON_5X = (
    "transformers 5.x prepends an <unk> that special_tokens_mask does not flag; the Ankh3 "
    "prefix path re-anchors on the residue count so the rows stay correct, but the values "
    "differ from those computed under 4.x"
)
_ANKH3_AT_450 = (
    "Ankh3 tokenization changed at transformers 4.50 (the spurious leading <unk> disappears); "
    "rows stay correctly aligned either way, but embeddings computed before 4.50 are not "
    "numerically comparable with ones computed from 4.50 onwards"
)


class _TfConstraint(NamedTuple):
    """One transformers-version window for one checkpoint. Bounds are (major, minor)."""

    at_least: Optional[Tuple[int, int]]  # inclusive lower bound, or None
    below: Optional[Tuple[int, int]]     # exclusive upper bound, or None
    hard: bool                           # raise (True) vs warn (False) — see the note above
    why: str                             # what actually goes wrong, for the message
    always: bool = False                 # violated at every version (no release loads this)


# Keyed by PLM_ENCODERS key. Checkpoints absent from this table have no known constraint
# inside the install pin. Every entry here is measured; a checkpoint whose behaviour was never
# observed across the version matrix gets no entry, because an unverified constraint is worse
# than none.
_TRANSFORMERS_CONSTRAINTS: Dict[str, Tuple[_TfConstraint, ...]] = {
    "protbert": (
        _TfConstraint(
            None, (5, 0), True,
            "the Rostlab/prot_bert tokenizer fails to instantiate on transformers 5.x",
        ),
    ),
    # Unconditional: 4.57 is the floor upstream documents, but the `esmc` model_type is
    # registered by none of 4.44, 4.57.6 or 5.15.1, so raising only below 4.57 would send the
    # caller to an install that still fails — several seconds later, inside transformers, with
    # the opaque error this table exists to pre-empt.
    "esmc_6b": (
        _TfConstraint(
            None, None, True,
            "no measured transformers release (4.44, 4.57.6, 5.15.1) registers the `esmc` "
            "model_type, so this checkpoint's transformers path does not load on any of them; "
            'use `esmc_600m` through the `esm` SDK instead (pip install "foldenv[esmc]")',
            always=True,
        ),
    ),
    "ankh": (
        _TfConstraint(None, (5, 0), True, _ANKH_ON_5X),
    ),
    "ankh_base": (
        _TfConstraint(None, (5, 0), True, _ANKH_ON_5X),
    ),
    "ankh3_large": (
        _TfConstraint(None, (5, 0), False, _ANKH3_ON_5X),
        _TfConstraint((4, 50), None, False, _ANKH3_AT_450),
    ),
    "ankh3_xl": (
        _TfConstraint(None, (5, 0), False, _ANKH3_ON_5X),
        _TfConstraint((4, 50), None, False, _ANKH3_AT_450),
    ),
}


def _installed_transformers_version() -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    """``(raw version string, (major, minor))`` of the installed transformers.

    Read from the distribution metadata, not ``transformers.__version__``: importing
    transformers costs seconds, and this runs before the (much larger) weight download decides
    anything. Dev/rc strings ("5.0.0.dev0") still parse; anything this regex cannot read
    degrades to ``None`` so an unrecognisable version never blocks a load that may be fine.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        raw = version("transformers")
    except PackageNotFoundError:
        return None, None
    m = re.match(r"(\d+)\.(\d+)", raw)
    return raw, ((int(m.group(1)), int(m.group(2))) if m else None)


# The `[plm]` extra's own ceiling. A remedy that omits it sends the caller to the latest
# release, which today is 5.x — breaking prot_bert and misaligning the Ankh checkpoints while
# "fixing" the constraint that prompted the message.
_PLM_CEILING = (5, 0)


def _fmt_bounds(c: _TfConstraint) -> str:
    """The constraint as a PEP 440 specifier fragment, e.g. ``>=4.57`` or ``<5.0``."""
    parts = []
    if c.at_least is not None:
        parts.append(f">={c.at_least[0]}.{c.at_least[1]}")
    if c.below is not None:
        parts.append(f"<{c.below[0]}.{c.below[1]}")
    return ",".join(parts)


def _fmt_remedy(c: _TfConstraint) -> str:
    """The constraint as an *installable* specifier — always bounded above by `_PLM_CEILING`."""
    below = c.below if c.below is not None else _PLM_CEILING
    return _fmt_bounds(c._replace(below=min(below, _PLM_CEILING)))


def check_transformers_version(model_name: str) -> None:
    """Apply `_TRANSFORMERS_CONSTRAINTS` for one PLM key: raise on hard, warn on soft.

    A no-op when transformers is absent (the caller guards availability separately) or when its
    version cannot be parsed — an unreadable version string is not evidence of a violation.
    """
    constraints = _TRANSFORMERS_CONSTRAINTS.get(model_name)
    if not constraints:
        return
    raw, parsed = _installed_transformers_version()
    if parsed is None:
        return
    model_id = C.PLM_ENCODERS.get(model_name, model_name)
    for c in constraints:
        violated = c.always or (
            (c.at_least is not None and parsed < c.at_least)
            or (c.below is not None and parsed >= c.below)
        )
        if not violated:
            continue
        if c.always:
            # No version satisfies this one, so there is no specifier to quote and no pip line
            # that would help; `why` carries the alternative instead.
            raise ImportError(
                f"PLM {model_name!r} ({model_id}) has no working transformers path on "
                f"transformers {raw}: {c.why}."
            )
        spec = _fmt_bounds(c)
        msg = (
            f"PLM {model_name!r} ({model_id}) requires transformers{spec}, but transformers "
            f"{raw} is installed: {c.why}."
        )
        remedy = _fmt_remedy(c)
        if c.hard:
            # ImportError, not ValueError: this is a dependency-environment problem, so it lands
            # in the same `except ImportError` as the missing-package error from the guard above.
            raise ImportError(f"{msg} Fix: pip install 'transformers{remedy}'")
        warnings.warn(
            f"{msg} Embeddings are still produced and still correctly aligned, but do not mix "
            f"them with ones computed under transformers{spec}. "
            f"To match those: pip install 'transformers{remedy}'",
            stacklevel=3,
        )


def get_device() -> torch.device:
    """Return the best available device, preferring Apple Silicon (MPS), then CUDA, then CPU.

    Set ``FOLDENV_FORCE_CPU=1`` to force CPU (e.g. to free MPS, or for MPS-vs-CPU parity).
    """
    if os.environ.get("FOLDENV_FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_available_plms() -> List[str]:
    """Return the names of the available pretrained language models."""
    return list(C.PLM_ENCODERS.keys())


def load_pretrained_plm(model_name: str, device: Optional[TorchDevice] = None):
    """Load ``(model, tokenizer)`` for a PLM encoder key, downloading weights on first use.

    Order matters: the key is validated first (a typo must not surface as a dependency error),
    then transformers' presence and this checkpoint's version window — both before the
    multi-GB download, so an environment that cannot run the model says so in seconds.
    """
    model_id = C.PLM_ENCODERS.get(model_name)
    if model_id is None:
        raise ValueError(
            f"Invalid model_name: {model_name}. Must be one of {get_available_plms()}"
        )
    require_plm_dependencies("transformers")
    check_transformers_version(model_name)
    if device is None:
        device = get_device()

    if "t5" in model_id.lower() or "ankh" in model_id.lower():
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(model_id)
        if "ankh" in model_id.lower():
            from transformers import AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except ImportError:
                # Some Ankh checkpoints (e.g. ankh3) ship only a slow SentencePiece
                # tokenizer; building the fast one needs protobuf. Fall back to the
                # slow tokenizer (sentencepiece only) — same per-residue token ids.
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        else:
            from transformers import T5Tokenizer

            tokenizer = T5Tokenizer.from_pretrained(model_id, do_lower_case=False)
    elif "esmc" in model_id.lower():
        # ESM Cambrian ships only a masked-LM head (ESMCForMaskedLM); per-residue
        # states come from output_hidden_states, not last_hidden_state. The native
        # `esmc` model_type is registered by no measured release, so the table above refuses
        # this checkpoint outright; this branch stands for if/when upstream ships it.
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model = AutoModelForMaskedLM.from_pretrained(model_id, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    model = model.to(device)
    model = model.eval()
    return model, tokenizer


@torch.inference_mode()
def embed_sequence(
    plm_model: PreTrainedModel, plm_tokenizer: PreTrainedTokenizerBase, sequence: str
):
    """Embed a sequence using a pretrained model; returns ``[1, L, dim]`` per-residue states."""
    sequence = sequence.upper()
    sequence = re.sub(r"[UZOB]", "X", sequence)  # always replace non-canonical AAs with X
    n_residues = len(sequence)  # capture before any prefixing/space-joining (per-residue count)
    # Pre-process sequence for ProtTrans models
    is_prostt5 = "ProstT5" in plm_tokenizer.name_or_path
    is_ankh3 = "ankh3" in plm_tokenizer.name_or_path.lower()
    if is_prostt5:
        # ProstT5 is bilingual: prepend the amino-acid-mode prefix and space-separate residues
        sequence = "<AA2fold> " + " ".join(sequence)
    elif is_ankh3:
        # Ankh3 is prefix-conditioned: its model card uses "[NLU]" for encoder embedding
        # extraction ("[S2S]" is an alternative it suggests may be stronger). Select via the
        # ANKH3_PREFIX env var (default "[NLU]"). Unlike ankh-large, ankh3 is NOT 1:1 raw.
        sequence = os.environ.get("ANKH3_PREFIX", "[NLU]") + sequence
    elif "Rostlab/prot" in plm_tokenizer.name_or_path:
        sequence = " ".join(sequence)
    inputs = plm_tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        return_special_tokens_mask=True,
    ).to(plm_model.device)
    outputs = plm_model(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    )
    # Encoder models expose `last_hidden_state`; masked-LM-head models (e.g. ESM C)
    # only carry per-residue states in `hidden_states` (final layer).
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        hidden = outputs.hidden_states[-1]
    embedding = hidden[~inputs["special_tokens_mask"].bool()].unsqueeze(0)
    if is_prostt5:
        # The <AA2fold> prefix is not flagged by `special_tokens_mask`, so drop its
        # leading position to keep only per-residue embeddings.
        embedding = embedding[:, 1:, :]
    elif is_ankh3:
        # The "[NLU]"/"[S2S]" prefix token — and, in some tokenizer versions, a spurious
        # leading <unk> — are not flagged by special_tokens_mask. Strip however many leading
        # non-residue tokens there actually are by anchoring on the residue count, so this is
        # correct whether or not the <unk> is emitted (older tokenizers emit it → strips 2;
        # newer ones don't → strips 1). A hardcoded strip drops the first residue otherwise.
        n_strip = embedding.shape[1] - n_residues
        embedding = embedding[:, n_strip:, :]
    return embedding
