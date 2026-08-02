"""PLM loading + per-residue embedding helpers.

Vendored so ``foldenv`` is self-contained: ``load_pretrained_plm`` builds the
(model, tokenizer) pair for a supported encoder, and ``embed_sequence`` runs one forward
pass and returns the per-residue embedding, handling each model's prefixes and slicing.
"""

from typing import List, Optional, Union

import os
import re

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from . import constants as C

TorchDevice = Union[str, torch.device]


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
    """Load ``(model, tokenizer)`` for a PLM encoder key, downloading weights on first use."""
    if device is None:
        device = get_device()
    model_id = C.PLM_ENCODERS.get(model_name)
    if model_id is None:
        raise ValueError(
            f"Invalid model_name: {model_name}. Must be one of {get_available_plms()}"
        )

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
        # states come from output_hidden_states, not last_hidden_state. Needs
        # transformers>=4.57 (native `esmc` model_type).
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
