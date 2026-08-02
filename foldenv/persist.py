"""L2 disk-persistence for the two expensive derived artifacts.

`context.py` holds parsed structure, DSSP, the contact KD-tree, and the one PLM forward
pass in in-memory dicts (L1). Those vanish with the process, so a multi-protein ×
multi-config sweep (the P1/P2 eval expansion) re-pays the two costly steps — **mkdssp** and
the **per-protein PLM forward pass** — every fresh run. This module adds a read-through
**disk** cache (L2) under the existing `.struct_context_cache/`, so those two are computed
once per (protein, relevant-config) and reused across processes.

Only DSSP and embeddings are persisted. The raw AlphaFold/RCSB mmCIF already persists
(`fetch.py`); the contact KD-tree is cheap to rebuild (not cached here, per §4).

Keying (what changes the artifact → what the on-disk path/version must include):

* **DSSP** depends on the structure + mkdssp + the RSA table (D3). All three now key the
  filename: the RSA table name (our MaxASA tables normalize DSSP's *absolute* ASA), the
  configured mkdssp **executable** (its build can shift the absolute ASA — and context.py's
  in-memory key already includes it, so the disk key must match or a run under a different
  binary would hit the file the other one wrote), and a **structure fingerprint** (size+mtime
  of the mmCIF — DSSP is entirely coordinate-derived, so a force-refetched/re-released
  structure must invalidate it, exactly as structure-aware embeddings already do). We key on
  the executable *string*, not its detected version (avoids a subprocess on every lookup); a
  correctness re-run across identically-named binaries still has the `persist:false` escape.
* **Embedding** depends on accession + model name; the tensor is CPU/device-agnostic
  (`embed_protein` returns `.cpu()`), so one forward pass is shared across *all* geometry
  configs (P1/P2/P4 all reuse it). The model name goes in the filename. **Structure-aware
  models** (SaProt) additionally derive a 3Di channel from the AF backbone, so their key also
  carries a fingerprint of the mmCIF file (size+mtime) — otherwise a same-length AF-DB
  re-release would be served the stale coordinates' embedding (the length guard can't catch
  same-length structural drift).

A ``_FORMAT`` tag is embedded in each filename so a schema change invalidates cleanly (old
files are simply never looked up). Corrupt/unreadable files are treated as a miss, never an
error. Persistence is on by default; disable per call via
``config.load(overrides={"cache": {"persist": False}})`` (the ``--no-cache`` escape for
correctness runs).
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

from .dssp import ResidueDSSP

# Bump when the *stored values* of an artifact can change: the on-disk schema, OR the
# computation that produces them — `run_dssp` / MaxASA tables (dssp.py) for DSSP, or
# `embed_protein`'s tokenization/slicing (embedding.py) for embeddings. Only the RSA-table
# *name* and model *name* are in the path; a change to the table *numbers* or the embed logic
# would otherwise silently serve stale disk artifacts, so bump the relevant _FORMAT then.
_DSSP_FORMAT = 1
_EMB_FORMAT = 1


def _write_atomic(path: Path, write_fn) -> None:
    """Write via a per-process-unique temp file, then atomically replace `path`.

    A fixed `.tmp` name would let two sweep processes computing the same artifact interleave
    their writes into one temp file; `mkstemp` gives each writer its own, so the final
    `replace()` is always a whole, self-consistent file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write_fn(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace


def enabled(cfg: dict) -> bool:
    """Whether the disk cache is active for this config (default on)."""
    return bool(cfg["cache"].get("persist", True))


def _root(cfg: dict) -> Path:
    """Cache root — the same directory `fetch.py` uses for mmCIF (resolved by config.load)."""
    return Path(cfg["cache"]["dir"])


# --- DSSP (M2) -------------------------------------------------------------------------

def _dssp_exe_tag(cfg: dict) -> str:
    """Filename-safe tag for the configured mkdssp executable (basename). Keyed so the
    disk cache matches context.py's in-memory key, which also includes the executable —
    different mkdssp builds can yield different absolute ASA (hence RSA)."""
    exe = str(cfg.get("dssp", {}).get("executable", "mkdssp"))
    base = os.path.basename(exe) or "mkdssp"
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def _dssp_path(cfg: dict, accession: str) -> Path:
    table = cfg["rsa"]["max_asa_table"]
    exe = _dssp_exe_tag(cfg)
    # DSSP is coordinate-derived → include the structure fingerprint so a re-fetched /
    # re-released mmCIF invalidates the stale RSA/SS (same defence as structure-aware
    # embeddings; "0" when absent leaves the key stable, e.g. for non-AF structures).
    fp = _structure_fingerprint(cfg, accession)
    return _root(cfg) / "dssp" / f"{accession}__{table}__{exe}__s{fp}__v{_DSSP_FORMAT}.json"


def load_dssp(cfg: dict, accession: str) -> dict[int, ResidueDSSP] | None:
    """Return the cached DSSP result for `accession`, or None on miss/disabled/corrupt."""
    if not enabled(cfg):
        return None
    path = _dssp_path(cfg, accession)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
        if blob.get("format_version") != _DSSP_FORMAT:
            return None
        result: dict[int, ResidueDSSP] = {}
        for resnum_str, r in blob["residues"].items():
            resnum = int(resnum_str)
            rsa = r["rsa"]
            result[resnum] = ResidueDSSP(
                resnum=resnum,
                aa=r["aa"],
                ss3=r["ss3"],
                ss8=r["ss8"],
                acc=r["acc"],
                rsa=float("nan") if rsa is None else rsa,  # None ↔ NaN (strict JSON)
            )
        return result
    except (OSError, ValueError, KeyError):
        # Corrupt / partially-written / schema-drifted file → treat as a miss, recompute.
        return None


def save_dssp(cfg: dict, accession: str, result: dict[int, ResidueDSSP]) -> None:
    """Persist a DSSP result. No-op when disabled. Best-effort (write failures are ignored)."""
    if not enabled(cfg):
        return
    path = _dssp_path(cfg, accession)
    residues = {
        str(resnum): {
            "aa": d.aa,
            "ss3": d.ss3,
            "ss8": d.ss8,
            "acc": d.acc,
            # NaN is not valid strict JSON → store null, restore to NaN on load.
            "rsa": None if (isinstance(d.rsa, float) and math.isnan(d.rsa)) else d.rsa,
        }
        for resnum, d in result.items()
    }
    blob = {
        "format_version": _DSSP_FORMAT,
        "max_asa_table": cfg["rsa"]["max_asa_table"],
        "residues": residues,
    }

    def _write(tmp: Path) -> None:
        with open(tmp, "w") as f:
            # allow_nan=False enforces the strict-JSON guarantee: any future NaN-capable field
            # that isn't pre-scrubbed (as rsa is above) fails loudly rather than writing `NaN`.
            json.dump(blob, f, allow_nan=False)

    try:
        _write_atomic(path, _write)
    except OSError:
        pass


# --- Per-protein embedding (M4) --------------------------------------------------------

def _structure_fingerprint(cfg: dict, accession: str) -> str:
    """Short fingerprint of the cached mmCIF (size+mtime) — distinguishes AF-DB re-releases.

    Mirrors `fetch.py`'s on-disk layout (`<dir>/alphafold/<ACC>.cif`). Only used to key
    structure-aware embeddings; `"0"` if the file isn't present (the embedding flow fetches it
    first, so this is just a defensive default).
    """
    cif = _root(cfg) / "alphafold" / f"{accession}.cif"
    try:
        st = cif.stat()
        return f"{st.st_size}-{int(st.st_mtime)}"
    except OSError:
        return "0"


def _emb_path(cfg: dict, accession: str, model_name: str) -> Path:
    # SaProt & other structure-aware models bake the AF backbone into the embedding, so their
    # key must track the structure file too (finding: same-length re-releases). Sequence-only
    # models are fully determined by accession+model, so we don't fingerprint them (a structure
    # refresh at equal length would needlessly invalidate an identical embedding).
    from .embedding import get_model_spec

    if get_model_spec(model_name).structure_aware:
        fp = _structure_fingerprint(cfg, accession)
        stem = f"{accession}__{model_name}__s{fp}__v{_EMB_FORMAT}.pt"
    else:
        stem = f"{accession}__{model_name}__v{_EMB_FORMAT}.pt"
    return _root(cfg) / "embeddings" / stem


def load_embedding(cfg: dict, accession: str, model_name: str):
    """Return the cached `[L, dim]` CPU tensor for (accession, model), or None on miss.

    The caller re-applies the per-residue length guard, so a stale/short tensor still fails
    loudly at assembly rather than silently misaligning positions.
    """
    if not enabled(cfg):
        return None
    path = _emb_path(cfg, accession, model_name)
    if not path.exists():
        return None
    import torch

    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # older torch has no weights_only kwarg
            return torch.load(path, map_location="cpu")
    except Exception:
        # Corrupt / truncated .pt → miss and recompute.
        return None


def evict_embedding(cfg: dict, accession: str, model_name: str) -> None:
    """Delete a cached embedding — e.g. after the caller's length guard rejects a stale/
    poisoned tensor, so it self-heals on the next run instead of failing forever. Best-effort;
    no-op when disabled or absent."""
    if not enabled(cfg):
        return
    try:
        _emb_path(cfg, accession, model_name).unlink(missing_ok=True)
    except OSError:
        pass


def save_embedding(cfg: dict, accession: str, model_name: str, tensor) -> None:
    """Persist a `[L, dim]` embedding tensor. No-op when disabled. Best-effort."""
    if not enabled(cfg):
        return
    if not hasattr(tensor, "shape"):  # only real tensors (skip stubs / None)
        return
    import torch

    path = _emb_path(cfg, accession, model_name)
    cpu_tensor = tensor.contiguous().cpu()
    try:
        _write_atomic(path, lambda tmp: torch.save(cpu_tensor, tmp))
    except (OSError, RuntimeError):
        pass
