"""L2 disk-persistence for the two expensive derived artifacts.

`context.py` holds parsed structure, DSSP, the contact KD-tree, and the one PLM forward
pass in in-memory dicts (L1). Those vanish with the process, so a multi-protein ×
multi-config sweep re-pays the two costly steps — **mkdssp** and
the **per-protein PLM forward pass** — every fresh run. This module adds a read-through
**disk** cache (L2) under `.foldenv_cache/`, so those two are computed
once per (protein, relevant-config) and reused across processes.

Only DSSP and embeddings are persisted. The raw AlphaFold/RCSB mmCIF already persists
(`fetch.py`); the contact KD-tree is cheap to rebuild, so it is not cached here (D4).

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
* **Embedding** depends on accession + model name + the **transformers version** that
  tokenized it; the tensor is CPU/device-agnostic (`embed_protein` returns `.cpu()`), so one
  forward pass is shared across *all* geometry configs (every contact/RSA variant reuses it).
  The model name and a `major.minor` transformers tag go in the filename: tokenization is not
  stable across the whole supported transformers range (Ankh3 loses a spurious leading `<unk>`
  at 4.50, and 5.x reintroduces one across the Ankh checkpoints), so the same accession+model
  under two versions is two different tensors and they must not land on the same path. Models
  loaded through the `esm` SDK take no tag — they never touch transformers, so tagging them
  would invalidate their cache on every unrelated upgrade. **Structure-aware models** (SaProt)
  additionally derive a 3Di channel from the AF backbone, so their key also carries a
  fingerprint of the mmCIF file (size+mtime) — otherwise a same-length AF-DB re-release would
  be served the stale coordinates' embedding (the length guard can't catch same-length
  structural drift).

A ``_FORMAT`` tag is embedded in each filename so a schema change invalidates cleanly (old
files are simply never looked up). Corrupt/unreadable files are treated as a miss, never an
error. Persistence is on by default; disable per call via
``config.load(overrides={"cache": {"persist": False}})`` (the opt-out for
correctness runs).
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from .dssp import SS8_TO_SS3, ResidueDSSP
from .fetch import _check_identifier

# Bump when the *stored values* of an artifact can change: the on-disk schema, OR the
# computation that produces them — `run_dssp` / MaxASA tables (dssp.py) for DSSP, or
# `embed_protein`'s tokenization/slicing (embedding.py) for embeddings. The path carries only
# the RSA-table *name*, the model *name*, and the transformers *major.minor*; a change to the
# table numbers, or to foldenv's own embed logic at a fixed transformers version, would
# otherwise silently serve stale disk artifacts, so bump the relevant _FORMAT then.
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

def _safe_tag(value: str) -> str:
    """A config value reduced to what is safe in a filename.

    Same treatment `_dssp_exe_tag` gives the executable. Applied to `max_asa_table` because it
    sits in the same generated path and is equally caller-supplied.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


def _dssp_exe_tag(cfg: dict) -> str:
    """Filename-safe tag for the configured mkdssp executable (basename). Keyed so the
    disk cache matches context.py's in-memory key, which also includes the executable —
    different mkdssp builds can yield different absolute ASA (hence RSA)."""
    exe = str(cfg.get("dssp", {}).get("executable", "mkdssp"))
    base = os.path.basename(exe) or "mkdssp"
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def _dssp_path(cfg: dict, accession: str) -> Path:
    # First statement, before `_structure_fingerprint` below stats a path built from it:
    # `context.get_dssp` consults this cache *before* it reaches `fetch_structure`, so a guard
    # living only in `fetch` never runs for it. `max_asa_table` gets the same filename
    # sanitising `_dssp_exe_tag` gives the executable -- both are caller-supplied config.
    accession = _check_identifier(accession, "accession")
    table = _safe_tag(cfg["rsa"]["max_asa_table"])
    exe = _dssp_exe_tag(cfg)
    # DSSP is coordinate-derived → include the structure fingerprint so a re-fetched /
    # re-released mmCIF invalidates the stale RSA/SS (same defence as structure-aware
    # embeddings; "0" when absent leaves the key stable, e.g. for non-AF structures).
    fp = _structure_fingerprint(cfg, accession)
    # Checked here, not at the callers: `context.get_dssp` consults this cache *before* it
    # reaches `fetch_structure`, so a guard that lives only in `fetch` never runs for it.
    # `max_asa_table` is sanitised the way `_dssp_exe_tag` sanitises the executable -- both are
    # caller-supplied config that lands in a filename.
    return _root(cfg) / "dssp" / f"{accession}__{table}__{exe}__s{fp}__v{_DSSP_FORMAT}.json"


#: The three-state alphabet `run_dssp` reduces to. `ss8` is checked for shape instead, because
#: `run_dssp` stores an unrecognised DSSP code verbatim and reduces it to "C".
_SS3_VALUES = frozenset("HEC")
_SS8_VALUES = frozenset(SS8_TO_SS3)


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
            # Validate on read. A cache file is not trusted input: it can be truncated, written
            # by an older format, or -- since the cache directory is CWD-relative by default --
            # edited by anyone who can write there. Values from here flow straight out through
            # `tool.invoke` to an LLM caller, and `OUTPUT_SCHEMA` promises `rsa` in [0, 1] and a
            # three-letter `ss3`, so a bad file must read as a miss rather than as an answer.
            # `ss8` is checked for shape, not membership: `run_dssp` maps an unrecognised DSSP
            # code to "C" via `SS8_TO_SS3.get(ss8, "C")` and stores the raw code, which
            # `dssp.py` documents as supported. Rejecting it here would refuse a record this
            # package legitimately writes, and the file would miss forever.
            if not isinstance(r["ss8"], str) or len(r["ss8"]) != 1:
                return None
            if r["ss3"] != SS8_TO_SS3.get(r["ss8"], "C"):
                return None
            if not isinstance(r["aa"], str) or len(r["aa"]) != 1:
                return None
            if not isinstance(r["acc"], (int, float)) or isinstance(r["acc"], bool):
                return None
            if rsa is not None and not (isinstance(rsa, (int, float))
                                        and not isinstance(rsa, bool) and 0.0 <= rsa <= 1.0):
                return None
            result[resnum] = ResidueDSSP(
                resnum=resnum,
                aa=r["aa"],
                ss3=r["ss3"],
                ss8=r["ss8"],
                acc=r["acc"],
                rsa=float("nan") if rsa is None else float(rsa),  # None ↔ NaN (strict JSON)
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


@lru_cache(maxsize=1)
def _transformers_tag() -> str:
    """`major.minor` of the installed transformers (e.g. `4.57`), or `na` when unavailable.

    Read from distribution metadata rather than by importing transformers, so a cache *lookup*
    stays free — this runs on every embedding hit, including the ones whose whole point is to
    skip loading the PLM stack. Cached because the installed version cannot change mid-process.
    `major.minor` is the right granularity: the tokenization differences that make two tensors
    incomparable land on minor releases, and patch-level keying would evict good caches.
    Deliberately NOT applied to the DSSP key — DSSP is mkdssp + coordinates only.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        raw = version("transformers")
    except PackageNotFoundError:
        return "na"
    m = re.match(r"(\d+)\.(\d+)", raw)
    return f"{m.group(1)}.{m.group(2)}" if m else "na"


def _emb_path(cfg: dict, accession: str, model_name: str) -> Path:
    accession = _check_identifier(accession, "accession")   # before any path is built from it
    # Two optional key components, each added only where it can actually change the tensor:
    #   `t<major.minor>` — the transformers version that tokenized it (see the module docstring);
    #     omitted for `esm`-SDK models, which never go through transformers.
    #   `s<size-mtime>`  — the structure fingerprint. SaProt & other structure-aware models bake
    #     the AF backbone into the embedding, so their key must track the structure file too
    #     (finding: same-length re-releases). Sequence-only models are fully determined by
    #     accession+model+version, so we don't fingerprint them (a structure refresh at equal
    #     length would needlessly invalidate an identical embedding).
    # `_EMB_FORMAT` is *not* bumped for this: adding the tag already makes every stale
    # transformers-path file unreachable, while a bump would also throw away SDK-path caches
    # that are still perfectly valid.
    from .embedding import get_model_spec

    spec = get_model_spec(model_name)
    parts = [accession, model_name]
    if not spec.sdk:
        parts.append(f"t{_transformers_tag()}")
    if spec.structure_aware:
        parts.append(f"s{_structure_fingerprint(cfg, accession)}")
    parts.append(f"v{_EMB_FORMAT}")
    return _root(cfg) / "embeddings" / ("__".join(parts) + ".pt")


#: The first four bytes of a PKZIP local file header. `torch.save` has written this container
#: since torch 1.6; the older `.tar` container is what CVE-2025-32434 bypasses `weights_only` on.
_ZIP_MAGIC = b"PK\x03\x04"


def _is_zip_pt(fh) -> bool:
    """Whether an open binary file starts with the zip magic `torch.save` writes.

    Deliberately not `torch.serialization._is_zipfile`: that is private, has moved between
    releases, and this is a four-byte check. Reads and rewinds, so the caller's handle is unmoved.
    """
    pos = fh.tell()
    try:
        return fh.read(4) == _ZIP_MAGIC
    finally:
        fh.seek(pos)


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
    from .embedding import _import_torch  # names the [plm] extra if torch is absent

    torch = _import_torch()

    try:
        # Refuse anything that is not a modern zip-format .pt, before torch.load sees it.
        #
        # `save_embedding` writes with plain `torch.save`, which has produced the zip format
        # since torch 1.6 -- so a legacy `.tar` file here was not written by this package. The
        # distinction matters: on torch <=2.5, `torch.load(..., weights_only=True)` does not
        # constrain the legacy tar path at all (CVE-2025-32434), so `weights_only` silently
        # stops being a defence for exactly the file shape we never emit. Later torch restricts
        # the legacy path properly, so this is belt-and-braces there -- but checking the container
        # closes it on every version rather than only for callers who have upgraded, which is why
        # this package does not pin a torch floor for it.
        with open(path, "rb") as fh:
            if not _is_zip_pt(fh):
                return None

        # `weights_only=True` always, with no fallback. The kwarg has existed since torch 1.13 and
        # the declared floor is 2.0, so a fallback for "older torch has no weights_only" guards a
        # case that cannot arise -- while turning a `TypeError` raised *inside* restricted
        # unpickling into a second, unrestricted load of the same bytes. On torch 2.0-2.5, where
        # the bare default is still `weights_only=False`, that second load executes whatever the
        # file contains. A cache file is untrusted input; there is nothing to fall back to.
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Corrupt / truncated / not-a-tensor .pt → miss and recompute.
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
    from .embedding import _import_torch

    torch = _import_torch()

    path = _emb_path(cfg, accession, model_name)
    cpu_tensor = tensor.contiguous().cpu()
    try:
        _write_atomic(path, lambda tmp: torch.save(cpu_tensor, tmp))
    except (OSError, RuntimeError):
        pass
