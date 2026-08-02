"""Load the locked decisions (D1–D6) from `decisions.yaml`.

The YAML holds the reasonable defaults; `load()` returns a nested dict and lets callers
override any leaf without touching the file. Keeping the decisions in data (not code) means
a results writeup can point at one file for "which cutoff / table / mask did you use?".
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

_DECISIONS_PATH = Path(__file__).with_name("decisions.yaml")


def _default_cache_dir() -> Path:
    """Default on-disk cache location, resolved fresh on each call.

    Overridable via the ``FOLDENV_CACHE_DIR`` env var or the ``cache.dir`` config leaf.
    Defaults under the current working directory so the package works the same whether
    installed into site-packages or run from a checkout. Resolving here (not at import
    time) means the env var and the working directory are read when ``load()`` runs, not
    when the module is first imported.
    """
    return Path(os.environ.get("FOLDENV_CACHE_DIR", Path.cwd() / ".foldenv_cache"))


def _deep_update(base: dict, overrides: Mapping[str, Any]) -> dict:
    """Recursively merge `overrides` into `base` (mutating and returning `base`)."""
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load(overrides: Mapping[str, Any] | None = None) -> dict:
    """Return the decisions dict, with `cache.dir` resolved to a concrete path.

    Args:
        overrides: nested dict merged over the file defaults, e.g.
            ``{"contacts": {"primary": "cb"}}``.
    """
    with open(_DECISIONS_PATH) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_update(cfg, copy.deepcopy(dict(overrides)))

    # Resolve the caching default here so downstream code always sees a real directory.
    if cfg["cache"].get("dir") is None:
        cfg["cache"]["dir"] = str(_default_cache_dir())

    # Allow the AlphaFold API base to be overridden by env (e.g. to point tests at an
    # unreachable host for deterministic, offline CI). An explicit `overrides` value wins.
    if (overrides is None or "alphafold" not in overrides) and os.environ.get(
        "FOLDENV_ALPHAFOLD_API_BASE"
    ):
        cfg["alphafold"]["api_base"] = os.environ["FOLDENV_ALPHAFOLD_API_BASE"]
    return cfg


def decisions_path() -> Path:
    """Path to the backing YAML (for logging which config a run used)."""
    return _DECISIONS_PATH
