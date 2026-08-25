"""Missing-dependency ergonomics for the optional PLM stack (nothing is uninstalled).

`torch`/`transformers` are the `[plm]` extra, so their absence is an expected state, not a
broken install — and `ModuleNotFoundError: No module named 'torch'` names neither the cause
nor the cure. Every entry into the PLM stack must instead raise an ImportError that names the
extra. Absence is simulated by monkeypatching `importlib.util.find_spec` (the guard's own
probe), so these run unchanged in an environment that has the whole stack installed.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from foldenv import embedding as E

# Guard on torch, not on `foldenv.plm`: the module's own optional-dependency guard raises a
# plain ImportError, which `importorskip` re-raises rather than skipping on.
pytest.importorskip("torch", reason="foldenv.plm needs torch (the [plm] extra)")

from foldenv import plm  # noqa: E402  — deliberately after the skip guard above

_ROOT = Path(__file__).resolve().parents[1]


def _hide(monkeypatch, *names):
    """Make `find_spec` report `names` as absent, leaving every other module real."""
    real = importlib.util.find_spec
    hidden = set(names)

    def fake(name, package=None):
        if name.split(".")[0] in hidden:
            return None
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def _boom(*a, **k):
    raise RuntimeError("got past the dependency guard")


# --- the guard itself ------------------------------------------------------------------

def test_missing_torch_names_the_extra(monkeypatch):
    _hide(monkeypatch, "torch")
    with pytest.raises(ImportError) as exc:
        plm.require_plm_dependencies("torch")
    msg = str(exc.value)
    assert "torch" in msg
    assert 'foldenv[plm]' in msg               # names the fix
    assert 'embedding.model = "none"' in msg   # and the way to skip the stack entirely


def test_missing_dependencies_are_reported_together(monkeypatch):
    # One traceback should list everything that is missing, not one package per attempt.
    _hide(monkeypatch, "torch", "transformers")
    with pytest.raises(ImportError) as exc:
        plm.require_plm_dependencies("torch", "transformers")
    msg = str(exc.value)
    assert "torch" in msg and "transformers" in msg


def test_only_the_missing_one_is_named(monkeypatch):
    _hide(monkeypatch, "transformers")
    with pytest.raises(ImportError) as exc:
        plm.require_plm_dependencies("torch", "transformers")
    assert "transformers" in str(exc.value)
    assert "torch" not in str(exc.value)


def test_guard_is_a_noop_when_everything_is_present():
    plm.require_plm_dependencies("torch")  # no exception


def test_error_is_not_a_bare_modulenotfounderror(monkeypatch):
    # ModuleNotFoundError subclasses ImportError, so `pytest.raises(ImportError)` alone would
    # also pass on the unhelpful error this design exists to replace.
    _hide(monkeypatch, "torch")
    with pytest.raises(ImportError) as exc:
        plm.require_plm_dependencies("torch")
    assert not isinstance(exc.value, ModuleNotFoundError)


# --- every PLM entry point routes through it -------------------------------------------

def test_load_pretrained_plm_without_transformers(monkeypatch):
    _hide(monkeypatch, "transformers")
    monkeypatch.setattr(plm, "get_device", _boom)
    with pytest.raises(ImportError, match=r"foldenv\[plm\]"):
        plm.load_pretrained_plm("ankh")


def test_load_embedder_saprot_without_transformers(monkeypatch):
    # SaProt builds its own EsmForMaskedLM pair instead of going through load_pretrained_plm,
    # so it needs — and must use — its own guard.
    _hide(monkeypatch, "transformers")
    with pytest.raises(ImportError, match=r"foldenv\[plm\]"):
        E.load_embedder("saprot", "cpu")


def test_load_embedder_transformers_path_without_transformers(monkeypatch):
    _hide(monkeypatch, "transformers")
    monkeypatch.setattr(plm, "get_device", _boom)
    with pytest.raises(ImportError, match=r"foldenv\[plm\]"):
        E.load_embedder("ankh", "cpu")


@pytest.mark.skipif(
    importlib.util.find_spec("esm") is not None,
    reason="the `esm` SDK is installed, so its absence cannot be observed here",
)
def test_esmc_sdk_path_names_its_own_extra():
    # ESM C via the SDK is transformers-free, so it must point at [esmc], not [plm].
    with pytest.raises(ImportError) as exc:
        E.load_embedder("esmc_600m", "cpu")
    msg = str(exc.value)
    assert "foldenv[esmc]" in msg
    assert "foldenv[plm]" not in msg


@pytest.mark.skipif(
    importlib.util.find_spec("esm") is not None,
    reason="the `esm` SDK is installed, so its absence cannot be observed here",
)
@pytest.mark.parametrize(
    "call",
    [
        lambda: E.embed_protein("esmc_600m", "MKTVRQ"),
        lambda: E.embed_residue("esmc_600m", "MKTVRQ", 1),
    ],
    ids=["embed_protein", "embed_residue"],
)
def test_public_helpers_name_the_sdk_extra_with_torch_hidden(monkeypatch, call):
    # The regression this locks: reaching torch before the `spec.sdk` dispatch would send an
    # ESM C user to download the multi-gigabyte [plm] stack for the wrong extra. Torch is
    # hidden explicitly so the assertion holds in a [plm] env too, where it would otherwise
    # pass for the wrong reason.
    _hide(monkeypatch, "torch")
    with pytest.raises(ImportError) as exc:
        call()
    msg = str(exc.value)
    assert "foldenv[esmc]" in msg
    assert "foldenv[plm]" not in msg


def test_import_torch_returns_the_module():
    assert E._import_torch().__name__ == "torch"


# --- the core install must not drag the stack in ---------------------------------------

def _in_subprocess(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_ROOT), capture_output=True, text=True,
    )


def test_core_import_pulls_in_neither_torch_nor_transformers():
    # The real guarantee behind the dependency split: `pip install foldenv` (no extras) must
    # still import. A fresh interpreter is the only honest way to ask — this process has torch
    # in sys.modules already. Checked here rather than by uninstalling anything.
    proc = _in_subprocess(
        "import sys, foldenv, foldenv.embedding, foldenv.persist, foldenv.context;"
        "print(sorted(m for m in ('torch', 'transformers') if m in sys.modules))"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_structural_path_does_not_import_the_plm_module():
    # `.plm` is where the torch guard lives, so importing it eagerly would make a bare install
    # fail at `import foldenv`.
    proc = _in_subprocess(
        "import sys, foldenv, foldenv.embedding;"
        "print('foldenv.plm' in sys.modules)"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"
