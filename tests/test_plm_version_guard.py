"""Per-checkpoint transformers-version guard (no weight downloads, no network).

The install pin is deliberately wide (`transformers>=4.27,<5`) because the per-checkpoint
requirements conflict — ESM C 6B needs >=4.57 while prot_bert needs <5 — so each checkpoint's
own window is enforced in `load_pretrained_plm` instead. These tests drive that table by
monkeypatching the *detected* transformers version, so they assert the policy on every
version in the matrix without installing any of them or fetching a single weight.
"""
import re
import warnings

import pytest

from foldenv import constants as C

# Guard on torch, not on `foldenv.plm`: the module's own optional-dependency guard raises a
# plain ImportError, which `importorskip` re-raises rather than skipping on.
pytest.importorskip("torch", reason="foldenv.plm needs torch (the [plm] extra)")
# transformers too, even though the version matrix is monkeypatched rather than installed: the
# two `load_pretrained_plm` tests below go through the loader, which requires transformers to be
# *present* before it reaches the version check. `foldenv[esmc]` is a real torch-without-
# transformers install, and this file must skip there rather than fail.
pytest.importorskip("transformers", reason="the loader tests need transformers (the [plm] extra)")

from foldenv import plm  # noqa: E402  — deliberately after the skip guard above


def _pretend_transformers(monkeypatch, raw):
    """Make the guard see transformers version `raw` (None = not installed)."""
    parsed = None
    if raw is not None:
        m = re.match(r"(\d+)\.(\d+)", raw)
        parsed = (int(m.group(1)), int(m.group(2))) if m else None
    monkeypatch.setattr(plm, "_installed_transformers_version", lambda: (raw, parsed))


def _assert_silent(model_name):
    """The check passes with neither an exception nor a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plm.check_transformers_version(model_name)
    assert caught == [], [str(w.message) for w in caught]


# --- the table itself ------------------------------------------------------------------

def test_constraint_keys_are_real_encoders():
    # A constraint keyed on a typo'd name is silently never applied.
    assert set(plm._TRANSFORMERS_CONSTRAINTS) <= set(C.PLM_ENCODERS)


def test_every_constraint_has_a_bound_and_a_reason():
    for name, constraints in plm._TRANSFORMERS_CONSTRAINTS.items():
        assert constraints, name
        for c in constraints:
            assert c.why.strip(), name
            if c.always:
                # An always-violated entry has nothing to bound: no version satisfies it, so it
                # must not pretend to by carrying a specifier.
                assert c.at_least is None and c.below is None, name
            else:
                assert (c.at_least is not None) or (c.below is not None), name
                assert plm._fmt_bounds(c), name


# --- hard constraints: raise -----------------------------------------------------------

def test_protbert_raises_on_transformers_5x(monkeypatch):
    _pretend_transformers(monkeypatch, "5.15.1")
    with pytest.raises(ImportError) as exc:
        plm.check_transformers_version("protbert")
    msg = str(exc.value)
    assert "protbert" in msg
    assert C.PLM_ENCODERS["protbert"] in msg      # names the checkpoint, not just the key
    assert "<5.0" in msg                          # names the requirement
    assert "5.15.1" in msg                        # names what is actually installed
    assert "pip install" in msg                   # names the fix


def test_protbert_ok_below_5(monkeypatch):
    for raw in ("4.27.0", "4.44.2", "4.50.0", "4.57.6"):
        _pretend_transformers(monkeypatch, raw)
        _assert_silent("protbert")


@pytest.mark.parametrize("raw", ["4.44.2", "4.57.0", "4.57.6", "5.15.1"])
def test_esmc_6b_raises_at_every_version(monkeypatch, raw):
    # The `esmc` model_type is registered by no measured release, so a floor-only rule would go
    # silent above 4.57 and let the caller hit the opaque upstream error this table pre-empts.
    _pretend_transformers(monkeypatch, raw)
    with pytest.raises(ImportError) as exc:
        plm.check_transformers_version("esmc_6b")
    msg = str(exc.value)
    assert "esmc_600m" in msg  # names the path that does work
    assert "pip install 'transformers" not in msg  # no version would help, so none is offered


def test_hard_violation_is_importerror_not_valueerror(monkeypatch):
    # ImportError so callers can catch the version problem in the same `except ImportError`
    # as the missing-package error; ValueError is reserved for a bad model key.
    _pretend_transformers(monkeypatch, "5.15.1")
    with pytest.raises(ImportError):
        plm.check_transformers_version("protbert")


@pytest.mark.parametrize("name", ["ankh", "ankh_base"])
def test_unanchored_ankh_raises_on_transformers_5x(monkeypatch, name):
    # 5.x prepends an <unk> that special_tokens_mask does not flag. These two have no
    # residue-count anchor, so the stray row survives and every residue shifts by one --
    # silent corruption, hence a raise. Measured on real 5.15.1: 7 rows for 6 residues.
    _pretend_transformers(monkeypatch, "5.15.1")
    with pytest.raises(ImportError, match=r"shifted by one"):
        plm.check_transformers_version(name)


@pytest.mark.parametrize("name", ["ankh3_large", "ankh3_xl"])
def test_anchored_ankh3_only_warns_on_transformers_5x(monkeypatch, name):
    # Same stray <unk>, but `embed_sequence`'s ankh3 branch re-anchors on the residue count and
    # drops it -- measured on real 5.15.1: 6 rows for 6 residues. Rows are right, values differ,
    # so this is comparability, not corruption, and must not block the run.
    _pretend_transformers(monkeypatch, "5.15.1")
    with pytest.warns(UserWarning, match=r"values"):
        plm.check_transformers_version(name)


def test_remedies_never_exceed_the_plm_ceiling(monkeypatch):
    # A remedy that omits the <5 ceiling installs the latest release, which is exactly the
    # version the constraint exists to keep out.
    _pretend_transformers(monkeypatch, "4.44.2")
    with pytest.warns(UserWarning) as rec:
        plm.check_transformers_version("ankh3_large")
    assert "pip install 'transformers>=4.50,<5.0'" in str(rec[0].message)

    _pretend_transformers(monkeypatch, "5.15.1")
    with pytest.raises(ImportError) as exc:
        plm.check_transformers_version("ankh")
    assert "pip install 'transformers<5.0'" in str(exc.value)


# --- soft constraints: warn, do not block ----------------------------------------------


def test_ankh_silent_inside_the_pin(monkeypatch):
    for raw in ("4.44.2", "4.50.0", "4.57.6"):
        _pretend_transformers(monkeypatch, raw)
        _assert_silent("ankh")


@pytest.mark.parametrize("name", ["ankh3_large", "ankh3_xl"])
def test_ankh3_warns_before_the_4_50_tokenization_change(monkeypatch, name):
    _pretend_transformers(monkeypatch, "4.44.2")
    with pytest.warns(UserWarning, match=r"4\.50"):
        plm.check_transformers_version(name)


@pytest.mark.parametrize("name", ["ankh3_large", "ankh3_xl"])
def test_ankh3_silent_from_4_50(monkeypatch, name):
    # The boundary fires only on the legacy side, so a current install is not nagged.
    for raw in ("4.50.0", "4.53.0", "4.57.6"):
        _pretend_transformers(monkeypatch, raw)
        _assert_silent(name)


def test_soft_violation_does_not_raise(monkeypatch):
    # The rows still land on the right residues; only the values are incomparable, so the
    # load must go through.
    _pretend_transformers(monkeypatch, "4.44.2")
    with pytest.warns(UserWarning):
        plm.check_transformers_version("ankh3_large")  # returns, no exception


# --- no constraint / no evidence: stay out of the way ----------------------------------

@pytest.mark.parametrize("name", ["prostt5", "prott5_xl_half", "esm", "saprot"])
def test_unconstrained_models_are_never_blocked(monkeypatch, name):
    for raw in ("4.27.0", "4.44.2", "5.15.1"):
        _pretend_transformers(monkeypatch, raw)
        _assert_silent(name)


def test_absent_transformers_is_a_noop(monkeypatch):
    # Presence is guarded separately; this check must not double-report it.
    _pretend_transformers(monkeypatch, None)
    _assert_silent("protbert")
    _assert_silent("esmc_6b")


def test_unparseable_version_is_a_noop(monkeypatch):
    # An unreadable version string is not evidence of a violation.
    _pretend_transformers(monkeypatch, "some-vendor-build")
    _assert_silent("protbert")
    _assert_silent("esmc_6b")


def test_dev_version_string_still_parses(monkeypatch):
    monkeypatch.setattr(
        plm, "_installed_transformers_version", lambda: ("5.0.0.dev0", (5, 0))
    )
    with pytest.raises(ImportError):
        plm.check_transformers_version("protbert")


# --- enforcement happens in the loader, before any download ----------------------------

def test_load_pretrained_plm_checks_version_before_touching_weights(monkeypatch):
    # `get_device` is the first thing after the two guards, so poisoning it proves the raise
    # came from the version check and that nothing downstream (download included) ran.
    _pretend_transformers(monkeypatch, "5.15.1")
    monkeypatch.setattr(plm, "get_device", _boom)
    with pytest.raises(ImportError, match=r"prot_bert"):
        plm.load_pretrained_plm("protbert")


def test_load_pretrained_plm_allows_a_satisfied_checkpoint_through(monkeypatch):
    # Same poison, opposite outcome: a satisfying version reaches `get_device`, i.e. the guard
    # is not blocking a legal load. `ankh` on 4.57.6 satisfies every entry that applies to it.
    _pretend_transformers(monkeypatch, "4.57.6")
    monkeypatch.setattr(plm, "get_device", _boom)
    with pytest.raises(RuntimeError, match="reached weight loading"):
        plm.load_pretrained_plm("ankh")


def test_bad_model_key_is_still_a_valueerror(monkeypatch):
    # The key is validated first: a typo must not surface as a dependency/version error.
    _pretend_transformers(monkeypatch, "5.15.1")
    monkeypatch.setattr(plm, "get_device", _boom)
    with pytest.raises(ValueError, match="Invalid model_name"):
        plm.load_pretrained_plm("not_a_plm")


def _boom():
    raise RuntimeError("reached weight loading")
