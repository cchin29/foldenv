"""Config-loader tests (no network, no heavy deps)."""
from pathlib import Path

from foldenv import config


def test_defaults_load():
    cfg = config.load()
    assert cfg["contacts"]["primary"] == "ca"
    assert cfg["contacts"]["ca_cutoff"] == 8.0
    assert cfg["contacts"]["cb_cutoff"] == 5.0
    assert cfg["plddt"]["mask_below"] == 50.0
    assert cfg["rsa"]["max_asa_table"] == "tien2013_theoretical"
    assert cfg["embedding"]["model"] == "ankh"
    assert cfg["embedding"]["device"] == "auto"
    assert cfg["dssp"]["output_format"] == "dssp"


def test_cache_dir_resolved():
    # null in YAML must resolve to a concrete path under the repo.
    cfg = config.load()
    assert cfg["cache"]["dir"] is not None
    assert cfg["cache"]["dir"].endswith(".foldenv_cache")


def test_overrides_deep_merge():
    cfg = config.load(overrides={"contacts": {"primary": "cb"}})
    assert cfg["contacts"]["primary"] == "cb"
    # sibling keys under the overridden section survive the deep-merge
    assert cfg["contacts"]["ca_cutoff"] == 8.0
    # untouched sections survive
    assert cfg["plddt"]["mask_below"] == 50.0


def test_overrides_do_not_mutate_file_defaults():
    config.load(overrides={"contacts": {"primary": "cb"}})
    assert config.load()["contacts"]["primary"] == "ca"


def test_decisions_path_exists():
    assert Path(config.decisions_path()).is_file()
