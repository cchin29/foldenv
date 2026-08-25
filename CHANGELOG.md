# Changelog

Notable changes to `foldenv`. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [semantic](https://semver.org/), with the 0.x caveat that a minor bump may carry
breaking changes.

## [0.2.0] — 2026-08-25

The dependency split: the structural pipeline no longer requires the deep-learning stack.
`pip install foldenv` is now `numpy`, `requests`, `pyyaml`, `biopython` and nothing else.

### Upgrading from 0.1.x — read this first

Two changes can surprise an in-place upgrade:

* **The default config now needs an extra.** `embedding.model` defaults to `ankh`, and the PLM
  path moved to `[plm]`. In a *fresh* environment, `get_structural_context("P04637", 175)` with
  default config raises `ImportError` until you `pip install "foldenv[plm]"`. Upgrading in an
  environment that already has torch and transformers is unaffected — nothing is uninstalled,
  and the import still resolves. Either install `foldenv[plm]`, or set
  `embedding.model = "none"` if you only need RSA, secondary structure, contacts and pLDDT.
* **Cached embeddings on the transformers path are invalidated, silently.** The on-disk key now
  includes the transformers `major.minor`, so existing `.pt` files are no longer found and the
  forward pass re-runs. Nothing is corrupted and nothing is deleted — but the first run after
  upgrading re-embeds, and the old files are left behind. To reclaim the space, delete the
  whole `.foldenv_cache/embeddings/` directory — it is a cache and rebuilds on demand. Do not
  try to delete the stale files selectively by filename pattern: entries produced through the
  `esm` SDK also carry no transformers tag, by design, and are still live.

### Changed — breaking

* `torch` and `transformers` are no longer core dependencies. They live in the new `[plm]`
  extra, which `[saprot]` implies. Every entry point into the PLM stack now raises an
  `ImportError` naming the extra rather than a bare `ModuleNotFoundError: No module named
  'torch'`.
* The embedding cache key carries the transformers `major.minor` for checkpoints loaded through
  transformers. Ankh3 tokenization changes at transformers 4.50, so the same protein and model
  under two versions are two different tensors and must not share a path. The DSSP cache is
  untouched by this and survives a transformers upgrade.
* Per-checkpoint transformers-version windows are enforced at load time, before any weight
  download. A checkpoint that cannot load, or that would load misaligned, now raises; one that
  loads but produces values incomparable with another version warns. This can refuse a
  checkpoint that previously appeared to load — deliberately: on transformers 5.x the Ankh
  tokenizers prepend an `<unk>` that `special_tokens_mask` does not flag, which shifts every
  residue's embedding by one position for `ankh` and `ankh_base`.
* `biopython>=1.80` is now required. `dssp_dict_from_pdb_file` gained the `dssp_version`
  argument in 1.80; on 1.79 `run_dssp` raises `TypeError` and every DSSP-backed field fails.

### Added

* `[plm]` extra — `torch>=2.0`, `transformers>=4.27,<5`, `sentencepiece`. `sentencepiece` is
  required, not optional: the T5/Ankh tokenizers behind ProstT5, `prott5_xl_half` and the Ankh3
  pair raise on load without it.
* `[esmc]` extra — the EvolutionaryScale `esm` SDK, for `esmc_600m`. It does not imply `[plm]`;
  that path never consults transformers. Note it resolves only on Python 3.10–3.12, since `esm`
  publishes no release for 3.9 or 3.13+.
* `docs/SETUP_NOTES.md` gains a per-checkpoint transformers constraint table and an
  environment-variable reference.
* `MANIFEST.in`, so the sdist ships `CITATION.cff`, `docs/`, `tests/TESTS.md` and
  `tests/__init__.py`.

### Changed

* `requires-python` is `>=3.9` — the `<3.13` ceiling is removed, making the package installable
  on 3.13 and 3.14. An upper bound here is a resolver gate rather than documentation: above it
  pip backtracks through the release history instead of reporting that the interpreter is too
  new, and it would make every CPython release require a foldenv release just to permit it. The
  tested range is stated by the classifiers and built by CI, which now covers 3.9 through 3.14.
* License metadata migrated to PEP 639 (`license = "CC-BY-NC-SA-4.0"` plus `license-files`),
  clearing two setuptools deprecations. The license itself is unchanged.

### Fixed

* `esmc_6b`'s transformers path is now reported honestly: no measured transformers release
  (4.44, 4.57.6, 5.15.1) registers the `esmc` model type, so it is refused with a pointer to
  `esmc_600m` rather than being documented as needing `transformers>=4.57`.
* Documentation claimed "ESM C is routed off MPS"; only `esmc_6b` is. `esmc_600m` runs on MPS.
* Documentation claimed Biopython auto-detects the mkdssp version on the path foldenv uses. It
  does not — `dssp_dict_from_pdb_file` only accepts the version, which is why `_detect_version`
  exists.

## [0.1.2] — 2026-08-22

* `foldenv.__version__` matches the packaging version.

## [0.1.1] — 2026-08-22

* Install from PyPI.

## [0.1.0] — 2026-08-02

* First tagged release: per-residue structural context from AlphaFold — relative solvent
  accessibility, secondary structure, contacts, pLDDT, and per-residue PLM embeddings.
  Tagged on GitHub only; 0.1.1 was the first release published to PyPI.

[0.2.0]: https://github.com/cchin29/foldenv/releases/tag/v0.2.0
[0.1.2]: https://github.com/cchin29/foldenv/releases/tag/v0.1.2
[0.1.1]: https://github.com/cchin29/foldenv/releases/tag/v0.1.1
[0.1.0]: https://github.com/cchin29/foldenv/releases/tag/v0.1.0
