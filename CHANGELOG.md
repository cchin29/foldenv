# Changelog

Notable changes to `foldenv`. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [semantic](https://semver.org/), with the 0.x caveat that a minor bump may carry
breaking changes.

## [0.3.0] — 2026-09-08

### Added


- `SiteStat.to_dict()` and `CrosscheckReport.to_dict()`. The README promised that all outputs
  were strict-JSON-safe; `analysis` and `validation` return dataclasses, which `json.dumps`
  refuses outright, and whose NaN fields it would otherwise write as a bare `NaN` token that is
  not valid JSON. The dataclasses are unchanged — this adds a serialisation path rather than
  replacing them — and the README now says which entry points return what.

- `notebooks/` — the evidence behind the defaults in `decisions.yaml`, as two runnable
  notebooks. `contact_and_rsa_decision_sweep.ipynb` sweeps the contact primary (D1), the pLDDT
  mask (D2) and the MaxASA table (D3) against functional-site and physical-sanity criteria;
  `skempi_interface_validation.ipynb` tests the contact cutoff at scale against SKEMPI 2.0's
  interface labels and measures what a monomer-only view misses at an interface. They import
  the installed package, ship no data, and are included in the sdist. Every cell executes: the
  DSSP-dependent sections are live, not recorded.
- Tests for the absent-DSSP path, which had none — six launch-failure modes, plus a check that
  `_detect_version` reached directly gives the same error `run_dssp` surfaces.
- `CITATION.cff` is now parsed by the test suite, not only read line-wise, and a duplicate key in
  any mapping fails. `date-released` is also checked against the changelog entry for the declared
  version rather than only for ISO shape, which is what a bump that edits three version strings
  and leaves the fourth behind actually looks like.
- Two entries to the `references:` in `CITATION.cff`: **SKEMPI 2.0**, which is not a dependency of
  the package but is what the shipped `skempi_interface_validation.ipynb` rests entirely on, and
  **Cuff & Barton 1999**, the source of the DSSP 8-state to 3-state reduction in `foldenv.dssp`.

### Security

- Cached DSSP records are validated on read: `ss3`, `ss8`, a single-character `aa`, and `rsa`
  within [0, 1]. These reach an LLM caller through `tool.invoke`, and `OUTPUT_SCHEMA` promises
  that range — so a cache file that violates it is recomputed rather than served.
- The `pypa/gh-action-pypi-publish` action is pinned to a commit rather than the `release/v1`
  branch. That branch's tip moves, and it runs while the job holds `id-token: write` for the
  PyPI environment.
- The embedding cache refuses any `.pt` that is not in the zip container `torch.save` writes.
  `weights_only=True` does not restrict torch's legacy `.tar` path on torch ≤ 2.5
  (CVE-2025-32434), so for exactly the file shape this package never emits, that flag stopped
  being a defence. Checking the container closes it on every torch version rather than only for
  callers who upgrade — which is why no torch floor is pinned for it. A legacy-format or
  malformed cache file now reads as a miss and is recomputed.
- Size bounds on what reaches the disk, where there were none: a 256 MiB ceiling per structure
  download, and a 512 MB expansion ceiling plus unconditional `0644` permissions on the notebooks'
  archive extraction. HTTP requests already carried a 30 s timeout; the `mkdssp --version` probe
  did not, and now takes one along with a closed stdin — it runs before every uncached `run_dssp`,
  so a binary that hung there blocked the caller with no error. This bounds the probe only: the
  DSSP run itself goes through Biopython, which accepts no timeout.
- `permissions:` blocks on both workflows. A workflow without one inherits the repository default,
  which on an older repository is still read-and-write; CI now declares `contents: read`, and the
  publish job declares the `contents: read` that naming `id-token: write` had implicitly removed.

### Fixed

- `analysis.summarize`'s own `mean_rsa` and `mean_contact_percentile` return `null` rather than
  `NaN` in the degenerate cases (no site with an RSA; an empty ranking). Converting the `sites`
  with `.to_dict()` — the recipe the README gives — was not sufficient on its own, so following
  the documentation could still produce invalid JSON.

- **A missing `mkdssp` now raises an error that says what to install.** It previously surfaced
  as a bare `FileNotFoundError` from inside Biopython's call stack, preceded by a warning that
  the version banner could not be parsed — which sent the reader after a banner that was never
  printed, with advice (upgrade, or set `dssp.executable`) that does not apply to a binary that
  is not there. DSSP is an external program, so it is the one missing dependency `pip` cannot
  supply, which makes the error text the whole remedy: it now names the binary, gives the
  per-platform install commands, points at the `dssp.executable` config leaf, and names which
  entry points stop working and which keep working. The launch-failure check catches `OSError`
  rather than naming subclasses: `FileNotFoundError` and `PermissionError` are the common two, but
  a path leading *through* a regular file raises `NotADirectoryError` and a binary built for
  another architecture raises plain `OSError`, and both of those otherwise reached exactly the
  spurious-banner failure this entry describes. The list of affected entry points now includes
  `get_dssp`, `tool.call` and `analysis.functional_site_stats`, and the message no longer claims
  v4 is required — `_detect_version` returns the parsed version precisely so Biopython can drive
  a 3.x binary with the legacy flag.
- `tool.invoke` validates the accession with `fullmatch` rather than `match`. `$` matches before a
  trailing newline, so `"P62593\n"` passed validation and reached a network fetch.

### Changed — breaking

The first two items change behaviour a caller can depend on; the rest of this section is
documentation and defaults.

- **Structure identifiers are validated where the cache path is built.** `fetch_structure`,
  `fetch_experimental_structure` and both cache-path builders now reject anything that is not
  alphanumerics plus an optional `-N` isoform suffix. Every public entry point sits behind one of
  those, so an identifier that previously reached a filesystem path — `../../elsewhere` among
  them — now raises `ValueError`. Real UniProt accessions and PDB ids are unaffected.
- **`tool.invoke` is stricter about arguments.** Unknown keys now raise `ValueError` — the
  published schema has always said `additionalProperties: false`, but the dispatcher did not
  enforce it, so a caller passing extra metadata alongside the schema keys will now see an error
  where it previously saw silence. The accession pattern is also bounded (`{1,12}` plus an optional
  `-[0-9]{1,3}` isoform suffix) in both `INPUT_SCHEMA` and the shipped `tool_spec.json`; every real
  UniProt accession still validates, but an over-long id is refused up front instead of reaching a
  filesystem path.
- **The reason recorded for the default embedding model (D5) has been withdrawn.** It previously
  cited a ΔΔG benchmark comparison; that comparison used a split that leaks between train and test,
  so it could not support a choice between encoders. The default is unchanged — Ankh-large,
  inherited from MuLAN — but the package no longer claims benchmark evidence for it, and does not
  rank encoders at all.

- The documented Python window for the `[esmc]` extra was wrong. It said `esm` publishes no
  release for 3.13+ and that current releases require 3.12 exactly, so the extra resolved only
  on 3.10–3.12. `esm` 3.4.0 requires `>=3.12` with **no upper bound**, so 3.13 and 3.14 now
  resolve to it — on interpreters this package has not measured the SDK against. The README and
  the `pyproject.toml` comment and `docs/SETUP_NOTES.md` now state the actual per-version
  behaviour and date it, since it is a window another project controls and will move again.

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
  publishes no release for 3.9 or 3.13+. *(Superseded: `esm` 3.4.0, released after this version,
  dropped the upper bound. See the `[0.3.0]` entry above for the current behaviour.)*
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

[Unreleased]: https://github.com/cchin29/foldenv/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/cchin29/foldenv/releases/tag/v0.3.0
[0.2.0]: https://github.com/cchin29/foldenv/releases/tag/v0.2.0
[0.1.2]: https://github.com/cchin29/foldenv/releases/tag/v0.1.2
[0.1.1]: https://github.com/cchin29/foldenv/releases/tag/v0.1.1
[0.1.0]: https://github.com/cchin29/foldenv/releases/tag/v0.1.0
