# foldenv — per-residue structural context from AlphaFold

[![PyPI](https://img.shields.io/pypi/v/foldenv.svg)](https://pypi.org/project/foldenv/)
[![Python](https://img.shields.io/pypi/pyversions/foldenv.svg)](https://pypi.org/project/foldenv/)
[![CI](https://github.com/cchin29/foldenv/actions/workflows/ci.yml/badge.svg)](https://github.com/cchin29/foldenv/actions/workflows/ci.yml)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**`foldenv`** computes a protein residue's **structural microenvironment** from its
AlphaFold-predicted fold: **relative solvent accessibility (RSA)**, **secondary structure**
(helix/strand/coil via DSSP), **spatial contacts / packing**, a **pLDDT** confidence flag,
and an optional **per-residue protein-language-model (PLM) embedding**. One call takes a
UniProt accession + residue position and returns a strict-JSON-safe dict — enough to tell a
**buried structural residue** apart from a **functional surface residue**, or to featurize a
mutation site for downstream ML.

```python
from foldenv import get_structural_context

get_structural_context("P04637", 175)   # TP53 R175
# → {"uniprot_id": "P04637", "position": 175, "wildtype_aa": "R",
#    "rsa": 0.022, "secondary_structure": "C", "contact_count": 14,
#    "nearest_contacts": [...], "embedding": [...1536-d...],
#    "embedding_model": "ankh", "plddt": 96.6}
```

The `embedding` field is the one part that needs a PLM extra (`[plm]`, or `[esmc]` for ESM C);
every other field comes from the base install. Set `embedding.model = "none"` to skip it.

Keywords: AlphaFold · relative solvent accessibility · RSA · secondary structure · DSSP ·
residue contacts · contact map · pLDDT · residue microenvironment · protein language model ·
per-residue embedding · structural bioinformatics · mutation effect featurization.

## Install

Requires **Python 3.9+**; built and tested through **3.14**.

```bash
pip install foldenv                 # structural path only — no torch, no transformers
pip install "foldenv[plm]"          # + torch, transformers, sentencepiece — PLM embeddings
pip install "foldenv[saprot]"       # + mini3di for SaProt's 3Di half (implies [plm])
pip install "foldenv[esmc]"         # + the EvolutionaryScale `esm` SDK — ESM C via `esmc_600m`
```

The base install is deliberately light — `numpy`, `requests`, `pyyaml`, `biopython` — and runs
the whole structural pipeline (RSA, secondary structure, contacts, pLDDT) with
`embedding.model = "none"`. Torch and transformers are multi-gigabyte wheels used only by the
PLM path, so they live in the `[plm]` extra: requesting any other `embedding.model` without it
raises an `ImportError` naming the extra rather than a bare `ModuleNotFoundError`.

`[esmc]` does not imply `[plm]`: the `esm` SDK path (`esmc_600m`) never consults transformers.
It does need torch, which `[esmc]` names directly. It is also the one extra with a narrower
Python window than the package — `esm` publishes no release for 3.9 or for 3.13+, and its
current releases require 3.12 exactly, so `[esmc]` resolves only on **3.10–3.12** (and only on
3.12 does it get the current SDK).

**External binary:** secondary structure + RSA need the **`mkdssp`** binary (DSSP v4).
- macOS: `brew tap brewsci/bio && brew install brewsci/bio/dssp`
- Linux (recommended): `conda install -c conda-forge -c bioconda dssp` — no sudo, no build
- Debian/Ubuntu: `apt-get install dssp` (provides `mkdssp`) — **verify it is v4**; apt has
  historically shipped 2.x/3.x, which works but needs `dssp.executable` set to the real name

Without it, the low-level contact and sequence helpers still work, but every documented entry
point — `get_structural_context`, `structural_profile`, `tool.invoke`, `analysis.summarize`,
`validation.crystal_crosscheck` — needs secondary structure and so fails. PLM weights
download on first use (Ankh-large ~2 GB).

## Usage

**Agent-tool wrapper (fast, interpretable).** `tool.invoke` omits the raw embedding by
default, so it returns quickly (no PLM download) and gives an LLM the interpretable fields:

```python
from foldenv import tool

tool.invoke({"uniprot_id": "P04637", "position": 175})            # dict or JSON string
tool.invoke({"uniprot_id": "P04637", "position": 175, "include_embedding": True})  # +1536-d vector
tool.tool_spec("anthropic")   # tool descriptor: {name, description, input_schema}  (also "openai"/"plain")
```

**Direct Python API.** `get_structural_context` computes the embedding by **default** (loads
the PLM, so it needs `[plm]`). Pass `embedding.model="none"` for the structural fields only —
that path runs on the base install:

```python
from foldenv import get_structural_context, config

cfg = config.load(overrides={"embedding": {"model": "none"}})   # skip the PLM forward pass
ctx = get_structural_context("P62593", 68, config=cfg)          # TEM-1 catalytic S70 (Ambler) = UniProt 68
ctx["rsa"], ctx["contact_count"], ctx["plddt"]                  # 0.052, 11, ...
```

**Other entry points:**

```python
from foldenv import structural_profile, analysis, validation

structural_profile("P62593")                     # {pos: {aa, rsa, ss3, contact_count, plddt}} for every residue
analysis.summarize("P04637")                      # functional-site structural signature
validation.crystal_crosscheck("P62593", "1BTL")   # AlphaFold-vs-experimental agreement report
```

Config is overridable per call via `config.load(overrides=...)`; all outputs are strict-JSON-safe.

## Returned fields

| Field | Meaning |
|---|---|
| `rsa` | relative solvent accessibility (Tien 2013 theoretical MaxASA) |
| `secondary_structure` | 3-state H/E/C from DSSP; G,I→H and B→E (Cuff & Barton Method A, as in MDTraj/MDAnalysis — see `SS8_TO_SS3`). SS-*prediction* benchmarks often send G,I,B→C instead, which reports less H and E |
| `contact_count` | number of residues in the Cα-8 Å neighborhood (pLDDT-masked) |
| `nearest_contacts` | list of `{resnum, aa, distance}` for the closest contacts |
| `embedding` / `embedding_model` | per-residue PLM vector + which model produced it (default Ankh-large, 1536-d) |
| `plddt` | AlphaFold per-residue confidence |

## Embedding models

`embedding.EMBEDDING_MODELS` registry — set `embedding.model` in `decisions.yaml` or via
`config.load(overrides=...)`:

| name | model | dim | notes |
|---|---|---|---|
| `ankh` | Ankh-large | 1536 | **default** |
| `ankh3_large` | Ankh3-large | 1536 | T5 encoder, `[NLU]` prefix |
| `ankh3_xl` | Ankh3-XL | 2560 | T5 encoder, `[NLU]` prefix |
| `prostt5_aa` | ProstT5 AA-mode | 1024 | |
| `saprot` | SaProt 650M | 1280 | structure-aware; 3Di from the AF backbone via **mini3di** (`[saprot]` extra) |
| `saprot_1.3b` | SaProt 1.3B | 1280 | structure-aware; deeper (not wider) than 650M |
| `esm2_3b` | ESM2-3B | 2560 | sequence-only |
| `esm2_650m` | ESM2-650M | 1280 | lighter sequence-only |
| `esmc_600m` | ESM C 600M | 1152 | via the `esm` SDK (`[esmc]` extra); transformers-independent |
| `esmc_6b` | ESM C 6B | 2560 | transformers path loads on no measured release — use `esmc_600m`; cluster/Forge target; never on Apple MPS |

**Transformers versions.** The `[plm]` extra pins `transformers>=4.27,<5`. The `>=4.27` floor is
inherited; the sequence models (Ankh, Ankh3×2, ProstT5, ESM2×2) are measured across 4.44 to
4.57.6. 5.x sets the ceiling: it fails to instantiate the prot_bert tokenizer, and the Ankh
tokenizers there prepend an `<unk>` that `special_tokens_mask` does not flag — which shifts
every residue's embedding by one for the plain Ankh checkpoints and changes the values (rows
stay correct) for the Ankh3 pair. (`protbert` and `ankh_base` are loader-only checkpoints,
reachable through `plm.load_pretrained_plm` but not selectable as `embedding.model`; the full
per-checkpoint table is in `docs/SETUP_NOTES.md` §5.)

Per-checkpoint requirements conflict inside that window — prot_bert and the unanchored Ankh
checkpoints need `<5`, while ESM C 6B's transformers path loads on no measured release at all —
so no single pin serves every caller and `foldenv.plm` checks each checkpoint's own window at
load time, before any weight download. A checkpoint that cannot load raises; one that loads
but produces values incomparable with another version warns. Ankh3 tokenization changes at
4.50: rows stay correctly aligned either way, but embeddings computed before 4.50 are not
numerically comparable with ones computed from 4.50 on, so the on-disk embedding cache is keyed
by transformers `major.minor` and the two cannot collide.

Device is picked automatically (MPS → CUDA → CPU); ESM C **6B** is routed off MPS
(`esmc_600m` runs on MPS fine).

## Configuration (decisions)

Defaults live in `foldenv/decisions.yaml` (overridable via `config.load(overrides=...)`):
Cα-8 Å contacts (Cβ-5 Å optional), mask pLDDT < 50, RSA via Tien 2013 theoretical MaxASA,
per-protein in-memory cache + optional L2 disk cache, Ankh-large embedding, mkdssp with
`--output-format=dssp`. The on-disk cache location is `./.foldenv_cache` by default,
overridable via `FOLDENV_CACHE_DIR` or the `cache.dir` config leaf.

## Tests

```bash
pip install -e ".[dev,saprot]"   # saprot pulls [plm]
pytest -q
```

Live tests self-skip when AlphaFold-DB is unreachable or `mkdssp` is absent; weight-loading
forward-pass tests are opt-in (`RUN_HEAVY_EMB=1`). See
[`tests/TESTS.md`](https://github.com/cchin29/foldenv/blob/main/tests/TESTS.md) for the full
inventory, and [`docs/SETUP_NOTES.md`](https://github.com/cchin29/foldenv/blob/main/docs/SETUP_NOTES.md)
for environment setup and the RSA/secondary-structure validation notes.

## Changelog

Release history and upgrade notes are in
[`CHANGELOG.md`](https://github.com/cchin29/foldenv/blob/main/CHANGELOG.md). **Upgrading from
0.1.x:** `torch`/`transformers` moved out of the core install into `[plm]`, and cached
embeddings on the transformers path are re-computed once — see the 0.2.0 entry.

## Citing

If `foldenv` contributes to work you publish, please cite it. GitHub's **Cite this repository**
button renders BibTeX and APA from [`CITATION.cff`](https://github.com/cchin29/foldenv/blob/main/CITATION.cff),
which also lists the upstream methods this package builds on (AlphaFold, DSSP, the MaxASA
reference tables, and the PLM checkpoints) — cite those alongside it where the relevant field
was used.

## Provenance & license

`foldenv` was developed inside a private research fork of
[MuLAN](https://github.com/GianLMB/mulan) and extracted here. Two helper modules
(`constants.py`, `plm.py`) adapt small routines from MuLAN — see [`NOTICE`](https://github.com/cchin29/foldenv/blob/main/NOTICE). Licensed under
**CC BY-NC-SA 4.0** (see [`LICENSE`](https://github.com/cchin29/foldenv/blob/main/LICENSE)),
the same license as MuLAN: free for attributed, non-commercial use; derivatives must share
alike.

## Acknowledgements

This work was carried out during a 2026 summer research internship at the Laboratoire de Biologie
Computationnelle, Quantitative et Synthétique — the Laboratory of Computational, Quantitative and
Synthetic Biology ([CQSB](https://lcqb.fr/), UMR 7238, CNRS–Sorbonne Université), Paris.

The internship was supported by a fellowship from the France-Stanford Center for Interdisciplinary
Studies, Stanford Global Studies Division, Stanford University.
