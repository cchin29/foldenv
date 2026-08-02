# foldenv — per-residue structural context from AlphaFold

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

Keywords: AlphaFold · relative solvent accessibility · RSA · secondary structure · DSSP ·
residue contacts · contact map · pLDDT · residue microenvironment · protein language model ·
per-residue embedding · structural bioinformatics · mutation effect featurization.

## Install

```bash
pip install foldenv                # core: structure fetch/parse, RSA, SS, contacts, embeddings
pip install "foldenv[saprot]"      # + mini3di, for SaProt's structure-aware 3Di embeddings
```

**External binary:** secondary structure + RSA need the **`mkdssp`** binary (DSSP v4).
- macOS: `brew tap brewsci/bio && brew install brewsci/bio/dssp`
- Debian/Ubuntu: `apt-get install dssp` (provides `mkdssp`)

Everything except DSSP runs without it; `mkdssp`-dependent fields raise a clear error if it's
missing. PLM weights download on first use (Ankh-large ~2 GB).

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
the PLM). Pass `embedding.model="none"` for the structural fields only:

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
| `secondary_structure` | 3-state H/E/C from DSSP |
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
| `esmc_600m` | ESM C 600M | 1152 | via the `esm` SDK |
| `esmc_6b` | ESM C 6B | 2560 | best PLM; needs `transformers>=4.57` (separate env); never on Apple MPS |

The sequence models (Ankh, Ankh3×2, ProstT5, ESM2×2) are tested on `transformers` 4.27–4.45.
Device is picked automatically (MPS → CUDA → CPU); ESM C is routed off MPS.

## Configuration (decisions)

Defaults live in `foldenv/decisions.yaml` (overridable via `config.load(overrides=...)`):
Cα-8 Å contacts (Cβ-5 Å optional), mask pLDDT < 50, RSA via Tien 2013 theoretical MaxASA,
per-protein in-memory cache + optional L2 disk cache, Ankh-large embedding, mkdssp with
`--output-format=dssp`. The on-disk cache location is `./.foldenv_cache` by default,
overridable via `FOLDENV_CACHE_DIR` or the `cache.dir` config leaf.

## Tests

```bash
pip install -e ".[dev,saprot]"
pytest -q
```

Live tests self-skip when AlphaFold-DB is unreachable or `mkdssp` is absent; weight-loading
forward-pass tests are opt-in (`RUN_HEAVY_EMB=1`). See
[`tests/TESTS.md`](https://github.com/cchin29/foldenv/blob/main/tests/TESTS.md) for the full
inventory, and [`docs/SETUP_NOTES.md`](https://github.com/cchin29/foldenv/blob/main/docs/SETUP_NOTES.md)
for environment setup and the RSA/secondary-structure validation notes.

## Provenance & license

`foldenv` was developed inside the [MuLAN](https://github.com/GianLMB/mulan) codebase and
extracted here. Two helper modules (`constants.py`, `plm.py`) adapt small routines from
MuLAN — see [`NOTICE`](https://github.com/cchin29/foldenv/blob/main/NOTICE). Licensed under
**CC BY-NC-SA 4.0** (see [`LICENSE`](https://github.com/cchin29/foldenv/blob/main/LICENSE)),
the same license as MuLAN: free for attributed, non-commercial use; derivatives must share
alike.
