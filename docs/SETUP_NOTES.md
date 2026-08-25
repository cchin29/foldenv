# Setup notes and gotchas

Environment setup and the implementation gotchas worth knowing before running the suite.

---

## 1. Python & venv

- **`requires-python` is `>=3.9` with no upper bound**, and the tested range is 3.9 through
  3.14 — every one of those built by the CI matrix, with the full suite green on 3.14 under
  torch 2.13 and transformers 4.57.6. The floor is enforced; the ceiling is deliberately not,
  because a `requires-python` cap is a resolver gate rather than documentation: above it pip
  backtracks through the release history instead of saying the interpreter is too new. A
  newer Python than the matrix covers is therefore untested, not blocked.
- **The practical constraint is wheel availability for the interpreter, not foldenv itself.** A
  brand-new `python3` can outrun the PLM stack's wheels, and the failure lands in the wrong
  place: `python -m venv` succeeds on such an interpreter, and only the later `pip install`
  fails or half-installs. Checking wheel availability before building the venv is cheaper than
  diagnosing a partial install.
  - This only bites environments that install `[plm]`/`[saprot]`. The base install is pure
    Python plus `numpy`/`biopython`, which have wheels far ahead of the PLM stack.
  - **`[esmc]` is the one extra narrower than the package.** `esm` publishes nothing for 3.9 or
    3.13+, and its current releases require 3.12 exactly, so `[esmc]` resolves only on
    3.10–3.12. On 3.10/3.11 pip silently backtracks to the older `esm 3.2.1.post1`.
- **Obtaining a specific interpreter** when the system one does not suit: pyenv
  (`pyenv install 3.12`), conda (`conda create -n foldenv python=3.12`), a distribution build
  (Homebrew or MacPorts `python@3.12`), or a cluster module system (`module load python/3.12`).
  Any of the four is fine — the venv only needs the interpreter, not a particular provenance.

- **A dedicated venv**, not one shared with unrelated projects — the PLM extras pin torch and
  transformers ranges that a sibling project may well contradict:
  ```
  python3.11 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -e ".[dev,saprot]"   # [saprot] pulls [plm] → torch + transformers
  ```
  A bare `pip install -e .` installs the structural stack only; the PLM tests then need
  `[plm]` (or `[saprot]`, which implies it). See §2.
- `.venv/` is gitignored (added to `.gitignore`).

## 2. Dependencies

The core install is the structural stack only — `numpy`, `requests`, `pyyaml`, `biopython`.
Everything the PLM path needs sits in extras (`[plm]`, `[saprot]`, `[esmc]`), so an env that
only computes RSA/SS/contacts/pLDDT never pulls torch or transformers.

The table records one known-good combination, not a set of pins. `pyproject.toml` constrains
only `biopython>=1.80`, `torch>=2.0` and `transformers>=4.27,<5`; every other version below is
simply what the combination was exercised at — useful for reproducing a result or bisecting a
regression.

| pkg | install | version | note |
|---|---|---|---|
| numpy | core | 2.4.6 | |
| requests | core | latest | AlphaFold-DB / RCSB fetch |
| pyyaml | core | latest | `decisions.yaml` loader |
| biopython | core | 1.87 | floor **`>=1.80`**, where the DSSP v4 auto-handling landed — see §3. Below it, `run_dssp` is a `TypeError` |
| torch | `[plm]`, `[esmc]` | 2.12.1 | `>=2.0`; CPU/MPS on Mac; on a CUDA node install the build matching it |
| transformers | `[plm]` | 4.44.2 | pinned **`>=4.27,<5`**; measured 4.44–4.57.6, `>=4.27` floor inherited — see §5 |
| sentencepiece | `[plm]` | latest | required, not optional: the T5/Ankh tokenizers behind ProstT5, prot_t5_xl_half and the Ankh3 pair raise on load without it |
| mini3di | `[saprot]` | latest | pure-Python 3Di encoder (SaProt structure half); no native deps |
| esm | `[esmc]` | 3.2.1.post1 | EvolutionaryScale SDK; the ESM C path that bypasses transformers |
| protobuf | *(optional)* | — | Not pulled in by any extra. Needed only for Ankh3's *fast* tokenizer — without it, Ankh3 falls back to the slow tokenizer (see gotcha in §7). ankh-large/ProstT5/ESM2/SaProt don't need it. |

Per-checkpoint transformers requirements are **not** expressible as one install pin (ESM C 6B
documents `>=4.57`, prot_bert needs `<5`), so `foldenv.plm` checks each checkpoint's own window at
load time and the pin stays wide. §5 has the table.

## 3. mkdssp (M2 — DSSP) — the D6 gotcha, per platform

**The gotcha:** mkdssp **v4 defaults to mmCIF output**, which older Biopython DSSP parsers
can't read. **Resolution:** from Biopython **1.80**, `dssp_dict_from_pdb_file` takes a
`dssp_version` argument and passes `--output-format=dssp` itself when told the executable is
≥4.0.0 — so **v4 works as-is, no 3.x needed**. An unparseable `mkdssp --version` banner is
assumed to be v4 (with a `RuntimeWarning`), so the case that actually misfires is a real 3.x
whose banner does not parse.

The `dssp_version` parameter is why `pyproject.toml` floors biopython at `>=1.80`: on 1.79 the call
raises `TypeError: dssp_dict_from_pdb_file() got an unexpected keyword argument 'dssp_version'`,
taking secondary structure and RSA down with it.

Detecting the version is the *caller's* job on this path: Biopython's `DSSP` class shells out to
`mkdssp --version` itself, but the lower-level `dssp_dict_from_pdb_file` that `foldenv/dssp.py`
uses only accepts the answer. `_detect_version` is where foldenv produces it.

- **macOS:** `brew tap brewsci/bio && brew install brewsci/bio/dssp` → a v4 `mkdssp` (4.6.1 at
  the time of writing) under the Homebrew prefix. The tap is necessary because homebrew-core
  dropped `dssp`.
- **Linux / cluster (recommended):** bioconda —
  `conda install -c conda-forge -c bioconda dssp` (installs a v4 `mkdssp`). No sudo, no build.
- **Linux via apt (fallback, verify version!):** `apt-get install dssp` — historically ships
  an **old 2.x/3.x** named `dssp` (or `mkdssp`). If <4, foldenv still works (Biopython uses
  the legacy command form), but set `dssp.executable` in `decisions.yaml` to the actual binary
  name (`dssp` vs `mkdssp`).
- **Source (last resort):** github.com/PDB-REDO/dssp — needs cmake, C++17, libcifpp, boost.
  Involved; prefer bioconda on the cluster.
- **Config:** `decisions.yaml → dssp.executable` (default `mkdssp`) lets each machine point at
  its binary name. RSA normalization is **machine-independent**: foldenv carries its own
  MaxASA tables (`dssp.MAX_ASA_TABLES`, verified to match Biopython's Wilke=Tien2013 / Sander
  exactly), so the D3 table choice doesn't depend on the DSSP build.

## 4. Compute / device (M4 — embeddings)

- `foldenv.plm.get_device()` prefers **MPS → CUDA → CPU**: MPS on Apple Silicon, CUDA on a GPU
  node, CPU where neither is present. Override with env `FOLDENV_FORCE_CPU=1`.
  - **⚠️ A GPU older than the torch build supports still gets picked.** `get_device()` tests for
    CUDA *availability*, not for whether the installed torch has kernels for that card's compute
    capability — so on an older GPU it selects CUDA and the embedding forward passes then crash
    with `CUDA error: no kernel image is available`. Recent torch builds target **sm_75+**, which
    excludes Pascal-era cards and below (a Quadro P400 is sm_61, and torch 2.12.1 fails on it
    exactly this way). **Run heavy/embedding work with `FOLDENV_FORCE_CPU=1`** on such a machine;
    the pure/live tests are unaffected, since they run no forward pass. A CPU-only torch wheel
    would also work, but forcing CPU is the one-liner the env override exists for.
- `foldenv/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (Mac-only effect; harmless on
  Linux/CUDA).
- **ESM C 6B never runs on MPS.** `embedding.EMBEDDING_MODELS["esmc_6b"].mps_ok=False` and
  `resolve_device` reroutes an MPS pick to CUDA (else CPU) with a warning. Intended targets: a
  large-RAM CPU host (bf16) or CUDA GPUs. Set `embedding.device` explicitly there.

## 5. Transformers version window and the per-checkpoint constraints

The `[plm]` extra pins **`transformers>=4.27,<5`**. The `>=4.27` floor is inherited; the
measured range is 4.44 through 4.57.6, plus 5.15.1 as the counter-example that sets the ceiling.
One window holds every checkpoint that loads at all, so the sequence models need no version
split. `esmc_6b` is the exception, and not for the reason the floor suggests: the `esmc`
`model_type` is registered by none of 4.44, 4.57.6 or 5.15.1, so its transformers path does not
load on any measured release. `esmc_600m` through the `esm` SDK (`[esmc]`) is the ESM C path
that works, and it does not consult transformers at all.

Requirements that *are* version-specific are enforced at load time by
`foldenv.plm.check_transformers_version`, before any weight download, rather than by the pin.

The table is keyed by `constants.PLM_ENCODERS` name (13 checkpoints the transformers loader
can build). `embedding.EMBEDDING_MODELS` exposes nine of those as `embedding.model`, sometimes
under a different name (`prostt5` here is `prostt5_aa` there), and adds `esmc_600m`, which is
not a `PLM_ENCODERS` entry at all — the `esm` SDK loads it under its own name. That leaves four
loader-only checkpoints, reachable through `plm.load_pretrained_plm` but not selectable as
`embedding.model`: `ankh_base`, `esm_35M`, `protbert`, `prott5_xl_half`. A † marks the three
of them that appear below; `esm_35M` has no version constraint and so has no row:

| checkpoint | window | severity | what goes wrong outside it |
|---|---|---|---|
| `protbert`† | `<5.0` | raise | the `Rostlab/prot_bert` tokenizer does not instantiate on 5.x |
| `esmc_6b` | *(none loads)* | raise | upstream documents a `>=4.57` floor, but the `esmc` `model_type` is registered by none of 4.44, 4.57.6 or 5.15.1 — so the entry is unconditional rather than a floor, and points at `esmc_600m` instead of a version to install |
| `ankh`, `ankh_base`† | `<5.0` | raise | 5.x prepends an `<unk>` that `special_tokens_mask` does not flag; these two have no residue-count anchor, so every residue shifts by one |
| `ankh3_large`, `ankh3_xl` | `<5.0` | warn | same stray `<unk>`, but the Ankh3 prefix path re-anchors on the residue count and drops it — rows stay correct, values differ |
| `ankh3_large`, `ankh3_xl` | `>=4.50` | warn | Ankh3 tokenization changed at 4.50; rows stay aligned either way, but embeddings from before 4.50 are not numerically comparable with ones from 4.50 on |

Raise vs warn splits on whether the rows still land on the right residues. A hard violation
means the checkpoint cannot load, or loads misaligned, so raising is the only honest outcome.
Alignment is checkpoint-specific: `embed_sequence` anchors on the residue count only for the
Ankh3 prefix path, and every other checkpoint drops non-residue rows through
`special_tokens_mask` alone — which is why a leading token the mask does not flag is a hard
violation rather than a soft one. A soft violation means the rows are correct and only the
*values* differ from another version. That is comparability, not correctness: one dataset
embedded end-to-end under one version is fine. The on-disk embedding cache key carries the
transformers `major.minor` (`persist.py`), so tensors from two versions cannot collide on the
same path.

`prostt5` and `prott5_xl_half`† carry no entry: they load through `T5Tokenizer` on every
version measured with `sentencepiece` and `protobuf` both present. Without `protobuf` — which
the `[plm]` extra does not install — they fail on 5.x for an unrelated reason (`tiktoken`),
which is a property of that environment rather than of the checkpoint, so the pin covers it
instead of the table.

**ESM C in practice.** Two paths reach ESM C, and only one goes through transformers:
- **`esmc_600m` — the `esm` SDK path** (`[esmc]` extra, `ESMC.from_pretrained("esmc_600m")`,
  registry flags `dim=1152`, `mps_ok=True`, `sdk=True`). It bypasses transformers entirely, so
  no version constraint applies and no cache-key version tag is added. It runs on **MPS in
  bf16 at ~1.4 s/protein** and passes the row-alignment gate (TEM-1 L150A, ratio 27).
  `embedding.py` branches on the `sdk` flag: forward is `encode` →
  `logits(return_embeddings=True)`, strip BOS/EOS, cast bf16→f32, CPU return. It needs the
  `esm` SDK but **not** `mini3di` (SaProt-only). This is the path for local ESM C smoke tests.
- **`esmc_6b` — the transformers path.** The `>=4.57` floor upstream documents has not been
  observed sufficient: on transformers **4.57.6** the load still fails inside transformers with
  *"model type `esmc` … does not recognize this architecture"* (and `trust_remote_code=True`
  does not help), which is why the guard entry is unconditional rather than a floor. The SDK
  exposes only 300M/600M for local use — 6B is a Forge-API or cluster target. Use `esmc_600m`
  anywhere else.

## 6. Network & weights (cluster compute nodes are often offline)

- **AlphaFold fetch** hits `alphafold.ebi.ac.uk` over HTTPS. Cluster compute nodes usually have
  **no outbound internet** → pre-fetch on a login node and populate the on-disk cache
  (`.foldenv_cache/alphafold/<ACC>.cif`), which the tool reuses by accession.
- **HF model weights** (Ankh ~2 GB, SaProt 650M, ESM C 6B large) download on first use. On the
  cluster, **pre-download on the login node** and set `HF_HOME`/`TRANSFORMERS_CACHE` to a shared
  path so compute nodes read from cache. Worth doing on any bandwidth-limited host.
- API detail: unknown/malformed accessions return **HTTP 400** (not just 404) — both map to
  `NoAlphaFoldModelError` in `fetch.py`.

## 7. Implementation gotchas (non-environment)

- **Residue numbering:** AF models use **UniProt canonical** numbering. Literature/PDB sites are
  often in other schemes — e.g. **TEM-1 catalytic S70/K73/E166 are Ambler-numbered**, ≠ UniProt
  (P62593 has a 23-aa signal peptide + Ambler insertions; not a linear offset). Convert via
  alignment/SIFTS before indexing structural output (bites M6/M7 validation).
- **pLDDT is in the B-factor column** of AF mmCIF (0–100). Used for the D2 mask (<50) and the
  returned `plddt`.
- **Glycine has no Cβ** → contact code (M3) must fall back to Cα (config
  `contacts.glycine_cb_fallback`).
- **Ankh3 tokenization has two independent switches.** *(a) protobuf:* with it, Ankh3 loads the
  fast tokenizer; without it, the slow one. *(b) transformers version:* below 4.50 the fast
  tokenizer emits a spurious `<unk>` after `[NLU]`, and from 4.50 on it does not (§5). The
  per-residue strip is anchored on the residue count so rows are aligned in every combination,
  but the four combinations produce *numerically different* embeddings. Reproducing an existing
  set of Ankh3 embeddings therefore means matching **both** switches, not just the transformers
  version — so record which combination produced them. The embedding cache path carries the
  transformers `major.minor`, so two versions' tensors never share a file; protobuf's presence
  is *not* in the key, so changing it within one transformers version needs a manual cache
  clear. (Full detail: the README "Embedding models" section and `foldenv/plm.py`.)

## 8. Verify a fresh machine

```
.venv/bin/python -m pytest tests -q
```
A base (no-extras) install runs the structural and packaging tests and skips the PLM ones; the
PLM tests need `[plm]`/`[saprot]`.

**Full-suite invocation** when `mkdssp` lives in a separate conda env and the GPU has to be
bypassed (§4):
```
PATH="$CONDA_PREFIX/envs/dssp/bin:$PATH" \
  RUN_HEAVY_EMB=1 FOLDENV_FORCE_CPU=1 \
  .venv/bin/python -m pytest tests -q
```
Live AlphaFold-fetch and mkdssp tests **self-skip** when offline / when `mkdssp` is absent, so a
green run on a fresh machine confirms the pure logic; a fully-online machine with mkdssp
exercises M1+M2 end-to-end. Heavy embedding forward-pass tests are opt-in: `RUN_HEAVY_EMB=1`.

## 9. Environment variables

| variable | read in | effect |
|---|---|---|
| `FOLDENV_CACHE_DIR` | `config.py` | supplies the on-disk cache root **when `cache.dir` is null** — an explicit `cache.dir` wins over it (default `./.foldenv_cache`) |
| `FOLDENV_FORCE_CPU` | `plm.py` | `=1` makes `get_device()` return CPU, bypassing MPS/CUDA selection |
| `FOLDENV_ALPHAFOLD_API_BASE` | `config.py` | overrides the AlphaFold-DB endpoint (note the opposite precedence to `FOLDENV_CACHE_DIR`: this one is applied unless `overrides["alphafold"]` is passed). Pointing it at an unreachable host is how the test suite forces the network-gated tests to self-skip |
| `ANKH3_PREFIX` | `plm.py` | the Ankh3 task prefix, default `[NLU]`; `[S2S]` selects the seq2seq prefix instead. Changing it changes the embeddings, and it is **not** part of the cache key — clear the cache when changing it |
| `RUN_HEAVY_EMB` | tests | `=1` opts the weight-downloading forward-pass tests in |
| `PYTORCH_ENABLE_MPS_FALLBACK` | *set*, not read, by `__init__.py` | `setdefault("1")` so an op with no MPS kernel falls back to CPU instead of raising; macOS-only effect |
