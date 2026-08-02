# Setup notes and gotchas

Environment setup and the implementation gotchas worth knowing before running the suite.

---

## 1. Python & venv

- **Use Python 3.11.** The macOS system `python3` is **3.14**, which has **no torch/transformers
  wheels yet** — `python -m venv` silently succeeds, then `pip install` fails/half-installs.
  On this Mac, 3.11 is MacPorts' `/opt/local/bin/python3.11`. On Linux/cluster, use the
  module system or a 3.11 toolchain (`module load python/3.11`, pyenv, or conda `python=3.11`).
  - **Linux CPU box (2026-07-08):** system `python3` is **3.12.3** (no 3.11 present); 3.12 has
    torch/transformers wheels, so `.venv` was built on 3.12.3 with the same pinned
    versions (torch 2.12.1, transformers 4.44.2, biopython 1.87) — all 42 pure tests pass.

- **Dedicated venv per machine**, not shared with the plot/train envs:
  ```
  python3.11 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -e ".[dev,saprot]"
  ```
- `.venv/` is gitignored (added to `.gitignore`).

## 2. Dependencies (versions that work here)

Runtime dependencies beyond MuLAN's: **`pyyaml`**, **`biopython`** (mini3di was
already there). Verified-good versions in `.venv`:

| pkg | version | note |
|---|---|---|
| torch | 2.12.1 | CPU/MPS on Mac; on cluster install the CUDA build matching the node |
| transformers | 4.44.2 | **pinned `<4.45`** by requirements — see ESM C caveat below |
| biopython | 1.87 | **must be recent** — the DSSP v4 auto-handling landed in newer Biopython |
| mini3di | latest | pure-Python 3Di encoder (SaProt structure half); no native deps |
| numpy | 2.4.6 | |
| protobuf | *(optional)* | **not installed here.** Needed only for Ankh3's *fast* tokenizer — without it, Ankh3 falls back to the slow tokenizer (see gotcha in §7). ankh-large/ProstT5/ESM2/SaProt don't need it. |

## 3. mkdssp (M2 — DSSP) — the D6 gotcha, per platform

**The gotcha:** mkdssp **v4 defaults to mmCIF output**, which older Biopython DSSP parsers
can't read. **Resolution:** Biopython **1.87's** wrapper runs `mkdssp --version`, detects
≥4.0.0, and passes `--output-format=dssp` itself — so **v4 works as-is, no 3.x needed**, as
long as Biopython is recent and `mkdssp --version` prints a parseable version. Our `dssp.py`
also detects the version (`_detect_version`) for the direct `dssp_dict_from_pdb_file` path.

- **macOS:** `brew tap brewsci/bio && brew install brewsci/bio/dssp` → **mkdssp 4.6.1** at
  `/opt/homebrew/bin/mkdssp`. (homebrew-core dropped `dssp`; MacPorts `port` was broken after
  the OS upgrade — needs `sudo port migrate`.)
- **Linux CPU box / cluster (recommended):** bioconda —
  `conda install -c conda-forge -c bioconda dssp` (installs a v4 `mkdssp`). No sudo, no build.
- **Linux via apt (fallback, verify version!):** `apt-get install dssp` — historically ships
  an **old 2.x/3.x** named `dssp` (or `mkdssp`). If <4, our code still works (Biopython uses
  the legacy command form), but set `dssp.executable` in `decisions.yaml` to the actual binary
  name (`dssp` vs `mkdssp`).
- **Source (last resort):** github.com/PDB-REDO/dssp — needs cmake, C++17, libcifpp, boost.
  Involved; prefer bioconda on the cluster.
- **Config:** `decisions.yaml → dssp.executable` (default `mkdssp`) lets each machine point at
  its binary name. RSA normalization is **machine-independent**: we carry our own MaxASA tables
  (`dssp.MAX_ASA_TABLES`, verified to match Biopython's Wilke=Tien2013 / Sander exactly), so the
  D3 table choice doesn't depend on the DSSP build.

## 4. Compute / device (M4 — embeddings)

- `foldenv.plm.get_device()` prefers **MPS → CUDA → CPU**. On the Linux CPU box it returns CPU;
  on the cluster, CUDA. Override with env `FOLDENV_FORCE_CPU=1`.
  - **⚠️ Linux box (2026-07-08):** this box has an old **Quadro P400 (CUDA capability sm_61)**,
    which the CUDA-build **torch 2.12.1** does *not* support (it targets sm_75+). `get_device()`
    still picks CUDA, so the heavy embedding forward passes crash with a
    `CUDA error: no kernel image is available` on this GPU. **Run heavy/embedding work with
    `FOLDENV_FORCE_CPU=1`** here (`RUN_HEAVY_EMB=1 FOLDENV_FORCE_CPU=1 pytest …` → 68/68 pass). The
    pure/live tests are unaffected (no forward pass). Alternative would be a CPU-only torch wheel,
    but forcing CPU is the one-liner and what the env override exists for.
- `foldenv/__init__.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` (Mac-only effect; harmless on
  Linux/CUDA).
- **ESM C 6B never runs on MPS.** `embedding.EMBEDDING_MODELS["esmc_6b"].mps_ok=False` and
  `resolve_device` reroutes an MPS pick to CUDA (else CPU) with a warning. Intended targets:
  the Linux CPU box (bf16/large-RAM) or cluster GPUs. Set `embedding.device` explicitly there.

## 5. ⚠️ ESM C needs a *different* transformers than the rest

**ESM C 6B requires `transformers >= 4.57`, but this package's `pyproject.toml` pins
`transformers < 4.45`** (repo-wide constraint for the other PLMs). These **cannot coexist** in
one env. Keep a separate **`.venv-esmc`** (currently `transformers 5.12.1`)
for ESM C work. So:
- Everything except ESM C (Ankh, Ankh3×2, ProstT5, SaProt×2, ESM2×2) → run in `.venv`.
- **ESM C 6B → run in a `transformers>=4.57` env (e.g. `.venv-esmc`) on Linux/cluster**, not in
  `.venv`. Keep the two envs separate; don't bump the structctx pin.

> **2026-07-10 — ESM C loader is transformers-version-fragile (found running P3 locally).** The
> current `.venv-esmc` has **transformers 5.12.1**, which **no longer registers the `esmc`
> `model_type`** — so the tool's loader (`foldenv.plm.load_pretrained_plm` →
> `AutoModelForMaskedLM.from_pretrained`) fails with *"model type `esmc` … does not recognize this
> architecture"* (and `trust_remote_code=True` does not help). So ESM C 6B is **not runnable on this
> Mac**: the transformers path needs the narrow ~4.57 window where `esmc` was native, and the
> installed `esm` SDK (3.2.1) only exposes **300M/600M** locally (6B is Forge-API/cluster). The
> cluster/benchmark ESM C runs used a working loader there; locally, treat ESM C 6B as
> **cluster-only**. Practical fix if a local ESM C run is ever needed: pin a transformers version
> that registers `esmc`, or use the `esm` SDK with a locally-available ESM C size.
>
> **2026-07-11 — `esmc_600m` added as an SDK-backed local ESM C (the practical fix, implemented).**
> The registry now has **`esmc_600m`** (dim 1152, `mps_ok=True`, `sdk=True`), loaded through the
> `esm` SDK (`ESMC.from_pretrained("esmc_600m")`) rather than transformers — so it sidesteps the
> 5.12.1 `esmc` model_type gap entirely. It runs on **MPS in bf16 at ~1.4 s/protein** and passes the
> P3 row-alignment gate (TEM-1 L150A, ratio 27). `embedding.py` branches on the `sdk` flag: forward
> is `encode` → `logits(return_embeddings=True)`, strip BOS/EOS, cast bf16→f32, CPU return. **Env:
> `esmc_600m` needs `.venv-esmc` (the `esm` SDK isn't in `.venv`); it does *not* need
> `mini3di` (SaProt-only).** So local ESM C smoke-testing uses `esmc_600m`; `esmc_6b` stays
> cluster-only.

## 6. Network & weights (cluster compute nodes are often offline)

- **AlphaFold fetch** hits `alphafold.ebi.ac.uk` over HTTPS. Cluster compute nodes usually have
  **no outbound internet** → pre-fetch on a login node and populate the on-disk cache
  (`.foldenv_cache/alphafold/<ACC>.cif`), which the tool reuses by accession.
- **HF model weights** (Ankh ~2 GB, SaProt 650M, ESM C 6B large) download on first use. On the
  cluster, **pre-download on the login node** and set `HF_HOME`/`TRANSFORMERS_CACHE` to a shared
  path so compute nodes read from cache. Same for the Linux box if it's bandwidth-limited.
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
- **Ankh3 tokenizer needs `protobuf` for the fast tokenizer.** With protobuf, Ankh3 loads the
  fast tokenizer (emits a spurious `<unk>` after `[NLU]`); without it, the slow tokenizer (no
  `<unk>`). The per-residue strip is anchored on the residue count so it's aligned either way,
  but the two produce *numerically different* embeddings. To reproduce the sweep's Ankh3
  embeddings, install protobuf. (Full detail: the README "Embedding models" section and `foldenv/plm.py`; the recorded
  Ankh3 sweep results were verified valid — fast tokenizer path.)

## 8. Verify a fresh machine

```
.venv/bin/python -m pytest tests -q
```
**Linux box full-suite invocation** (mkdssp lives in a dedicated conda env; force CPU for the
old GPU — see §4):
```
PATH="$CONDA_PREFIX/envs/dssp/bin:$PATH" \
  RUN_HEAVY_EMB=1 FOLDENV_FORCE_CPU=1 \
  .venv/bin/python -m pytest tests -q      # → 68 passed
```
Live AlphaFold-fetch and mkdssp tests **self-skip** when offline / when `mkdssp` is absent, so a
green run on a fresh box confirms the pure logic; a fully-online box with mkdssp exercises M1+M2
end-to-end. Heavy embedding forward-pass test is opt-in: `RUN_HEAVY_EMB=1`.
