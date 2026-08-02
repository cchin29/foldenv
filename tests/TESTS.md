# `foldenv` — test suite

Reference for the tests under `tests/`. They cover `get_structural_context(uniprot_id,
position) -> dict` and its supporting modules, from AlphaFold fetch through the tool wrapper.

## Overview

**83 tests** across 12 files (`test_config`, `test_config_nondefaults`, `test_fetch`,
`test_dssp`, `test_contacts`, `test_embedding`, `test_context`, `test_validation`,
`test_analysis`, `test_tool`, `test_persist`, `test_context_persist_wiring`; plus an empty
`__init__.py`). Each test falls into one of three **categories**, which decide whether it
actually runs on a given machine:

- **pure / offline** — no network, no external binary, no weight downloads. Exercise logic,
  schemas, registries, table math, device routing, and deterministic synthetic geometry.
  These **always run** (a few `pytest.importorskip("Bio"/"mini3di")` only if a dep is absent).
- **live** — hit AlphaFold-DB / RCSB and/or shell out to `mkdssp`. They **self-skip** when the
  network or binary is missing, via the guard pattern:
  ```python
  def _online():
      try: requests.get(api_base + "/P62593", timeout=15); return True
      except Exception: return False
  live = pytest.mark.skipif(not _online() or shutil.which("mkdssp") is None, reason=...)
  ```
  (`test_validation` also pings `files.rcsb.org` for `1BTL`.) Live tests use TEM-1 (`P62593`)
  and TP53 (`P04637`) as fixtures.
- **heavy** — real PLM forward passes that download multi-GB weights. **Opt-in only** via
  `RUN_HEAVY_EMB=1`, using:
  ```python
  heavy = pytest.mark.skipif(os.environ.get("RUN_HEAVY_EMB") != "1", reason="set RUN_HEAVY_EMB=1")
  ```
  A heavy test may also be `@live` (needs weights *and* network/mkdssp).

**Current counts** (this machine, network + `mkdssp` present, no `RUN_HEAVY_EMB`):
`76 passed, 6 skipped`. The 6 skips are exactly the heavy tests (4 in `test_embedding`, 1 in
`test_context`, 1 in `test_tool`). With **`RUN_HEAVY_EMB=1`** and network + `mkdssp` available,
all **83** run. Offline / without `mkdssp`, the live tests self-skip instead. (The 14
persistence tests — `test_persist` + `test_context_persist_wiring` — are all pure/offline, so
they always run.)

## Summary table

| File | Covers (milestone / module) | Tests | Categories |
|---|---|---:|---|
| `test_config.py` | config loader + `decisions.yaml` (D1–D6) | 5 | pure |
| `test_config_nondefaults.py` | non-default D2/D3/D4 choices through the assembler | 3 | live (3) |
| `test_fetch.py` | M1 — AlphaFold-DB fetch/parse + isoform selector | 7 | pure (4), live (3) |
| `test_dssp.py` | M2 — SS8→SS3 map, MaxASA/RSA tables, mkdssp run | 6 | pure (5), live (1) |
| `test_contacts.py` | M3 — Cα/Cβ KD-tree contacts, pLDDT mask | 11 | pure (9), live (2) |
| `test_embedding.py` | PLM registry, SA-string, 3Di, device routing, forward passes | 15 | pure (11), heavy (4) |
| `test_context.py` | M5 — assembler / public API dict | 7 | live (6), heavy+live (1) |
| `test_validation.py` | M6 — crystal cross-check (AF vs experimental) | 2 | live (2) |
| `test_analysis.py` | M7 — Tier-2 functional-site analysis | 6 | live (6) |
| `test_tool.py` | M8 — tool wrapper, JSON-Schema, validation | 7 | pure (4), live (2), heavy (1) |
| `test_persist.py` | §4 — L2 disk cache (`persist.py`) in isolation | 10 | pure (10) |
| `test_context_persist_wiring.py` | §4 — `context.py` read-through wiring | 4 | pure (4) |

## Per-file detail

### `test_config.py` (5, pure)
- `test_defaults_load` — `config.load()` yields the locked D1–D6 defaults (Cα primary, 8.0/5.0 Å cutoffs, pLDDT mask 50, Tien-2013 table, `ankh`, device `auto`, dssp output-format).
- `test_cache_dir_resolved` — YAML `null` cache dir resolves to a concrete path ending in `.foldenv_cache`.
- `test_overrides_deep_merge` — an override deep-merges: overridden key changes, sibling and untouched sections survive.
- `test_overrides_do_not_mutate_file_defaults` — overrides don't mutate the shared file defaults (a fresh `load()` still reads `ca`).
- `test_decisions_path_exists` — `config.decisions_path()` points at a real file.

### `test_config_nondefaults.py` (3, live)
Covers non-default `decisions.yaml` choices end-to-end (the defaults are exercised elsewhere).
- `test_nondefault_rsa_table_changes_rsa_and_cache_is_table_keyed` — a non-default MaxASA table (`sander_rost1994`) changes the assembled `rsa` vs the default (`tien2013_theoretical`), same SS; the DSSP cache is keyed by table so both coexist (distinct cached objects).
- `test_in_memory_false_is_uncached_but_correct` — with `cache.in_memory=false`, `get_structure` returns a fresh parse each call and the module caches stay empty, yet the assembled dict is still correct (`contact_count==13`) — guards the self-exclusion-across-structure-instances fix.
- `test_nondefault_plddt_mask_through_config` — a mask threshold above the pLDDT range (200) masks every partner → 0 contacts; the default (50) leaves a normal shell.

### `test_fetch.py` (7 — pure 4, live 3)
- `test_select_model_entry_single` (pure) — single entry is selected and `own` has 1 element.
- `test_select_model_entry_filters_isoforms` (pure) — **isoform selector**: TP53-shaped list (canonical + `-2..-9` isoforms) picks the canonical and keeps `own` at 1, so no spurious multi-fragment warning.
- `test_select_model_entry_true_fragments` (pure) — genuine length-fragments sharing the accession (F1/F2) keep both in `own`.
- `test_select_model_entry_warns_on_fallback` (pure) — no exact accession match warns ("falling back") and returns the fallback entry.
- `test_fetch_tem1_parses` (live) — fetch TEM-1: accession + cached `.cif` path, >100 residues, B-factor/pLDDT column in 0–100.
- `test_fetch_is_cached` (live) — second fetch reuses the cached file (mtime unchanged).
- `test_unknown_accession_raises` (live) — a nonexistent accession raises `NoAlphaFoldModelError`.

### `test_dssp.py` (6 — pure 5, live 1)
- `test_ss8_to_ss3_mapping` (pure) — DSSP 8-state → 3-state map (H/G/I→H, E/B→E, T/S/-/P/space→C).
- `test_max_asa_tables_present_and_complete` (pure) — every MaxASA table covers all 20 standard residues exactly.
- `test_max_asa_lookup_and_unknown` (pure) — `max_asa("A", tien2013_theoretical)==129.0`; non-standard `X` → `None`.
- `test_tables_differ_between_references` (pure) — Sander/Rost and Tien tables give different values for the same residue.
- `test_run_dssp_rejects_unknown_table` (pure) — an unknown table name raises `ValueError`.
- `test_dssp_on_tem1` (live) — real `mkdssp` on TEM-1: 286 records keyed 1..286, ss3 ∈ {H,E,C}, rsa ∈ [0,1], and the **disulfide-cysteine regression** — DSSP lowercases disulfide-bonded Cys (a/b/c…); `run_dssp` must renormalize them to canonical uppercase `C` (TEM-1 has a Cys–Cys bridge), so all aa are canonical and at least one `C` remains.

### `test_contacts.py` (11 — pure 9, live 2)
Synthetic deterministic scenes: Cα atoms on the x-axis at 0/4/7/12/16 Å.
- `test_ca_contact_count_and_ordering` (pure) — from x=0, exactly the two neighbors within 8 Å, ordered ascending by distance; reports `atom_mode=CA`, `cutoff=8.0`.
- `test_self_excluded` (pure) — the query residue is never its own contact.
- `test_n_nearest_truncates` (pure) — full `contact_count` counts the whole shell; `nearest_contacts` is truncated to `n_nearest`.
- `test_plddt_mask_drops_low_confidence_partner` (pure) — a partner with pLDDT<50 is masked out of the shell.
- `test_glycine_cb_fallback_to_ca` (pure) — a Gly (no CB) in cb-mode falls back to CA instead of crashing.
- `test_prebuilt_index_matches_oneoff` (pure) — a prebuilt contact index gives identical results to per-call construction.
- `test_index_self_excluded` (pure) — the shared KD-tree still excludes the query's own atom (by residue id, so it holds even when the index was built from a different structure instance, e.g. `cache.in_memory=false`).
- `test_missing_residue_raises` (pure) — an absent residue number raises `KeyError`.
- `test_bad_mode_raises` (pure) — an invalid mode raises `ValueError`.
- `test_contacts_on_tem1_invariants` (live) — on TEM-1 pos 150: non-empty, nearest are ascending, within cutoff, self-excluded, ≤`n_nearest`, and a sane upper bound (<40) for an 8 Å shell.
- `test_contacts_symmetric_tem1` (live) — Tier-1 invariant: with the pLDDT mask off, contacts are geometrically symmetric (150→p implies p→150).

### `test_embedding.py` (14 — pure 10, heavy 4)
- `test_registry_has_expected_models` (pure) — `EMBEDDING_MODELS` is exactly the 9 expected names with the right dims; structure-aware set is exactly the SaProt family.
- `test_ankh_is_default` (pure) — config default embedding model is `ankh`.
- `test_saprot_is_structure_aware_and_others_not` (pure) — SaProt `structure_aware=True`; ankh/prostt5/esmc `False`.
- `test_esmc_flagged_mps_incompatible` (pure) — `esmc_6b.mps_ok=False`, `ankh.mps_ok=True`.
- `test_unknown_model_raises` (pure) — `get_model_spec` on an unknown name raises `ValueError`.
- `test_sa_sequence_interleaves_and_masks_noncanonical` (pure) — SaProt SA string interleaves AA(upper)+3Di(lower); non-canonical `X` → `#` AA-half; 2 chars/residue.
- `test_esmc_reroutes_off_mps` (pure) — with MPS mocked and no CUDA, resolving `esmc_6b` warns and reroutes to CPU.
- `test_mps_ok_model_keeps_mps` (pure) — an mps-ok model (`ankh`) keeps MPS.
- `test_explicit_device_honored` (pure) — an explicit `device="cpu"` is honored.
- `test_structure_to_3di_length_matches_seq` (pure) — mini3di on a tiny synthetic structure yields `len==seq_len`, all lowercase.
- `test_ankh_forward_shape` (heavy) — Ankh forward pass returns `(len(seq), 1536)`.
- `test_embedding_alignment_mutation_sensitive` (heavy) — **row↔residue alignment / off-by-one**: a point mutation at pos 150 changes the embedding *most* at row 150 (`argmax+1==p`) and dominates (`>5×` median), proving `per_res[pos-1]` indexes the right residue.
- `test_ankh3_length_alignment_regression` (heavy) — **ankh3 regression**: `embed_protein("ankh3_large")` returns exactly `len(seq)` rows (was L-1 before the `[NLU]`-prefix-strip fix), and the mutation still localizes to its row.
- `test_saprot_structure_aware_forward_and_alignment` (heavy) — SaProt path (3Di from AF backbone → SA tokens → forward): shape `(L,1280)`, finite, and a mutation (flips the SA AA-half) changes row 150 the most.

### `test_context.py` (7 — live 6, heavy+live 1)
Assembly runs with `embedding.model: none` so no weights download (except the heavy test).
- `test_sequence_from_structure_tem1` (live) — sequence from structure is 286 aa over the canonical alphabet ∪ `X`.
- `test_assemble_full_dict_tem1` (live) — full dict: ids/position, wildtype identity agrees across sequence and DSSP, ss ∈ {H,E,C}, rsa ∈ [0,1], `contact_count==13`, ≤5 nearest contacts with the right keys, pLDDT 0–100, embedding fields `None`, and **strict JSON** (`json.dumps(..., allow_nan=False)` — no NaN/Inf leaks).
- `test_gap_position_degrades_gracefully` (live) — **graceful degradation**: a position present in the sequence but absent from the structure (an `X` gap-fill) nulls the structural fields (`rsa`/`ss`/`plddt`=None, `contact_count`=0, `nearest_contacts`=[]) instead of raising, and stays strict-JSON.
- `test_gap_position_nulls_embedding` (live) — **C1**: at a gap-fill position the `embedding`/`embedding_model` degrade to null *with* the other structural fields (a stubbed non-null embedder proves it isn't just the `none`-model path).
- `test_position_out_of_range_raises` (live) — position 10000 and 0 both raise `IndexError` (1-based).
- `test_nearest_contacts_consistent_with_count` (live) — nearest distances are sorted and `len(nearest) <= contact_count`.
- `test_assemble_with_real_embedding` (heavy+live) — default cfg (Ankh) end-to-end: `embedding_model=="ankh"`, embedding is a 1536-list, and a second position is served from the cached forward pass.

### `test_validation.py` (2 — live 2)
M6 crystal cross-check, AF `P62593` vs experimental `1BTL` (needs AlphaFold-DB + RCSB + mkdssp).
- `test_crystal_crosscheck_tem1` — chain `A`, `n_compared>200` (~263 ordered), and the **cross-check thresholds**: `ss3_agreement>0.85` (observed 0.996), `rsa_pearson>0.85` (observed 0.984), `rsa_mae<0.10` (observed 0.027) — margin under calibration to catch a broken pipeline / numbering.
- `test_crosscheck_numbering_alignment_not_identity` — the mapping must reconcile Ambler↔UniProt numbering by alignment, not assume `exp_resnum==af_pos` (else agreement collapses); re-asserts the ss3/rsa thresholds.

### `test_analysis.py` (6 — live 6)
M7 Tier-2 functional-site signatures on TEM-1 and TP53.
- `test_structural_profile_shape_and_consistency` — per-residue profile has 286 entries; a sampled `contact_count` matches a direct `get_contacts` call; rsa/ss3 fields well-formed.
- `test_tem1_catalytic_all_buried` — catalytic sites map to {68,71,164} with identities S/K/E (confirms Ambler S70/K73/E166 → UniProt), all buried, rsa<0.15 (observed ~0.05).
- `test_numbering_mismatch_raises` — a wrong expected identity at a real position raises `ValueError("expected")` (guards a bad offset).
- `test_out_of_range_site_raises` — an out-of-range site raises `ValueError("out of range")`.
- `test_tp53_hotspots_split_structural_vs_contact` — R175 buried/packed (rsa<0.10, contact percentile>0.9) vs R248 exposed DNA-contact (not buried, rsa>0.4), distinguishable on the burial axis (`r248.rsa > r175.rsa + 0.3`).
- `test_summarize_fields` — `summarize("P62593")` reports `n_sites==3`, `all_buried True`, mean contact percentile ∈ [0,1].

### `test_tool.py` (7 — pure/offline 4, live 2, heavy 1)
- `test_spec_styles` (pure) — `tool_spec` for `anthropic`/`openai`/`plain` shapes; unknown style raises `ValueError`.
- `test_committed_spec_in_sync` (pure) — **spec drift-guard**: committed `tool_spec.json` must equal `tool.tool_spec("plain")`.
- `test_invoke_validation_offline` (pure) — **argument validation**: missing args → `ValueError`; non-int / bool position, non-str id, stringy bool `include_embedding` → `TypeError`; position 0 and a malformed accession → `ValueError` (all fail fast before any network).
- `test_input_schema_required_fields` (pure) — `INPUT_SCHEMA` requires `{uniprot_id, position}` and forbids additional properties.
- `test_invoke_default_omits_embedding` (live) — default invoke returns structural fields, omits embedding (null, no forward pass), int `contact_count`, strict JSON.
- `test_invoke_accepts_json_string` (live) — a JSON-string argument is accepted (pos 1 → `wildtype_aa=="M"`).
- `test_invoke_with_embedding` (heavy) — `include_embedding=True` returns `embedding_model=="ankh"` and a 1536-list.

### `test_persist.py` (10, pure)
L2 disk-persistence layer (`persist.py`) round-tripped in isolation — no network, mkdssp, or
PLM weights (DSSP records and a small `torch` tensor are built by hand).
- `test_dssp_roundtrip` — `save_dssp`→`load_dssp` reconstructs every `ResidueDSSP` field; a non-standard residue's NaN `rsa` is restored as NaN (not None/0).
- `test_dssp_written_json_is_strict` — the file contains no invalid `NaN` token, parses under a strict JSON parser, and the NaN residue's `rsa` is stored as `null`.
- `test_dssp_table_keys_are_distinct` — different MaxASA tables map to different files; the other table stays a miss (no cross-table collision).
- `test_embedding_roundtrip` — `save_embedding`→`load_embedding` reproduces a `[286,1536]` tensor exactly.
- `test_embedding_model_keys_are_distinct` — an embedding saved under one model isn't served for another.
- `test_seqonly_embedding_path_has_no_structure_fingerprint` — sequence-only models (`ankh`) are keyed on accession+model only (no `__s` segment), so a structure refresh doesn't needlessly invalidate them.
- `test_structure_aware_embedding_keyed_on_structure` — SaProt's path carries a mmCIF fingerprint (`__s…`); changing the structure file (size/mtime) changes the key → miss, not a stale-coordinate hit.
- `test_persist_disabled_is_noop` — `cache.persist=false` writes nothing and loads short-circuit to `None` (no `dssp/`/`embeddings/` dirs created).
- `test_corrupt_files_are_a_miss_not_an_error` — a truncated JSON / non-torch `.pt` is treated as a miss (recompute), never an exception.
- `test_format_version_bump_invalidates` — bumping `_DSSP_FORMAT` changes the filename so an old-format file is never read (clean schema invalidation).

### `test_context_persist_wiring.py` (4, pure)
Verifies `context.py` consults the disk cache *before* the expensive path — the costly
callables (`run_dssp`, `load_embedder`, `embed_protein`, `get_structure`) are monkeypatched to
raise, so a disk hit is proven by their never being reached. No network / mkdssp / PLM.
- `test_get_dssp_uses_disk_hit` — after `save_dssp` + `clear_cache`, `get_dssp` returns the cached result with `run_dssp` **and** `get_structure` poisoned; NaN `rsa` restored.
- `test_protein_embedding_uses_disk_hit` — a pre-seeded tensor is returned with the embedder (`load_embedder`/`embed_protein`) and `get_structure` poisoned (`get_sequence` stubbed to the matching length).
- `test_protein_embedding_length_guard_rejects_stale_tensor` — a disk tensor whose row count ≠ sequence length raises `ValueError` (the guard covers disk-loaded tensors, not just fresh ones).
- `test_persist_false_ignores_disk` — with `persist=false`, both `get_dssp` and `_protein_embedding` fall through to the (poisoned) recompute path despite files being present on disk.

## How to run

Use a Python 3.11 virtualenv (`pip install -e ".[dev,saprot]"`).

```bash
# default suite — pure tests always run; live tests self-skip if AF-DB/RCSB/mkdssp missing
.venv/bin/python -m pytest tests -q

# heavy suite — also downloads PLM weights and runs real forward passes
RUN_HEAVY_EMB=1 .venv/bin/python -m pytest tests -q

# a single file / a single test
.venv/bin/python -m pytest tests/test_tool.py -q
.venv/bin/python -m pytest \
  tests/test_dssp.py::test_dssp_on_tem1 -q
```

- **Live** tests need network (AlphaFold-DB; `test_validation` also RCSB `1BTL`) and the
  **`mkdssp`** binary (macOS: `brew install brewsci/bio/dssp` → v4.6.1). Without them they skip.
- **Heavy** tests download weights on first use — Ankh ~2 GB (`test_ankh_forward_shape`,
  the alignment/mutation tests, the context/tool end-to-end); Ankh3-large and SaProt-650M are
  additional multi-GB downloads. Weights are HuggingFace-cached, so re-runs reuse them; AF
  structures are disk-cached under the config cache dir.

## Covered vs. not

- **ESM C (`esmc_6b`)** is only exercised at the registry/device-routing level (spec flags,
  the off-MPS reroute); there is **no forward pass** — it needs `transformers≥4.57` and a
  separate env / cluster GPU (this venv pins `transformers<4.45`).
- **`esm2_650m` / `esm2_3b` / `prostt5_aa` / `ankh3_xl` / `saprot_1.3b`** are covered by the
  registry/dim assertions but have **no dedicated forward-pass test here** (the heavy forward
  tests use `ankh`, `ankh3_large`, and `saprot`). Nothing is pre-cached; weights download on
  first heavy run.
- The **M7 biological tests** (`test_analysis.py`) are **descriptive** — they encode observed
  TEM-1/TP53 findings with margin, not a general-purpose classifier; the structural signal is
  complementary to conservation, not standalone.
- Binding the tool into any particular agent framework is out of scope here; only the
  framework-neutral wrapper (`invoke` / `tool_spec` / schema drift-guard) is tested.
