# Contributing

Issues and pull requests are welcome — bug reports, reproducible failure cases, and
documentation corrections especially.

## Licensing of contributions

`foldenv` is licensed **CC BY-NC-SA 4.0**, not an OSI open-source license, so the terms for
inbound contributions are worth stating rather than leaving to convention: by opening a pull
request you agree that your contribution is licensed under CC BY-NC-SA 4.0, on the same terms
as the rest of the project. The ShareAlike clause is inherited from the routines adapted from
MuLAN and cannot be relaxed — see [`NOTICE`](NOTICE).

## Development setup

`docs/SETUP_NOTES.md` covers environments, the `mkdssp` dependency, and the per-checkpoint
`transformers` constraints. `tests/TESTS.md` describes what the suite covers and what skips
where. In short:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,saprot]"    # [saprot] implies [plm]
.venv/bin/python -m pytest -q
```

Live tests self-skip without network or `mkdssp`; weight-downloading tests are opt-in with
`RUN_HEAVY_EMB=1`. A change that alters stored artifacts — the DSSP tables or the embedding
tokenization — must bump the relevant `_FORMAT` in `foldenv/persist.py`, or stale cache entries
will be served silently.

## Style

Match the surrounding code: 100-column soft limit, double quotes, and comments that say *why*
rather than *what*. There is no enforced formatter; please do not reformat unrelated code in a
pull request, as it buries the actual change.
