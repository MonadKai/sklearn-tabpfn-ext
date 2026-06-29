# sklearn-tabpfn-ext

Shared, transform-only TabPFN preprocessing operators + JSON/NPZ codec,
extracted from vldm. See the ldmkit design spec.

## Tests

```sh
uv run --extra dev python -m pytest -q -m "not tabpfn"   # core (sklearn/numpy/pydantic only)
uv run --extra dev --extra tabpfn python -m pytest -q -m tabpfn   # translator lane (needs torch)
```

## Migration: cross-impl parity (vldm ↔ sklearn-tabpfn-ext)

`tools/cross_impl_parity_check.py` verifies (torch-free) that artifacts are
loadable+transform-identical across both codecs, with on-disk op_ids staying
`vldm.preprocessing.*` — the Phase B backward-compatibility gate. Requires the
vldm source on the path:

```sh
VLDM_SRC=/path/to/vldm uv run --extra dev python tools/cross_impl_parity_check.py
```
