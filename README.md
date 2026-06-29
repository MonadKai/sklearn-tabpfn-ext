# sklearn-tabpfn-ext

Shared, transform-only TabPFN preprocessing operators + JSON/NPZ codec,
extracted from vldm. See the ldmkit design spec.

## License & provenance

Apache-2.0 (see [LICENSE](LICENSE)). This library was extracted and adapted from
[vldm](https://github.com/MonadKai/vldm)'s `vldm/preprocessing/` subsystem and
`scripts/tabpfn_translator.py` (also Apache-2.0) — moved verbatim except for
import-namespace rewrites and a codec change that keeps on-disk op_ids as
`vldm.preprocessing.*` for interoperability. See [NOTICE](NOTICE) for attribution.

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
