# sklearn-tabpfn-ext

Shared, transform-only TabPFN preprocessing operators + JSON/NPZ codec,
extracted from vldm. See the ldmkit design spec.

## Release status

This package is currently pre-0.1.  A `v0.1.0` release should mean the
preprocessing implementation is complete for both vldm and ldmkit:

- vldm can hard-depend on this package at runtime;
- ldmkit can ingest TabPFN models through this package;
- old vldm-written artifacts, new ext-written artifacts, and ldmkit-written
  model repositories pass the compatibility gate.

The 2026-07-09 readiness pass completed those compatibility gates against the
pre-0.1 commit line. The package version remains below `0.1.0` until the
release tag is cut intentionally.

Current downstream pin while pre-0.1:

```text
sklearn-tabpfn-ext @ git+https://github.com/MonadKai/sklearn-tabpfn-ext.git@62202cb
```

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

For local cross-repository readiness, the real TabPFN lane can also be run from
the `ldmkit` environment with this checkout on `PYTHONPATH` to avoid repeated
torch/CUDA downloads.

## Migration: cross-impl parity (vldm ↔ sklearn-tabpfn-ext)

`tools/cross_impl_parity_check.py` verifies (torch-free) that artifacts are
loadable+transform-identical across both codecs, with on-disk op_ids staying
`vldm.preprocessing.*` — the Phase B backward-compatibility gate. Requires the
vldm source on the path:

```sh
VLDM_SRC=/path/to/vldm uv run --extra dev python tools/cross_impl_parity_check.py
```
