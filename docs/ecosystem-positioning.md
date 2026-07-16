# LDM Ecosystem Positioning

This document records where `sklearn-tabpfn-ext` sits in the LDM ecosystem.
The guiding rule is: preprocessing semantics belong in one reusable library,
not separately inside ingest, serving, and platform projects.

Documentation consistency audit: [`docs/documentation-consistency-audit.md`](documentation-consistency-audit.md).

## Repository Roles

| Repository | Primary role |
| --- | --- |
| `sklearn-tabpfn-ext` | TabPFN preprocessing semantics library. |
| `ldmkit` | LDM source repository and ingest toolkit. |
| `vldm` | executable artifact compiler and inference runtime. |
| `vldm-sigma-server` | SIGMA wire adapter for vldm. |
| `sigma-perf` | black-box SIGMA protocol verification and performance harness. |
| `ldm-platform` | deployment control plane. |

## sklearn-tabpfn-ext Boundary

`sklearn-tabpfn-ext` owns:

- sklearn-compatible preprocessing transform operators.
- JSON/NPZ codec for preprocessing artifacts.
- Operator registry and stable on-disk operator semantics.
- Input sanitizer representation.
- TabPFN-to-operator translation used by ingest and conversion flows.

It does not own:

- LDM repository metadata or validation profiles; those belong to `ldmkit`.
- `.vldm` bake, runtime loading, capacity analysis, or inference; those belong to `vldm`.
- SIGMA protocol mapping; that belongs to `vldm-sigma-server`.
- Deployment gate policy, rollout, retention, or Kubernetes orchestration; those belong to `ldm-platform`.

The artifact identity namespace can remain `vldm.preprocessing.*` for backward
compatibility even when the implementation owner is this package.

## Capability Handoff

`ldmkit` and `vldm` should depend on this package for preprocessing behavior.
They should not fork transform semantics or infer ownership from Python module
names written into old artifacts.
