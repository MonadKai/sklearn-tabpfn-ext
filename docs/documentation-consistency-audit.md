# Documentation Consistency Audit

Date: 2026-07-17

Scope: LDM ecosystem positioning and capability ownership across `vldm`,
`ldmkit`, `sklearn-tabpfn-ext`, `vldm-sigma-server`, `sigma-perf`, and
`ldm-platform`.

Reference boundary: [`docs/ecosystem-positioning.md`](ecosystem-positioning.md).

## Round 1

Reviewed README, CHANGELOG, CONTEXT, and ecosystem positioning docs for wording
that could assign repository storage, vldm bake/runtime, SIGMA protocol, or
deployment policy to this package.

Findings:

- The existing README already described this package as shared transform-only
  preprocessing operators plus codec.
- No conflicting design docs or ADRs exist in this repository.

Fixes:

- No conflict-specific fixes were needed beyond the newly added ecosystem
  positioning and CONTEXT glossary.

## Round 2

Re-scanned for `runtime`, `repository`, `deployment`, `serving`, `policy`,
`vldm`, and `ldmkit`.

Result:

- Remaining references are aligned: this package owns preprocessing semantics
  and codec behavior only.
