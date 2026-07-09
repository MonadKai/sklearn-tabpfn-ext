# Changelog

## Unreleased

- Lower the supported Python floor to `>=3.10`, matching the vldm integration
  target.
- Mark the package as pre-0.1.  The `v0.1.0` release is reserved for complete
  vldm and ldmkit preprocessing support.
- Expose the TabPFN translator contract through `sklearn_tabpfn_ext.tabpfn` so
  downstream callers do not import private translator modules.
- Extend cross-implementation parity coverage to `InputSanitizer` sidecars.
- Record the 2026-07-09 downstream readiness pass: vldm facade/runtime gates,
  ldmkit ingest -> vldm validate/bake/validate gate, old artifact smoke, new
  ext-written artifact gate, and real TabPFN translator lane all passed against
  the pre-0.1 commit line.
