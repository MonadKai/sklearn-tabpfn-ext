# Changelog

## v0.1.0 - 2026-07-09

- Lower the supported Python floor to `>=3.10`, matching the vldm integration
  target.
- Release the first complete vldm + ldmkit preprocessing support boundary.
- Expose the TabPFN translator contract through `sklearn_tabpfn_ext.tabpfn` so
  downstream callers do not import private translator modules.
- Extend cross-implementation parity coverage to `InputSanitizer` sidecars.
- Record the 2026-07-09 downstream readiness pass: vldm facade/runtime gates,
  ldmkit ingest -> vldm validate/bake/validate gate, old artifact smoke, new
  ext-written artifact gate, and real TabPFN translator lane all passed against
  the restored `v0.1.0` release line.
- Record the 2026-07-10 readiness refresh: core/default gates passed across
  `sklearn-tabpfn-ext`, `ldmkit`, and `vldm`; `ldmkit` continues to pin
  `sklearn-tabpfn-ext@v0.1.0` for TabPFN `6.3.2`, `6.4.0`, and `6.4.1`.
