# sklearn-tabpfn-ext Context

This glossary captures project-specific language for the preprocessing package
shared by `ldmkit` and `vldm`.

## Language

**Preprocessing Semantics Library**:
The package that owns executable TabPFN preprocessing operators, transform behavior, codec behavior, operator registry, input sanitizer representation, and TabPFN-to-operator translation.
_Avoid_: Runtime engine, model repository

**Canonical Operator ID**:
The stable on-disk operator identifier written into preprocessing artifacts. It may remain `vldm.preprocessing.*` for compatibility even when the implementation owner is `sklearn-tabpfn-ext`.
_Avoid_: Python import path, implementation module

**Transform-Only Package**:
A package that represents and executes preprocessing transforms without owning model repository storage, artifact bake, serving, or deployment policy.
_Avoid_: Inference runtime, platform adapter
