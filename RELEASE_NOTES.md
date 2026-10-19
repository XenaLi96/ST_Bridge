# 0.1.0rc1 — 2026-10-17

- Portable `bridge-st` CLI for synthetic examples, data validation, supervised training, label-independent prediction, evaluation, legacy HD import, and tensor-checkpoint conversion.
- Original hierarchical and C2F model definitions retained with source hashes and reference training scripts.
- Configurations for baseline C2F, the later cell-type/activation/quantile recipe, a hierarchical example, and a fast synthetic example.
- Explicit training/held-out spot isolation; no test-expression calibration in prediction.
- Numeric/string NPZ interfaces, ordered gene checks, strict model loading, and separate per-gene/pooled evaluation definitions.
- CPU end-to-end tests and HD-format/checkpoint compatibility tests; see `docs/validation.md` for actual results.

Remaining author decisions: confirm the final manuscript source version, select the source license, and decide which real trained weights and datasets can accompany a public repository. Full real-data manuscript reproduction was not performed as part of packaging. The archive has not been pushed, published on PyPI, or assigned a DOI.
