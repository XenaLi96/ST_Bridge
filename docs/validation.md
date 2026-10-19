# Validation record

Date: 2026-10-17. Release: `0.1.0rc1`.

## Environment and scope

CPU validation used Python 3.9.19, PyTorch 2.4.1, PyG 2.5.3, NumPy 1.26.4, SciPy 1.13.1, and scikit-learn 1.5.2. The HD import test also used pandas 2.0.3, scanpy 1.10.2, anndata 0.10.9, h5py 3.11.0, and pyarrow 17.0.0.

All test inputs were generated synthetic arrays or a generated miniature 10x-style HDF5/Parquet fixture. No real tissue data or trained manuscript weights were used. GPU execution and Python 3.10/3.11 were not tested. No full real-data training run or manuscript-table reproduction was performed.

## Automated checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Result: **9 tests passed**, final run 3.101 seconds.

- Train C2F with expression gating, cell-type prototypes, and quantile matching, then predict and evaluate.
- Obtain identical predictions from labeled and label-free input files.
- Align shuffled predictions by cell ID and obtain identical evaluation reports.
- Perturb only held-out cell-expression labels and obtain identical trained state dictionaries, including the cell-type/activation/quantile/variance/ranking path.
- Train the hierarchical model and convert a generated legacy checkpoint without changing predictions.
- Convert C2F checkpoints with optional node GNN, position head, expression head, and hard gate without changing predictions.
- Reject incompatible gene order, nonempty training outputs, and invalid configurations.
- Handle a single-cell group without local edges and preserve explicit cached positional features.
- Check metric identities, undefined constant-vector R², and zero MMD for identical large vectors after paired subsampling.
- Convert a generated legacy HD fixture, preserve H5 or explicit panel gene order, retain positional features, and include background expression in coarse sums. The legacy pickle import requires an explicit trust flag.

Some tests cover several checks within one workflow; the suite contains nine test methods.

## Wheel installation and CLI workflow

Built a wheel with:

```bash
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir ../BRIDGE-code-release-dist
```

Installed that wheel using `pip install --no-deps` into a separate temporary virtual environment with `--system-site-packages`. Existing scientific dependencies were reused; this was not a fresh download/resolution of all dependencies. Verified that `bridge_st` imported from the new environment's `site-packages`, rather than this source tree.

Ran `bash scripts/smoke_test.sh` from outside the source directory with an empty `PYTHONPATH`. All CLI stages passed: version, synthetic data generation, input validation, two training epochs, label-free prediction, and evaluation. The fixture contained 80 cells, 16 spots, six genes, and 384 image features. Evaluation selected 20 held-out cells. The training loss decreased from 4.2901013 to 3.5443656; these short synthetic results demonstrate execution only.

## Source and archive checks

- Verified all eight original-source hashes in `provenance/source-manifest.json`.
- Confirmed the two packaged model definitions and two reference training scripts are byte-identical to their recorded sources.
- Scanned the release source for personal server paths, known notification-secret identifiers, private-key headers, and API-key assignments; no matches were found. Research datasets, weights, caches, and environment files are excluded from the source ZIP.
- `scripts/make_release.py` validates ZIP integrity and checks each archived source file against the included `SHA256SUMS`; an external ZIP checksum is also generated.

Remaining release decisions are recorded in `RELEASE_NOTES.md` and `LICENSING.md`: authors must confirm the final manuscript code snapshot and software license. Real pretrained weights and dataset access information are not supplied by this validation.
