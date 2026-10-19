# Third-party components

The release imports PyTorch, PyTorch Geometric, NumPy, SciPy, and scikit-learn. The optional HD importer additionally uses pandas, Scanpy, AnnData, h5py, and PyArrow. These are dependencies installed separately, not vendored source. Their respective distributions retain their original attribution and licenses.

CellViT is an upstream segmentation/feature-extraction tool discussed in the manuscript. Its source, pretrained weights, images, and derived feature caches are **not included** here. Obtain it from [the original CellViT project](https://github.com/TIO-IKIM/CellViT), consult [its license](https://github.com/TIO-IKIM/CellViT/blob/main/LICENSE), and cite its publication when used. Record the exact extractor version for each trained model.

No iStar, THItoGene, Hist2ST, or BLEEP source/weights are bundled. No patient images, real expression matrices, trained manuscript weights, private API keys, or notification configurations are part of this archive.

Recovered project-owned model definitions are identified in `provenance/source-manifest.json`; their contributor ownership and final license must be confirmed by the authors as explained in `LICENSING.md`.
