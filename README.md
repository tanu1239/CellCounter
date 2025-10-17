# CellCounter

CellCounter is a package that infers the number of cells in spatial spots from Visium spatial images. CellCounter is built on top of stardist and incorporates additional normalization and processing steps. More features to be included. 

## Run the code
Download the code              
Run `pip install .`      



We use pytest for automated testing. To install development dependencies (pytest and black):

`pip install .[dev]`

All tests, including data fixtures, live under the tests/ directory:

```
tests/
├── data
│   ├── adata_img.h5ad
│   └── adata_no_img.h5ad
├── test_extraction_patches.py
└── test_utils.py
```

Run the full test suite with:

`pytest --maxfail=1 --disable-warnings -q`

You can customize pytest behavior via the pytest.ini file in the project root.

