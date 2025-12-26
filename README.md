# CellCounter

CellCounter is a package that infers the number of cells in spatial spots from Visium spatial transcriptomic histology images. CellCounter is built on top of a StarDist and OpenCV-based computer vision model and incorporates additional normalization and processing steps. 

Current features grouped by file include:
- **utils.py** - Includes functions for reading in histology image files from an AnnData file, image path, or tif file returning a numpy array.
- **extraction.py** - Includes various functions to crop an input image in the form of a numpy array. The user must specify coordinates corresponding to what portion of the input image they are interested in and a crop of the input image corresponding to those coordinates will be returned in the form of another numpy array. This is used to run the pipeline on specific portions of histology images.
- **preprocessing.py** - Includes various functions that accept images in the form of numpy arrays and apply transformations to them, returning the resulting image also in the form of a numpy array. Currently supported image transformations include applying Contrast Limited Adaptive Histogram Equalization (CLAHE), unsharp masking, laplacian filtering, histogram equalization, and normalization.
- **stardist_model.py** - Includes the main function for applying the StarDist and OpenCV-based computer vision model to an input image in the form of a numpy array. It will infer the number of cells within the image and provide information such as area, coordinates, intensity, bounding box and more for each cell prediction.
- **analysis.py** - Includes various functions for evaluating the performance of the model. Contains functions to determine the predicted number of cells per spatial spot in the histology image by integrating the predictions received from the model and spot coordinates from the histology image (which are known when histology slices are prepared). Also, includes functions to determine annotated number of cells per spatial spot in histology image by taking in json file containing manual annotation coordinates and integrating with spot coordinates from histology images. Then can compare the correspondance between predicted cell coordinates and annotated coordinates by Intersection over Union (IoU) measurements and analyzing whether the predicted number of cells in each spot and annotated number of cells (ground truth) correspond to each other.

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

