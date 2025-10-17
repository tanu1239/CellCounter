# tests/test_utils.py
import numpy as np
import pytest
from anndata import read_h5ad
from cellcounter.utils import _read_img_adata, _read_img_path, _read_large_tif_img_path

DATA_DIR = "tests/data"


def test_read_img_from_adata_success():
    """Test reading a spatial image from an AnnData object."""
    # Load the .h5ad that *does* contain a spatial image under .uns["spatial"]
    adata = read_h5ad(f"{DATA_DIR}/adata_img.h5ad")
    img = _read_img_adata(adata, img_key="A1")
    assert isinstance(img, np.ndarray)
    assert img.size > 0


def test_read_img_from_adata_missing_key():
    # Even if .uns["spatial"] exists, requesting a non-existent img_key raises KeyError
    adata = read_h5ad(f"{DATA_DIR}/adata_img.h5ad")
    with pytest.raises(KeyError):
        _read_img_adata(adata, img_key="this_key_does_not_exist")


def test_read_img_path_success():
    """Test reading an image from a file path."""
    img = _read_img_path(f"{DATA_DIR}/img.png")
    assert isinstance(img, np.ndarray)
    assert img.size > 0


def test_read_large_tif_img_invalid_path():
    """Test that a FileNotFoundError is raised when provided with an invalid path"""
    invalid_path = f"{DATA_DIR}/invalid_image.tif"
    with pytest.raises(FileNotFoundError):
        _read_large_tif_img_path(invalid_path)


def test_read_large_tif_img_path_success():
    """Test reading an image from a file path"""
    img = _read_large_tif_img_path(f"{DATA_DIR}/A2_10_patch.tiff")
    assert isinstance(img, np.ndarray)
    assert img.size > 0


test_read_img_from_adata_success()
test_read_img_from_adata_missing_key()
test_read_img_path_success()
test_read_large_tif_img_invalid_path()
test_read_large_tif_img_path_success()
