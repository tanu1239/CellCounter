# tests/test_extraction_patches.py
import numpy as np
import pytest
from skimage import color
from cellcounter.utils import _read_img_path
from cellcounter.extraction import (
    extract_patches,
    crop_image_cartesian,
    crop_image_screen,
    crop_image,
)


def test_non_array_input():
    """Test that a TypeError is raised when the input is not a NumPy array."""
    with pytest.raises(TypeError):
        extract_patches("not an array")


def test_invalid_dimensionality():
    """Test that a ValueError is raised when the input array has an invalid number of dimensions."""
    arr = np.zeros((2, 2, 2, 2))
    with pytest.raises(ValueError):
        extract_patches(arr)


def test_exact_patch_size():
    """Test that the function returns the correct patch when the input image is the same size as the patch."""
    dim = 4
    img = np.arange(dim * dim * 1).reshape((dim, dim, 1))
    patches, _ = extract_patches(img, dimension=dim)
    assert isinstance(patches, np.ndarray)
    assert patches.shape == (1, dim, dim, 1)
    np.testing.assert_array_equal(patches[0], img)


def test_smaller_than_dimension():
    """Test that the function returns a zero-padded patch when the input image is smaller than the patch size."""
    dim = 4
    h, w, c = 2, 3, 3
    img = np.ones((h, w, c), dtype=int)
    patches, _ = extract_patches(img, dimension=dim)
    assert patches.shape == (1, dim, dim, 3)
    patch = patches[0]
    np.testing.assert_array_equal(patch[:h, :w], img)
    assert np.all(patch[h:, :] == 0)
    assert np.all(patch[:, w:] == 0)


def test_multiple_patches():
    """Test that the function returns multiple patches when the input image is larger than the patch size."""
    dim = 3
    img = np.arange(2 * 5 * 3).reshape((2, 5, 3))
    patches, _ = extract_patches(img, dimension=dim)
    assert patches.shape == (2, dim, dim, 3)
    expected0 = np.zeros((dim, dim, 3), dtype=img.dtype)
    expected0[:2, :3, :] = img[:2, :3, :]
    expected1 = np.zeros((dim, dim, 3), dtype=img.dtype)
    expected1[:2, :2, :] = img[:2, 3:5, :]
    np.testing.assert_array_equal(patches[0], expected0)
    np.testing.assert_array_equal(patches[1], expected1)


def test_crop_image_cartesian():
    """Test that the crop_image_cartesian function works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    ground_truth_cropped = _read_img_path("tests/data/A2_10_patch.tiff")
    np.testing.assert_array_equal(test_image_cropped.img, ground_truth_cropped)


def test_invalid_crop_image_cartesian_dimensionality():
    """Test that a ValueError is raised when the given cartesian coordinates are invalid."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    with pytest.raises(ValueError):
        crop_image_cartesian(test_image, 6000, 6000, 8000, 8000)


def test_crop_image_screen():
    """Test that the crop_image_screen function works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_screen(test_image, 2443, 2674, 2561, 2809)
    ground_truth_cropped = _read_img_path("tests/data/A2_10_patch.tiff")
    np.testing.assert_array_equal(test_image_cropped.img, ground_truth_cropped)


def test_invalid_crop_image_screen_dimensionality():
    """Test that a ValueError is raised when the given image coordinates are invalid."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    with pytest.raises(ValueError):
        crop_image_screen(test_image, 2000, 3000, 4000, 2500)


def test_crop_image():
    """Test that the crop_image function works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image(test_image, 2443, 4436, 4275, 2674)
    ground_truth_cropped = _read_img_path("tests/data/A2_10_patch.tiff")
    np.testing.assert_array_equal(test_image_cropped.img, ground_truth_cropped)


def test_invalid_crop_image_dimensionality():
    """Test that a ValueError is raised when the given cropping dimensions are invalid."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    with pytest.raises(ValueError):
        crop_image_screen(test_image, 3500, 4200, 4000, 2500)


test_non_array_input()
test_invalid_dimensionality()
test_exact_patch_size()
test_smaller_than_dimension()
test_multiple_patches()
test_crop_image_cartesian()
test_invalid_crop_image_cartesian_dimensionality()
test_crop_image_screen()
test_invalid_crop_image_screen_dimensionality()
test_crop_image()
test_invalid_crop_image_dimensionality()
