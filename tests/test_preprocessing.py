# tests/test_preprocessing.py
import numpy as np
import pytest
from skimage.util import img_as_float32
from skimage import color
from cellcounter.utils import _read_img_path
from cellcounter.extraction import (
    extract_patches,
    crop_image_cartesian,
    crop_image_screen,
    crop_image,
)
from cellcounter.preprocessing import (
    preprocessing_clahe,
    preprocessing_unsharp_masking,
    preprocessing_laplacian_filtering,
    preprocessing_histogram_equalization,
    preprocessing_normalization,
)
from cellcounter.ImageBundle import ImageBundle, ImageBundle_equal


def test_preprocessing_clahe_invalid_clip_limit():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for clip_limit."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    with pytest.raises(ValueError):
        preprocessing_clahe(
            test_image_cropped,
            (
                test_image_cropped.img.shape[1] // 8,
                test_image_cropped.img.shape[0] // 8,
            ),
            -0.01,
        )


def test_preprocessing_clahe():
    """Test preprocessing_clahe works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    ground_truth_clahe = np.load("tests/data/clahe_image_test_cropped.npy")
    clahe_image = preprocessing_clahe(
        test_image_cropped,
        (test_image_cropped.img.shape[1] // 8, test_image_cropped.img.shape[0] // 8),
        0.01,
    )
    ground_truth_clahe_ImageBundle = ImageBundle(
        ground_truth_clahe, 2443, 4410, 2561, 4275
    )
    assert ImageBundle_equal(ground_truth_clahe_ImageBundle, clahe_image)


def test_preprocessing_unsharp_masking_invalid_radius():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for radius."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    with pytest.raises(ValueError):
        preprocessing_unsharp_masking(test_image_cropped, -1, 1)


def test_preprocessing_unsharp_masking():
    """Test preprocessing_unsharp_masking works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    # ground_truth_unsharp_masking = np.load("tests/data/unsharp_masking_image_test.npy")
    ground_truth_unsharp_masking = np.load(
        "tests/data/unsharp_masking_test_cropped.npy"
    )
    unsharp_masking_image = preprocessing_unsharp_masking(test_image_cropped, 1, 1)
    ground_truth_unsharp_masking_ImageBundle = ImageBundle(
        ground_truth_unsharp_masking, 2443, 4410, 2561, 4275
    )
    # unsharp_masking_image.img = color.rgba2rgb(unsharp_masking_image.img)
    assert ImageBundle_equal(
        ground_truth_unsharp_masking_ImageBundle, unsharp_masking_image
    )


def test_preprocessing_laplacian_invalid_ksize():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for ksize."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    with pytest.raises(ValueError):
        preprocessing_laplacian_filtering(test_image_cropped, 2, None)


def test_preprocessing_laplacian():
    """Test preprocessing_laplacian_filtering works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    # ground_truth_unsharp_masking = np.load("tests/data/unsharp_masking_image_test.npy")
    ground_truth_laplacian = np.load("tests/data/laplacian_test_cropped.npy")
    laplacian_image = preprocessing_laplacian_filtering(test_image_cropped, 3, None)
    ground_truth_laplacian_ImageBundle = ImageBundle(
        ground_truth_laplacian, 2443, 4410, 2561, 4275
    )
    # unsharp_masking_image.img = color.rgba2rgb(unsharp_masking_image.img)
    assert ImageBundle_equal(ground_truth_laplacian_ImageBundle, laplacian_image)


def test_preprocessing_histo_eq_invalid_nbins():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for nbins."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    with pytest.raises(ValueError):
        preprocessing_histogram_equalization(test_image_cropped, 0, None)


def test_preprocessing_histo_equalization():
    """Test preprocessing_histogram_equalization works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    ground_truth_histo_eq = np.load("tests/data/histo_eq_image_test_cropped.npy")
    histo_eq_image = preprocessing_histogram_equalization(test_image_cropped, 256, None)
    ground_truth_histo_eq_ImageBundle = ImageBundle(
        ground_truth_histo_eq, 2443, 4410, 2561, 4275
    )
    assert ImageBundle_equal(ground_truth_histo_eq_ImageBundle, histo_eq_image)


def test_preprocessing_normalization_invalid_lower_percentile():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for lower_percentile."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    with pytest.raises(ValueError):
        preprocessing_normalization(test_image_cropped, -1, 46)


def test_preprocessing_normalization():
    """Test preprocessing_normalization works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    ground_truth_normalized = np.load("tests/data/normalized_image_test_cropped.npy")
    normalized_image = preprocessing_normalization(test_image_cropped, 3, 99.8)
    ground_truth_normalized_ImageBundle = ImageBundle(
        ground_truth_normalized, 2443, 4410, 2561, 4275
    )
    assert ImageBundle_equal(ground_truth_normalized_ImageBundle, normalized_image)


test_preprocessing_clahe_invalid_clip_limit()
test_preprocessing_clahe()
test_preprocessing_unsharp_masking_invalid_radius()
test_preprocessing_unsharp_masking()
test_preprocessing_laplacian_invalid_ksize()
test_preprocessing_laplacian()
test_preprocessing_histo_eq_invalid_nbins()
test_preprocessing_histo_equalization()
test_preprocessing_normalization_invalid_lower_percentile()
test_preprocessing_normalization()
