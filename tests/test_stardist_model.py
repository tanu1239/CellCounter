# tests/test_stardist_model.py
import numpy as np
import pytest
import pandas as pd
from skimage import io
import pandas.testing as pdt
from skimage import color
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from cellcounter.utils import _read_img_path
from cellcounter.extraction import (
    extract_patches,
    crop_image_cartesian,
    crop_image_screen,
    crop_image,
)
from cellcounter.stardist_model import run_model, result_plot


def test_run_model_invalid_prob_thresh():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for prob_thresh."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    with pytest.raises(ValueError):
        run_model(test_image_cropped, (4, 4, 1), -0.55, 10)


def test_run_model():
    """Test run_model works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    labels, details, results = run_model(test_image_cropped, (4, 4, 1), 0.55, 10)
    ground_truth_labels = np.load("tests/data/he_labels_A2_10.npy")
    loaded = np.load("tests/data/details_big_up_A2_10.npz", allow_pickle=True)
    ground_truth_details = dict(loaded)
    ground_truth_results = pd.read_csv("tests/data/ground_truth_stardist_run_test.csv")
    np.testing.assert_array_equal(labels, ground_truth_labels)
    for key in details:
        # np.testing.assert_array_equal(
        #     details[key], ground_truth_details[key], err_msg=f"Mismatch in key: {key}", rtol=1e-4, atol=1e-6
        # )
        np.testing.assert_allclose(
            details[key],
            ground_truth_details[key],
            err_msg=f"Mismatch in key: {key}",
            rtol=1e-4,
            atol=1e-6,
        )
    results_df = pd.DataFrame(results)

    # Drop 'coords' column from both DataFrames before comparison
    results_df_no_coords = results_df.drop(columns=["coords", "label"], errors="ignore")
    ground_truth_results_no_coords = ground_truth_results.drop(
        columns=["coords", "label"], errors="ignore"
    )

    pdt.assert_frame_equal(
        results_df_no_coords,
        ground_truth_results_no_coords,
        check_dtype=False,
        check_exact=False,
        rtol=1e-4,
        atol=1e-6,
    )


def test_result_plot_invalid_spot_diameter():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for spot_diameter."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    labels = io.imread("tests/data/he_labels_A2_10.tiff")
    spot_coords_path = "tests/data/spot_coords.csv"
    with pytest.raises(ValueError):
        result_plot(test_image_cropped, labels, spot_coords_path, -2, 5)


def test_result_plot():
    # Load and preprocess test image
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    labels, details, results = run_model(test_image_cropped, (4, 4, 1), 0.55, 10)

    # Generate test result plot (as array)
    test_result_plot = result_plot(
        test_image_cropped, labels, "tests/data/spot_coords.csv", 54.039344274655456, 10
    )

    # Load pre-generated ground truth image
    ground_truth_result_plot = imread("tests/data/he_labels_with_spots_FIXED.tiff")

    # Ensure it’s RGB (remove alpha if present)
    if ground_truth_result_plot.shape[2] == 4:
        ground_truth_result_plot = ground_truth_result_plot[..., :3]

    # Resize both to same shape
    target_shape = (512, 512)
    test_resized = resize(
        test_result_plot, target_shape, preserve_range=True, anti_aliasing=True
    ).astype(np.float64)
    gt_resized = resize(
        ground_truth_result_plot, target_shape, preserve_range=True, anti_aliasing=True
    ).astype(np.float64)

    # Convert to grayscale
    gray_test = rgb2gray(test_resized)
    gray_gt = rgb2gray(gt_resized)

    # Compute SSIM
    score, _ = ssim(gray_test, gray_gt, full=True, data_range=1.0)

    # Assert similarity
    assert score > 0.995, f"Images do not look similar enough (SSIM={score:.4f})"


test_run_model_invalid_prob_thresh()
test_run_model()
test_result_plot_invalid_spot_diameter()
test_result_plot()
