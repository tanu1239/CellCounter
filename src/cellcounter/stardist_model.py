# src/cellcounter/stardist.py
"""
Runs Stardist on preprocessed images, produces a regionprops table with prediction results and data and plots the result
"""

import numpy as np
from stardist.models import StarDist2D
import pandas as pd
from PIL import Image
import io
from skimage.measure import regionprops_table, regionprops
import matplotlib.pyplot as plt
from skimage.transform import rescale
from csbdeep.utils import normalize
from typing import Tuple
from .ImageBundle import ImageBundle, is_valid_ImageBundle


def run_model(
    image_data: ImageBundle, n_tiles: tuple, prob_thresh: float, upscaling_factor: float
) -> Tuple[np.ndarray, dict, pd.DataFrame]:
    """
    Performs prediction on the image in the ImageBundle, transforms results back to original coordinates
    and converts results to a pandas DataFrame.

    Parameters
    ----------
    image_data : ImageBundle
        The ImageBundle containing the image.
    n_tiles : tuple
        Number of tiles to split image into during prediction.
    prob_thresh : float
        Prediction threshold for classification as a predicted cell.
    upscaling_factor : float
        Upscaling factor applied to the input image.

    Returns
    -------
    he_labels : np.ndarray
        Labeled segmentation mask from StarDist.
    details : dict
        Additional StarDist output.
    regions_df : pd.DataFrame
        DataFrame containing region properties and transformed coordinates.
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError(f"image_data is not a valid ImageBundle object")

    if prob_thresh < 0:
        raise ValueError(
            f"Threshold for cell detection cannot be negative, got: {prob_thresh}"
        )

    if upscaling_factor < 0:
        raise ValueError(
            f"Factor for upscaling image cannot be negative, got: {upscaling_factor}"
        )

    # Upscale the image
    img_upscaled = rescale(
        image_data.img, upscaling_factor, anti_aliasing=False, channel_axis=2
    )

    # Run StarDist prediction
    model = StarDist2D.from_pretrained("2D_versatile_he")
    he_labels, details = model.predict_instances(
        normalize(img_upscaled), n_tiles=n_tiles, prob_thresh=prob_thresh
    )

    # Get scalar properties using regionprops_table
    regions_table = regionprops_table(
        he_labels,
        intensity_image=normalize(img_upscaled),
        properties=[
            "area",
            "bbox",
            "bbox_area",
            "centroid",
            "convex_area",
            "coords",
            "eccentricity",
            "equivalent_diameter",
            "extent",
            "filled_area",
            "label",
            "local_centroid",
            "major_axis_length",
            "max_intensity",
            "mean_intensity",
            "min_intensity",
            "minor_axis_length",
            "inertia_tensor",
            "inertia_tensor_eigvals",
        ],
    )

    regions_df = pd.DataFrame(regions_table)

    # Transform scalar coordinate properties
    for col in ["bbox-0", "bbox-2", "centroid-0", "local_centroid-0"]:
        if col in regions_df:
            regions_df[col] = (
                regions_df[col] / upscaling_factor
            ) + image_data.bottom_right_y
    for col in ["bbox-1", "bbox-3", "centroid-1", "local_centroid-1"]:
        if col in regions_df:
            regions_df[col] = (
                regions_df[col] / upscaling_factor
            ) + image_data.top_left_x

    # Extract per-region coords and transform
    regions_props = regionprops(he_labels, intensity_image=normalize(img_upscaled))
    global_coords = []
    for region in regions_props:
        coords = region.coords  # (row, col) format
        rows, cols = coords[:, 0], coords[:, 1]
        x = cols / upscaling_factor + image_data.top_left_x
        y = rows / upscaling_factor + image_data.bottom_right_y
        global_coords.append(np.stack([x, y], axis=1))  # shape (N_points, 2), (x, y)

    regions_df["coords"] = global_coords

    return he_labels, details, regions_df


def result_plot(
    image_data: ImageBundle,
    labels: np.ndarray,
    spot_coords_path: str,
    spot_diameter: float,
    upscaling_factor: int,
) -> np.ndarray:
    """
    Generates a plot of the predictions with spots overlayed using matplotlib

    Parameters
    ----------
    image_data: ImageBundle
        The ImageBundle containing the image
    labels: np.ndarray
        Array containing integer labels corresponding to detected instances in the input image
    spot_coords_path: str
        Path to a csv containing spot locations (x-coordinates and y-coordinates)
    spot_diameter: float
        The diameter of each spot
    upscaling_factor: int
        How much the image in the ImageBundle was upscaled by

    Returns
    -------
    img_array: np.ndarray
        Numpy array of the plot of the predictions with spots overlayed
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError(f"image_data is not a valid ImageBundle object")

    plt.subplot(1, 2, 2)

    try:
        coords = pd.read_csv(spot_coords_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The spot coordinates file was not found: {spot_coords_path}"
        )

    if spot_diameter < 0:
        raise ValueError(
            f"The diameter of the spots in the image cannot be negative, got: {spot_diameter}"
        )

    if upscaling_factor < 0:
        raise ValueError(
            f"Factor for upscaling image cannot be negative, got:{upscaling_factor}"
        )

    coords["x"] = (coords["x"] - image_data.top_left_x) * upscaling_factor
    coords["y"] = (coords["y"] - image_data.bottom_right_y) * upscaling_factor

    circle_count = 0
    fig, ax = plt.subplots(figsize=(15, 15))
    for i, row in coords.iterrows():
        circle = plt.Circle(
            (row.x, row.y),
            radius=(spot_diameter) * upscaling_factor / 2,
            color="red",
            fill=False,
            lw=0.5,
        )
        ax.add_artist(circle)
    plt.imshow(labels)
    plt.axis("off")
    plt.title("prediction")
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="tiff",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=False,
    )
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img_array = np.array(img)
    buf.close()
    plt.show(block=False)

    return img_array
