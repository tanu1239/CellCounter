# src/cellcounter/analysis.py
"""
Functions for analysis of number of predictions per spot, filtering of annotations, and validating accuracy of predictions
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
import numpy as np
from typing import Tuple
from rtree import index
from .ImageBundle import is_valid_ImageBundle, ImageBundle


def cells_per_spot(
    spot_coords_path: str,
    analysis_results: pd.DataFrame,
    spot_diameter: float,
    image_data: ImageBundle,
) -> pd.DataFrame:
    """
    Returns the number of predictions in each spot within the region to which Stardist was applied

    Parameters
    ----------
    spot_coords_path: str
        Path to a csv containing spot locations (x-coordinates and y-coordinates)
    analysis_results : pd.DataFrame
        The dataframe containing prediction results
    spot_diameter: float
        The diameter of each spot
    image_data: ImageBundle
        The ImageBundle containing the image containing the spots of interest (whose cells per spot are being calculated)

    Returns
    -------
    cell_count_per_spot : pd.DataFrame
        A DataFrame containing coordinates of each spot, number of predictions in each spot, and coordinates of each of those predictions
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError("image_data is not a valid ImageBundle object")

    if spot_diameter <= 0:
        raise ValueError(
            f"diameter of spots must be a positive value, got: {spot_diameter}"
        )

    # Read and filter spot coordinates
    try:
        coords = pd.read_csv(spot_coords_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The spot coordinates file was not found: {spot_coords_path}"
        )

    x_min, x_max = (
        image_data.top_left_x,
        image_data.top_left_x + image_data.img.shape[1],
    )
    y_min, y_max = (
        image_data.bottom_right_y,
        image_data.bottom_right_y + image_data.img.shape[0],
    )

    coords = coords[
        (coords["x"] >= x_min)
        & (coords["x"] <= x_max)
        & (coords["y"] >= y_min)
        & (coords["y"] <= y_max)
    ]

    if coords.empty:
        return pd.DataFrame(columns=["x", "y", "predict_count", "predictions_in_spot"])

    # Extract spot and prediction coordinates
    spot_x = coords["x"].values
    spot_y = coords["y"].values
    pred_x = analysis_results["centroid-1"].values  # x (col)
    pred_y = analysis_results["centroid-0"].values  # y (row)
    pred_coords = analysis_results["coords"].values  # ndarray of (N_pixels, 2)

    # Vectorized distance calculation
    diff_x2 = (spot_x[:, None] - pred_x[None, :]) ** 2
    diff_y2 = (spot_y[:, None] - pred_y[None, :]) ** 2
    distances = np.sqrt(diff_x2 + diff_y2)

    # Find predictions within spot_diameter / 2
    spot_preds_mask = distances < (spot_diameter / 2)  # N x M boolean array
    predict_counts = spot_preds_mask.sum(axis=1)  # Number of predictions per spot

    # Initialize output DataFrame
    N = len(coords)
    cell_count_per_spot = pd.DataFrame(
        {
            "x": spot_x,
            "y": spot_y,
            "predict_count": predict_counts,
            "predictions_in_spot": [[] for _ in range(N)],  # Empty lists for POLYGONs
        }
    )

    # Process predictions for each spot
    for i in range(N):
        if predict_counts[i] == 0:
            continue

        matched_coords = pred_coords[spot_preds_mask[i]]
        # Filter valid coordinates and create POLYGONs
        polygons = [
            Polygon([(round(x, 2), round(y, 2)) for x, y in arr])
            for arr in matched_coords
            if len(arr) >= 3
        ]
        cell_count_per_spot.at[i, "predictions_in_spot"] = polygons

    return cell_count_per_spot


def annotations_per_spot(
    spot_coords_path: str,
    annotations: gpd.GeoDataFrame,
    spot_diameter: float,
    image_data: ImageBundle,
) -> pd.DataFrame:
    """
    Returns the number of annotations in each spot within the region to which Stardist was applied

    Parameters
    ----------
    spot_coords_path: str
        Path to a csv containing spot locations (x-coordinates and y-coordinates)
    annotations: gpd.GeoDataFrame
        The dataframe containing all annotations in the region
    spot_diameter: float
        The diameter of each spot
    image_data: ImageBundle
        The ImageBundle containing the image containing the spots of interest (whose annotations per spot are being calculated)

    Returns
    -------
    annot_count_per_spot : pd.DataFrame
        A DataFrame containing coordinates, number of annotations, and coordinates of annotations in each spot
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError("image_data is not a valid ImageBundle object")

    if spot_diameter <= 0:
        raise ValueError(
            f"diameter of spots must be a positive value, got: {spot_diameter}"
        )

    # Load and filter spot coordinates
    try:
        coords = pd.read_csv(spot_coords_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The spot coordinates file was not found: {spot_coords_path}"
        )

    annotations.crs = None

    x_min, x_max = (
        image_data.top_left_x,
        image_data.top_left_x + image_data.img.shape[1],
    )
    y_min, y_max = (
        image_data.bottom_right_y,
        image_data.bottom_right_y + image_data.img.shape[0],
    )

    coords = coords[
        (coords["x"] >= x_min)
        & (coords["x"] <= x_max)
        & (coords["y"] >= y_min)
        & (coords["y"] <= y_max)
    ]

    if coords.empty:
        return pd.DataFrame(columns=["x", "y", "annot_count", "annotations_in_spot"])

    # Compute annotation centroids
    centroids = annotations["geometry"].centroid
    centroid_x = centroids.x.values
    centroid_y = centroids.y.values

    # Vectorized distance computation
    spot_x = coords["x"].values
    spot_y = coords["y"].values
    diff_x2 = (spot_x[:, None] - centroid_x[None, :]) ** 2
    diff_y2 = (spot_y[:, None] - centroid_y[None, :]) ** 2
    distances = np.sqrt(diff_x2 + diff_y2)
    distances = np.round(distances, 4)

    # Find annotations within spot_diameter / 2
    spot_mask = distances < (spot_diameter / 2)  # N x M boolean array
    annot_counts = spot_mask.sum(axis=1)  # Number of annotations per spot

    # Initialize output DataFrame
    N = len(coords)
    annot_count_per_spot = pd.DataFrame(
        {
            "x": spot_x,
            "y": spot_y,
            "annot_count": annot_counts,
            "annotations_in_spot": [[] for _ in range(N)],  # Empty lists for POLYGONs
        }
    )

    # Process annotations for each spot
    ann_geometries = annotations["geometry"].values
    for i in range(N):
        if annot_counts[i] == 0:
            continue
        matched_anns = ann_geometries[spot_mask[i]]
        # Round coordinates to 2 decimal places
        polygons = [
            Polygon([(round(x, 2), round(y, 2)) for x, y in poly.exterior.coords])
            for poly in matched_anns
        ]
        annot_count_per_spot.at[i, "annotations_in_spot"] = polygons

    return annot_count_per_spot


def fix_self_intersections(poly: Polygon) -> Polygon:
    """
    Fix self-intersecting polygon by applying buffer(0).
    If the fix results in multiple polygons, return the largest one.
    Returns the fixed polygon or original if already valid.

    Parameters
    ----------
    poly: Polygon
        The input polygon that needs to be fixed

    Returns
    -------
    fixed_poly : Polygon
        A non-self intersecting polygon that is an approximation of the input polygon (or the input polygon itself if it is not self-intersecting)
    """
    if poly.is_valid:
        return poly

    fixed_poly = poly.buffer(0)

    # If result is a MultiPolygon, pick the largest polygon by area
    if isinstance(fixed_poly, MultiPolygon):
        largest = max(fixed_poly.geoms, key=lambda p: p.area)
        return largest

    # Otherwise (Polygon or empty), return as is
    return fixed_poly


def compute_iou_matrix(pred_coords: list, ann_geometries: list) -> pd.DataFrame:
    """
    Computes an IoU matrix for all predictions and annotations in the spot with rows corresponding to predictions, columns corresponding to annotations, and entries corresponding to IOU values

    Parameters
    ----------
    pred_coords: list
        List of Shapely Polygon objects representing prediction geometries in a single spot.

    ann_geometries: list
        List of Shapely Polygon (or geometry) objects representing annotation geometries in a single spot.

    Returns
    -------
    iou_matrix_df : pd.DataFrame
        A DataFrame an iou matrix consisting of all predictions and annotations in the spot
    """
    # Create prediction polygons
    raw_pred_polys = [pred for pred in pred_coords]

    fixed_pred_polys = []
    for i, poly in enumerate(raw_pred_polys):
        if poly.is_empty:
            continue
        fixed_poly = fix_self_intersections(poly)
        if fixed_poly.is_empty or not fixed_poly.is_valid:
            print(f"Prediction polygon {i} could not be fixed.")
            continue
        fixed_pred_polys.append(fixed_poly)

    # Filter valid annotation geometries
    ann_polys = [ann for ann in ann_geometries if not ann.is_empty and ann.is_valid]

    # If either list is empty, return empty DataFrame
    if not fixed_pred_polys or not ann_polys:
        return pd.DataFrame()

    # Build R-tree index for annotation polygons
    rtree_index = index.Index()
    for i, ann in enumerate(ann_polys):
        rtree_index.insert(i, ann.bounds)

    # Initialize IoU matrix
    iou_matrix = np.zeros((len(fixed_pred_polys), len(ann_polys)))

    # Compute IoU values using spatial indexing
    for i, pred_poly in enumerate(fixed_pred_polys):
        pred_bbox = pred_poly.bounds

        candidates = list(rtree_index.intersection(pred_bbox))

        for j in candidates:
            ann_poly = ann_polys[j]
            if ann_poly.is_empty or not ann_poly.is_valid:
                continue

            intersection_area = pred_poly.intersection(ann_poly).area
            union_area = pred_poly.union(ann_poly).area
            if union_area > 0:
                iou = intersection_area / pred_poly.area
                print(f"IOU[{i},{j}] = {iou:.3f}")
                iou_matrix[i, j] = iou

    return pd.DataFrame(iou_matrix)


def IOU_calculations(
    cell_count_per_spot: pd.DataFrame,
    annot_count_per_spot: pd.DataFrame,
    iou_threshold: float,
) -> pd.DataFrame:
    """
    Takes in a dataframe of all predictions and annotations in a region, and produces a dataframe containing
    number of predictions that overlap with annotations (IOU > 0.5) and number of annotations that overlap
    with predictions (IOU > 0.5)

    Parameters
    ----------
    cell_count_per_spot: pd.DataFrame
        DataFrame containing coordinates of each spot and number of predictions in each spot
    annot_count_per_spot: pd.DataFrame
        DataFrame containing coordinates of each spot and number of annotations in each spot
    iou_threshold: float
        Minimum intersection over union required for a prediction to be considered a correct match with an annotation and vice versa.

    Returns
    -------
    IOU_calculations : pd.DataFrame
        DataFrame consisting of number of predictions per spot, number of annotations per spot, number of predictions
        that overlaps with annotations (IOU > 0.5) per spot, and number of annotations that overlaps with predictions (IOU > 0.5)
        per spot
    """

    if iou_threshold < 0 or iou_threshold > 1:
        raise ValueError(
            f"diameter of spots must be a between 0 and 1, got {iou_threshold}"
        )

    # Extract coordinates for spots, predictions, and annotations
    spot_x = cell_count_per_spot["x"].values
    spot_y = cell_count_per_spot["y"].values

    all_preds_in_spot = cell_count_per_spot["predictions_in_spot"].values
    all_anns_in_spot = annot_count_per_spot["annotations_in_spot"].values

    # Initialize result arrays
    predict_counts = np.zeros(len(cell_count_per_spot), dtype=int)
    annot_counts = np.zeros(len(cell_count_per_spot), dtype=int)
    correct_predict_counts = np.zeros(len(cell_count_per_spot), dtype=int)
    correct_annot_counts = np.zeros(len(cell_count_per_spot), dtype=int)

    # Vectorized IoU calculation across all spots
    for idx in range(len(cell_count_per_spot)):
        # Extract predictions and annotations for current spot
        preds_in_spot = all_preds_in_spot[idx]
        anns_in_spot = all_anns_in_spot[idx]

        # Store the prediction and annotation counts
        predict_counts[idx] = len(preds_in_spot)
        annot_counts[idx] = len(anns_in_spot)

        # If there are no predictions or annotations in this spot, skip it
        if len(preds_in_spot) == 0 or len(anns_in_spot) == 0:
            continue

        # Calculate IoU matrix for current spot (predictions x annotations)
        iou_matrix = compute_iou_matrix(preds_in_spot, anns_in_spot)

        # Find correct predictions and annotations based on IoU threshold
        correct_predict_mask = (iou_matrix >= iou_threshold).any(axis=1)
        correct_annot_mask = (iou_matrix >= iou_threshold).any(axis=0)

        # Store correct counts
        correct_predict_counts[idx] = np.sum(correct_predict_mask)
        correct_annot_counts[idx] = np.sum(correct_annot_mask)

        # Print matched annotations' coordinates
        print(f"Spot {idx}: Matched Annotations Coordinates:")
        for j, matched in enumerate(correct_annot_mask):
            if matched:
                coords = (
                    list(anns_in_spot[j].exterior.coords)
                    if hasattr(anns_in_spot[j], "exterior")
                    else anns_in_spot[j].coords
                )
                print(f"  Annotation {j} coords: {coords}")

    # Create the final DataFrame with the results
    IOU_calculations = pd.DataFrame(
        {
            "x": spot_x,
            "y": spot_y,
            "predict_count": predict_counts,
            "annot_count": annot_counts,
            "correct_predict_count": correct_predict_counts,
            "correct_annot_count": correct_annot_counts,
        }
    )

    return IOU_calculations


def transform_polygon(
    image_data: ImageBundle, geopandas_polygon: Polygon, upscaling_factor: float
) -> Polygon:
    """
    Transforms a polygon by subtracting top-left coordinates of place in original image and applying upscaling factor

    Parameters
    ----------
    image_data: ImageBundle
       The ImageBundle containing the image on which Stardist is run
    geopandas_polygon: Polygon
        The polygon given to the function to transform
    upscaling_factor: float
        How much the image in the ImageBundle was upscaled by

    Returns
    -------
    transformed_polygon: Polygon
        The transformed polygon
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError(f"image_data is not a valid ImageBundle object")

    if upscaling_factor < 0:
        raise ValueError(
            f"Factor for upscaling image cannot be negative, got:{upscaling_factor}"
        )

    transformed_coords = [
        (
            (x - image_data.top_left_x) * upscaling_factor,
            (y - image_data.bottom_right_y) * upscaling_factor,
        )
        for x, y in geopandas_polygon.exterior.coords
    ]

    transformed_polygon = Polygon(transformed_coords)
    return transformed_polygon


def is_valid_polygon(
    top_left_x: int,
    top_left_y: int,
    geopandas_polygon: Polygon,
    region_width: int,
    region_height: int,
    polygon_to_remove: list,
) -> bool:
    """
    Checks whether a polygon is not the bounding box around the region of interest and whether the polygon is in the region of interest

    Parameters
    ----------
    top_left_x: int
        X-coordinate of top-left corner of region of interest
    top_left_y: int
        Y-coordinate of top-left corner of region of interest
    geopandas_polygon: Polygon
        The polygon given to the function
    region_width: int
        Width of region from which annotations are being extracted
    region_height: int
        Height of region from which annotations are being extracted
    polygon_to_remove: list
        A list of coordinates representing a polygon to remove before analysis such as artificial annotation outlining an area of interest

    Returns
    -------
    valid_polygon: bool
        Whether or not the polygon is a valid polygon with all of its coordinates in the region of interest or not
    """

    if top_left_x < 0:
        raise ValueError(f"invalid value for top_left_x coordinate, got {top_left_x}")

    if top_left_y < 0:
        raise ValueError(f"invalid value for top_left_y coordinate, got {top_left_y}")

    if region_width < 0:
        raise ValueError(
            f"invalid value for region_width dimension, got {region_width}"
        )

    if region_height < 0:
        raise ValueError(
            f"invalid value for region_height dimension, got {region_height}"
        )

    if geopandas_polygon.equals(polygon_to_remove):
        return False
    for x, y in geopandas_polygon.exterior.coords:
        if (
            x < top_left_x
            or y > top_left_y
            or x > (top_left_x + region_width)
            or y < (top_left_y - region_height)
        ):
            return False
    return True


def extract_annotations_in_region(
    geojson_path: str,
    polygon_to_remove: list,
    top_left_x: int,
    top_left_y: int,
    region_width: int,
    region_height: int,
) -> gpd.GeoDataFrame:
    """
    Returns a new Geopandas data frame containing annotations that come from the same region as the image in image_data

    Parameters
    ----------
    geojson_path: str
        Path to a geojson file containing all annotations in an image
    polygon_to_remove : list
        A list of coordinates representing a polygon to remove before analysis such as artificial annotation outlining an area of interest
    top_left_x: int
        X-coordinate of top-left corner of region of interest
    top_left_y: int
        Y-coordinate of top-left corner of region of interest
    region_width: int
        Width of region from which annotations are being extracted
    region_height: int
        Height of region from which annotations are being extracted

    Returns
    -------
    output_gdf : gpd.GeoDataFrame
        The output geopandas data frame containing only annotations from the same region sa the image in image_data
    """

    if top_left_x < 0:
        raise ValueError(f"invalid value for top_left_x coordinate, got {top_left_x}")

    if top_left_y < 0:
        raise ValueError(f"invalid value for top_left_y coordinate, got {top_left_y}")

    if region_width < 0:
        raise ValueError(
            f"invalid value for region_width dimension, got {region_width}"
        )

    if region_height < 0:
        raise ValueError(
            f"invalid value for region_height dimension, got {region_height}"
        )

    # Load GeoJSON
    try:
        gdf = gpd.read_file(geojson_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The annotations geojson file was not found: {geojson_path}"
        )

    # Keep only Polygon geometries
    gdf = gdf[gdf.geometry.type == "Polygon"]

    # Define the exact polygon to remove
    polygon_to_remove = Polygon(polygon_to_remove)

    # Filter polygons
    output_gdf = gdf[
        gdf["geometry"].apply(
            lambda poly: is_valid_polygon(
                top_left_x,
                top_left_y,
                poly,
                region_width,
                region_height,
                polygon_to_remove,
            )
        )
    ]

    return output_gdf
