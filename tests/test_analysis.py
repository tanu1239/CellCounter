# tests/test_analysis.py
import pytest
import pandas as pd
import numpy as np
from skimage import color
import geopandas as gpd
from shapely import wkt
import re
from shapely.geometry import Polygon
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
import pandas.testing as pdt
from cellcounter.utils import _read_img_path
from cellcounter.stardist_model import run_model
from cellcounter.extraction import (
    extract_patches,
    crop_image_cartesian,
    crop_image_screen,
    crop_image,
)
from cellcounter.analysis import (
    cells_per_spot,
    annotations_per_spot,
    fix_self_intersections,
    IOU_calculations,
    compute_iou_matrix,
    transform_polygon,
    is_valid_polygon,
    extract_annotations_in_region,
)


def test_cells_per_spot_invalid_spot_diameter():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for spot_diameter."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    labels, details, results = run_model(test_image_cropped, (4, 4, 1), 0.55, 2)
    with pytest.raises(ValueError):
        cells_per_spot("data/spot_coords.csv", results, -2, test_image_cropped)


def normalize_polygon_string(polygon_data):
    """
    Normalize a POLYGON string by rounding floating-point numbers to 2 decimal places
    and standardizing whitespace. Handles both string and list inputs.

    Args:
        polygon_data (str or list): Either a string containing POLYGON objects
                                   (e.g., "[<POLYGON ((5146.1 2610, ...)>]") or a list
                                   of POLYGON objects.

    Returns:
        str: Normalized string with floats rounded to 2 decimal places and consistent whitespace
    """
    # Convert list to string if necessary
    if isinstance(polygon_data, list):
        polygon_str = str(polygon_data)
    else:
        polygon_str = polygon_data

    # Round floating-point numbers to 2 decimal places
    def round_number(match):
        return f"{float(match.group(0)):.2f}"

    normalized = re.sub(r"\d+\.\d+", round_number, polygon_str)
    # Remove extra whitespace and standardize to single spaces
    normalized = re.sub(r"\s+", " ", normalized.strip())
    return normalized


def test_cells_per_spot():
    """Test cells_per_spot works correctly on an example case."""
    spot_coords_path = "tests/data/spot_coords.csv"
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 5110, 2648, 5209, 2547)
    labels, details, results = run_model(test_image_cropped, (4, 4, 1), 0.55, 10)
    spot_diameter = 54.379
    test_cells_per_spot = cells_per_spot(
        spot_coords_path, results, spot_diameter, test_image_cropped
    )
    ground_truth_cells_per_spot = pd.read_csv("tests/data/A2_spot2.csv")

    # Normalize predictions_in_spot for both DataFrames
    test_cells_per_spot["predictions_in_spot"] = test_cells_per_spot[
        "predictions_in_spot"
    ].apply(normalize_polygon_string)
    ground_truth_cells_per_spot["predictions_in_spot"] = ground_truth_cells_per_spot[
        "predictions_in_spot"
    ].apply(normalize_polygon_string)

    # Save to CSV for reference (optional)

    # Compare DataFrames
    pdt.assert_frame_equal(
        test_cells_per_spot.reset_index(drop=True),
        ground_truth_cells_per_spot.reset_index(drop=True),
        check_dtype=False,
        check_like=True,
    )


def test_annotations_per_spot_invalid_spot_diameter():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for spot_diameter."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    annotations = gpd.read_file("tests/data/A2runrun.geojson")
    with pytest.raises(ValueError):
        annotations_per_spot(
            "tests/data/spot_coords.csv", annotations, -2, test_image_cropped
        )


def normalize_polygon_string(polygon_data):
    """
    Normalize a POLYGON string by rounding floating-point numbers to 2 decimal places
    and standardizing whitespace. Handles both string and list inputs.

    Args:
        polygon_data (str or list): Either a string containing POLYGON objects
                                   (e.g., "[<POLYGON ((5146.1 2610, ...)>]") or a list
                                   of POLYGON objects.

    Returns:
        str: Normalized string with floats rounded to 2 decimal places and consistent whitespace

    Raises:
        TypeError: If polygon_data is neither a string nor a list
    """
    if not isinstance(polygon_data, (str, list)):
        raise TypeError(f"Expected string or list, got {type(polygon_data)}")

    # Convert list to string if necessary
    if isinstance(polygon_data, list):
        polygon_str = str(polygon_data)
    else:
        polygon_str = polygon_data

    # Round floating-point numbers to 2 decimal places
    def round_number(match):
        return f"{float(match.group(0)):.2f}"

    normalized = re.sub(r"\d+\.\d+", round_number, polygon_str)
    # Remove extra whitespace and standardize to single spaces
    normalized = re.sub(r"\s+", " ", normalized.strip())
    return normalized


def test_annotations_per_spot():
    """Test annotations_per_spot works correctly on an example case."""
    spot_coords_path = "tests/data/spot_coords.csv"
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 5110, 2648, 5209, 2547)
    test_gdf = extract_annotations_in_region(
        "tests/data/A2runrun.geojson",
        [(5110, 2648), (5209, 2648), (5209, 2547), (5110, 2547)],
        5110,
        2648,
        99,
        101,
    )
    annotations = gpd.read_file("tests/data/A2runrun.geojson")
    spot_diameter = 54.379
    test_annotations_per_spot = annotations_per_spot(
        spot_coords_path, test_gdf, spot_diameter, test_image_cropped
    )
    ground_truth_annotations_per_spot = pd.read_csv("tests/data/annotate2.csv")

    # Normalize annotations_in_spot
    if "annotations_in_spot" in test_annotations_per_spot.columns:
        test_annotations_per_spot["annotations_in_spot"] = test_annotations_per_spot[
            "annotations_in_spot"
        ].apply(normalize_polygon_string)
        ground_truth_annotations_per_spot["annotations_in_spot"] = (
            ground_truth_annotations_per_spot["annotations_in_spot"].apply(
                normalize_polygon_string
            )
        )

    # Compare DataFrames
    pdt.assert_frame_equal(
        test_annotations_per_spot.reset_index(drop=True),
        ground_truth_annotations_per_spot.reset_index(drop=True),
        check_dtype=False,
        check_like=True,
        rtol=1e-5,
        atol=1e-8,
    )


def test_IOU_calculations_invalid_threshold():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for iou_threshold."""
    cell_count_per_spot = pd.read_csv("tests/data/A2_spot2.csv")
    annot_count_per_spot = pd.read_csv("tests/data/annotate2.csv")
    with pytest.raises(ValueError):
        IOU_calculations(cell_count_per_spot, annot_count_per_spot, -0.25)


def test_IOU_calculations():
    """Test IOU_calculations works correctly on an example case."""
    spot_coords_path = "tests/data/spot_coords.csv"
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 5110, 2648, 5209, 2547)
    labels, details, results = run_model(test_image_cropped, (4, 4, 1), 0.55, 10)
    spot_diameter = 54.379
    test_cells_per_spot = cells_per_spot(
        spot_coords_path, results, spot_diameter, test_image_cropped
    )
    annotations = gpd.read_file("tests/data/A2runrun.geojson")
    test_gdf = extract_annotations_in_region(
        "tests/data/A2runrun.geojson",
        [(5110, 2648), (5209, 2648), (5209, 2547), (5110, 2547)],
        5110,
        2648,
        99,
        101,
    )
    test_annotations_per_spot = annotations_per_spot(
        spot_coords_path, test_gdf, spot_diameter, test_image_cropped
    )
    test_IOU_calculations = IOU_calculations(
        test_cells_per_spot, test_annotations_per_spot, 0.5
    )
    ground_truth_IOU_calculations = pd.read_csv("tests/data/IOU_compare2.csv")
    pd.testing.assert_frame_equal(
        test_IOU_calculations, ground_truth_IOU_calculations, check_dtype=False
    )


def test_transform_polygon_invalid_upscaling_factor():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for upscaling_factor."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 2443, 4410, 2561, 4275)
    test_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(ValueError):
        transform_polygon(test_image_cropped, test_polygon, -2)


def test_transform_polygon():
    """Test transform_polygon works correctly on an example case."""
    test_image = np.load("tests/data/fullres.npz")["arr"]
    test_image = color.rgba2rgb(test_image)
    test_image_cropped = crop_image_cartesian(test_image, 5110, 2648, 5209, 2547)
    test_polygon = Polygon([(5294, 2692), (5389, 2692), (5389, 2825), (5294, 2825)])
    test_transformed_polygon = transform_polygon(test_image_cropped, test_polygon, 10)
    ground_truth_transformed_polygon = Polygon(
        [(1840, 1450), (2790, 1450), (2790, 2780), (1840, 2780)]
    )
    assert test_transformed_polygon.equals(ground_truth_transformed_polygon)


def test_is_valid_polygon_invalid_top_left_x():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for top_left_x."""
    top_left_x = -2
    top_left_y = 256
    region_width = 99
    region_height = 125
    test_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    polygon_to_remove = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    with pytest.raises(ValueError):
        is_valid_polygon(
            top_left_x,
            top_left_y,
            test_polygon,
            region_width,
            region_height,
            polygon_to_remove,
        )


def test_is_valid_polygon():
    """Test is_valid_polygon works correctly on an example case."""
    top_left_x = 0
    top_left_y = 10
    region_width = 10
    region_height = 10
    test_polygon = Polygon([(6, 6), (6, 7), (7, 7), (7, 6)])
    polygon_to_remove = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    assert is_valid_polygon(
        top_left_x,
        top_left_y,
        test_polygon,
        region_width,
        region_height,
        polygon_to_remove,
    )


def test_extract_annotations_invalid_region_width():
    """Test that a ValueError is raised when an invalid value is passed as a parameter such as for region_width."""
    top_left_x = -2
    top_left_y = 256
    region_width = 99
    region_height = 125
    polygon_to_remove = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    with pytest.raises(ValueError):
        extract_annotations_in_region(
            "tests/data/A2runrun.geojson",
            polygon_to_remove,
            top_left_x,
            top_left_y,
            region_width,
            region_height,
        )


def _round_geometry(geom: BaseGeometry, decimals: int) -> BaseGeometry:
    """Round coordinates of geometry to a fixed number of decimal places."""

    def _round_coords(x, y, z=None):
        if z is not None:
            return (round(x, decimals), round(y, decimals), round(z, decimals))
        return (round(x, decimals), round(y, decimals))

    return transform(_round_coords, geom)


def normalize_geometry(geom: BaseGeometry, decimals: int = 6) -> dict:
    """Convert geometry to a rounded dict for consistent comparison."""
    if geom.is_empty:
        return {}
    return mapping(_round_geometry(geom, decimals))


def geodataframes_functionally_equal(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, geom_tol: float = 1e-6
) -> bool:
    try:
        # Ensure both GeoDataFrames have the same columns
        if set(gdf1.columns) != set(gdf2.columns):
            print("Column sets differ.")
            return False

        # Ensure same column order
        gdf2 = gdf2[gdf1.columns]

        # Make copies so we don't modify the originals
        gdf1 = gdf1.copy()
        gdf2 = gdf2.copy()

        # Normalize geometry
        geom_col = gdf1.geometry.name
        gdf1[geom_col] = gdf1[geom_col].apply(
            lambda g: normalize_geometry(g, decimals=6)
        )
        gdf2[geom_col] = gdf2[geom_col].apply(
            lambda g: normalize_geometry(g, decimals=6)
        )

        for col in gdf1.columns:
            # Normalize numeric types
            if pd.api.types.is_numeric_dtype(gdf1[col]):
                gdf1[col] = gdf1[col].astype("float64").round(6)
                gdf2[col] = gdf2[col].astype("float64").round(6)

            # Normalize string types
            elif pd.api.types.is_string_dtype(gdf1[col]):
                gdf1[col] = gdf1[col].str.strip().fillna("")
                gdf2[col] = gdf2[col].str.strip().fillna("")

        # Sort by all columns to make row order irrelevant
        gdf1_sorted = gdf1.sort_values(by=gdf1.columns.tolist()).reset_index(drop=True)
        gdf2_sorted = gdf2.sort_values(by=gdf2.columns.tolist()).reset_index(drop=True)

        # Compare final results
        return gdf1_sorted.equals(gdf2_sorted)

    except Exception as e:
        print(f"Comparison failed: {e}")
        return False


def safe_convert_geometry_column_to_wkt(
    df: gpd.GeoDataFrame, geom_col: str = "geometry"
) -> pd.DataFrame:
    """
    Ensures the geometry column contains valid Shapely geometries and converts them to WKT strings.
    Returns a pandas DataFrame with WKT strings in the 'geometry' column.
    """
    df = df.copy()

    def to_wkt_safe(geom):
        if isinstance(geom, BaseGeometry):
            return geom.wkt
        try:
            return shape(geom).wkt  # In case it's GeoJSON-like
        except Exception:
            return None  # Drop or mark invalid entries

    df[geom_col] = df[geom_col].apply(to_wkt_safe)
    return pd.DataFrame(df)


def normalize_wkt(geometry):
    """Normalize WKT string to ensure consistent floating-point representation."""
    if not isinstance(geometry, str):
        geometry = geometry.wkt
    geom = wkt.loads(geometry)
    coords = [(float(x), float(y)) for x, y in geom.exterior.coords]
    formatted_coords = ", ".join(f"{x:.1f} {y:.1f}" for x, y in coords)
    return f"POLYGON (({formatted_coords}))"


def test_extract_annotations_in_region():
    """Test extract_annotations_in_region works correctly on an example case."""
    geojson_path = "tests/data/A2runrun.geojson"
    polygon_to_remove = [(5110, 2648), (5209, 2648), (5209, 2547), (5110, 2547)]
    top_left_x = 5110
    top_left_y = 2648
    region_width = 99
    region_height = 101

    test_extract_annotations = extract_annotations_in_region(
        geojson_path,
        polygon_to_remove,
        top_left_x,
        top_left_y,
        region_width,
        region_height,
    )

    # Convert GeoDataFrame to DataFrame and reset index
    test_pd = pd.DataFrame(test_extract_annotations).reset_index(drop=True)
    # Convert geometry column to WKT strings and normalize
    test_pd["geometry"] = test_pd["geometry"].apply(normalize_wkt)

    # Load ground truth and normalize
    ground_truth_df = pd.read_csv("tests/data/extracted2.csv")
    ground_truth_df["geometry"] = ground_truth_df["geometry"].apply(normalize_wkt)

    # Check row counts
    if len(test_pd) != len(ground_truth_df):
        raise AssertionError(
            f"Row count mismatch: test_pd has {len(test_pd)} rows, "
            f"ground_truth_df has {len(ground_truth_df)} rows"
        )

    # Compare DataFrames
    pd.testing.assert_frame_equal(test_pd, ground_truth_df, check_dtype=False)


test_cells_per_spot_invalid_spot_diameter()
test_cells_per_spot()
test_annotations_per_spot_invalid_spot_diameter()
test_annotations_per_spot()
test_IOU_calculations_invalid_threshold()
test_IOU_calculations()
test_transform_polygon_invalid_upscaling_factor()
test_transform_polygon()
test_is_valid_polygon_invalid_top_left_x()
test_is_valid_polygon()
test_extract_annotations_invalid_region_width()
test_extract_annotations_in_region()
