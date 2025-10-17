# src/cellcounter/extraction.py
"""
Module for extracting non-overlapping patches from spatial images.
"""
import numpy as np
from skimage import util
from .ImageBundle import ImageBundle


def extract_patches(
    img: np.ndarray, dimension: int = 4096
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Cut the image into non-overlapping patches of size `dimension x dimension`.
    Regions exceeding the image boundaries are zero-padded.

    Parameters
    ----------
    img : np.ndarray
        Input image as a 3D (H, W, C) array.
    dimension : int, optional
        The height and width of each square patch. Default is 4096.

    Returns
    -------
    tuple[np.ndarray, list[tuple[int, int]]]
        A tuple containing:
        - A 4D array of shape (n_patches, dimension, dimension, C) with the extracted patches.
        - A list of tuples (top, left) indicating the coordinates of each patch in the original image.
    The list has length n_patches, where n_patches is the number of patches extracted.
    The patches are ordered from top to bottom and left to right.

    Raises
    ------
    TypeError
        If `img` is not a NumPy array.
    ValueError
        If `img` has neither 2 nor 3 dimensions.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("`img` must be a NumPy array")

    # Determine image shape
    if img.ndim == 3:
        H, W, C = img.shape
    else:
        raise ValueError(f"`img` must be a 3D array, got {img.ndim}D")

    # Compute number of patches along each axis
    n_h = (H + dimension - 1) // dimension
    n_w = (W + dimension - 1) // dimension

    # Create a zero-padded canvas
    pad_H = n_h * dimension
    pad_W = n_w * dimension
    padded = np.zeros((pad_H, pad_W, C), dtype=img.dtype)
    padded[:H, :W, :] = img

    # Extract patches
    patches = []
    patch_coordinates = []
    for i in range(n_h):
        for j in range(n_w):
            top, left = i * dimension, j * dimension
            patch = padded[top : top + dimension, left : left + dimension, :]
            patches.append(patch)
            patch_coordinates.append((top, left))

    # Stack and return patches and coordinates
    return np.stack(patches, axis=0), patch_coordinates


def crop_image_cartesian(
    img: np.ndarray,
    top_left_x: int,
    top_left_y: int,
    bottom_right_x: int,
    bottom_right_y: int,
) -> ImageBundle:
    """
    Crops the input image to return an ImageBundle object containing a specific portion of the image determined by the given coordinates
    Note: Since later analysis will be done by exporting annotations as geojson files, this function uses geographic/cartesian coordinates
    so top_left_y will be greater than bottom_right_y (y coordinates increase upward)

    Parameters
    ----------
    img : np.ndarray
        Input image as a 3D (H, W, C) array.
    int: top_left_x
        X-coordinate of top-left corner of interest
    int: top_left_y
        Y-coordinate of top-left corner of interest
    int: bottom_right_x
        X-coordinate of bottom-right corner of interest
    int: bottom_right_y
        Y-coordinate of bottom-right corner of interest

    Returns
    -------
    image_data: ImageBundle
        instance of ImageBundle class storing the cropped image and x and y-coordinates of top-left corner

    Raises
    ------
    TypeError
        If `img` is not a NumPy array.
    ValueError
        If `img` has neither 2 nor 3 dimensions.
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("`img` must be a NumPy array")

    # Determine image shape
    if img.ndim == 3:
        H, W, C = img.shape
    else:
        raise ValueError(f"`img` must be a 3D array, got {img.ndim}D")

    if top_left_x < 0 or top_left_x > img.shape[1] or top_left_x > bottom_right_x:
        raise ValueError(f"Invalid top_left_x value provided for cropping")

    if top_left_y < 0 or top_left_y > img.shape[0] or top_left_y < bottom_right_y:
        raise ValueError(f"Invalid top_left_y value provided for cropping")

    if (
        bottom_right_x < 0
        or bottom_right_x > img.shape[1]
        or bottom_right_x < top_left_x
    ):
        raise ValueError(f"Invalid bottom_right_x value provided for cropping")

    if (
        bottom_right_y < 0
        or bottom_right_y > img.shape[0]
        or bottom_right_y > top_left_y
    ):
        raise ValueError(f"Invalid bottom_right_y value provided for cropping")

    crop_left = top_left_x
    crop_right = img.shape[1] - bottom_right_x
    crop_top = bottom_right_y
    crop_bottom = img.shape[0] - top_left_y
    cropped_image = util.crop(
        img, ((crop_top, crop_bottom), (crop_left, crop_right), (0, 0)), copy=False
    )

    image_data = ImageBundle(
        cropped_image, top_left_x, top_left_y, bottom_right_x, bottom_right_y
    )

    return image_data


def crop_image_screen(
    img: np.ndarray,
    top_left_x: int,
    top_left_y: int,
    bottom_right_x: int,
    bottom_right_y: int,
) -> ImageBundle:
    """
    Crops the input image to return an ImageBundle object containing a specific portion of the image determined by the given coordinates
    Note: This function uses the image/screen coordinate system so bottom_right_y will be greater than top_left_y (y coordinates increase downward)

    Parameters
    ----------
    img : np.ndarray
        Input image as a 3D (H, W, C) array.
    int: top_left_x
        X-coordinate of top-left corner of interest
    int: top_left_y
        Y-coordinate of top-left corner of interest
    int: bottom_right_x
        X-coordinate of bottom-right corner of interest
    int: bottom_right_y
        Y-coordinate of bottom-right corner of interest

    Returns
    -------
    image_data: ImageBundle
        instance of ImageBundle class storing the cropped image and x and y-coordinates of top-left corner

    Raises
    ------
    TypeError
        If `img` is not a NumPy array.
    ValueError
        If `img` has neither 2 nor 3 dimensions.
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("`img` must be a NumPy array")

    # Determine image shape
    if img.ndim == 3:
        H, W, C = img.shape
    else:
        raise ValueError(f"`img` must be a 3D array, got {img.ndim}D")

    if top_left_x < 0 or top_left_x > img.shape[1] or top_left_x > bottom_right_x:
        raise ValueError(f"Invalid top_left_x value provided for cropping")

    if top_left_y < 0 or top_left_y > img.shape[0] or top_left_y > bottom_right_y:
        raise ValueError(f"Invalid top_left_y value provided for cropping")

    if (
        bottom_right_x < 0
        or bottom_right_x > img.shape[1]
        or bottom_right_x < top_left_x
    ):
        raise ValueError(f"Invalid bottom_right_x value provided for cropping")

    if (
        bottom_right_y < 0
        or bottom_right_y > img.shape[0]
        or bottom_right_y < top_left_y
    ):
        raise ValueError(f"Invalid bottom_right_y value provided for cropping")

    crop_left = top_left_x
    crop_right = img.shape[1] - bottom_right_x
    crop_top = img.shape[0] - bottom_right_y
    crop_bottom = top_left_y
    cropped_image = util.crop(
        img, ((crop_top, crop_bottom), (crop_left, crop_right), (0, 0)), copy=False
    )

    image_data = ImageBundle(
        cropped_image,
        top_left_x,
        img.shape[0] - top_left_y,
        bottom_right_x,
        img.shape[0] - bottom_right_y,
    )

    return image_data


def crop_image(
    img: np.ndarray, crop_left: int, crop_right: int, crop_top: int, crop_bottom: int
) -> ImageBundle:
    """
    Crops the input image to return an ImageBundle object containing a specific portion of the image determined by the given cropping dimensions
    Note: This function takes inputs corresponding to the amount by which to crop the 4 sides of the input image

    Parameters
    ----------
    img : np.ndarray
        Input image as a 3D (H, W, C) array.
    int: top_left_x
        X-coordinate of top-left corner of interest
    int: top_left_y
        Y-coordinate of top-left corner of interest
    int: bottom_right_x
        X-coordinate of bottom-right corner of interest
    int: bottom_right_y
        Y-coordinate of bottom-right corner of interest

    Returns
    -------
    image_data: ImageBundle
        instance of ImageBundle class storing the cropped image and x and y-coordinates of top-left corner

    Raises
    ------
    TypeError
        If `img` is not a NumPy array.
    ValueError
        If `img` has neither 2 nor 3 dimensions.
    """

    if not isinstance(img, np.ndarray):
        raise TypeError("`img` must be a NumPy array")

    # Determine image shape
    if img.ndim == 3:
        H, W, C = img.shape
    else:
        raise ValueError(f"`img` must be a 3D array, got {img.ndim}D")

    if crop_left < 0 or crop_left > img.shape[1]:
        raise ValueError(f"Invalid crop_left value provided for cropping")

    if crop_top < 0 or crop_top > img.shape[0]:
        raise ValueError(f"Invalid crop_top value provided for cropping")

    if crop_right < 0 or crop_right > img.shape[1]:
        raise ValueError(f"Invalid crop_right value provided for cropping")

    if crop_bottom < 0 or crop_bottom > img.shape[0]:
        raise ValueError(f"Invalid crop_bottom value provided for cropping")

    if crop_left + crop_right > img.shape[1]:
        raise ValueError(
            f"crop_left and crop_right values are too large (more than entire image is cropped)"
        )

    if crop_top + crop_bottom > img.shape[0]:
        raise ValueError(
            f"crop_top and crop_bottom values are too large (more than entire image is cropped)"
        )

    cropped_image = util.crop(
        img, ((crop_top, crop_bottom), (crop_left, crop_right), (0, 0)), copy=False
    )

    image_data = ImageBundle(
        cropped_image,
        crop_left,
        img.shape[0] - crop_bottom,
        img.shape[1] - crop_right,
        crop_top,
    )

    return image_data
