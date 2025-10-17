# src/cellcounter/preprocessing.py
"""
Preprocessing functions for images in ImageBundle class.
"""

import numpy as np
from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.filters import unsharp_mask, laplace
from skimage import color
from csbdeep.utils import normalize
from typing import Optional, Tuple
from .ImageBundle import ImageBundle, is_valid_ImageBundle


def preprocessing_clahe(
    image_data: ImageBundle,
    kernel_size: Optional[Tuple[int, int]] = None,
    clip_limit: float = 0.01,
) -> ImageBundle:
    """
    Load an ImageBundle and returns a new ImageBundle with the image preprocessed using Contrast Limited Adaptive Histogram Equalization

    Parameters
    ----------
    image_data: ImageBundle
        The ImageBundle containing the image
    kernel_size: tuple
        Defines the shape of contextual regions used in the algorithm
    clip_limit: int
        Helps set the threshold of contrast enhancement such that it does not become too extreme

    Returns
    -------
    clahe_image_data: ImageBundle
        A new ImageBundle that contains the processed image
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError(f"image_data is not a valid ImageBundle object")

    if kernel_size is None:
        h, w = image_data.img.shape[0], image_data.img.shape[1]
        kernel_size = (w // 8, h // 8)

    elif not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
        raise TypeError(
            f"kernel_size must be a tuple of (int, int), got: {kernel_size}"
        )

    if kernel_size[0] <= 0 or kernel_size[1] <= 0:
        raise ValueError(f"kernel_size values must be positive, got: {kernel_size}")

    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be > 0, got: {clip_limit}")

    clahe_image = equalize_adapthist(image_data.img, kernel_size, clip_limit)
    clahe_image_data = ImageBundle(
        clahe_image,
        image_data.top_left_x,
        image_data.top_left_y,
        image_data.bottom_right_x,
        image_data.bottom_right_y,
    )

    return clahe_image_data


def preprocessing_unsharp_masking(
    image_data: ImageBundle, radius: float = 1, amount: float = 1
) -> ImageBundle:
    """
    Load an ImageBundle and returns a new ImageBundle with the image preprocessed using unsharp masking

    Parameters
    ----------
    image_data: ImageBundle
        The ImageBundle containing the image
    radius: float
        Defines the radius of the gaussian blur with higher values meaning more blurring
    amount: float
        Determines intensity of sharpening effect (how much edge-enhancement is added back to the image)

    Returns
    -------
    unsharp_image_data: ImageBundle
        A new ImageBundle that contains the processed image
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError(f"image_data is not a valid ImageBundle object")

    if radius <= 0:
        raise ValueError(f"radius must be > 0, got: {radius}")

    if amount < 0:
        raise ValueError(f"amount must be >= 0, got: {amount}")

    unsharp_image = unsharp_mask(image_data.img, radius, amount)
    unsharp_image_data = ImageBundle(
        unsharp_image,
        image_data.top_left_x,
        image_data.top_left_y,
        image_data.bottom_right_x,
        image_data.bottom_right_y,
    )

    return unsharp_image_data


def preprocessing_laplacian_filtering(
    image_data: ImageBundle, ksize: int = 3, mask: np.ndarray = None
) -> ImageBundle:
    """
    Load an ImageBundle and returns a new ImageBundle with the image preprocessed using laplacian filtering

    Parameters
    ----------
    image_data: ImageBundle
        The ImageBundle containing the image
    ksize: int
        The size of the Laplacian operator so that it will have a size of ksize * ImageData.img.ndim
    mask: np.ndarray
        An optional mask that restricts filter application to a certain area

    Returns
    -------
    laplacian_image_data : ImageBundle
        A new ImageBundle that contains the processed image
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError(f"image_data is not a valid ImageBundle object")

    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got: {ksize}")

    if mask is not None:
        if mask.shape != image_data.img.shape or mask.dtype != bool:
            raise ValueError(
                f"mask must be a boolean array having same size as input image"
            )

    laplacian_image = laplace(image_data.img, ksize, mask)
    laplacian_image_data = ImageBundle(
        laplacian_image,
        image_data.top_left_x,
        image_data.top_left_y,
        image_data.bottom_right_x,
        image_data.bottom_right_y,
    )

    return laplacian_image_data


def preprocessing_histogram_equalization(
    image_data: ImageBundle, nbins: int, mask: np.ndarray = None
) -> ImageBundle:
    """
    Load an ImageBundle and returns a new ImageBundle with the image preprocessed using histogram equalization

    Parameters
    ----------
    image_data: ImageBundle
        The ImageBundle containing the image
    nbins: int
        Number of bins for image histogram
    mask: np.ndarray
        Array of same shape as the image containing 0s and 1s. Points at which mask == True are used for equalization

    Returns
    -------
    histo_equalization_image_data : ImageBundle
        A new ImageBundle that contains the processed image
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError(f"image_data is not a valid ImageBundle object")

    if nbins <= 0:
        raise ValueError(f"nbins must be a positive integer, got: {nbins}")

    if mask is not None:
        if mask.shape != image_data.img.shape | mask.dtype != bool:
            raise ValueError(
                f"mask must be a boolean array having same size as input image, got"
            )

    if image_data.img.ndim == 2:
        # Grayscale
        histo_image = equalize_hist(image_data.img, nbins=nbins, mask=mask)

    elif image_data.img.ndim == 3 and image_data.img.shape[-1] in (3, 4):
        # RGB or RGBA
        rgb = image_data.img[..., :3]  # Drop alpha if present
        hsv = color.rgb2hsv(rgb)
        hsv[..., 2] = equalize_hist(hsv[..., 2], nbins=nbins, mask=mask)
        histo_image = color.hsv2rgb(hsv)

    histo_equalization_image_data = ImageBundle(
        histo_image,
        image_data.top_left_x,
        image_data.top_left_y,
        image_data.bottom_right_x,
        image_data.bottom_right_y,
    )

    return histo_equalization_image_data


def preprocessing_normalization(
    image_data: ImageBundle, lower_percentile: float, upper_percentile: float
) -> ImageBundle:
    """
    Load an ImageBundle and returns a new ImageBundle with the normalized image based on given percentiles
    Parameters
    ----------
    image_data: ImageBundle
        The ImageBundle containing the image
    lower_percentile: float
        Performs normalization with this percentile becoming 0
    upper_percentile: float
        Performs normalization with this percentile becoming 1

    Returns
    -------
    normalized_image_data: ImageBundle
        A new ImageBundle that contains the processed image
    """

    if not is_valid_ImageBundle(image_data):
        raise ValueError(f"image_data is not a valid ImageBundle object")

    if (
        lower_percentile < 0
        or lower_percentile >= 100
        or lower_percentile > upper_percentile
    ):
        raise ValueError(
            f"lower_percentile must be 0 <= pmin < 100 and must not be greater than upper percentile, got: {lower_percentile}"
        )

    if (
        upper_percentile <= 0
        or upper_percentile > 100
        or upper_percentile < lower_percentile
    ):
        raise ValueError(
            f"upper_percentile must be 0 < pmax <= 100 and must not be lower than lower percentile, got: {upper_percentile}"
        )

    normalized_image = normalize(
        image_data.img, lower_percentile, upper_percentile, axis=(0, 1)
    )
    normalized_image_data = ImageBundle(
        normalized_image,
        image_data.top_left_x,
        image_data.top_left_y,
        image_data.bottom_right_x,
        image_data.bottom_right_y,
    )

    return normalized_image_data
