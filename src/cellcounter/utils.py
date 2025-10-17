# src/cellcounter/utils.py
"""
Utility functions for reading images from AnnData objects or file paths.
"""
import numpy as np
from PIL import Image
from anndata import AnnData
from skimage import color
import tifffile


def _read_img_adata(adata: AnnData, img_key: str) -> np.ndarray:
    """
    Read a spatial image stored in an AnnData object and return it as a NumPy array.
    Looks under adata.uns["spatial"][img_key]["images"]["fullres"] or falls back to "hires".

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial image data.
    img_key : str
        The key in adata.uns["spatial"] for the spatial image (e.g., the tissue section name).

    Returns
    -------
    img_array : np.ndarray
        The image data as a NumPy array.

    Raises
    ------
    KeyError
        If the spatial data or image resolution keys are not found.
    ValueError
        If the loaded data cannot be converted to a NumPy array.
    """
    # Check for spatial entry
    if "spatial" not in adata.uns:
        raise KeyError("Anndata object has no 'spatial' entry in .uns")
    spatial = adata.uns["spatial"]
    if img_key not in spatial:
        raise KeyError(f"Spatial key '{img_key}' not found in adata.uns['spatial']")

    imgs = spatial[img_key].get("images", {})
    # Try fullres first, then hires
    for res in ("fullres", "hires"):
        if res in imgs:
            img_data = imgs[res]
            break
    else:
        raise KeyError(
            f"Neither 'fullres' nor 'hires' images found for key '{img_key}'"
        )

    # Convert to numpy array
    if isinstance(img_data, np.ndarray):
        return img_data
    try:
        return np.asarray(img_data)
    except Exception as e:
        raise ValueError(f"Could not convert image data to array: {e}")


def _read_img_path(img_path: str) -> np.ndarray:
    """
    Load an image from disk and return it as a NumPy array.

    Parameters
    ----------
    img_path : str
        Path to the image file.

    Returns
    -------
    img_array : np.ndarray
        The loaded image as a NumPy array.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the image cannot be opened or converted.
    """
    try:
        with Image.open(img_path) as img:
            img_array = np.array(img)
            if img_array.shape[-1] == 4:  # Has alpha channel
                rgb_image = color.rgba2rgb(img_array)
            else:
                rgb_image = img_array
            return rgb_image
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading image from {img_path}: {e}")


def _read_large_tif_img_path(img_path: str) -> np.ndarray:
    """
    Load an large image (several gigabytes) from disk and return it as a NumPy array.

    Parameters
    ----------
    img_path : str
        Path to the image file.

    Returns
    -------
    img_array : np.ndarray
        The loaded image as a NumPy array.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If the image cannot be opened or converted.
    """
    try:
        image = tifffile.imread(img_path)
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)
        return image
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading image from {img_path}: {e}")
