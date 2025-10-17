import numpy as np


class ImageBundle:
    def __init__(
        self,
        img: np.ndarray,
        top_left_x: int,
        top_left_y: int,
        bottom_right_x: int,
        bottom_right_y: int,
    ):
        self.img = img
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y


def is_valid_ImageBundle(image_data: ImageBundle) -> bool:
    """
    Specification function that tests whether the input ImageBundle is valid

    Parameters
    ----------
    image_data : ImageBundle
        The ImageBundle containing the image.

    Returns
    -------
    is_valid: bool
        Whether or not the intput ImageBundle is valid
    """

    img = image_data.img
    top_left_x = image_data.top_left_x
    top_left_y = image_data.top_left_y
    bottom_right_x = image_data.bottom_right_x
    bottom_right_y = image_data.bottom_right_y

    if not isinstance(img, np.ndarray):
        raise TypeError("`img` must be a NumPy array")

    # Determine image shape
    if img.ndim == 3:
        H, W, C = img.shape
    else:
        raise ValueError(f"`img` must be a 3D array, got {img.ndim}D")

    if top_left_x < 0 | top_left_x > img.shape[1] | top_left_x > bottom_right_x:
        return False
    if top_left_y < 0 | top_left_y > img.shape[0] | top_left_y < bottom_right_y:
        return False
    if bottom_right_x < 0 | bottom_right_x > img.shape[1]:
        return False
    if bottom_right_y < 0 | bottom_right_y > img.shape[0]:
        return False

    if bottom_right_x - top_left_x != img.shape[1]:
        return False

    if top_left_y - bottom_right_y != img.shape[0]:
        return False

    return True


def ImageBundle_equal(
    image_data1: ImageBundle, image_data2: ImageBundle, tol: float = 1
) -> bool:
    """
    Specification function that tests whether 2 input ImageBundles are equal

    Parameters
    ----------
    image_data1 : ImageBundle
        One ImageBundle object being tested for equality
    image_data2: ImageBundle
        The other ImageBundle object being tested for equality

    Returns
    -------
    is_equal: bool
        Whether or not the two input ImageBundles are equal
    """

    if not np.allclose(image_data1.img, image_data2.img, atol=tol):
        return False

    if image_data1.top_left_x != image_data2.top_left_x:
        return False

    if image_data1.top_left_y != image_data2.top_left_y:
        return False

    if image_data1.bottom_right_x != image_data2.bottom_right_x:
        return False

    if image_data1.bottom_right_y != image_data2.bottom_right_y:
        return False

    return True
