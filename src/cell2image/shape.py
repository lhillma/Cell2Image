from typing import Callable

import numpy as np
from numba import njit
from scipy.signal import convolve2d

import cell2image.image as cimg

# re-exported: create_mask_procedurally used to live here
from cell2image.neighbourhood import create_mask_procedurally  # noqa: F401


def calc_xi_numerically(neighbour_order: int) -> int:
    """
    Numerically calculate the correction factor xi for a given neighbour order.

    Args:
        neighbour_order (int): The neighbour order for which to calculate the
            correction factor.

    Returns:
        int: The correction factor xi.

    Examples:

    >>> calc_xi_numerically(1)
    1

    >>> calc_xi_numerically(2)
    3

    >>> calc_xi_numerically(8)
    36

    >>> calc_xi_numerically(100)
    2850

    >>> calc_xi_numerically(1000)
    130181
    """
    mask = create_mask_procedurally(neighbour_order)
    pivot = mask.shape[0] // 2
    weights = np.arange(pivot, 0, -1)
    submask = mask[:pivot]

    xi = int(np.sum(weights @ submask))
    return xi


def crop_cell_shape(field: np.ndarray, cell_id: int) -> np.ndarray:
    img = (
        cimg._shift_cell_id_to_centre(field, cell_id, size=field.shape[0]) == cell_id
    ).astype(np.uint8)

    xcoords, ycoords = np.where(img != 0)
    x_min, x_max = xcoords.min(), xcoords.max() + 1
    y_min, y_max = ycoords.min(), ycoords.max() + 1
    return img[x_min:x_max, y_min:y_max]


def get_perimeter_estimator(
    neighbour_order: int = 8,
) -> Callable[[np.ndarray, int], float]:
    """
    Create a perimeter estimator function for a given neighbour order.

    A perimeter estimator function estimates the perimeter of a cell in a field. It
    takes the field and the cell id as arguments and returns the estimated perimeter of
    the cell with the given id.

    Args:
        neighbour_order (int, optional): The neighbour order for which to create
            the perimeter estimator. Defaults to 8.

    Returns:
        Callable[[np.ndarray, int], float]: The perimeter estimator function.
    """
    mask = create_mask_procedurally(neighbour_order)
    correction = calc_xi_numerically(neighbour_order)

    def calc_perimeter(
        field: np.ndarray,
        cell_id: int,
    ):
        cell_img = crop_cell_shape(field, cell_id)
        convoluted = convolve2d(cell_img, mask, mode="full", boundary="fill")

        mx, my = mask.shape

        outline = convoluted
        outline[mx // 2 : -mx // 2 + 1, my // 2 : -my // 2 + 1] = np.where(
            cell_img != 0, 0, outline[mx // 2 : -mx // 2 + 1, my // 2 : -my // 2 + 1]
        )

        return np.sum(outline) / correction

    return calc_perimeter


def calc_area(
    field: np.ndarray,
    cell_id: int,
):
    return np.sum(field == cell_id)


def calc_shape_index(
    field: np.ndarray,
    cell_id: int,
    perimeter_estimator: Callable[[np.ndarray, int], float],
):
    perimeter = perimeter_estimator(field, cell_id)
    area = calc_area(field, cell_id)
    return perimeter / area**0.5
