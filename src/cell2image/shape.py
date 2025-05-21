from typing import Callable

import numpy as np
from numba import njit
from scipy.signal import convolve2d

import cell2image.image as cimg


@njit
def create_mask_procedurally(neighbour_order: int) -> np.ndarray:
    """
    Procedurally create a mask for a given neighbour order.
    Args:
        neighbour_order (int): The neighbour order for which to create the mask.
    Returns:
        np.ndarray: The mask for the given neighbour order.
    Examples:
    >>> create_mask_procedurally(1)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])
    >>> create_mask_procedurally(2)
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]])
    """

    # Step through the neighbouhood of a pixel at the origin (0, 0). Due to symmetry,
    # it is sufficient to consider the first quadrant. Add pixels on the boundary that
    # need to be checked in the next step to the open list. A neighbour order of n
    # defines the set of pixels that are nth closest to the origin. Therefore, we keep
    # track of the number of times we see an increase in the distance to the origin.
    positions = []
    open = [(0, 0)]
    max_distance = 0
    counter_distance_increased = 0
    while open:
        x, y = open.pop()
        if (x, y) in positions:
            continue  # skip already visited pixels

        # add current pixel to the list of positions
        positions.append((x, y))

        # check right and top neighbours (first quadrant is sufficient)
        for new_x, new_y in zip([x, x + 1], [y + 1, y]):
            new_distance = np.sqrt(new_x**2 + new_y**2)

            if new_distance <= max_distance:
                # add pixel to the list of positions if it is within the current
                # distance
                open.append((new_x, new_y))
            elif counter_distance_increased < neighbour_order:
                # The pixel is further away than the current distance. However, we
                # still haven't moved to the final distance level, thus add it to the
                # list
                max_distance = new_distance
                counter_distance_increased += 1
                open.append((new_x, new_y))

    positions_np = np.array(positions)

    # Sometimes, we check a pixel that is further away than the true maximum distance.
    # In this case, we need to remove the pixels that are too far away (all pixels
    # closer than that are guaranteed to be checked in the algorithm above).

    # Compuce distances for all points
    distances = np.sqrt(positions_np[:, 0] ** 2 + positions_np[:, 1] ** 2)

    # Find the unique
    bins = np.sort(np.unique(distances))
    if len(bins) > neighbour_order + 1:
        # Cut off
        positions_np = positions_np[distances <= bins[neighbour_order]]

    # construct the mask from the positions
    size = positions_np.max() + 1
    mask_quadrant = np.zeros((size, size), dtype=np.int_)
    for x, y in positions_np:
        mask_quadrant[x, y] = 1

    # Construct the full mask by mirroring the quadrant
    mask = np.zeros((2 * size - 1, 2 * size - 1), dtype=np.int_)
    mask[size - 1 : 2 * size, size - 1 : 2 * size] = mask_quadrant
    mask[size - 1 : 2 * size, 0:size] = mask_quadrant[::, ::-1]
    mask[0:size, size - 1 : 2 * size] = mask_quadrant[::-1, ::]
    mask[0:size, 0:size] = mask_quadrant[::-1, ::-1]

    return mask.astype(np.int_)


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
