"""
Lattice neighbourhoods of a given neighbour order.

A neighbour order of n contains all sites that are among the n smallest distinct
(Euclidean) distances from a site: order 1 is the von Neumann neighbourhood (4
sites), order 2 the Moore neighbourhood (8 sites), order 3 adds the sites at
distance 2 (12 sites), and so on.
"""

from functools import lru_cache

import numpy as np
from numba import njit


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


@lru_cache(maxsize=None)
def neighbourhood_offsets(neighbour_order: int, steps: int = 1) -> np.ndarray:
    """
    Offsets of all sites reachable in `steps` steps through a neighbourhood.

    One step goes to any site of the neighbourhood of the given order, so
    `steps` = 1 gives the neighbourhood itself and larger values grow it (the
    `steps`-fold dilation of the neighbourhood mask). The origin is excluded.

    Args:
        neighbour_order (int): Neighbour order of a single step (>= 1)
        steps (int, optional): Number of steps (>= 0). Defaults to 1.

    Returns:
        np.ndarray: Integer array of shape (n, 2) with the (dx, dy) offsets,
            sorted by distance from the origin. Empty for `steps` = 0.

    Examples:

    >>> neighbourhood_offsets(1).tolist()
    [[-1, 0], [0, -1], [0, 1], [1, 0]]

    >>> len(neighbourhood_offsets(2)), len(neighbourhood_offsets(1, steps=2))
    (8, 12)
    """
    if neighbour_order < 1:
        raise ValueError(f"neighbour_order must be >= 1, got {neighbour_order}")
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")

    mask = create_mask_procedurally(neighbour_order)
    r = mask.shape[0] // 2
    step = [(x - r, y - r) for x, y in zip(*np.nonzero(mask))]

    reachable = {(0, 0)}
    for _ in range(steps):
        reachable = {(x + dx, y + dy) for x, y in reachable for dx, dy in step}
    reachable.discard((0, 0))

    offsets = sorted(reachable, key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    return np.array(offsets, dtype=np.int64).reshape(-1, 2)
