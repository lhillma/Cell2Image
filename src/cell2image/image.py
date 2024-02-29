from pathlib import Path
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from numba import njit, uint64

from .types import SimulationFrame


def read_vtk_frame(vtk_path: Path) -> SimulationFrame:
    """
    Read a single frame from a vtk file

    Parse the vtk file into numpy arrays and pack them into a SimulationFrame dataclass.

    Args:
        vtk_path (Path): Path to the vtk file

    Returns:
        SimulationFrame: A dataclass containing the parsed frame
    """
    step = _get_step_from_path(vtk_path)
    cell_type, cell_id, cluster_id = _vtk_to_numpy(vtk_path)
    return SimulationFrame(step, cell_type, cell_id, cluster_id)


def crop_cell_neighbourhood(
    image_in: np.ndarray, frame: SimulationFrame, cell_id: int, size: int = 50
) -> np.ndarray:
    """
    Crop a size x size area from `image_in` around a cell and its neighbours

    This requires `image_in` to correspond to the same frame as `frame`. It also
    requires frame.cell_id to be a 2D array of the same shape as `image_in`. Finally,
    the cell_id must be present in `frame.cell_id`.

    Args:
        image_in (np.ndarray): The image to crop from
        frame (SimulationFrame): The frame containing the cell data
        cell_id (int): The id of the cell to crop around
        size (int, optional): The size of the cropped image. Defaults to 50.

    Returns:
        np.ndarray: The cropped image
    """
    cx, cy = np.where(frame.cell_id == cell_id)
    cx = int(cx.mean())
    cy = int(cy.mean())

    image = np.roll(np.roll(image_in.copy(), cx, axis=0), cy, axis=1)[0:size, 0:size]

    return image[:, :, np.newaxis] * np.array([255, 255, 255], dtype=np.uint8)


def crop_cell_and_neighbours(
    frame: SimulationFrame, cell_id: int, neighbour_order=1, size: int = 50
) -> np.ndarray:
    """
    Crop a size x size area from `image_in` around a cell and its neighbours

    This differs from `crop_cell_neighbourhood` in that it does not only crop the entire
    image, which could contain arbitrary cells, but only the cells with the given id and
    its neighbours up to a given order. This is useful for training a model to classify
    cells based on their local environment.

    Args:
        frame (SimulationFrame): The frame containing the cell data
        cell_id (int): The id of the cell to crop around
        neighbour_order (int, optional): The order of neighbours to include. Defaults
            to 1 for direct neighbours (first order).
        size (int, optional): The size of the cropped image. Defaults to 50.

    Returns:
        np.ndarray: The cropped image
    """
    neighbours = get_cell_neighbour_ids(frame.cell_id, cell_id, neighbour_order)
    return crop_cells_by_id(frame, cell_id, neighbours, size)


def crop_cells_by_id(
    frame: SimulationFrame,
    central_cell_id: int,
    additional_cells: set = set(),
    size: int = 50,
) -> np.ndarray:
    """
    Crop a size x size area from `image_in` with the given cell in the center

    Args:
        frame (SimulationFrame): The frame containing the cell data
        central_cell_id (int): The id of the cell to crop around
        additional_cells (set, optional): Additional cells to include in the cropped
            image. Defaults to empty set().
        size (int, optional): The size of the cropped image. Defaults to 50.

    Returns:
        np.ndarray: The cropped image
    """
    s_cell_id = _shift_cell_id_to_centre(frame.cell_id, central_cell_id, size)

    image = get_cell_outlines(
        s_cell_id, s_cell_id.shape, np.array(list(additional_cells) + [central_cell_id])
    )

    return image[:, :, np.newaxis] * np.array([255, 255, 255], dtype=np.uint8)


def get_cell_type_by_id(frame: SimulationFrame, cell_id: int) -> int:
    """
    Get the cell type of a cell by its id

    Args:
        frame (SimulationFrame): The frame containing the cell data
        cell_id (int): The id of the cell to get the type of

    Returns:
        int: The type of the cell
    """
    return frame.cell_type[frame.cell_id == cell_id][0]


@njit
def get_cell_neighbour_ids(
    cell_ids: np.ndarray, cell_id: int, neighbour_order=1
) -> set[int]:
    """
    Get the ids of the neighbours of a cell up to a given order

    Args:
        cell_ids (np.ndarray): The cell ids
        cell_id (int): The id of the cell to get the neighbours of
        neighbour_order (int, optional): The order of neighbours to include. Defaults
            to 1 for direct neighbours (first order).

    Returns:
        set[int]: The ids of the neighbours
    """
    neighbours = set()

    if neighbour_order == 0:
        return neighbours

    for x in range(cell_ids.shape[0]):
        for y in range(cell_ids.shape[1]):
            if cell_ids[x, y] == cell_id:
                if x > 0 and cell_ids[x - 1, y] != cell_id:
                    neighbours.add(uint64(cell_ids[x - 1, y]))
                if x < cell_ids.shape[0] - 1 and cell_ids[x + 1, y] != cell_id:
                    neighbours.add(uint64(cell_ids[x + 1, y]))
                if y > 0 and cell_ids[x, y - 1] != cell_id:
                    neighbours.add(uint64(cell_ids[x, y - 1]))
                if y < cell_ids.shape[1] - 1 and cell_ids[x, y + 1] != cell_id:
                    neighbours.add(uint64(cell_ids[x, y + 1]))

    if neighbour_order > 1:
        new_neighbours = set()
        for n in neighbours:
            new_neighbours.update(
                get_cell_neighbour_ids(cell_ids, n, neighbour_order - 1)
            )
        neighbours.update(new_neighbours)
    return neighbours


@njit
def get_cell_outlines(
    cell_id: np.ndarray, dim: tuple[int, int], cell_ids: np.ndarray | None = None
) -> np.ndarray:
    img = np.ones(dim, dtype=np.uint8)
    """
    Extract an image of the cell outlines from a simulation frame.

    In a Cellular Potts model, each pixel is assigned a cell-id value, which
    determines which cell occupies this pixel at a given time. In order to
    determine the outlines, this function takes the cell-id values and compares
    the neighbouring cell-ids with the cell-id at each pixel. If all neighbours
    have the same cell-id as the pixel itself, the pixel is in the center of a
    cell and thus remains white. If there is a neighbour with a different
    cell-id, the pixel is part of the boundary and therefore set to black.
    Applying this function results in a 2px wide boundary of the cells.

    Args:
        cell_id (np.ndarray): The cell ids
        dim (tuple[int, int]): The dimensions of the image
        cell_ids (np.ndarray, optional): The cell ids to extract the outlines for.
            Defaults to None.

    Returns:
        np.ndarray: The extracted image
    """

    for x in range(dim[0]):
        for y in range(dim[1]):
            if cell_ids is not None and cell_id[x, y] not in cell_ids:
                img[x, y] = 1
                continue
            if x > 0 and cell_id[x - 1, y] != cell_id[x, y]:
                img[x, y] = 0
                continue
            if x < dim[0] - 1 and cell_id[x + 1, y] != cell_id[x, y]:
                img[x, y] = 0
                continue
            if y > 0 and cell_id[x, y - 1] != cell_id[x, y]:
                img[x, y] = 0
                continue
            if y < dim[1] - 1 and cell_id[x, y + 1] != cell_id[x, y]:
                img[x, y] = 0
                continue

    return img


def _shift_cell_id_to_centre(
    cell_ids: np.ndarray, cell_id: int, size: int = -1
) -> np.ndarray:
    """
    Shift the cell ids such that the given cell is in the center of the image

    Args:
        cell_ids (np.ndarray): The cell ids
        cell_id (int): The id of the cell to shift to the center
        size (int, optional): The size of the image. Defaults to -1.

    Returns:
        np.ndarray: The shifted cell ids
    """
    if size < 0:
        size = np.min(cell_ids.shape)
    s_cell_id = cell_ids.copy()
    cx, cy = np.where(cell_ids == cell_id)
    dx = cx.max() - cx.min()
    dy = cy.max() - cy.min()

    if dx > cell_ids.shape[0] / 2:
        s_cell_id = np.roll(s_cell_id, int(cell_ids.shape[0] / 2), axis=0)

    if dy > cell_ids.shape[1] / 2:
        s_cell_id = np.roll(s_cell_id, int(cell_ids.shape[1] / 2), axis=1)

    cx, cy = np.where(s_cell_id == cell_id)
    cx = cx.mean()
    cy = cy.mean()

    s_cell_id = np.roll(
        np.roll(s_cell_id, int(-cx + size / 2), axis=0), int(-cy + size / 2), axis=1
    )[0:size, 0:size]
    return s_cell_id


def _vtk_to_numpy(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a vtk file into numpy arrays

    Args:
        path (Path): Path to the vtk file

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The parsed cell type, cell id and
            cluster id
    """
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(path.as_posix())
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    data = reader.GetOutput()
    cell_grid = dsa.WrapDataObject(data)

    cell_type = cell_grid.PointData["CellType"]
    cell_id = cell_grid.PointData["CellId"]
    cluster_id = cell_grid.PointData["ClusterId"]

    shape = (int(np.sqrt(cell_type.size)), int(np.sqrt(cell_type.size)))

    return (
        np.flip(np.array(cell_type).reshape(shape), axis=0),
        np.flip(np.array(cell_id).reshape(shape), axis=0),
        np.flip(np.array(cluster_id).reshape(shape), axis=0),
    )


def _get_step_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])
