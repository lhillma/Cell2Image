from pathlib import Path
import numpy as np
from PIL import Image
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from numba import njit, uint64
from matplotlib import pyplot as plt

from .types import SimulationFrame


def crop_cell_neighbourhood(
    frame: SimulationFrame, cell_id: int, size: int = 50
) -> np.ndarray:
    """Get an image of a single cell"""
    cx, cy = np.where(frame.cell_id == cell_id)
    cx = int(cx.mean())
    cy = int(cy.mean())

    image = np.roll(np.roll(frame.image.copy(), cx, axis=0), cy, axis=1)[0:size, 0:size]

    return image[:, :, np.newaxis] * np.array([255, 255, 255], dtype=np.uint8)


def get_cell_type(frame: SimulationFrame, cell_id: int) -> int:
    """Get the cell type of a cell"""
    return frame.cell_type[frame.cell_id == cell_id][0]


def crop_cell_and_neighbours(
    frame: SimulationFrame, cell_id: int, n_neighbours=1, size: int = 50
) -> np.ndarray:
    """Get an image of a single cell and its neighbors"""
    neighbours = get_cell_neighbour_ids(frame.cell_id, cell_id, n_neighbours)
    return crop_single_cell(frame, cell_id, size, neighbours)

    # # crop a 50x50 image around the cell's center of mass
    # cx, cy = np.where(frame.cell_id == cell_id)
    # cx = cx.mean()
    # cy = cy.mean()
    # image = np.prod(
    #     [_get_single_cell_outline(frame, n, (cx, cy), size) for n in neighbours], axis=0
    # )

    # return image[:, :, np.newaxis] * np.array([255, 255, 255], dtype=np.uint8)


@njit
def get_cell_neighbour_ids(
    cell_ids: np.ndarray, cell_id: int, n_neighbours=1
) -> set[int]:
    neighbours = set()

    if n_neighbours == 0:
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

    if n_neighbours > 1:
        new_neighbours = set()
        for n in neighbours:
            new_neighbours.update(get_cell_neighbour_ids(cell_ids, n, n_neighbours - 1))
        neighbours.update(new_neighbours)
    return neighbours


def crop_single_cell(
    frame: SimulationFrame,
    cell_id: int,
    size: int = 50,
    neighbours: set = set(),
) -> np.ndarray:
    """Get an image of a single cell"""
    s_cell_id = centre_cell_id(frame.cell_id, cell_id, size)

    image = get_cell_outlines(
        s_cell_id, s_cell_id.shape, np.array(list(neighbours) + [cell_id])
    )

    return image[:, :, np.newaxis] * np.array([255, 255, 255], dtype=np.uint8)


def centre_cell_id(cell_ids: np.ndarray, cell_id: int, size: int = -1) -> np.ndarray:
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


def _get_single_cell_outline(frame, cell_id, center, size):
    cell = np.zeros(frame.image.shape[:2])
    cell[frame.cell_id == cell_id] = 1
    image = get_cell_outlines(cell, frame.image.shape[:2])

    # crop a 50x50 image around the cell's center of mass
    s_cell_id = frame.cell_id.copy()
    cx, cy = np.where(s_cell_id == cell_id)
    if center is None:
        cx_s = cx.mean()
        cy_s = cy.mean()
    else:
        cx_s = center[0]
        cy_s = center[1]

    dx = cx.max() - cx.min()
    dy = cy.max() - cy.min()

    if dx > frame.image.shape[0] / 2:
        s_cell_id = np.roll(s_cell_id, int(frame.image.shape[0] / 2), axis=0)
        image = np.roll(image, int(frame.image.shape[0] / 2), axis=0)
        cx_s += int(frame.image.shape[0] / 2)

    if dy > frame.image.shape[1] / 2:
        s_cell_id = np.roll(s_cell_id, int(frame.image.shape[1] / 2), axis=1)
        image = np.roll(image, int(frame.image.shape[1] / 2), axis=1)
        cy_s += int(frame.image.shape[1] / 2)

    # cx, cy = np.where(s_cell_id == cell_id)
    # cx = cx.mean()
    # cy = cy.mean()

    # roll image such that the cell's center of mass is in the center of the image
    image = np.roll(
        np.roll(image, int(-cx_s + size / 2), axis=0), int(-cy_s + size / 2), axis=1
    )[0:size, 0:size]
    return image


def draw_cell_outlines(frame: SimulationFrame):
    """Draw the outlines of the cells in the frame"""
    cell_id = np.flip(frame.cell_id.reshape(frame.image.shape[:2], order="C"), axis=0)
    image = get_cell_outlines(cell_id, frame.image.shape[:2])
    plt.imshow(image)


@njit
def get_cell_outlines(
    cell_id: np.ndarray, dim: tuple[int, int], cell_ids: np.ndarray | None = None
) -> np.ndarray:
    img = np.ones(dim, dtype=np.uint8)

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


def track_cell_by_type(frame: SimulationFrame, cell_type: int) -> SimulationFrame:
    """Track a cell by its cell_id"""
    ret_frame = SimulationFrame(
        frame.step,
        frame.image.copy(),
        frame.cell_type.copy(),
        frame.cell_id.copy(),
        frame.cluster_id.copy(),
    )

    ret_frame.image[
        (frame.cell_type == cell_type) & (frame.image[:, :, 0] != 0)
    ] = np.array([255, 0, 0])

    return ret_frame


def track_cell_by_index(frame: SimulationFrame, cell_id: int) -> SimulationFrame:
    """Track a cell by its cell_id"""
    ret_frame = SimulationFrame(
        frame.step,
        frame.image.copy(),
        frame.cell_type.copy(),
        frame.cell_id.copy(),
        frame.cluster_id.copy(),
    )

    ret_frame.image[
        (frame.cell_id == cell_id) & (frame.image[:, :, 0] != 0)
    ] = np.array([255, 0, 0])

    return ret_frame


def read_vtk_frame(vtk_path: Path) -> SimulationFrame:
    step = get_step(vtk_path)
    cell_type, cell_id, cluster_id = read_vtk(vtk_path)
    image = get_cell_outlines(np.array(cell_id), cell_id.shape[:2])
    return SimulationFrame(step, image, cell_type, cell_id, cluster_id)


def read_simulation_frame(
    img_path: Path, vtk_path: Path, crop: tuple[int, int, int, int] = (0, 0, -1, -1)
) -> SimulationFrame:
    step = get_step(img_path)
    image = read_image(img_path, crop)
    cell_type, cell_id, cluster_id = read_vtk(vtk_path)
    return SimulationFrame(step, image, cell_type, cell_id, cluster_id)


def read_vtk(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def read_image(path, crop: tuple[int, int, int, int] = (0, 0, -1, -1)) -> np.ndarray:
    im_frame = Image.open(path)
    return np.array(im_frame)[crop[0] : crop[2], crop[1] : crop[3]]


def get_step(path: Path) -> int:
    return int(path.stem.split("_")[-1])
