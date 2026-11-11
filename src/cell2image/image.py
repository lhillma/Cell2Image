from pathlib import Path
from typing import Iterable, cast

import h5py
import numpy as np
from numba import njit, uint64

from .neighbourhood import neighbourhood_offsets
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


def read_h5_lattice_snapshots(
    file_path: Path, frame_indices: Iterable[int] | None = None
) -> list[SimulationFrame]:
    """
    Read frames from an h5 file containing lattice snapshots

    Args:
        file_path (Path): Path to the h5 file
        frame_indices (Iterable[int], optional): The positional indices (0-based,
            in file order) of the frames to read. Defaults to None, which reads
            all frames. Only the batches overlapping the requested indices are
            read from disk.

    Returns:
        list[SimulationFrame]: A list of SimulationFrames containing the parsed
            frames, in ascending order of their position in the file
    """
    wanted = None if frame_indices is None else set(frame_indices)

    with h5py.File(file_path, "r") as f:
        frames: list[SimulationFrame] = []
        batch_size = int(f.attrs["batch_size"])
        n_batches = len(f.keys())
        global_index = 0
        for batch_number in range(n_batches):
            batch_name = f"batch_{batch_number}"
            batch = f[batch_name]

            if wanted is None:
                local_indices = list(range(batch_size))
            else:
                local_indices = sorted(
                    i for i in range(batch_size) if (global_index + i) in wanted
                )
                if not local_indices:
                    global_index += batch_size
                    continue

            timesteps: np.ndarray = cast(np.ndarray, batch.attrs["timestep"])
            cell_types = cast(np.ndarray, batch["cell_type"][local_indices])
            cell_ids = cast(np.ndarray, batch["cell_id"][local_indices])
            cluster_ids = cast(np.ndarray, batch["cluster_id"][local_indices])
            for local_i, i in enumerate(local_indices):
                step = timesteps[i]
                cell_type = cell_types[local_i]
                cell_id = cell_ids[local_i]
                cluster_id = cluster_ids[local_i]
                frames.append(SimulationFrame(step, cell_type, cell_id, cluster_id))

            global_index += batch_size

    return frames


def frame_to_image(
    frame: SimulationFrame,
    colors: dict[int, np.ndarray],
    plot_cell_outlines: bool = True,
    plot_nuclear_outlines: bool = True,
    neighbour_order: int = 1,
    line_width: int = 2,
) -> np.ndarray:
    """
    Convert a SimulationFrame to an image with cell outlines

    Args:
        frame (SimulationFrame): The frame to convert
        cyt_type (int): The cell type id of the cytoplasm
        c_color (np.ndarray): The color of the cytoplasm (RGB array)
        nuc_type (int): The cell type id of the nucleus
        n_color (np.ndarray): The color of the nucleus (RGB array)
        plot_cell_outlines (bool, optional): Whether to include cell outlines.
            Defaults to True.
        plot_nuclear_outlines (bool, optional): Whether to include nuclear
            outlines. Defaults to True.
        neighbour_order (int, optional): Neighbour order in which a pixel looks
            for pixels of other cells (see `get_cell_outlines`). Defaults to 1.
        line_width (int, optional): Width of the outlines in pixels (see
            `get_cell_outlines`). Defaults to 2.

    Returns:
        np.ndarray: The image with cell outlines as RGB array
    """
    cell_outlines = get_cell_outlines(
        frame.cluster_id,
        frame.cluster_id.shape,
        neighbour_order=neighbour_order,
        line_width=line_width,
    )
    all_outlines = get_cell_outlines(
        frame.cell_id,
        frame.cell_id.shape,
        neighbour_order=neighbour_order,
        line_width=line_width,
    )
    nuclear_outlines = (all_outlines == cell_outlines).astype(np.uint8)
    outlines = (
        cell_outlines
        if plot_cell_outlines
        else np.ones(cell_outlines.shape, dtype=np.uint8)
    ) * (
        nuclear_outlines
        if plot_nuclear_outlines
        else np.ones(nuclear_outlines.shape, dtype=np.uint8)
    )

    colored_types: list[np.ndarray] = []
    for cell_type_id, color in colors.items():
        colored_types.append(
            np.ones(outlines.shape + (3,))
            * (frame.cell_type[:, :, None] == cell_type_id)
            * color[None, None, :]
        )

    outlines = np.clip(outlines[:, :, None] * sum(colored_types), 0, 1)
    return outlines


def upscale_frame(frame: SimulationFrame, grid_scale: int) -> SimulationFrame:
    """
    Upscale the lattice of a frame by an integer factor

    Every lattice site becomes a `grid_scale` x `grid_scale` block of sites with
    the same cell type, cell id and cluster id. Rendering the upscaled frame
    (instead of upscaling the rendered image) keeps the outlines thin relative to
    the cells, since their width is set in pixels of the rendered lattice.

    Args:
        frame (SimulationFrame): The frame to upscale
        grid_scale (int): Upscaling factor (>= 1)

    Returns:
        SimulationFrame: The upscaled frame (the input frame if grid_scale is 1)
    """
    if grid_scale < 1 or int(grid_scale) != grid_scale:
        raise ValueError(f"grid_scale must be a positive integer, got {grid_scale}")
    if grid_scale == 1:
        return frame

    def up(field: np.ndarray) -> np.ndarray:
        return field.repeat(grid_scale, axis=0).repeat(grid_scale, axis=1)

    return SimulationFrame(
        frame.step, up(frame.cell_type), up(frame.cell_id), up(frame.cluster_id)
    )


def crop_cell_neighbourhood(
    image_in: np.ndarray, cell_ids: np.ndarray, cell_id: int, size: int = 50
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
    return _shift_image_to_centre(image_in, cell_ids, cell_id, size)


def crop_cell_and_neighbours(
    cell_ids: np.ndarray, cell_id: int, neighbour_order=1, size: int = 50
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
    neighbours = get_cell_neighbour_ids(cell_ids, cell_id, neighbour_order)
    return crop_cells_by_id(cell_ids, cell_id, neighbours, size)


def crop_cells_by_id(
    cell_ids: np.ndarray,
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
    s_cell_id = _shift_cell_id_to_centre(cell_ids, central_cell_id, size)

    image = get_cell_outlines(
        s_cell_id, s_cell_id.shape, np.array(list(additional_cells) + [central_cell_id])
    )

    return image


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


def get_motility_labels(cluster_id: np.ndarray, indices: np.ndarray):
    """
    Get the motility labels for a given set of indices

    Args:
        cluster_id (np.ndarray): The cluster ids
        indices (np.ndarray): The indices of the active clusters

    Returns:
        np.ndarray: The motility labels
    """
    cluster_id_sorted = np.sort(np.unique(cluster_id))
    active_clusters = cluster_id_sorted[indices]
    return np.where(np.isin(cluster_id, active_clusters), 1, 0)


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

                # Periodic boundary conditions
                if x == 0 and cell_ids[cell_ids.shape[0] - 1, y] != cell_id:
                    neighbours.add(uint64(cell_ids[cell_ids.shape[0] - 1, y]))
                if x == cell_ids.shape[0] - 1 and cell_ids[0, y] != cell_id:
                    neighbours.add(uint64(cell_ids[0, y]))
                if y == 0 and cell_ids[x, cell_ids.shape[1] - 1] != cell_id:
                    neighbours.add(uint64(cell_ids[x, cell_ids.shape[1] - 1]))
                if y == cell_ids.shape[1] - 1 and cell_ids[x, 0] != cell_id:
                    neighbours.add(uint64(cell_ids[x, 0]))

    if neighbour_order > 1:
        new_neighbours = set()
        for n in neighbours:
            new_neighbours.update(
                get_cell_neighbour_ids(cell_ids, n, neighbour_order - 1)
            )
        neighbours.update(new_neighbours)
    return neighbours


def get_cell_outlines(
    cell_id: np.ndarray,
    dim: tuple[int, int],
    cell_ids: np.ndarray | None = None,
    neighbour_order: int = 1,
    line_width: int = 2,
) -> np.ndarray:
    """
    Extract an image of the cell outlines from a simulation frame.

    In a Cellular Potts model, each pixel is assigned a cell-id value, which
    determines which cell occupies this pixel at a given time. A pixel is part of
    an outline (set to 0, black) if a pixel of another cell lies in its
    neighbourhood; otherwise it is in the interior of a cell and remains 1
    (white).

    The neighbourhood is that of the given neighbour order (1: von Neumann,
    2: Moore, ...; see `cell2image.neighbourhood`), grown by `line_width // 2`
    steps. Both cells along a boundary are marked, so the outline is
    `line_width` pixels wide in the direction of a lattice axis. For an odd
    `line_width`, the cell with the larger id looks one step further, which adds
    the remaining pixel on its side only. With the defaults (neighbour_order=1,
    line_width=2) this marks every pixel with a von Neumann neighbour of another
    cell, giving a 2px wide outline; line_width=1 gives a 1px wide one.

    Pixels outside the array are not considered (no periodic boundaries).

    Args:
        cell_id (np.ndarray): The cell ids
        dim (tuple[int, int]): The dimensions of the image
        cell_ids (np.ndarray, optional): The cell ids to extract the outlines for.
            Defaults to None (all cells).
        neighbour_order (int, optional): Neighbour order in which a pixel looks
            for pixels of other cells. Defaults to 1.
        line_width (int, optional): Width of the outlines in pixels (>= 1).
            Defaults to 2.

    Returns:
        np.ndarray: The extracted image
    """
    if line_width < 1:
        raise ValueError(f"line_width must be >= 1, got {line_width}")
    near = neighbourhood_offsets(neighbour_order, line_width // 2)
    # odd widths: the cell with the larger id also checks one step further
    far = (
        neighbourhood_offsets(neighbour_order, line_width // 2 + 1)
        if line_width % 2
        else np.zeros((0, 2), dtype=np.int64)
    )
    return _cell_outlines(cell_id, dim, cell_ids, near, far)


@njit
def _cell_outlines(
    cell_id: np.ndarray,
    dim: tuple[int, int],
    cell_ids: np.ndarray | None,
    near: np.ndarray,
    far: np.ndarray,
) -> np.ndarray:
    """
    Outline image of `get_cell_outlines`.

    A pixel is an outline pixel if a pixel at one of the `near` offsets belongs to
    another cell, or one at the `far` offsets belongs to a cell with a smaller id.
    """
    img = np.ones(dim, dtype=np.uint8)

    for x in range(dim[0]):
        for y in range(dim[1]):
            own = cell_id[x, y]
            if cell_ids is not None and own not in cell_ids:
                continue
            found = False
            for k in range(near.shape[0]):
                nx = x + near[k, 0]
                ny = y + near[k, 1]
                if 0 <= nx < dim[0] and 0 <= ny < dim[1] and cell_id[nx, ny] != own:
                    found = True
                    break
            if not found:
                for k in range(far.shape[0]):
                    nx = x + far[k, 0]
                    ny = y + far[k, 1]
                    if 0 <= nx < dim[0] and 0 <= ny < dim[1] and cell_id[nx, ny] < own:
                        found = True
                        break
            if found:
                img[x, y] = 0

    return img


def _shift_image_to_centre(
    image: np.ndarray, cell_ids: np.ndarray, cell_id: int, size: int = -1
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
    s_image = image.copy()
    s_cell_id = cell_ids.copy()
    cx, cy = np.where(cell_ids == cell_id)
    dx = cx.max() - cx.min()
    dy = cy.max() - cy.min()

    if dx > cell_ids.shape[0] / 2:
        s_image = np.roll(s_image, int(cell_ids.shape[0] / 2), axis=0)
        s_cell_id = np.roll(s_cell_id, int(cell_ids.shape[0] / 2), axis=0)

    if dy > cell_ids.shape[1] / 2:
        s_image = np.roll(s_image, int(cell_ids.shape[1] / 2), axis=1)
        s_cell_id = np.roll(s_cell_id, int(cell_ids.shape[1] / 2), axis=1)

    cx, cy = np.where(s_cell_id == cell_id)
    cx = cx.mean()
    cy = cy.mean()

    s_image = np.roll(
        np.roll(s_image, int(-cx + size / 2), axis=0), int(-cy + size / 2), axis=1
    )[0:size, 0:size]
    return s_image


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

    Raises:
        ImportError: If vtk is not installed (install cell2image[vtk]).
    """
    try:
        import vtk
        from vtk.numpy_interface import dataset_adapter as dsa
    except ImportError as e:
        raise ImportError(
            "Reading vtk files requires vtk: pip install 'cell2image[vtk]'"
        ) from e

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
