import numpy as np
import pytest

from cell2image.image import get_cell_outlines, upscale_frame
from cell2image.neighbourhood import neighbourhood_offsets
from cell2image.rendering import render_frame_image
from cell2image.types import SimulationFrame


def outlines_von_neumann(cell_id: np.ndarray) -> np.ndarray:
    """The original outline algorithm: 0 where a 4-neighbour belongs to another cell."""
    img = np.ones(cell_id.shape, dtype=np.uint8)
    nx, ny = cell_id.shape
    for x in range(nx):
        for y in range(ny):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= x + dx < nx and 0 <= y + dy < ny:
                    if cell_id[x + dx, y + dy] != cell_id[x, y]:
                        img[x, y] = 0
    return img


def two_cells(n: int = 10) -> np.ndarray:
    """Cell 1 in the left half (columns < n // 2), cell 2 in the right half."""
    field = np.ones((n, n), dtype=np.int64)
    field[:, n // 2 :] = 2
    return field


def marked_columns(img: np.ndarray) -> list[int]:
    return sorted(set(np.nonzero(img == 0)[1].tolist()))


def test_defaults_match_original_algorithm():
    rng = np.random.default_rng(0)
    field = rng.integers(0, 5, size=(40, 30))
    assert np.array_equal(
        get_cell_outlines(field, field.shape), outlines_von_neumann(field)
    )


@pytest.mark.parametrize(
    "line_width, columns",
    [(1, [5]), (2, [4, 5]), (3, [4, 5, 6]), (4, [3, 4, 5, 6])],
)
def test_line_width_across_straight_boundary(line_width, columns):
    # odd widths put the extra pixel on the side of the larger id (cell 2)
    field = two_cells()
    img = get_cell_outlines(field, field.shape, line_width=line_width)
    assert marked_columns(img) == columns
    # every row is marked the same way
    assert np.all(img == img[:1])


@pytest.mark.parametrize("neighbour_order, n_marked", [(1, 5), (2, 9), (3, 13)])
def test_neighbour_order_around_single_pixel(neighbour_order, n_marked):
    # a single pixel cell: it and its neighbourhood (of the given order) are marked
    field = np.zeros((9, 9), dtype=np.int64)
    field[4, 4] = 1
    img = get_cell_outlines(field, field.shape, neighbour_order=neighbour_order)
    assert np.sum(img == 0) == n_marked


def test_cell_ids_restricts_outlines():
    field = np.zeros((10, 10), dtype=np.int64)
    field[1:4, 1:4] = 1
    field[6:9, 6:9] = 2
    img = get_cell_outlines(field, field.shape, cell_ids=np.array([1]))
    assert np.all(img[5:, 5:] == 1)
    assert np.any(img[:5, :5] == 0)


def test_invalid_line_width():
    field = two_cells()
    with pytest.raises(ValueError):
        get_cell_outlines(field, field.shape, line_width=0)


def test_neighbourhood_offsets():
    assert neighbourhood_offsets(1).tolist() == [[-1, 0], [0, -1], [0, 1], [1, 0]]
    assert len(neighbourhood_offsets(2)) == 8
    assert len(neighbourhood_offsets(3)) == 12
    # two von Neumann steps reach all sites with Manhattan distance <= 2
    assert len(neighbourhood_offsets(1, steps=2)) == 12
    assert len(neighbourhood_offsets(1, steps=0)) == 0


def frame(field: np.ndarray) -> SimulationFrame:
    return SimulationFrame(0, (field > 0).astype(np.int64), field, field)


def test_upscale_frame():
    field = two_cells(4)
    up = upscale_frame(frame(field), 3)
    assert up.cell_id.shape == (12, 12)
    assert np.array_equal(up.cell_id[::3, ::3], field)
    assert np.array_equal(up.cluster_id, up.cell_id)
    with pytest.raises(ValueError):
        upscale_frame(frame(field), 0)


def test_grid_scale_keeps_lines_thin():
    colors = {1: np.array([1.0, 0.0, 0.0])}
    f = frame(two_cells(10))
    image_scale = np.asarray(render_frame_image(f, colors, scale=4))
    grid_scale = np.asarray(render_frame_image(f, colors, grid_scale=4))
    assert image_scale.shape == grid_scale.shape == (40, 40, 3)

    def black_columns(rgb):
        return marked_columns(np.where(rgb.sum(axis=2) == 0, 0, 1))

    # scale upscales the 2px line, grid_scale renders it at 2px
    assert len(black_columns(image_scale)) == 8
    assert len(black_columns(grid_scale)) == 2
