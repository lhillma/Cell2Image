import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

from .image import frame_to_image
from .types import SimulationFrame


def list_vtk_files(directory: Path) -> list[Path]:
    """
    Find all vtk files in a directory (recursively) in sorted order.

    Args:
        directory (Path): Directory to search for vtk files

    Returns:
        list[Path]: Sorted list of vtk file paths
    """
    vtk_files = list(directory.glob("**/*.vtk"))
    vtk_files.sort()
    return vtk_files


def check_color_available(color: str) -> None:
    """
    Check if the color is available in matplotlib CSS4 colors and exit if it is not.

    A leading `#` is treated as a hex color code and is always accepted.
    """
    if color.startswith("#"):
        return
    if color in mcolors.CSS4_COLORS:
        return

    print(f"Color {color} is not available in matplotlib CSS4 colors.")
    print("Available colors are:")
    print(list(mcolors.CSS4_COLORS.keys()))
    print()
    print(
        "Check the list of available colors at: https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors"
    )
    sys.exit(1)


def convert_color_to_rgb(color: str) -> np.ndarray:
    """
    Convert a color name or hex code to an RGB numpy array.

    Args:
        color (str): Color name (matplotlib CSS4) or hex code

    Returns:
        np.ndarray: RGB color as a numpy array
    """
    return np.array(mcolors.to_rgb(mcolors.CSS4_COLORS.get(color, color)))


def parse_type_colors(colors: list[tuple[int, str]]) -> dict[int, np.ndarray]:
    """
    Validate and convert a list of (cell_type, color) pairs to an RGB color map.

    Args:
        colors (list[tuple[int, str]]): Cell type and color name/hex pairs

    Returns:
        dict[int, np.ndarray]: Mapping from cell type id to RGB color
    """
    type_colors: dict[int, np.ndarray] = {}
    for cell_type, color in colors:
        check_color_available(color)
        type_colors[cell_type] = convert_color_to_rgb(color)
    return type_colors


def render_frame_image(
    frame: SimulationFrame,
    colors: dict[int, np.ndarray],
    scale: float = 1,
) -> Image.Image:
    """
    Render a SimulationFrame to a (optionally upscaled) PIL image.

    Args:
        frame (SimulationFrame): The frame to render
        colors (dict[int, np.ndarray]): Mapping from cell type id to RGB color
        scale (float, optional): Upscaling factor. Defaults to 1.

    Returns:
        Image.Image: The rendered image
    """
    rgb = frame_to_image(frame, colors)
    image = Image.fromarray((rgb * 255).astype(np.uint8))
    if scale != 1:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.NEAREST,
        )
    return image
