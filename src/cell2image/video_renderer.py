import io
import sys
from pathlib import Path

import click
import ffmpy
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from tqdm import tqdm

from cell2image.image import (
    SimulationFrame,
    frame_to_image,
    read_h5_lattice_snapshots,
    read_vtk_frame,
)


@click.command()
@click.argument(
    "in_path",
    type=click.Path(exists=True),
)
@click.option(
    "--output", type=click.Path(), default="output.mp4", help="Output file name"
)
@click.option(
    "--color",
    "-c",
    "colors",
    type=(int, str),
    multiple=True,
    help="Cell type and color pairs (e.g. -c 1 red -c 2 blue)",
    default=[(1, "blue"), (2, "green")],
)
@click.option("--scale", default=1, help="Upscale the image")
def main(
    in_path: str,
    output: str,
    colors: list[tuple[int, str]],
    scale: float,
) -> None:
    input_path = Path(in_path)

    type_colors = {c_type: color for c_type, color in colors}

    if input_path.is_file() and input_path.suffix == ".h5":
        render_video_from_h5_file(
            input_path,
            type_colors=type_colors,
            output=output,
            scale=scale,
        )
    elif input_path.is_dir():
        render_video_from_vtk_directory(
            input_path,
            type_colors=type_colors,
            output=output,
            scale=scale,
        )
    else:
        print(
            "Input path must be a directory containing vtk files or a single h5 file."
        )
        sys.exit(1)


def render_video_from_vtk_directory(
    input_path: Path,
    type_colors: dict[int, str],
    output: str,
    scale: float,
) -> None:
    """
    Render the video from a list of local vtk files.

    Args:
        vtk_files (list[Path]): List of vtk files
        **kwargs: Keyword arguments to pass to render_video
    """
    vtk_files = list(input_path.glob("**/*.vtk"))
    vtk_files.sort()
    frames = [read_vtk_frame(file) for file in vtk_files]
    render_video(frames, type_colors=type_colors, output=output, scale=scale)


def render_video_from_h5_file(
    input_path: Path,
    type_colors: dict[int, str],
    output: str,
    scale: float,
) -> None:
    """
    Render the video from a single h5 file.
    Args:
        input_path (Path): Path to the h5 file
        **kwargs: Keyword arguments to pass to render_video
    """
    frames = read_h5_lattice_snapshots(input_path)
    render_video(frames, type_colors=type_colors, output=output, scale=scale)


def render_video(
    frames: list[SimulationFrame],
    type_colors: dict[int, str],
    output: str = "output.mp4",
    scale: float = 1,
) -> None:
    """
    Render the video from the vtk files.

    Args:
        vtk_files (list[Path]): List of vtk files
        cyt_type (int, optional): Cell type id of the cytoplasm. Defaults to 1.
        cyt_color (str, optional): Color of the cytoplasm. Defaults to "green".
        nuc_type (int, optional): Cell type id of the nucleus. Defaults to 2.
        nuc_color (str, optional): Color of the nucleus. Defaults to "green".
        output (str, optional): Output file name. Defaults to "output.mp4".
    """

    colors = {
        c_type: _convert_color_to_rgb(color) for c_type, color in type_colors.items()
    }

    images = io.BytesIO()
    for frame in tqdm(frames):
        outlines = frame_to_image(frame, colors)

        image = Image.fromarray((outlines * 255).astype(np.uint8))
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.NEAREST,
        )
        image.save(images, format="png")

    ff = ffmpy.FFmpeg(
        inputs={"pipe:0": "-y -f image2pipe -r 25"},
        outputs={output: "-c:v libx264 -pix_fmt yuv420p -movflags faststart"},
    )
    ff.run(input_data=images.getbuffer())


def _check_color_available(color: str) -> None:
    """
    Check if the color is available in matplotlib CSS4 colors and exit if it is not.
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


def _convert_color_to_rgb(color: str) -> np.ndarray:
    """
    Convert a color string or a numpy array to an RGB numpy array.
    Args:
        color (str | list[float]): Color string or list of RGB values
    Returns:
        np.ndarray: RGB color as a numpy array
    """
    return np.array(mcolors.to_rgb(mcolors.CSS4_COLORS.get(color, color)))


if __name__ == "__main__":
    main()
