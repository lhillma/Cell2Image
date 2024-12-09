import io
import sys
from pathlib import Path

import click
import ffmpy
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from tqdm import tqdm

from cell2image.image import SimulationFrame, get_cell_outlines, read_vtk_frame


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option(
    "--output", type=click.Path(), default="output.mp4", help="Output file name"
)
@click.option("--cyt-color", default="blue", help="Color of the cytoplasm")
@click.option("--nuc-color", default="green", help="Color of the nucleus")
@click.option("--cyt-type", default=1, help="Cell type id of the cytoplasm")
@click.option("--nuc-type", default=2, help="Cell type id of the nucleus")
@click.option("--scale", default=1, help="Upscale the image")
def main(
    input_dir: str,
    output: str,
    cyt_color: str,
    nuc_color: str,
    cyt_type: int,
    nuc_type: int,
    scale: float,
) -> None:
    input_path = Path(input_dir)
    vtk_files = list(input_path.glob("**/*.vtk"))
    vtk_files.sort()

    _check_color_available(cyt_color)
    _check_color_available(nuc_color)

    render_video_from_file_list(
        vtk_files,
        cyt_type=cyt_type,
        cyt_color=cyt_color,
        nuc_type=nuc_type,
        nuc_color=nuc_color,
        output=output,
        scale=scale,
    )


def render_video_from_file_list(vtk_files: list[Path], **kwargs) -> None:
    """
    Render the video from a list of local vtk files.

    Args:
        vtk_files (list[Path]): List of vtk files
        **kwargs: Keyword arguments to pass to render_video
    """
    frames = [read_vtk_frame(file) for file in vtk_files]
    render_video(frames, **kwargs)


def render_video(
    frames: list[SimulationFrame],
    cyt_type: int = 1,
    cyt_color: str = "blue",
    nuc_type: int = 2,
    nuc_color: str = "green",
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

    n_color = np.array(mcolors.to_rgb(mcolors.CSS4_COLORS.get(nuc_color, nuc_color)))
    c_color = np.array(mcolors.to_rgb(mcolors.CSS4_COLORS.get(cyt_color, cyt_color)))

    images = io.BytesIO()
    for frame in tqdm(frames):
        outlines = get_cell_outlines(frame.cell_id, frame.cell_id.shape)
        cytoplasm = (
            np.ones(outlines.shape + (3,))
            * (frame.cell_type[:, :, None] == cyt_type)
            * c_color[None, None, :]
        )
        nucleus = (
            np.ones(outlines.shape + (3,))
            * (frame.cell_type[:, :, None] == nuc_type)
            * n_color[None, None, :]
        )

        outlines = np.clip(outlines[:, :, None] * (cytoplasm + nucleus), 0, 1)

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


if __name__ == "__main__":
    main()
