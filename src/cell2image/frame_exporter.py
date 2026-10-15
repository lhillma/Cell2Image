import sys
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from cell2image.image import read_h5_lattice_snapshots, read_vtk_frame
from cell2image.rendering import list_vtk_files, parse_type_colors, render_frame_image
from cell2image.types import SimulationFrame


@click.command()
@click.argument(
    "in_path",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="frames",
    help="Output directory for rendered images",
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
@click.option(
    "--frame",
    "frame_index",
    type=int,
    default=None,
    help=(
        "0-based position (in file order) of a single frame to export. Required "
        "in multi-frame mode (a directory of vtk files or an h5 file). Mutually "
        "exclusive with --range."
    ),
)
@click.option(
    "--range",
    "frame_range",
    type=(int, int),
    default=None,
    help=(
        "0-based, inclusive START END range of frame positions to export. Required "
        "in multi-frame mode (a directory of vtk files or an h5 file). Mutually "
        "exclusive with --frame."
    ),
)
def main(
    in_path: str,
    output_dir: str,
    colors: list[tuple[int, str]],
    scale: float,
    frame_index: int | None,
    frame_range: tuple[int, int] | None,
) -> None:
    input_path = Path(in_path)
    out_dir = Path(output_dir)

    if frame_index is not None and frame_range is not None:
        print("Specify either --frame or --range, not both.")
        sys.exit(1)

    type_colors = parse_type_colors(list(colors))

    if input_path.is_file() and input_path.suffix == ".h5":
        indices = _require_indices(
            frame_index, frame_range, "exporting frames from an h5 file"
        )
        export_frames_from_h5_file(input_path, indices, type_colors, out_dir, scale)
    elif input_path.is_file() and input_path.suffix == ".vtk":
        if frame_index is not None or frame_range is not None:
            print(
                "--frame and --range are not valid when exporting a single vtk file."
            )
            sys.exit(1)
        export_single_vtk_file(input_path, type_colors, out_dir, scale)
    elif input_path.is_dir():
        indices = _require_indices(
            frame_index, frame_range, "exporting frames from a directory of vtk files"
        )
        export_frames_from_vtk_directory(
            input_path, indices, type_colors, out_dir, scale
        )
    else:
        print(
            "Input path must be a single vtk file, a directory containing vtk "
            "files, or a single h5 file."
        )
        sys.exit(1)


def export_single_vtk_file(
    vtk_path: Path,
    type_colors: dict[int, np.ndarray],
    out_dir: Path,
    scale: float,
) -> None:
    """
    Export a single frame read from one vtk file as an image.

    Args:
        vtk_path (Path): Path to the vtk file
        type_colors (dict[int, np.ndarray]): Mapping from cell type id to RGB color
        out_dir (Path): Directory to write the rendered image to
        scale (float): Upscaling factor
    """
    frame = read_vtk_frame(vtk_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_frame(frame, type_colors, out_dir, scale)


def export_frames_from_vtk_directory(
    input_path: Path,
    indices: set[int],
    type_colors: dict[int, np.ndarray],
    out_dir: Path,
    scale: float,
) -> None:
    """
    Export a selection of frames read from a directory of vtk files as images.

    Args:
        input_path (Path): Directory containing the vtk files
        indices (set[int]): 0-based positions (in sorted file order) of the
            frames to export
        type_colors (dict[int, np.ndarray]): Mapping from cell type id to RGB color
        out_dir (Path): Directory to write the rendered images to
        scale (float): Upscaling factor
    """
    vtk_files = list_vtk_files(input_path)
    _validate_indices(indices, len(vtk_files))

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(sorted(indices)):
        frame = read_vtk_frame(vtk_files[i])
        _save_frame(frame, type_colors, out_dir, scale)


def export_frames_from_h5_file(
    input_path: Path,
    indices: set[int],
    type_colors: dict[int, np.ndarray],
    out_dir: Path,
    scale: float,
) -> None:
    """
    Export a selection of frames read from a single h5 lattice file as images.

    Args:
        input_path (Path): Path to the h5 file
        indices (set[int]): 0-based positions (in file order) of the frames to export
        type_colors (dict[int, np.ndarray]): Mapping from cell type id to RGB color
        out_dir (Path): Directory to write the rendered images to
        scale (float): Upscaling factor
    """
    frames = read_h5_lattice_snapshots(input_path, frame_indices=indices)
    if len(frames) < len(indices):
        print(
            f"Warning: only found {len(frames)} of the {len(indices)} requested "
            f"frame(s) in {input_path}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in tqdm(frames):
        _save_frame(frame, type_colors, out_dir, scale)


def _save_frame(
    frame: SimulationFrame,
    type_colors: dict[int, np.ndarray],
    out_dir: Path,
    scale: float,
) -> None:
    image = render_frame_image(frame, type_colors, scale)
    image.save(out_dir / f"frame_{frame.step}.png")


def _require_indices(
    frame_index: int | None, frame_range: tuple[int, int] | None, context: str
) -> set[int]:
    if frame_index is None and frame_range is None:
        print(f"--frame or --range is required when {context}.")
        sys.exit(1)
    if frame_index is not None:
        return {frame_index}
    start, end = frame_range  # type: ignore[misc]
    return set(range(start, end + 1))


def _validate_indices(indices: set[int], n_frames: int) -> None:
    for i in indices:
        if i < 0 or i >= n_frames:
            print(f"Frame index {i} is out of range (0-{n_frames - 1}).")
            sys.exit(1)


if __name__ == "__main__":
    main()
