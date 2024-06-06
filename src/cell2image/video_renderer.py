import io
import ffmpy
import click
from pathlib import Path
from PIL import Image

from .image import get_cell_outlines, read_vtk_frame


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
def main(input_dir: str):
    input_path = Path(input_dir)
    vtk_files = list(input_path.glob("**/*.vtk"))
    vtk_files.sort()

    images = io.BytesIO()

    for i, vtk_file in enumerate(vtk_files):
        frame = read_vtk_frame(vtk_file)
        outlines = get_cell_outlines(frame.cell_id, frame.cell_id.shape)
        image = Image.fromarray(outlines * 255)
        image.save(images, format="png")

    ff = ffmpy.FFmpeg(
        inputs={"pipe:0": "-y -f image2pipe -r 25"},
        outputs={"output.mp4": None},
    )
    ff.run(input_data=images.getbuffer())


if __name__ == "__main__":
    main()
