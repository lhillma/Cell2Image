# Cell2Image

Python module to convert cell grid data from simulations of the Cellular Potts
model to (binary) images.


## Installation

Install the package with pip:

```bash
pip install git@https://github.com/lhillma/Cell2Image.git
```

To use the video creation functionality, you need to install the `ffmpeg` package.
On Ubuntu, you can install it with:

```bash
sudo apt-get install ffmpeg
```

There is no standard way to install `ffmpeg` on Windows. There is a standalone
installer available here: https://github.com/oop7/ffmpeg-install-guide

## Usage

**Note**: Please install the package first (see above).

### Creating a video in the command line

The `cell2image` package provides a command line interface to create a video from
a list of vtk files. You can use the c2i-video command to create a video from a list
of vtk files:

```bash
c2i-video /path/to/simulation/output --output output.mp4
```

You can also create a video from a h5 data file containing lattice data:

```bash
c2i-video /path/to/lattice.h5 --output output.mp4
```

These commands will create a video `output.mp4` from the vtk files in the directory
`/path/to/simulation/output`. Note that the command finds all vtk files even in
subdirectories, so it is sufficient to point to the top level output directory of your
simulation - no need to add `/cc3d_output/LatticeData` ;)

In order to see a list of available options for colouring the cells, see the output
of `c2i-video --help`:

```bash
> c2i-video --help
Usage: c2i-video [OPTIONS] IN_PATH

Options:
  --output PATH                  Output file name
  -c, --color <INTEGER TEXT>...  Cell type and color pairs (e.g. -c 1 red -c 2
                                 blue)
  --scale INTEGER                Upscale the image
  --help                         Show this message and exit.
```

### Creating a video in Python

You can also create a video from a list of vtk files directly in Python with the
`render_video` function in the `cell2image.video_renderer` module, for example:

```python
from cell2image.video_renderer import render_video

# ... load a list of simulation frames

render_video(frames, type_colors = {1: "blue", 2: "green"}, output='output.mp4')
```
This code will create a video `output.mp4` from the vtk files in the list.
Have a look at the docstring of the `render_video` function for more information on
the available parameters.

### Measuring cell shape properties

You can measure the area, perimeter, and shape index of individual cells using
the `cell2image.shape` module. First, load a simulation frame, then create a
perimeter estimator and compute the metrics:

```python
from pathlib import Path
from cell2image.image import read_h5_lattice_snapshots
from cell2image.shape import (
    get_perimeter_estimator,
    calc_area,
    calc_shape_index,
)

frame_list = read_h5_lattice_snapshots(Path("/path/to/simulation/lattice.h5"))
frame = frame_list[0]  # get the first simulation frame
field = frame.cell_id  # label array where each cell has a unique integer id

# Create a perimeter estimator (default is neighbour_order=8)
perimeter_estimator = get_perimeter_estimator()

# Measure a cell by its id (e.g. cell 100)
cell_id = 100
area = calc_area(field, cell_id)
perimeter_value = perimeter_estimator(field, cell_id)
shape_index = calc_shape_index(field, cell_id, perimeter_estimator)

print(f"Area: {area}, Perimeter: {perimeter_value:.2f}, Shape index: {shape_index:.3f}")
```

The shape index is defined as `perimeter / sqrt(area)`, which provides a
dimensionless measure of how compact a cell is (a circle has the lowest value of
~3.54).
