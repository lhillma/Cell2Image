# Cell2Image

Python module to convert cell grid data from simulations of the Cellular Potts
model to (binary) images.


## Installation

Clone the repository and install the package with pip:

```bash
git clone git@gitlab.tue.nl:smb/cell-migration/cell2image.git
```

Navigate to the root of the repository and install the package with pip:

```bash
pip install .
```

To use the video creation functionality, you need to install the `ffmpeg` package.
On Ubuntu, you can install it with:

```bash
sudo apt-get install ffmpeg
```

**Note**: If you'd like to install the package in development mode, use the `-e`
flag with pip:

```bash
pip install -e .
```

That way, changes to the source code will be automatically reflected in the
installed package upon save (refer to the
[pip documentation](https://pip.pypa.io/en/stable/cli/pip_install/#options) and
[setuptools documentation](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
for more information).

## Usage

**Note**: Please install the package first (see above).

### Creating a video in the command line

The `cell2image` package provides a command line interface to create a video from
a list of vtk files. You can use the c2i-video command to create a video from a list
of vtk files:

```bash
c2i-video /path/to/simulation/output --output output.mp4
```

This command will create a video `output.mp4` from the vtk files in the directory
`/path/to/simulation/output`. Note that the command finds all vtk files even in
subdirectories, so it is sufficient to point to the top level output directory of your
simulation - no need to add `/cc3d_output/LatticeData` ;)

In order to see a list of available options for colouring the cells, see the output
of `c2i-video --help`:

```bash
> c2i-video --help
Usage: c2i-video [OPTIONS] INPUT_DIR

Options:
  --output PATH       Output file name
  --cyt-color TEXT    Color of the cytoplasm
  --nuc-color TEXT    Color of the nucleus
  --cyt-type INTEGER  Cell type id of the cytoplasm
  --nuc-type INTEGER  Cell type id of the nucleus
  --help              Show this message and exit.
```

### Creating a video in Python

You can also create a video from a list of vtk files directly in Python with the
`render_video` function in the `cell2image.video_renderer` module, for example:

```python
from cell2image.video_renderer import render_video

# List of vtk files
vtk_files = ['file1.vtk', 'file2.vtk', 'file3.vtk']

render_video(vtk_files, output='output.mp4')
```
This code will create a video `output.mp4` from the vtk files in the list.
Have a look at the docstring of the `render_video` function for more information on
the available parameters.
