import numpy as np

from cell2image import image as cimg


def get_adjacency_from_labelled_image(img: np.ndarray):
    cell_ids = np.unique(img)
    adjacency = {
        cell_id: cimg.get_cell_neighbour_ids(img, cell_id) for cell_id in cell_ids
    }
    return adjacency
