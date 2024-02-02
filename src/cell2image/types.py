import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationFrame:
    step: int
    image: np.ndarray
    cell_type: np.ndarray
    cell_id: np.ndarray
    cluster_id: np.ndarray
