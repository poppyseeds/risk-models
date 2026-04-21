from typing import Dict, List

import numpy as np
import pandas as pd

from preprocessing.contracts import HARDWARE_SEQUENCE_FEATURES, NETWORK_FEATURES, PROCESS_FEATURES


def network_frame(network: Dict[str, float]) -> pd.DataFrame:
    frame = pd.DataFrame([network])
    return frame[NETWORK_FEATURES]


def process_matrix(process_sequence: List[Dict[str, float]], window_size: int) -> np.ndarray:
    frame = pd.DataFrame(process_sequence)[PROCESS_FEATURES]
    values = frame.values.astype(float)
    if len(values) < window_size:
        pad_count = window_size - len(values)
        padding = np.repeat(values[[0]], pad_count, axis=0)
        values = np.vstack([padding, values])
    else:
        values = values[-window_size:]
    return values


def hardware_matrix(hardware_sequence: List[Dict[str, float]], window_size: int) -> np.ndarray:
    frame = pd.DataFrame(hardware_sequence)[HARDWARE_SEQUENCE_FEATURES]
    values = frame.values.astype(float)
    if len(values) < window_size:
        pad_count = window_size - len(values)
        padding = np.repeat(values[[0]], pad_count, axis=0)
        values = np.vstack([padding, values])
    else:
        values = values[-window_size:]
    return values
