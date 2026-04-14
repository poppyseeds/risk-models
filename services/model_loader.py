from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import tensorflow as tf

from config.settings import settings


def _safe_load(load_fn, path: str, state: Dict[str, Any], key: str):
    try:
        if not Path(path).exists():
            state["errors"].append(f"missing file: {path}")
            return None
        return load_fn(path)
    except Exception as exc:
        state["errors"].append(f"failed to load {path}: {exc}")
        return None


def load_inference_assets() -> Dict[str, Any]:
    state: Dict[str, Any] = {"errors": []}

    network_model = _safe_load(
        lambda p: tf.keras.models.load_model(p, compile=False),
        settings.network_model_path,
        state,
        "network_model",
    )
    network_scaler = _safe_load(joblib.load, settings.network_scaler_path, state, "network_scaler")
    network_columns = _safe_load(joblib.load, settings.network_columns_path, state, "network_columns")

    process_model = _safe_load(
        lambda p: tf.keras.models.load_model(p, compile=False),
        settings.process_model_path,
        state,
        "process_model",
    )
    process_scaler = _safe_load(joblib.load, settings.process_scaler_path, state, "process_scaler")
    process_threshold = _safe_load(np.load, settings.process_threshold_path, state, "process_threshold")

    return {
        "network_model": network_model,
        "network_scaler": network_scaler,
        "network_columns": network_columns,
        "process_model": process_model,
        "process_scaler": process_scaler,
        "process_threshold": process_threshold,
        "ready": len(state["errors"]) == 0,
        "errors": state["errors"],
    }


def align_network_columns(df, expected_columns: List[str]):
    aligned = df.copy()
    for col in expected_columns:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[expected_columns]
