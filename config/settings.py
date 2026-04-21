import os
from dataclasses import dataclass


@dataclass
class Settings:
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    model_dir: str = os.getenv("MODEL_DIR", "models")
    network_model_path: str = os.getenv("NETWORK_MODEL_PATH", "models/ram_model.h5")
    network_scaler_path: str = os.getenv("NETWORK_SCALER_PATH", "models/scaler.pkl")
    network_columns_path: str = os.getenv("NETWORK_COLUMNS_PATH", "models/columns.pkl")
    process_model_path: str = os.getenv("PROCESS_MODEL_PATH", "models/lstm_model.h5")
    process_scaler_path: str = os.getenv("PROCESS_SCALER_PATH", "models/lstm_scaler.pkl")
    process_threshold_path: str = os.getenv("PROCESS_THRESHOLD_PATH", "models/threshold.npy")
    process_window_size: int = int(os.getenv("PROCESS_WINDOW_SIZE", "10"))
    hardware_model_path: str = os.getenv("HARDWARE_MODEL_PATH", "models/hardware_model.keras")
    hardware_scaler_path: str = os.getenv("HARDWARE_SCALER_PATH", "models/hardware_scaler.pkl")
    hardware_threshold_path: str = os.getenv("HARDWARE_THRESHOLD_PATH", "models/hardware_threshold.npy")
    hardware_window_size: int = int(os.getenv("HARDWARE_WINDOW_SIZE", os.getenv("PROCESS_WINDOW_SIZE", "10")))
    history_db_path: str = os.getenv("HISTORY_DB_PATH", "data/incidents.db")
    top_signals_count: int = int(os.getenv("TOP_SIGNALS_COUNT", "5"))
    # Blend NN output with heuristics so demos differ when models are weak/untrained
    score_blend_network_model: float = float(os.getenv("SCORE_BLEND_NETWORK_MODEL", "0.35"))
    score_blend_process_model: float = float(os.getenv("SCORE_BLEND_PROCESS_MODEL", "0.35"))
    score_blend_hardware_model: float = float(os.getenv("SCORE_BLEND_HARDWARE_MODEL", "0.45"))
    # process loss / (threshold * mult) capped at 1 before blending
    process_loss_saturation_mult: float = float(os.getenv("PROCESS_LOSS_SATURATION_MULT", "20.0"))
    hardware_loss_saturation_mult: float = float(os.getenv("HARDWARE_LOSS_SATURATION_MULT", "12.0"))


settings = Settings()
