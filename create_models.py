from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing.contracts import HARDWARE_SEQUENCE_FEATURES, NETWORK_FEATURES, PROCESS_FEATURES


def train_scalers_from_csv(csv_path: str) -> None:
    frame = pd.read_csv(csv_path)
    missing_net = [c for c in NETWORK_FEATURES if c not in frame.columns]
    missing_proc = [c for c in PROCESS_FEATURES if c not in frame.columns]
    if missing_net or missing_proc:
        raise ValueError(f"CSV missing required columns. network={missing_net}, process={missing_proc}")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    network_scaler = StandardScaler().fit(frame[NETWORK_FEATURES].values)
    process_scaler = StandardScaler().fit(frame[PROCESS_FEATURES].values)

    joblib.dump(network_scaler, model_dir / "scaler.pkl")
    joblib.dump(NETWORK_FEATURES, model_dir / "columns.pkl")
    joblib.dump(process_scaler, model_dir / "lstm_scaler.pkl")

    missing_hw = [c for c in HARDWARE_SEQUENCE_FEATURES if c not in frame.columns]
    if not missing_hw:
        hardware_scaler = StandardScaler().fit(frame[HARDWARE_SEQUENCE_FEATURES].values)
        joblib.dump(hardware_scaler, model_dir / "hardware_scaler.pkl")
        print("Saved network, process, and hardware scaler artifacts into models/")
    else:
        print(
            f"Saved network/process scaler artifacts into models/; skipped hardware scaler, missing columns: {missing_hw}"
        )


if __name__ == "__main__":
    input_csv = "samples/training_reference.csv"
    if not Path(input_csv).exists():
        raise FileNotFoundError(
            "Missing samples/training_reference.csv. Provide real industrial baseline data before creating scalers."
        )
    train_scalers_from_csv(input_csv)
