# RAM Antivirus - Industrial Cyber Defense MVP

RAM Antivirus is a Flask-based demo for industrial cyber defense that fuses:
- Network anomaly score
- Process anomaly score
- Context-aware fusion risk score with severity and response recommendation

## Problem

Industrial environments can be attacked in two correlated ways:
- IT/OT network intrusion patterns
- Unsafe process behavior at PLC/sensor level

This MVP demonstrates a single pipeline that scores both domains and returns one explainable risk object for operators.

## Architecture

- `app.py`: Flask app bootstrap, CORS, sample file serving
- `api/routes.py`: API contract, unified detect endpoint, history/status endpoints
- `preprocessing/contracts.py`: named industrial input schema and payload validation
- `preprocessing/transform.py`: network/process shaping
- `services/model_loader.py`: safe model/scaler loading with error status
- `services/inference.py`: single inference pipeline
- `riskscore.py`: fusion engine used directly by API
- `services/logging_store.py`: SQLite incident logging for dashboard timeline
- `response/recommendations.py`: severity + recommendation mapping
- `templates/index.html`, `static/script.js`: dashboard UI
- `samples/`: demo payloads and CSV examples

## Data Contract (No anonymous f0..f40)

### Network features
- `packet_rate`
- `bytes_per_sec`
- `avg_packet_size`
- `tcp_syn_rate`
- `failed_login_rate`
- `new_connection_rate`
- `external_ip_ratio`
- `dns_query_rate`

### Process sequence features
- `reactor_temp_c`
- `reactor_pressure_bar`
- `valve_position_pct`
- `motor_current_a`

### Required payload shape
```json
{
  "site_id": "plant-a",
  "asset_id": "plc-01",
  "timestamp": "2026-04-15T09:00:00Z",
  "network": { "...named network features..." : 0.0 },
  "process_sequence": [
    { "...named process features..." : 0.0 }
  ]
}
```

## API

### `POST /api/detect`
Returns:
- `fused_risk_score`
- `network_score` and `process_score`
- normalized sub-scores
- `severity`: low/medium/high/critical
- `anomaly_reason`
- top contributing signals
- recommendation
- model status

### `GET /api/history`
Returns recent incidents from SQLite.

### `GET /api/status`
Returns model/system readiness state.

## Setup

1. Create and activate virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Add real model artifacts in `models/`:
   - `ram_model.h5`
   - `scaler.pkl`
   - `columns.pkl`
   - `lstm_model.h5`
   - `lstm_scaler.pkl`
   - `threshold.npy`
4. Run:
   - `python app.py`

## Demo Flow

1. Open dashboard at `http://127.0.0.1:5000`
2. Click one of the sample buttons:
   - normal
   - suspicious network
   - suspicious process
   - combined attack
3. Run detection.
4. Inspect:
   - live/latest detection panel
   - fused risk and sub-scores
   - anomaly reason and recommendation
   - timeline/history entries

## Model Artifact Notes

- `create_models.py` now builds scalers from `samples/training_reference.csv` (not random data).
- `create_models_v2.py` saves architecture templates only; retraining on real data is mandatory for credible results.

## Configuration

Environment variables:
- `DEBUG`
- `CORS_ORIGINS`
- `MODEL_DIR`
- `NETWORK_MODEL_PATH`
- `NETWORK_SCALER_PATH`
- `NETWORK_COLUMNS_PATH`
- `PROCESS_MODEL_PATH`
- `PROCESS_SCALER_PATH`
- `PROCESS_THRESHOLD_PATH`
- `PROCESS_WINDOW_SIZE`
- `HISTORY_DB_PATH`
- `TOP_SIGNALS_COUNT`

## Tests

Run:
- `pytest`

Coverage includes:
- payload validation
- preprocessing shape contract
- fusion scoring output contract
- API response structure and invalid payload handling

## Limitations

- Explanation layer is heuristic (top signal values), not SHAP.
- Without trained production models, outputs are pipeline-valid but not operationally calibrated.
- Current UI is MVP and not authenticated.
