# RAM Antivirus - Industrial Cyber Defense MVP

RAM Antivirus is a Flask-based industrial cyber defense demo that fuses three domains:

- Network anomaly scoring
- Process anomaly scoring
- Hardware anomaly and tamper scoring

The system returns one explainable fused risk object with severity, reason, and recommendation.

## What It Detects

- IT/OT network intrusion behavior
- Unsafe or manipulated process behavior on PLC/sensor loops
- Hardware-side compromise signals such as chassis tamper, debug-port activation, or rogue USB devices

## Architecture

- `app.py`: Flask app bootstrap, CORS, dashboard route, sample serving
- `api/routes.py`: API endpoints (`/api/detect`, `/api/history`, `/api/status`)
- `preprocessing/contracts.py`: request validation and payload contract
- `preprocessing/transform.py`: network/process/hardware transforms
- `services/model_loader.py`: model/scaler/threshold loading with readiness errors
- `services/heuristics.py`: domain heuristics and hardware hard-rule triggers
- `services/inference.py`: end-to-end scoring pipeline and fusion input preparation
- `riskscore.py`: final fusion logic and override behavior
- `services/logging_store.py`: SQLite incident history
- `templates/index.html`, `static/script.js`: dashboard UI
- `samples/`: demo payloads including `hardware_tamper.json`

## Payload Contract

Required top-level fields:

- `site_id`
- `asset_id`
- `timestamp`
- `network`
- `process_sequence`
- `hardware_sequence`
- `hardware_state`

`hardware_sequence` holds analog telemetry windows (voltage/current/timing/etc.) and `hardware_state` holds discrete tamper facts (`chassis_open`, `jtag_active`, `uart_active`, `unexpected_usb`, etc.).

Use `samples/hardware_tamper.json` as the reference shape.

## API

### `POST /api/detect`

Returns:

- `fused_risk_score`
- `network_score`, `process_score`, `hardware_score`
- normalized scores
- `severity` (`low`, `medium`, `high`, `critical`)
- `anomaly_reason` (includes hardware-critical reasons when applicable)
- `top_contributors` for all domains
- `recommendation`
- `model_status` and per-domain readiness

### `GET /api/history`

Returns recent incidents from SQLite.

### `DELETE /api/history`

Clears timeline history.

### `GET /api/status`

Returns system/model readiness and load errors.

## Setup and Run (Windows / GitHub Friendly)

### 1) Clone

```bash
git clone <your-repo-url>
cd ram-antivirus
```

### 2) Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Place model artifacts in `models/`

Network:

- `ram_model.h5`
- `scaler.pkl`
- `columns.pkl`

Process:

- `lstm_model.h5`
- `lstm_scaler.pkl`
- `threshold.npy`

Hardware:

- `hardware_model.keras` (preferred)
- `hardware_scaler.pkl`
- `hardware_threshold.npy`

Note: hardware loader supports `.keras` by default. If your config still points to `.h5`, the loader also tries the `.keras` file automatically.

### 5) Run the app

```bash
python app.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 6) Optional: run tests

```bash
python -m pytest -q
```

## Dashboard Demo Flow

1. Open the dashboard.
2. Load a sample (`normal`, `suspicious_network`, `suspicious_process`, `combined_attack`, `hardware_tamper`).
3. Click **Run detection**.
4. Inspect fused risk, domain scores, reason text, and timeline.

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
- `HARDWARE_MODEL_PATH`
- `HARDWARE_SCALER_PATH`
- `HARDWARE_THRESHOLD_PATH`
- `HARDWARE_WINDOW_SIZE`
- `SCORE_BLEND_NETWORK_MODEL`
- `SCORE_BLEND_PROCESS_MODEL`
- `SCORE_BLEND_HARDWARE_MODEL`
- `PROCESS_LOSS_SATURATION_MULT`
- `HARDWARE_LOSS_SATURATION_MULT`
- `HISTORY_DB_PATH`
- `TOP_SIGNALS_COUNT`

## Push to GitHub

If you want to push your local changes:

```bash
git add .
git commit -m "Integrate hardware anomaly detection and update docs"
git push
```

If this is a new branch:

```bash
git push -u origin <branch-name>
```

## Limitations

- Explanations are heuristic/top-signal based, not SHAP/feature-attribution grade.
- Model quality depends entirely on real baseline and attack datasets.
- This is an MVP dashboard (no auth/SSO/hard multi-tenant controls yet).
