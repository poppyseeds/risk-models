"""Ensure fusion network windows are isolated per site/asset."""

from unittest.mock import MagicMock

from services.inference import InferenceService


def _minimal_assets():
    return {
        "network_model": MagicMock(),
        "network_scaler": MagicMock(),
        "network_columns": None,
        "process_model": MagicMock(),
        "process_scaler": MagicMock(),
        "process_threshold": None,
        "ready": True,
        "errors": [],
    }


def test_network_windows_are_per_site_asset():
    assets = _minimal_assets()
    assets["network_scaler"].transform.return_value = [[0.0] * 8]
    assets["network_model"].predict.return_value = [[0.5]]
    assets["process_scaler"].n_features_in_ = 4
    assets["process_scaler"].transform.return_value = [[0.0] * 4] * 10
    assets["process_model"].predict.return_value = [[[0.0] * 4] * 10]

    svc = InferenceService(assets)
    base = {
        "timestamp": "t",
        "network": {k: 1.0 for k in [
            "packet_rate", "bytes_per_sec", "avg_packet_size", "tcp_syn_rate",
            "failed_login_rate", "new_connection_rate", "external_ip_ratio", "dns_query_rate",
        ]},
        "process_sequence": [
            {"reactor_temp_c": 72, "reactor_pressure_bar": 8, "valve_position_pct": 44, "motor_current_a": 12}
        ],
    }

    a = {**base, "site_id": "s1", "asset_id": "plc-a"}
    b = {**base, "site_id": "s1", "asset_id": "plc-b"}

    svc.run(a)
    svc.run(a)
    assert "s1:plc-a" in svc._windows
    assert "s1:plc-b" not in svc._windows
    assert len(svc._windows["s1:plc-a"]) == 2

    svc.run(b)
    assert len(svc._windows["s1:plc-a"]) == 2
    assert len(svc._windows["s1:plc-b"]) == 1
