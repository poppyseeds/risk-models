"""Ensure fusion network windows are isolated per site/asset."""

import numpy as np
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
    assets["hardware_model"] = MagicMock()
    assets["hardware_model"].input_shape = (None, 512, 2)
    assets["hardware_model"].predict.side_effect = lambda seq, verbose=0: np.zeros_like(seq)
    assets["hardware_scaler"] = MagicMock()
    assets["hardware_scaler"].n_features_in_ = 2
    assets["hardware_scaler"].transform.side_effect = lambda matrix: matrix
    assets["hardware_threshold"] = None

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
        "hardware_sequence": [
            {
                "vcc_voltage_v": 5.0,
                "cpu_current_ma": 180.0,
                "clock_jitter_ns": 2.0,
                "board_temp_c": 40.0,
                "brownout_flag": 0.0,
                "reset_count_delta": 0.0,
            }
        ],
        "hardware_state": {
            "chassis_open": 0,
            "tamper_switch": 0,
            "jtag_active": 0,
            "uart_active": 0,
            "unexpected_usb": 0,
            "usb_hid_burst": 0,
        },
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


def test_hardware_sequence_matches_model_input_length():
    assets = _minimal_assets()
    assets["network_scaler"].transform.return_value = [[0.0] * 8]
    assets["network_model"].predict.return_value = [[0.5]]
    assets["process_scaler"].n_features_in_ = 4
    assets["process_scaler"].transform.return_value = [[0.0] * 4] * 10
    assets["process_model"].predict.return_value = [[[0.0] * 4] * 10]
    assets["hardware_model"] = MagicMock()
    assets["hardware_model"].input_shape = (None, 512, 2)

    def _predict(seq, verbose=0):
        assert seq.shape == (1, 512, 2)
        return np.zeros_like(seq)

    assets["hardware_model"].predict.side_effect = _predict
    assets["hardware_scaler"] = MagicMock()
    assets["hardware_scaler"].n_features_in_ = 2
    assets["hardware_scaler"].transform.side_effect = lambda matrix: matrix[:, :2]
    assets["hardware_threshold"] = None

    svc = InferenceService(assets)
    payload = {
        "site_id": "s1",
        "asset_id": "plc-a",
        "timestamp": "t",
        "network": {k: 1.0 for k in [
            "packet_rate", "bytes_per_sec", "avg_packet_size", "tcp_syn_rate",
            "failed_login_rate", "new_connection_rate", "external_ip_ratio", "dns_query_rate",
        ]},
        "process_sequence": [
            {"reactor_temp_c": 72, "reactor_pressure_bar": 8, "valve_position_pct": 44, "motor_current_a": 12}
        ],
        "hardware_sequence": [
            {
                "vcc_voltage_v": 5.0,
                "cpu_current_ma": 180.0,
                "clock_jitter_ns": 2.0,
                "board_temp_c": 40.0,
                "brownout_flag": 0.0,
                "reset_count_delta": 0.0,
            }
        ],
        "hardware_state": {
            "chassis_open": 0,
            "tamper_switch": 0,
            "jtag_active": 0,
            "uart_active": 0,
            "unexpected_usb": 0,
            "usb_hid_burst": 0,
        },
    }

    result = svc.run(payload)
    assert "explanation" in result
    assert result["explanation"]["hardware"]["rule_hits"] == []


def test_explanation_includes_hardware_rule_hits():
    assets = _minimal_assets()
    assets["network_scaler"].transform.return_value = [[0.0] * 8]
    assets["network_model"].predict.return_value = [[0.5]]
    assets["process_scaler"].n_features_in_ = 4
    assets["process_scaler"].transform.return_value = [[0.0] * 4] * 10
    assets["process_model"].predict.return_value = [[[0.0] * 4] * 10]
    assets["hardware_model"] = MagicMock()
    assets["hardware_model"].input_shape = (None, 512, 2)
    assets["hardware_model"].predict.side_effect = lambda seq, verbose=0: np.zeros_like(seq)
    assets["hardware_scaler"] = MagicMock()
    assets["hardware_scaler"].n_features_in_ = 2
    assets["hardware_scaler"].transform.side_effect = lambda matrix: matrix[:, :2]
    assets["hardware_threshold"] = None

    svc = InferenceService(assets)
    payload = {
        "site_id": "s1",
        "asset_id": "plc-a",
        "timestamp": "t",
        "network": {k: 1.0 for k in [
            "packet_rate", "bytes_per_sec", "avg_packet_size", "tcp_syn_rate",
            "failed_login_rate", "new_connection_rate", "external_ip_ratio", "dns_query_rate",
        ]},
        "process_sequence": [
            {"reactor_temp_c": 72, "reactor_pressure_bar": 8, "valve_position_pct": 44, "motor_current_a": 12}
        ],
        "hardware_sequence": [
            {
                "vcc_voltage_v": 5.0,
                "cpu_current_ma": 180.0,
                "clock_jitter_ns": 2.0,
                "board_temp_c": 40.0,
                "brownout_flag": 0.0,
                "reset_count_delta": 0.0,
            }
        ],
        "hardware_state": {
            "chassis_open": 1,
            "tamper_switch": 1,
            "jtag_active": 1,
            "uart_active": 0,
            "unexpected_usb": 0,
            "usb_hid_burst": 0,
        },
    }

    result = svc.run(payload)
    rule_signals = {item["signal"] for item in result["explanation"]["hardware"]["rule_hits"]}
    assert rule_signals == {"chassis_open", "tamper_switch", "jtag_active"}
    assert result["explanation"]["summary"]


def test_top_contributors_use_anomaly_scores_not_raw_magnitude():
    assets = _minimal_assets()
    assets["network_scaler"].transform.return_value = [[0.0] * 8]
    assets["network_model"].predict.return_value = [[0.1]]
    assets["process_scaler"].n_features_in_ = 4
    assets["process_scaler"].transform.return_value = [[0.0] * 4] * 10
    assets["process_model"].predict.return_value = [[[0.0] * 4] * 10]
    assets["hardware_model"] = MagicMock()
    assets["hardware_model"].input_shape = (None, 10, 6)
    assets["hardware_model"].predict.side_effect = lambda seq, verbose=0: np.zeros_like(seq)
    assets["hardware_scaler"] = MagicMock()
    assets["hardware_scaler"].n_features_in_ = 6
    assets["hardware_scaler"].transform.side_effect = lambda matrix: matrix
    assets["hardware_threshold"] = None

    svc = InferenceService(assets)
    payload = {
        "site_id": "s1",
        "asset_id": "plc-a",
        "timestamp": "2026-04-22T00:00:00Z",
        "network": {
            "packet_rate": 600.0,
            "bytes_per_sec": 100000.0,
            "avg_packet_size": 300.0,
            "tcp_syn_rate": 4.0,
            "failed_login_rate": 0.0,
            "new_connection_rate": 4.0,
            "external_ip_ratio": 0.04,
            "dns_query_rate": 2.0,
        },
        "process_sequence": [
            {"reactor_temp_c": 72, "reactor_pressure_bar": 8, "valve_position_pct": 44, "motor_current_a": 12}
        ],
        "hardware_sequence": [
            {
                "vcc_voltage_v": 5.0,
                "cpu_current_ma": 180.0,
                "clock_jitter_ns": 2.0,
                "board_temp_c": 40.0,
                "brownout_flag": 0.0,
                "reset_count_delta": 0.0,
            }
        ],
        "hardware_state": {
            "chassis_open": 0,
            "tamper_switch": 0,
            "jtag_active": 0,
            "uart_active": 0,
            "unexpected_usb": 0,
            "usb_hid_burst": 0,
        },
    }

    result = svc.run(payload)
    assert result["top_contributors"]["network"][0]["signal"] == "packet_rate"
    assert result["top_contributors"]["network"][0]["score"] > 0.0
