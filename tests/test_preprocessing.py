import pytest

from preprocessing.contracts import PayloadValidationError, validate_payload
from preprocessing.transform import process_matrix


def _payload():
    return {
        "site_id": "plant-a",
        "asset_id": "plc-01",
        "timestamp": "2026-04-15T00:00:00Z",
        "network": {
            "packet_rate": 100,
            "bytes_per_sec": 1000,
            "avg_packet_size": 100,
            "tcp_syn_rate": 1,
            "failed_login_rate": 0,
            "new_connection_rate": 2,
            "external_ip_ratio": 0.1,
            "dns_query_rate": 3,
        },
        "process_sequence": [
            {
                "reactor_temp_c": 72,
                "reactor_pressure_bar": 8.2,
                "valve_position_pct": 44,
                "motor_current_a": 12.2,
            }
        ],
    }


def test_validate_payload_success():
    validated = validate_payload(_payload())
    assert validated["site_id"] == "plant-a"


def test_validate_payload_missing_key():
    bad = _payload()
    bad.pop("network")
    with pytest.raises(PayloadValidationError):
        validate_payload(bad)


def test_validate_payload_rejects_non_numeric_feature():
    bad = _payload()
    bad["process_sequence"][0]["motor_current_a"] = "oops"
    with pytest.raises(PayloadValidationError):
        validate_payload(bad)


def test_validate_payload_rejects_bad_timestamp():
    bad = _payload()
    bad["timestamp"] = "not-a-timestamp"
    with pytest.raises(PayloadValidationError):
        validate_payload(bad)


def test_process_matrix_pads_to_window():
    matrix = process_matrix(_payload()["process_sequence"], window_size=4)
    assert matrix.shape == (4, 4)
