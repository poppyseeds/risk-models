from typing import Dict, List


NETWORK_FEATURES: List[str] = [
    "packet_rate",
    "bytes_per_sec",
    "avg_packet_size",
    "tcp_syn_rate",
    "failed_login_rate",
    "new_connection_rate",
    "external_ip_ratio",
    "dns_query_rate",
]

PROCESS_FEATURES: List[str] = [
    "reactor_temp_c",
    "reactor_pressure_bar",
    "valve_position_pct",
    "motor_current_a",
]

REQUIRED_KEYS = {"site_id", "asset_id", "network", "process_sequence", "timestamp"}


class PayloadValidationError(ValueError):
    pass


def _validate_network(network: Dict[str, float]) -> None:
    if not isinstance(network, dict):
        raise PayloadValidationError("network must be an object with named features")
    missing = [f for f in NETWORK_FEATURES if f not in network]
    if missing:
        raise PayloadValidationError(f"network missing features: {missing}")


def _validate_process_sequence(process_sequence: List[Dict[str, float]]) -> None:
    if not isinstance(process_sequence, list) or len(process_sequence) == 0:
        raise PayloadValidationError("process_sequence must be a non-empty list")
    for i, row in enumerate(process_sequence):
        if not isinstance(row, dict):
            raise PayloadValidationError(f"process_sequence[{i}] must be an object")
        missing = [f for f in PROCESS_FEATURES if f not in row]
        if missing:
            raise PayloadValidationError(f"process_sequence[{i}] missing features: {missing}")


def validate_payload(payload: Dict) -> Dict:
    if not isinstance(payload, dict):
        raise PayloadValidationError("payload must be a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise PayloadValidationError(f"missing required keys: {missing}")
    _validate_network(payload["network"])
    _validate_process_sequence(payload["process_sequence"])
    return payload
