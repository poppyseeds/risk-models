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

HARDWARE_SEQUENCE_FEATURES: List[str] = [
    "vcc_voltage_v",
    "cpu_current_ma",
    "clock_jitter_ns",
    "board_temp_c",
    "brownout_flag",
    "reset_count_delta",
]

HARDWARE_STATE_FEATURES: List[str] = [
    "chassis_open",
    "tamper_switch",
    "jtag_active",
    "uart_active",
    "unexpected_usb",
    "usb_hid_burst",
]

DEFAULT_HARDWARE_SEQUENCE_ROW: Dict[str, float] = {
    "vcc_voltage_v": 5.0,
    "cpu_current_ma": 180.0,
    "clock_jitter_ns": 2.0,
    "board_temp_c": 40.0,
    "brownout_flag": 0.0,
    "reset_count_delta": 0.0,
}

DEFAULT_HARDWARE_STATE: Dict[str, int] = {
    "chassis_open": 0,
    "tamper_switch": 0,
    "jtag_active": 0,
    "uart_active": 0,
    "unexpected_usb": 0,
    "usb_hid_burst": 0,
}

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


def _validate_hardware_sequence(hardware_sequence: List[Dict[str, float]]) -> None:
    if not isinstance(hardware_sequence, list) or len(hardware_sequence) == 0:
        raise PayloadValidationError("hardware_sequence must be a non-empty list")
    for i, row in enumerate(hardware_sequence):
        if not isinstance(row, dict):
            raise PayloadValidationError(f"hardware_sequence[{i}] must be an object")
        missing = [f for f in HARDWARE_SEQUENCE_FEATURES if f not in row]
        if missing:
            raise PayloadValidationError(f"hardware_sequence[{i}] missing features: {missing}")


def _validate_hardware_state(hardware_state: Dict[str, int]) -> None:
    if not isinstance(hardware_state, dict):
        raise PayloadValidationError("hardware_state must be an object with named flags")
    missing = [f for f in HARDWARE_STATE_FEATURES if f not in hardware_state]
    if missing:
        raise PayloadValidationError(f"hardware_state missing features: {missing}")


def _normalize_hardware(payload: Dict) -> Dict:
    normalized = dict(payload)
    if normalized.get("hardware_sequence") is None:
        normalized["hardware_sequence"] = [dict(DEFAULT_HARDWARE_SEQUENCE_ROW)]
    if normalized.get("hardware_state") is None:
        normalized["hardware_state"] = dict(DEFAULT_HARDWARE_STATE)
    return normalized


def validate_payload(payload: Dict) -> Dict:
    if not isinstance(payload, dict):
        raise PayloadValidationError("payload must be a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise PayloadValidationError(f"missing required keys: {missing}")
    normalized = _normalize_hardware(payload)
    _validate_network(normalized["network"])
    _validate_process_sequence(normalized["process_sequence"])
    _validate_hardware_sequence(normalized["hardware_sequence"])
    _validate_hardware_state(normalized["hardware_state"])
    return normalized
