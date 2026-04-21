"""Bounded 0-1 anomaly hints from raw industrial features."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from preprocessing.contracts import HARDWARE_SEQUENCE_FEATURES, NETWORK_FEATURES, PROCESS_FEATURES

_NETWORK_SOFT_MAX: Dict[str, float] = {
    "packet_rate": 280.0,
    "bytes_per_sec": 110_000.0,
    "avg_packet_size": 520.0,
    "tcp_syn_rate": 14.0,
    "failed_login_rate": 3.0,
    "new_connection_rate": 18.0,
    "external_ip_ratio": 0.18,
    "dns_query_rate": 14.0,
}

_PROCESS_TYPICAL_PTP: Dict[str, float] = {
    "reactor_temp_c": 1.2,
    "reactor_pressure_bar": 0.35,
    "valve_position_pct": 8.0,
    "motor_current_a": 1.2,
}

_HARDWARE_TYPICAL_PTP: Dict[str, float] = {
    "vcc_voltage_v": 0.08,
    "cpu_current_ma": 35.0,
    "clock_jitter_ns": 3.5,
    "board_temp_c": 6.0,
    "brownout_flag": 0.25,
    "reset_count_delta": 0.5,
}

_HARDWARE_CRITICAL_FLAGS: Dict[str, str] = {
    "chassis_open": "Chassis opened",
    "tamper_switch": "Tamper switch triggered",
    "jtag_active": "JTAG debug port active",
    "uart_active": "UART debug port active",
    "unexpected_usb": "Unexpected USB device attached",
    "usb_hid_burst": "USB HID burst detected",
}


def network_heuristic_anomaly(network: Dict[str, float]) -> float:
    parts: List[float] = []
    for key in NETWORK_FEATURES:
        soft = _NETWORK_SOFT_MAX[key]
        v = float(network[key])
        if v <= soft:
            parts.append(0.0)
        else:
            excess = (v - soft) / (soft + 1e-8)
            parts.append(float(np.clip(excess / 2.0, 0.0, 1.0)))
    return float(np.mean(parts))


def network_feature_contributions(network: Dict[str, float]) -> List[Dict[str, float]]:
    contributions: List[Dict[str, float]] = []
    for key in NETWORK_FEATURES:
        soft = _NETWORK_SOFT_MAX[key]
        value = float(network[key])
        if value <= soft:
            score = 0.0
        else:
            excess = (value - soft) / (soft + 1e-8)
            score = float(np.clip(excess / 2.0, 0.0, 1.0))
        contributions.append({"signal": key, "value": value, "expected_max": float(soft), "score": score})
    return sorted(contributions, key=lambda item: item["score"], reverse=True)


def process_heuristic_anomaly(process_sequence: List[Dict[str, float]]) -> float:
    if len(process_sequence) < 2:
        return 0.0
    arr = np.array([[float(row[k]) for k in PROCESS_FEATURES] for row in process_sequence], dtype=float)
    ptp = np.ptp(arr, axis=0)
    scales = np.array([_PROCESS_TYPICAL_PTP[k] for k in PROCESS_FEATURES], dtype=float)
    ratios = np.clip(ptp / (scales + 1e-8), 0.0, 4.0) / 4.0
    return float(np.mean(ratios))


def hardware_heuristic_anomaly(hardware_sequence: List[Dict[str, float]]) -> float:
    if not hardware_sequence:
        return 0.0
    if len(hardware_sequence) < 2:
        row = hardware_sequence[-1]
        brownout = float(np.clip(float(row["brownout_flag"]), 0.0, 1.0))
        resets = float(np.clip(float(row["reset_count_delta"]) / 2.0, 0.0, 1.0))
        return float(max(brownout, resets))
    arr = np.array([[float(row[k]) for k in HARDWARE_SEQUENCE_FEATURES] for row in hardware_sequence], dtype=float)
    ptp = np.ptp(arr, axis=0)
    scales = np.array([_HARDWARE_TYPICAL_PTP[k] for k in HARDWARE_SEQUENCE_FEATURES], dtype=float)
    ratios = np.clip(ptp / (scales + 1e-8), 0.0, 4.0) / 4.0
    return float(np.mean(ratios))


def hardware_rule_hits(hardware_state: Dict[str, int]) -> List[str]:
    hits: List[str] = []
    for key, label in _HARDWARE_CRITICAL_FLAGS.items():
        if int(hardware_state.get(key, 0)) == 1:
            hits.append(label)
    return hits


def hardware_rule_contributions(hardware_state: Dict[str, int]) -> List[Dict[str, int]]:
    contributions: List[Dict[str, int]] = []
    for key, label in _HARDWARE_CRITICAL_FLAGS.items():
        value = int(hardware_state.get(key, 0))
        if value == 1:
            contributions.append({"signal": key, "label": label, "value": value, "score": 1})
    return contributions


def hardware_rule_anomaly(hardware_state: Dict[str, int]) -> float:
    return 1.0 if hardware_rule_hits(hardware_state) else 0.0
