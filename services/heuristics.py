"""Bounded0–1 anomaly hints from raw industrial features (complements NN scores for credible demos)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from preprocessing.contracts import NETWORK_FEATURES, PROCESS_FEATURES

# Soft upper envelope for “steady-state” OT network; exceed → ramps toward 1.0
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

# Typical peak-to-peak over a short window under normal control (used to score process drift)
_PROCESS_TYPICAL_PTP: Dict[str, float] = {
    "reactor_temp_c": 1.2,
    "reactor_pressure_bar": 0.35,
    "valve_position_pct": 8.0,
    "motor_current_a": 1.2,
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


def process_heuristic_anomaly(process_sequence: List[Dict[str, float]]) -> float:
    if len(process_sequence) < 2:
        return 0.0
    arr = np.array([[float(row[k]) for k in PROCESS_FEATURES] for row in process_sequence], dtype=float)
    ptp = np.ptp(arr, axis=0)
    scales = np.array([_PROCESS_TYPICAL_PTP[k] for k in PROCESS_FEATURES], dtype=float)
    ratios = np.clip(ptp / (scales + 1e-8), 0.0, 4.0) / 4.0
    return float(np.mean(ratios))
