from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import settings
from preprocessing.contracts import (
    DEFAULT_HARDWARE_SEQUENCE_ROW,
    DEFAULT_HARDWARE_STATE,
    HARDWARE_SEQUENCE_FEATURES,
    NETWORK_FEATURES,
    PROCESS_FEATURES,
)
from preprocessing.transform import hardware_matrix, network_frame, process_matrix
from response.recommendations import classify_severity, recommendation_for_severity
from riskscore import FusionEngine
from services.heuristics import (
    hardware_heuristic_anomaly,
    hardware_rule_contributions,
    hardware_rule_anomaly,
    hardware_rule_hits,
    network_feature_contributions,
    network_heuristic_anomaly,
    process_heuristic_anomaly,
)
from services.model_loader import align_network_columns


class InferenceError(RuntimeError):
    pass


class InferenceService:
    """Per-(site_id, asset_id) network score windows so unrelated assets do not share fusion state."""

    _MAX_WINDOW_KEYS = 512

    def __init__(self, assets: Dict[str, Any]):
        self.assets = assets
        self.fusion = FusionEngine()
        self._windows: "OrderedDict[str, List[float]]" = OrderedDict()
        self.window_size = settings.process_window_size
        self.hardware_window_size = settings.hardware_window_size

    def _window_key(self, payload: Dict[str, Any]) -> str:
        return f"{payload['site_id']}:{payload['asset_id']}"

    def _append_network_window(self, key: str, network_score: float) -> List[float]:
        if key in self._windows:
            self._windows.move_to_end(key)
            window = self._windows[key]
        else:
            window = []
            self._windows[key] = window
            while len(self._windows) > self._MAX_WINDOW_KEYS:
                self._windows.popitem(last=False)
        window.append(network_score)
        if len(window) > self.window_size:
            del window[: len(window) - self.window_size]
        return window

    def _top_network_signals(self, network_payload: Dict[str, float]) -> List[Dict]:
        return [
            {
                "signal": item["signal"],
                "value": float(item["value"]),
                "score": float(item["score"]),
            }
            for item in network_feature_contributions(network_payload)[: settings.top_signals_count]
        ]

    def _top_process_signals(self, process_contribs: List[Dict[str, float]]) -> List[Dict]:
        return [
            {
                "signal": item["signal"],
                "value": float(item["value"]),
                "score": float(item["score"]),
            }
            for item in process_contribs[: settings.top_signals_count]
        ]

    def _top_hardware_signals(
        self,
        hardware_contribs: List[Dict[str, float]],
        hardware_rule_details: List[Dict[str, int]],
    ) -> List[Dict]:
        combined: List[Dict[str, float]] = [
            {
                "signal": item["signal"],
                "value": float(item["value"]),
                "score": float(item["score"]),
            }
            for item in hardware_contribs
        ]
        combined.extend(
            {
                "signal": item["signal"],
                "label": item["label"],
                "value": float(item["value"]),
                "score": float(item["score"]),
            }
            for item in hardware_rule_details
        )
        combined.sort(key=lambda item: float(item["score"]), reverse=True)
        return combined[: settings.top_signals_count]

    def _hardware_sequence_length(self) -> int:
        model = self.assets.get("hardware_model")
        input_shape = getattr(model, "input_shape", None)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if isinstance(input_shape, tuple) and len(input_shape) >= 2:
            sequence_length = input_shape[1]
            if isinstance(sequence_length, int) and sequence_length > 0:
                return sequence_length
        return self.hardware_window_size

    @staticmethod
    def _align_sequence_feature_count(matrix: np.ndarray, expected_features: int) -> np.ndarray:
        if matrix.shape[1] > expected_features:
            return matrix[:, :expected_features]
        if matrix.shape[1] < expected_features:
            pad = np.zeros((matrix.shape[0], expected_features - matrix.shape[1]), dtype=float)
            return np.hstack([matrix, pad])
        return matrix

    def _sequence_model_analysis(
        self,
        sequence: List[Dict[str, float]],
        window_size: int,
        feature_names: List[str],
        scaler,
        model,
        transform_fn,
        domain: str,
    ) -> tuple[float, List[Dict[str, float]]]:
        if model is None or scaler is None:
            raise InferenceError(f"{domain} inference assets are unavailable")

        try:
            matrix = transform_fn(sequence, window_size)
            expected_features = getattr(scaler, "n_features_in_", matrix.shape[1])
            matrix = self._align_sequence_feature_count(matrix, expected_features)

            scaled = scaler.transform(matrix)
            seq = np.expand_dims(scaled, axis=0)
            recon = np.asarray(model.predict(seq, verbose=0))

            if recon.shape == seq.shape:
                error = np.abs(seq - recon)
                raw_loss = float(np.mean(error))
                feature_scores = np.mean(error, axis=(0, 1))
            elif recon.ndim == 2 and recon.shape[-1] == seq.shape[-1]:
                target = seq[:, -1, :]
                error = np.abs(target - recon)
                raw_loss = float(np.mean(error))
                feature_scores = np.ravel(error[0])
            else:
                raise InferenceError(
                    f"{domain} model output shape mismatch: expected {seq.shape} or {(1, seq.shape[-1])}, got {recon.shape}"
                )

            latest_row = matrix[-1]
            contributions: List[Dict[str, float]] = []
            for idx, feature in enumerate(feature_names[: len(feature_scores)]):
                contributions.append(
                    {
                        "signal": feature,
                        "value": float(latest_row[idx]),
                        "score": float(feature_scores[idx]),
                    }
                )
            contributions.sort(key=lambda item: item["score"], reverse=True)
            return raw_loss, contributions[: settings.top_signals_count]
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"{domain} scoring failed: {exc}") from exc

    def _network_score(self, payload: Dict[str, float]) -> float:
        if self.assets["network_model"] is None or self.assets["network_scaler"] is None:
            raise InferenceError("network inference assets are unavailable")

        try:
            frame = network_frame(payload)
            expected = self.assets["network_columns"] or NETWORK_FEATURES
            frame = align_network_columns(frame, expected)
            # Use array input to avoid feature-name warnings when scaler was fit without names.
            scaled = self.assets["network_scaler"].transform(frame.to_numpy(dtype=float))
            pred = self.assets["network_model"].predict(scaled, verbose=0)
        except Exception as exc:
            raise InferenceError(f"network scoring failed: {exc}") from exc

        prob = float(np.ravel(pred)[0])
        if prob < 0.0 or prob > 1.0:
            # if model output is a logit, convert to sigmoid probability
            prob = float(1.0 / (1.0 + np.exp(-prob)))
        return prob

    def _process_analysis(self, payload: List[Dict[str, float]]) -> tuple[float, List[Dict[str, float]]]:
        """Reconstruction loss divided by threshold (or raw loss if no threshold). Unbounded; cap later."""
        loss, contributions = self._sequence_model_analysis(
            payload,
            self.window_size,
            PROCESS_FEATURES,
            self.assets["process_scaler"],
            self.assets["process_model"],
            process_matrix,
            "process",
        )

        threshold = self.assets["process_threshold"]
        if threshold is None:
            return float(loss), contributions
        t = float(np.ravel(np.asarray(threshold))[0])
        return float(loss / max(t, 1e-8)), contributions

    def _process_raw_ratio(self, payload: List[Dict[str, float]]) -> float:
        ratio, _ = self._process_analysis(payload)
        return ratio

    def _hardware_analysis(self, payload: List[Dict[str, float]]) -> tuple[Optional[float], List[Dict[str, float]]]:
        """Reconstruction loss divided by threshold; returns None when hardware model assets are unavailable."""
        if self.assets.get("hardware_model") is None or self.assets.get("hardware_scaler") is None:
            return None, []

        loss, contributions = self._sequence_model_analysis(
            payload,
            self._hardware_sequence_length(),
            HARDWARE_SEQUENCE_FEATURES,
            self.assets["hardware_scaler"],
            self.assets["hardware_model"],
            hardware_matrix,
            "hardware",
        )

        threshold = self.assets.get("hardware_threshold")
        if threshold is None:
            return float(loss), contributions
        t = float(np.ravel(np.asarray(threshold))[0])
        return float(loss / max(t, 1e-8)), contributions

    def _hardware_raw_ratio(self, payload: List[Dict[str, float]]) -> Optional[float]:
        ratio, _ = self._hardware_analysis(payload)
        return ratio

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        detected_at = datetime.now(timezone.utc).isoformat()
        hardware_sequence = payload.get("hardware_sequence") or [dict(DEFAULT_HARDWARE_SEQUENCE_ROW)]
        hardware_state = payload.get("hardware_state") or dict(DEFAULT_HARDWARE_STATE)

        net_model = float(np.clip(self._network_score(payload["network"]), 0.0, 1.0))
        net_heur = float(np.clip(network_heuristic_anomaly(payload["network"]), 0.0, 1.0))
        wn = float(np.clip(settings.score_blend_network_model, 0.0, 1.0))
        network_combined = float(np.clip(wn * net_model + (1.0 - wn) * net_heur, 0.0, 1.0))

        proc_raw, process_contribs = self._process_analysis(payload["process_sequence"])
        proc_model_01 = float(
            np.clip(proc_raw / max(settings.process_loss_saturation_mult, 1e-8), 0.0, 1.0)
        )
        proc_heur = float(np.clip(process_heuristic_anomaly(payload["process_sequence"]), 0.0, 1.0))
        wp = float(np.clip(settings.score_blend_process_model, 0.0, 1.0))
        process_combined = float(np.clip(wp * proc_model_01 + (1.0 - wp) * proc_heur, 0.0, 1.0))

        hardware_raw, hardware_contribs = self._hardware_analysis(hardware_sequence)
        hardware_model_01 = (
            0.0
            if hardware_raw is None
            else float(
                np.clip(hardware_raw / max(settings.hardware_loss_saturation_mult, 1e-8), 0.0, 1.0)
            )
        )
        hardware_heur = float(np.clip(hardware_heuristic_anomaly(hardware_sequence), 0.0, 1.0))
        hardware_rule = float(np.clip(hardware_rule_anomaly(hardware_state), 0.0, 1.0))
        wh = float(np.clip(settings.score_blend_hardware_model, 0.0, 1.0))
        hardware_combined = float(
            max(hardware_rule, np.clip(wh * hardware_model_01 + (1.0 - wh) * hardware_heur, 0.0, 1.0))
        )
        hardware_hits = hardware_rule_hits(hardware_state)
        hardware_rule_details = hardware_rule_contributions(hardware_state)

        key = self._window_key(payload)
        net_window = self._append_network_window(key, network_combined)

        fusion_out = self.fusion.fuse(net_window, process_combined, hardware_combined, hardware_hits)
        risk_score = float(max(0.0, min(1.0, fusion_out["risk_score"])))
        severity = classify_severity(risk_score)

        top_network = self._top_network_signals(payload["network"])
        top_process = self._top_process_signals(process_contribs)
        top_hardware = self._top_hardware_signals(hardware_contribs, hardware_rule_details)

        explanation = {
            "summary": fusion_out["reason"],
            "network": {
                "branch": "heuristic-rank",
                "top_factors": network_feature_contributions(payload["network"])[
                    : settings.top_signals_count
                ],
            },
            "process": {
                "branch": "reconstruction-error",
                "top_factors": process_contribs,
            },
            "hardware": {
                "branch": "reconstruction-error + rule-based flags",
                "top_factors": hardware_contribs,
                "rule_hits": hardware_rule_details,
            },
        }

        return {
            "detected_at": detected_at,
            "timestamp": payload["timestamp"],
            "site_id": payload["site_id"],
            "asset_id": payload["asset_id"],
            "system_status": "healthy" if self.assets["ready"] else "degraded",
            "fused_risk_score": risk_score,
            "network_score": network_combined,
            "process_score": process_combined,
            "hardware_score": hardware_combined,
            "score_breakdown": {
                "network_model": net_model,
                "network_heuristic": net_heur,
                "process_model_raw_ratio": proc_raw,
                "process_model_01": proc_model_01,
                "process_heuristic": proc_heur,
                "hardware_model_raw_ratio": hardware_raw,
                "hardware_model_01": hardware_model_01,
                "hardware_heuristic": hardware_heur,
                "hardware_rule": hardware_rule,
                "hardware_rule_hits": hardware_hits,
            },
            "network_normalized": fusion_out["network_norm"],
            "process_normalized": fusion_out["process_norm"],
            "hardware_normalized": fusion_out["hardware_norm"],
            "severity": severity,
            "anomaly_reason": fusion_out["reason"],
            "top_contributors": {"network": top_network, "process": top_process, "hardware": top_hardware},
            "explanation": explanation,
            "recommendation": recommendation_for_severity(severity),
            "model_status": {
                "ready": self.assets["ready"],
                "errors": self.assets["errors"],
                "domain_readiness": {
                    "network": self.assets["network_model"] is not None and self.assets["network_scaler"] is not None,
                    "process": self.assets["process_model"] is not None and self.assets["process_scaler"] is not None,
                    "hardware": self.assets.get("hardware_model") is not None
                    and self.assets.get("hardware_scaler") is not None,
                },
            },
        }
