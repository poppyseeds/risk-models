from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from config.settings import settings
from preprocessing.contracts import NETWORK_FEATURES, PROCESS_FEATURES
from preprocessing.transform import network_frame, process_matrix
from response.recommendations import classify_severity, recommendation_for_severity
from riskscore import FusionEngine
from services.heuristics import network_heuristic_anomaly, process_heuristic_anomaly
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
        sorted_items = sorted(network_payload.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        return [{"signal": k, "value": float(v)} for k, v in sorted_items[: settings.top_signals_count]]

    def _top_process_signals(self, process_sequence: List[Dict[str, float]]) -> List[Dict]:
        last = process_sequence[-1]
        sorted_items = sorted(last.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        return [{"signal": k, "value": float(v)} for k, v in sorted_items[: settings.top_signals_count]]

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

    def _process_raw_ratio(self, payload: List[Dict[str, float]]) -> float:
        """Reconstruction loss divided by threshold (or raw loss if no threshold). Unbounded; cap later."""
        if self.assets["process_model"] is None or self.assets["process_scaler"] is None:
            raise InferenceError("process inference assets are unavailable")

        try:
            matrix = process_matrix(payload, self.window_size)
            scaler = self.assets["process_scaler"]
            expected_features = getattr(scaler, "n_features_in_", matrix.shape[1])
            if matrix.shape[1] > expected_features:
                matrix = matrix[:, :expected_features]
            elif matrix.shape[1] < expected_features:
                pad = np.zeros((matrix.shape[0], expected_features - matrix.shape[1]), dtype=float)
                matrix = np.hstack([matrix, pad])

            scaled = scaler.transform(matrix)
            seq = np.expand_dims(scaled, axis=0)
            recon = self.assets["process_model"].predict(seq, verbose=0)
            recon = np.asarray(recon)

            if recon.shape == seq.shape:
                loss = float(np.mean((seq - recon) ** 2))
            elif recon.ndim == 2 and recon.shape[-1] == seq.shape[-1]:
                seq_target = seq[:, -1, :]
                loss = float(np.mean((seq_target - recon) ** 2))
            else:
                raise InferenceError(
                    f"process model output shape mismatch: expected {seq.shape} or {(1, seq.shape[-1])}, got {recon.shape}"
                )
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"process scoring failed: {exc}") from exc

        threshold = self.assets["process_threshold"]
        if threshold is None:
            return float(loss)
        t = float(np.ravel(np.asarray(threshold))[0])
        return float(loss / max(t, 1e-8))

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        detected_at = datetime.now(timezone.utc).isoformat()

        net_model = float(np.clip(self._network_score(payload["network"]), 0.0, 1.0))
        net_heur = float(np.clip(network_heuristic_anomaly(payload["network"]), 0.0, 1.0))
        wn = float(np.clip(settings.score_blend_network_model, 0.0, 1.0))
        network_combined = float(np.clip(wn * net_model + (1.0 - wn) * net_heur, 0.0, 1.0))

        proc_raw = self._process_raw_ratio(payload["process_sequence"])
        proc_model_01 = float(
            np.clip(proc_raw / max(settings.process_loss_saturation_mult, 1e-8), 0.0, 1.0)
        )
        proc_heur = float(np.clip(process_heuristic_anomaly(payload["process_sequence"]), 0.0, 1.0))
        wp = float(np.clip(settings.score_blend_process_model, 0.0, 1.0))
        process_combined = float(np.clip(wp * proc_model_01 + (1.0 - wp) * proc_heur, 0.0, 1.0))

        key = self._window_key(payload)
        net_window = self._append_network_window(key, network_combined)

        fusion_out = self.fusion.fuse(net_window, process_combined)
        risk_score = float(max(0.0, min(1.0, fusion_out["risk_score"])))
        severity = classify_severity(risk_score)

        top_network = self._top_network_signals(payload["network"])
        top_process = self._top_process_signals(payload["process_sequence"])

        return {
            "detected_at": detected_at,
            "timestamp": payload["timestamp"],
            "site_id": payload["site_id"],
            "asset_id": payload["asset_id"],
            "system_status": "healthy" if self.assets["ready"] else "degraded",
            "fused_risk_score": risk_score,
            "network_score": network_combined,
            "process_score": process_combined,
            "score_breakdown": {
                "network_model": net_model,
                "network_heuristic": net_heur,
                "process_model_raw_ratio": proc_raw,
                "process_model_01": proc_model_01,
                "process_heuristic": proc_heur,
            },
            "network_normalized": fusion_out["network_norm"],
            "process_normalized": fusion_out["process_norm"],
            "severity": severity,
            "anomaly_reason": fusion_out["reason"],
            "top_contributors": {"network": top_network, "process": top_process},
            "recommendation": recommendation_for_severity(severity),
            "model_status": {"ready": self.assets["ready"], "errors": self.assets["errors"]},
        }
