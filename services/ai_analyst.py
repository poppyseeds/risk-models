import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib import error, request

from config.settings import settings


class AIAnalystService:
    def __init__(self):
        self.enabled = settings.ai_analyst_enabled
        self.provider = "ollama"
        self.model = settings.ai_analyst_model
        self.base_url = settings.ai_analyst_base_url.rstrip("/")
        self.timeout_sec = settings.ai_analyst_timeout_sec

    def status(self) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "ready": False,
                "provider": self.provider,
                "model": self.model,
                "detail": "AI analyst is disabled by configuration",
                "available_models": [],
            }

        try:
            tags = self._request_json("GET", "/api/tags")
            models = [item.get("name", "") for item in tags.get("models", []) if item.get("name")]
        except Exception as exc:
            return {
                "enabled": True,
                "ready": False,
                "provider": self.provider,
                "model": self.model,
                "detail": f"Could not reach Ollama at {self.base_url}: {exc}",
                "available_models": [],
            }

        if self.model not in models:
            return {
                "enabled": True,
                "ready": False,
                "provider": self.provider,
                "model": self.model,
                "detail": f"Configured Ollama model '{self.model}' is not installed",
                "available_models": models,
            }

        return {
            "enabled": True,
            "ready": True,
            "provider": self.provider,
            "model": self.model,
            "detail": None,
            "available_models": models,
        }

    def analyze(self, payload: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
        status = self.status()
        if not status["ready"]:
            return {
                "status": "unavailable",
                "provider": self.provider,
                "model": self.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": "Local AI analyst unavailable",
                "what_happened": "The local AI analyst could not run for this detection.",
                "where_it_happened": f"{risk.get('site_id', '-')}/{risk.get('asset_id', '-')}",
                "why_it_happened": status.get("detail") or "The local AI model is not ready",
                "evidence": [],
                "immediate_actions": [risk.get("recommendation", "Inspect the affected asset manually.")],
                "recovery_plan": ["Make sure the configured Ollama model is installed and retry the detection."],
                "confidence": "low",
            }

        try:
            body = {
                "model": self.model,
                "system": self._system_prompt(),
                "prompt": self._build_user_input(payload, risk),
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.2,
                },
            }
            response = self._request_json("POST", "/api/generate", body)
            raw = str(response.get("response") or "").strip()
            parsed = json.loads(raw)
            return self._normalize_result(parsed, risk)
        except Exception as exc:
            return {
                "status": "error",
                "provider": self.provider,
                "model": self.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": "Local AI analyst request failed",
                "what_happened": risk.get("anomaly_reason", "Detection completed but the local AI analyst failed."),
                "where_it_happened": f"{risk.get('site_id', '-')}/{risk.get('asset_id', '-')}",
                "why_it_happened": f"Local AI analyst error: {exc}",
                "evidence": [],
                "immediate_actions": [risk.get("recommendation", "Inspect the affected asset manually.")],
                "recovery_plan": ["Check the local Ollama service and model configuration, then retry."],
                "confidence": "low",
            }

    def _request_json(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

    def _system_prompt(self) -> str:
        return (
            "You are an industrial cyber incident analyst running locally. "
            "Use only the provided payload and detector evidence. "
            "Do not invent unsupported causes. "
            "If the evidence is incomplete, say that clearly. "
            "Return valid JSON only with these exact keys: "
            "summary, what_happened, where_it_happened, why_it_happened, evidence, immediate_actions, recovery_plan, confidence. "
            "evidence must be an array of short strings. "
            "immediate_actions and recovery_plan must be arrays of short strings. "
            "confidence must be one of low, medium, or high."
        )

    def _build_user_input(self, payload: Dict[str, Any], risk: Dict[str, Any]) -> str:
        context = {
            "site_id": payload.get("site_id"),
            "asset_id": payload.get("asset_id"),
            "timestamp": payload.get("timestamp"),
            "network": payload.get("network"),
            "process_sequence": payload.get("process_sequence"),
            "hardware_sequence": payload.get("hardware_sequence"),
            "hardware_state": payload.get("hardware_state"),
            "detector_output": {
                "severity": risk.get("severity"),
                "anomaly_reason": risk.get("anomaly_reason"),
                "fused_risk_score": risk.get("fused_risk_score"),
                "network_score": risk.get("network_score"),
                "process_score": risk.get("process_score"),
                "hardware_score": risk.get("hardware_score"),
                "top_contributors": risk.get("top_contributors"),
                "score_breakdown": risk.get("score_breakdown"),
                "explanation": risk.get("explanation"),
                "recommendation": risk.get("recommendation"),
            },
        }
        return (
            "Analyze this industrial cyber incident and explain where, what, and why the problem arose. "
            "Then provide immediate actions and a short recovery plan for the current situation. "
            "Return JSON only.\n\n"
            f"{json.dumps(context, indent=2)}"
        )

    def _normalize_result(self, parsed: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
        def _list_of_strings(value: Any, fallback: List[str]) -> List[str]:
            if not isinstance(value, list):
                return fallback
            items = [str(item).strip() for item in value if str(item).strip()]
            return items or fallback

        confidence = str(parsed.get("confidence", "medium")).lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"

        return {
            "status": "available",
            "provider": self.provider,
            "model": self.model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": str(parsed.get("summary") or risk.get("anomaly_reason") or "Incident analysis available"),
            "what_happened": str(parsed.get("what_happened") or risk.get("anomaly_reason") or "An anomaly was detected."),
            "where_it_happened": str(
                parsed.get("where_it_happened") or f"{risk.get('site_id', '-')}/{risk.get('asset_id', '-')}"
            ),
            "why_it_happened": str(
                parsed.get("why_it_happened")
                or "The likely cause was inferred from the detector's top signals and rule hits."
            ),
            "evidence": _list_of_strings(parsed.get("evidence"), [risk.get("anomaly_reason", "Detector evidence available")]),
            "immediate_actions": _list_of_strings(
                parsed.get("immediate_actions"),
                [risk.get("recommendation", "Inspect the affected asset manually.")],
            ),
            "recovery_plan": _list_of_strings(
                parsed.get("recovery_plan"),
                ["Review the highest-scoring contributors and restore the asset to a known-good state."],
            ),
            "confidence": confidence,
        }
