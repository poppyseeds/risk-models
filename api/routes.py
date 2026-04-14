from typing import Optional

from flask import Blueprint, jsonify, request

from preprocessing.contracts import PayloadValidationError, validate_payload
from services.inference import InferenceError

_HISTORY_LIMIT_MIN = 1
_HISTORY_LIMIT_MAX = 500


def _parse_history_limit(raw: Optional[str], default: int = 20) -> Optional[int]:
    if raw is None or raw == "":
        return default
    try:
        n = int(raw, 10)
    except (TypeError, ValueError):
        return None
    if n < _HISTORY_LIMIT_MIN or n > _HISTORY_LIMIT_MAX:
        return None
    return n


def create_api(inference_service, logging_store):
    bp = Blueprint("api", __name__)

    @bp.route("/detect", methods=["POST"])
    def detect():
        try:
            payload = request.get_json(silent=True)
            payload = validate_payload(payload)
            result = inference_service.run(payload)
            logging_store.record_incident(result)
            return jsonify({"risk": result}), 200
        except PayloadValidationError as exc:
            return jsonify({"error": "invalid_payload", "detail": str(exc)}), 400
        except InferenceError as exc:
            return jsonify({"error": "inference_failure", "detail": str(exc)}), 503
        except Exception as exc:
            return jsonify({"error": "unexpected_error", "detail": str(exc)}), 500

    @bp.route("/history", methods=["GET"])
    def history():
        limit = _parse_history_limit(request.args.get("limit"))
        if limit is None:
            return (
                jsonify(
                    {
                        "error": "invalid_query",
                        "detail": f"limit must be an integer between {_HISTORY_LIMIT_MIN} and {_HISTORY_LIMIT_MAX}",
                    }
                ),
                400,
            )
        return jsonify({"history": logging_store.latest_incidents(limit=limit)}), 200

    @bp.route("/history", methods=["DELETE"])
    def history_clear():
        try:
            cleared = logging_store.clear_incidents()
            return jsonify({"ok": True, "cleared": cleared}), 200
        except Exception as exc:
            return jsonify({"error": "clear_failed", "detail": str(exc)}), 500

    @bp.route("/status", methods=["GET"])
    def status():
        return jsonify(
            {
                "system_status": "healthy" if inference_service.assets["ready"] else "degraded",
                "model_errors": inference_service.assets["errors"],
            }
        )

    return bp
